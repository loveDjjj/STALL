"""固定观察、共享状态更新；计算窗口之间不创建伪连续帧。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ('looped', 'untied', 'no_refresh', 'spatial_only', 'state_only', 'no_time_position')


def position_encoding(t, side, width, device, dtype, temporal=True):
    """独立时间/行/列正弦位置，不使用标签、域名或视频长度描述。"""
    axes = torch.meshgrid(torch.arange(t, device=device), torch.arange(side, device=device),
                          torch.arange(side, device=device), indexing='ij')
    bands = math.ceil(width / 6)
    freq = torch.exp(-math.log(10000) * torch.arange(bands, device=device) / max(bands - 1, 1))
    components=[]
    for i,a in enumerate(axes):
        angle=a[...,None]*freq
        components.extend((angle.sin(),angle.cos()) if temporal or i else
                          (torch.zeros_like(angle),torch.zeros_like(angle)))
    p = torch.cat(components, -1)
    return p.reshape(t, side * side, -1)[..., :width].to(dtype)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('隐藏维度必须能被头数整除')
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def prepare(self, memory):
        k, v = self.kv(memory).chunk(2, -1)
        return tuple(x.unflatten(-1, (self.heads, -1)).transpose(-3, -2) for x in (k, v))

    def forward(self, query, memory=None, kv=None, mask=None):
        q = self.q(query).unflatten(-1, (self.heads, -1)).transpose(-3, -2)
        k, v = self.prepare(memory) if kv is None else kv
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.out(y.transpose(-3, -2).flatten(-2))


def tiles(x, tile, temporal=True):
    """2×2空间组内读取全窗口时间；空间层负责跨组交互。"""
    b, t, n, d = x.shape
    s = math.isqrt(n)
    if s * s != n or s % tile:
        raise ValueError('Patch网格与读取组尺寸不兼容')
    g = s // tile
    z = x.reshape(b, t, g, tile, g, tile, d)
    if temporal:
        return z.permute(0, 2, 4, 1, 3, 5, 6).reshape(b * g * g, t * tile * tile, d)
    return z.permute(0, 1, 2, 4, 3, 5, 6).reshape(b * t * g * g, tile * tile, d)


def untiles(z, shape, tile, temporal=True):
    b, t, n, d = shape
    g = math.isqrt(n) // tile
    if temporal:
        x = z.reshape(b, g, g, t, tile, tile, d).permute(0, 3, 1, 4, 2, 5, 6)
    else:
        x = z.reshape(b, t, g, g, tile, tile, d).permute(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(shape)


class RelationBlock(nn.Module):
    def __init__(self, width, heads, tile=2, mlp_ratio=2, temporal=True):
        super().__init__()
        self.tile, self.temporal = tile, temporal
        self.snorm = nn.LayerNorm(width)
        self.qnorm = nn.LayerNorm(width)
        self.enorm = nn.LayerNorm(width)
        self.fnorm = nn.LayerNorm(width)
        self.spatial = Attention(width, heads)
        self.read = Attention(width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, width * mlp_ratio), nn.GELU(),
                                 nn.Linear(width * mlp_ratio, width))
        self.gates = nn.Parameter(torch.full((3, width), .1))

    def prepare(self, evidence, valid):
        memory = tiles(self.enorm(evidence), self.tile, self.temporal)
        mask = tiles(valid[..., None, None].expand(*evidence.shape[:-1], 1), self.tile,
                     self.temporal).squeeze(-1).bool()
        return self.read.prepare(memory), mask[:, None, None, :]

    def forward(self, h, valid, prepared=None):
        b, t, n, d = h.shape
        z = self.snorm(h).reshape(b * t, n, d)
        h = h + self.gates[0] * self.spatial(z, z).reshape_as(h)
        q = tiles(self.qnorm(h), self.tile, self.temporal)
        if prepared is None:
            prepared = self.prepare(h, valid)
        kv, mask = prepared
        h = h + self.gates[1] * untiles(self.read(q, kv=kv, mask=mask), h.shape,
                                      self.tile, self.temporal)
        h = h + self.gates[2] * self.ffn(self.fnorm(h))
        return h.masked_fill(~valid[..., None, None], 0)


class LoopedDetector(nn.Module):
    def __init__(self, input_dim=1024, width=256, heads=4, loops=4, tile=2,
                 mlp_ratio=2, variant='looped'):
        super().__init__()
        if variant not in VARIANTS or loops < 1:
            raise ValueError('未知变体或非法循环数')
        self.variant, self.loops, self.width = variant, loops, width
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.blocks = nn.ModuleList([RelationBlock(width, heads, tile, mlp_ratio,
                                   temporal=variant != 'spatial_only')
                                   for _ in range(loops if variant == 'untied' else 1)])
        # state_only将第二份观察置为同一状态，保持读出参数预算与完整模型相同。
        self.readout = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width),
                                     nn.GELU(), nn.Linear(width, 1))

    def forward(self, patches, frame_valid, window_owner=None, videos=None):
        """输入为展平的有效窗口[W,T,N,D]；掩码为[W,T]。"""
        if patches.ndim != 4 or frame_valid.shape != patches.shape[:2]:
            raise ValueError('输入形状/掩码不一致')
        if not frame_valid.any(1).all():
            raise ValueError('不能将空窗口当真实观察')
        x = patches.masked_fill(~frame_valid[..., None, None], 0)
        e = self.projection(x)
        e = e + position_encoding(e.shape[1], math.isqrt(e.shape[2]), self.width, e.device, e.dtype,
                                  temporal=self.variant != 'no_time_position')
        e = e.masked_fill(~frame_valid[..., None, None], 0)
        h = e
        prepared = None
        if self.variant != 'no_refresh' and len(self.blocks) == 1:
            prepared = self.blocks[0].prepare(e, frame_valid)
        for r in range(self.loops):
            block = self.blocks[r] if len(self.blocks) > 1 else self.blocks[0]
            memory = block.prepare(e, frame_valid) if len(self.blocks) > 1 else prepared
            h = block(h, frame_valid, memory)
        original = h if self.variant == 'state_only' else e
        # BF16用于矩阵运算，统计聚合保持FP32，避免窗口/视频分数额外量化同分。
        token_logits = self.readout(torch.cat((original, h), -1)).squeeze(-1).float()
        window_logits = (token_logits * frame_valid[..., None]).sum((1, 2)) / (
            frame_valid.sum(1) * patches.shape[2])
        if window_owner is None:
            return window_logits
        if videos is None or window_owner.shape != window_logits.shape:
            raise ValueError('视频/窗口归属缺失')
        sums = torch.zeros(videos, device=window_logits.device, dtype=window_logits.dtype)
        counts = torch.zeros_like(sums)
        sums.index_add_(0, window_owner, window_logits)
        counts.index_add_(0, window_owner, torch.ones_like(window_logits))
        if (counts == 0).any():
            raise ValueError('批次包含无观察视频')
        return sums / counts
