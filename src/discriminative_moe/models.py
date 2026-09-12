"""相同输入的线性、MLP、均匀多头与判别MoE。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, dimension, kind='mlp', experts=1, hidden_budget=128, dropout=.1):
        super().__init__()
        if kind not in ('linear','mlp','uniform','moe') or experts<1:
            raise ValueError('分类器配置非法')
        self.kind=kind
        self.count=experts if kind in ('uniform','moe') else 1
        width=max(1,hidden_budget//self.count)
        self.experts=nn.ModuleList([
            nn.Linear(dimension,1) if kind=='linear' else
            nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,1))
            for _ in range(self.count)])
        self.router=nn.Linear(dimension,self.count) if kind=='moe' else None
        if self.router is not None:
            # 从均匀路由开始，但不同专家独立初始化，梯度可打破对称。
            nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)

    def forward(self, features, mask):
        if features.ndim!=3 or mask.shape!=features.shape[:2] or not mask.any(dim=1).all():
            raise ValueError('需要有效窗口mask，不能给视频补伪观察')
        logits=torch.cat([e(features) for e in self.experts],dim=-1)
        log_gate=F.log_softmax(self.router(features),dim=-1) if self.router is not None else torch.full_like(logits,-math.log(self.count))
        # 在log域构造概率混合，随后视频内部按有效窗口等权。
        log_fake=torch.logsumexp(log_gate+F.logsigmoid(logits),dim=-1)
        log_real=torch.logsumexp(log_gate+F.logsigmoid(-logits),dim=-1)
        normalizer=mask.sum(1).log()
        vf=torch.logsumexp(log_fake.masked_fill(~mask,-torch.inf),dim=1)-normalizer
        vr=torch.logsumexp(log_real.masked_fill(~mask,-torch.inf),dim=1)-normalizer
        gate=log_gate.exp()
        video_gate=(gate*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        mean_gate=video_gate.mean(0)
        balance=(self.count*(mean_gate.square().sum())-1) if self.router is not None else features.new_zeros(())
        return dict(log_fake=vf,log_real=vr,p_fake=vf.exp(),gate=video_gate,window_gate=gate,
                    expert_probability=logits.sigmoid(),balance=balance)

    @staticmethod
    def loss(output, fake_labels, balance_weight=.01):
        y=fake_labels.to(output['log_fake'].dtype)
        bce=-(y*output['log_fake']+(1-y)*output['log_real']).mean()
        return bce+balance_weight*output['balance']


def training_standardizer(features, mask):
    """只传训练行；每视频窗口总权重为1，不按K多寡改变拟合权重。"""
    if not mask.any(1).all():raise ValueError('空训练视频')
    weights=mask.to(torch.float64)/mask.sum(1,keepdim=True)
    x=features.to(torch.float64)
    mean=(x*weights[...,None]).sum((0,1))/len(x)
    variance=((x-mean).square()*weights[...,None]).sum((0,1))/len(x)
    scale=variance.sqrt().clamp_min(1e-6)
    return mean.float(),scale.float()
