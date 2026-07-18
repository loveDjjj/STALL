import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DINO_V3_MODEL_NAME = "dinov3_vitl16"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认路径；可通过 DINO_V3_REPO_DIR / DINO_V3_WEIGHTS 环境变量覆盖。
DINO_V3_REPO_DIR = os.getenv(
    "DINO_V3_REPO_DIR", os.path.join(_REPO_ROOT, "dinov3")
)
DINO_V3_WEIGHTS = os.getenv(
    "DINO_V3_WEIGHTS",
    os.path.join(_REPO_ROOT, "dinov3", "weights", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
)

DINOV3_GITHUB_URL = "https://github.com/facebookresearch/dinov3"

AGG_STR2FN = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}


# ─────────────────────────────────────────────────────────────────────────────
# DINOv3
# ─────────────────────────────────────────────────────────────────────────────

def create_dinov3_transform(resize_size: int = 224):
    """DINOv3 LVD-1689M 预训练模型使用的标准 ImageNet eval transform。"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_dinov3_model(device: str, repo_dir: str = None, weights: str = None):
    """从本地 dinov3 repo clone 加载 DINOv3 ViT-L/16 模型。

    Args:
        device:    目标设备字符串（"cuda" / "cpu"）。
        repo_dir:  本地 dinov3 repo clone 路径；默认使用 DINO_V3_REPO_DIR。
        weights:   .pth 权重路径；默认使用 DINO_V3_WEIGHTS。

    返回：
        (model, transform)
    """
    repo_dir = repo_dir or DINO_V3_REPO_DIR
    weights = weights or DINO_V3_WEIGHTS

    if not os.path.exists(repo_dir):
        raise ValueError(
            f"未在 '{repo_dir}' 找到 DINOv3 repo。\n"
            f"请从 {DINOV3_GITHUB_URL} clone，并把权重放到 weights/。\n"
            f"可通过 --dino-repo 或 DINO_V3_REPO_DIR 环境变量覆盖路径。"
        )
    if not os.path.exists(weights):
        raise ValueError(
            f"未在 '{weights}' 找到 DINOv3 权重。\n"
            f"请从 {DINOV3_GITHUB_URL} 下载 dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth。\n"
            f"可通过 --dino-weights 或 DINO_V3_WEIGHTS 环境变量覆盖路径。"
        )

    # 直接导入 backbone，而不通过 torch.hub.load；后者会导入 hubconf.py，
    # 并带入依赖 torchvision.transforms.v2（torchvision >= 0.15）的
    # detectors/segmentors。
    import sys
    sys.path.insert(0, repo_dir)
    try:
        from dinov3.hub.backbones import dinov3_vitl16
        model = dinov3_vitl16(weights=weights)
    finally:
        if repo_dir in sys.path:
            sys.path.remove(repo_dir)
    model = model.to(device).eval()
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    transform = create_dinov3_transform()
    if device == "cuda":
        print(f"DINOv3 模型已加载到 {device} ({torch.cuda.device_count()} 个可见 GPU)")
    else:
        print(f"DINOv3 模型已加载到 {device}")
    return model, transform


# ─────────────────────────────────────────────────────────────────────────────
# 数学工具
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood(array):
    """N(0,I) 下的高斯 log-likelihood。输入 [N, T, D]，返回 [N, T]。"""
    D = array.shape[-1]
    return -0.5 * (D * np.log(2.0 * np.pi) + (array ** 2).sum(axis=-1))


def whitening_transform(emb, mu, W):
    """应用预拟合白化：(emb - mu) @ W。"""
    return np.matmul(emb - mu, W)


def raw_emb_to_log_likelihoods(raw_emb, mu, W, preprocess_fn=None):
    if preprocess_fn is not None:
        raw_emb = preprocess_fn(raw_emb)
    return log_likelihood(whitening_transform(raw_emb, mu, W))


def diff_vec(arr):
    """连续帧差分。[N, T, D] → [N, T-1, D]。"""
    return arr[:, 1:, :] - arr[:, :-1, :]


def l2_normalize(arr):
    arr = arr.astype(np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return arr / norm


def diff_normalized_embeddings(arr):
    """L2-normalized 连续帧 embedding 差分。[N,T,D] → [N,T-1,D]。"""
    return l2_normalize(diff_vec(arr))


def get_percentile_score(inf_scores, sorted_calib_scores):
    """Fraction of calibration scores <= each inference score. Both 1-D arrays."""
    positions = np.searchsorted(sorted_calib_scores, inf_scores, side="right")
    return positions / len(sorted_calib_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_video_frames(video_path, frame_indices=None):
    """Load frames from an MP4 file.

    Args:
        video_path:    Path to .mp4 file.
        frame_indices: Optional list of 0-based frame indices to load.
                       If None, all frames are loaded sequentially.

    返回：
        np.ndarray of shape [T, H, W, C] in BGR order.
    """
    # 部分 OpenCV build 使用默认 backend 时会错误处理 percent-encoded 文件名；
    # 另一些 build 又拒绝显式 FFMPEG capture-by-name。这里两种都尝试。
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if frame_indices is None:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    frame_indices_sorted = sorted(frame_indices)
    frames_dict = {}
    for idx in frame_indices_sorted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_dict[idx] = frame
    cap.release()
    return np.array([frames_dict[i] for i in frame_indices if i in frames_dict])


# ─────────────────────────────────────────────────────────────────────────────
# STALL 检测器
# ─────────────────────────────────────────────────────────────────────────────

class STALL:
    """用于 AI 生成视频检测的 Spatial-Temporal Anomaly Log-Likelihood 检测器。

    ``final_score`` 越高，视频越接近真实视频（score direction: HIGHER_IS_REAL）。

    Args:
        device:     Torch 设备字符串。
        data_dict:  Dict / npz，包含 W_spat, mu_spat, W_temp, mu_temp,
                    calib_ll_spat, calib_ll_temp.
        spat_agg:   空间分支跨帧聚合方式（"max"）。
        temp_agg:   时序分支跨 frame-pair 聚合方式（"min"）。
        dino_repo:  可选 DINOv3 repo 路径覆盖。
        dino_weights: 可选 DINOv3 权重路径覆盖。
    """

    # 类级 cache，使多个 STALL 实例共享同一个已加载模型。
    _shared_model = None
    _shared_transform = None

    def __init__(
        self,
        device,
        data_dict: dict,
        spat_agg: str = "max",
        temp_agg: str = "min",
        dino_repo: str = None,
        dino_weights: str = None,
        load_dino: bool = True,
    ):
        if load_dino:
            if STALL._shared_model is None:
                STALL._shared_model, STALL._shared_transform = load_dinov3_model(
                    device, repo_dir=dino_repo, weights=dino_weights
                )
            self.model = STALL._shared_model
            self.transform = STALL._shared_transform
        else:
            self.model = None
            self.transform = None

        self.device = device

        self.w_spat = data_dict["W_spat"]
        self.mu_spat = data_dict["mu_spat"]
        self.w_temp = data_dict["W_temp"]
        self.mu_temp = data_dict["mu_temp"]

        self.spat_agg = spat_agg
        self.temp_agg = temp_agg

        calib_spat = data_dict["calib_ll_spat"]
        calib_temp = data_dict["calib_ll_temp"]
        self.calib_spat_sorted = np.sort(AGG_STR2FN[spat_agg](calib_spat, axis=1))
        self.calib_temp_sorted = np.sort(AGG_STR2FN[temp_agg](calib_temp, axis=1))

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _embed_flat_frames(self, flat_frames: list, batch_size: int) -> np.ndarray:
        """Run DINOv3 on a flat list of BGR frames. Returns [total_frames, D]."""
        device = next(self.model.parameters()).device
        flat_embs = []
        with torch.no_grad():
            for start in range(0, len(flat_frames), batch_size):
                batch = flat_frames[start : start + batch_size]
                tensors = [
                    self.transform(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                    for fr in batch
                ]
                flat_embs.append(self.model(torch.stack(tensors).to(device)))
        return torch.cat(flat_embs, dim=0).cpu().numpy()

    def frames_to_embeddings(self, video_arrays, batch_size=32):
        """Extract DINOv3 embeddings for a list of videos.

        Args:
            video_arrays: List of np.ndarray, each [T, H, W, C] BGR.
            batch_size:   Frames per GPU batch.

        返回：
            np.ndarray of shape [N, T, D].
        """
        lengths = [len(v) for v in video_arrays]
        flat_frames = [frame for video in video_arrays for frame in video]
        flat_embs = self._embed_flat_frames(flat_frames, batch_size)
        cursor = 0
        all_embeddings = []
        for length in lengths:
            all_embeddings.append(flat_embs[cursor : cursor + length])
            cursor += length
        return np.stack(all_embeddings)

    # ── Branch log-likelihoods ────────────────────────────────────────────────

    def spatial_log_likelihood(self, embs):
        """空间 LL：白化 raw embeddings → N(0,I)。形状 [N, T]。"""
        return raw_emb_to_log_likelihoods(embs, mu=self.mu_spat, W=self.w_spat)

    def temporal_log_likelihood(self, embs):
        """时序 LL：白化 L2-normalized frame diff → N(0,I)。形状 [N, T-1]。"""
        zero_mask = (np.linalg.norm(diff_vec(embs), axis=-1) == 0)  # [N, T-1]
        ll = raw_emb_to_log_likelihoods(
            embs, mu=self.mu_temp, W=self.w_temp, preprocess_fn=diff_normalized_embeddings
        )
        # 完全相同的连续帧会产生零差分，没有时序信息。
        # 将其 LL 设为 +inf，可在 min 聚合中排除它们，使其不会被选中。
        # 注意：如果把 temp_agg 从 "min" 改成 "mean" 或 "max"，必须重审这里；
        # +inf 会污染这些聚合，需要换策略。
        ll[zero_mask] = np.inf
        return ll

    # ── 共享打分逻辑 ─────────────────────────────────────────────────────────

    def _scores_from_embs(self, embs):
        """从 [1, T, D] embedding 数组计算所有 STALL 分数。

        返回与 inference() 相同结构的 dict。
        """
        spat_ll = self.spatial_log_likelihood(embs)
        temp_ll = self.temporal_log_likelihood(embs)

        spat_agg = AGG_STR2FN[self.spat_agg](spat_ll, axis=1)
        temp_agg = AGG_STR2FN[self.temp_agg](temp_ll, axis=1)

        spat_pct = get_percentile_score(spat_agg, self.calib_spat_sorted)
        temp_pct = get_percentile_score(temp_agg, self.calib_temp_sorted)

        return {
            "embs": embs,
            "spat_ll": spat_ll,
            "temp_ll": temp_ll,
            "spat_ll_agg": spat_agg,
            "temp_ll_agg": temp_agg,
            "spat_percentile": spat_pct,
            "temp_percentile": temp_pct,
            "final_score": 0.5 * (spat_pct + temp_pct),
        }

    # ── 单视频推理 ───────────────────────────────────────────────────────────

    def inference(self, video_path, frame_indices=None):
        """在单个视频文件上运行 STALL。

        返回包含以下 key 的 dict：
            embs, spat_ll, temp_ll,
            spat_ll_agg, temp_ll_agg,
            spat_percentile, temp_percentile,
            final_score   (0-1，越高越接近真实视频，HIGHER_IS_REAL)
        """
        frames = load_video_frames(video_path, frame_indices)
        embs = self.frames_to_embeddings([frames])
        return self._scores_from_embs(embs)

    # ── 批量推理 ─────────────────────────────────────────────────────────────

    def batch_inference(self, video_paths, frame_indices_list=None, batch_size=32):
        """高效地一次处理多个视频。

        所有视频帧会被展平成一个序列，并一起送入 DINOv3（每次 forward 处理
        ``batch_size`` 帧），之后再拆回逐视频 embedding。分数逐视频独立计算，
        因此数值上等价于对每个视频单独调用 ``inference()``。

        Args:
            video_paths:        .mp4 文件路径列表。
            frame_indices_list: 可选帧索引列表，每个视频一个列表。``None`` 或省略时
                                加载该视频全部帧。
            batch_size:         每次 DINOv3 forward 的帧数。

        返回：
            与 ``inference()`` 格式相同的 result dict 列表，顺序与输入视频一致。
        """
        if frame_indices_list is None:
            frame_indices_list = [None] * len(video_paths)

        # 1. 加载所有帧 ───────────────────────────────────────────────────────
        all_frames = [
            load_video_frames(path, fidx)
            for path, fidx in zip(video_paths, frame_indices_list)
        ]
        lengths = [len(f) for f in all_frames]          # 每个视频的帧数
        flat_frames = [frame for vid in all_frames for frame in vid]

        # 2. 对所有帧执行一次 batched DINOv3 处理 ─────────────────────────────
        flat_embs = self._embed_flat_frames(flat_frames, batch_size)  # [total_frames, D]

        # 3. 将 embeddings 拆回逐视频并打分 ───────────────────────────────────
        results = []
        cursor = 0
        for length in lengths:
            embs = flat_embs[cursor : cursor + length][np.newaxis]  # [1, T, D]
            cursor += length
            results.append(self._scores_from_embs(embs))
        return results

    # ── Debug 输出 ───────────────────────────────────────────────────────────

    def print_score_debug(self, result: dict):
        """打印 inference result dict 中的中间打分值。

        传入 inference() 或 _scores_from_embs() 返回的 dict，用于检查：
          - Log-likelihood 是否为负（符合 Gaussian LL 预期）
          - 白化 embedding 每维方差是否接近 1
          - 聚合 LL 在校准范围中的位置
          - 百分位分数（真实通常应高于约 0.5，生成通常更低）
        """
        embs    = result["embs"]
        spat_ll = result["spat_ll"]
        temp_ll = result["temp_ll"]
        spat_agg = result["spat_ll_agg"]
        temp_agg = result["temp_ll_agg"]
        spat_pct = result["spat_percentile"]
        temp_pct = result["temp_percentile"]

        D = embs.shape[-1]
        w_embs_spat = whitening_transform(embs, self.mu_spat, self.w_spat)
        w_embs_temp = whitening_transform(
            diff_normalized_embeddings(embs), self.mu_temp, self.w_temp
        )
        print(f"  [debug] emb shape:           {embs.shape}  (D={D})")
        print(f"  [debug] whitened_spat norm²  mean={np.mean(w_embs_spat**2):.3f}  (expected ~1.0 per dim)")
        print(f"  [debug] whitened_temp norm²  mean={np.mean(w_embs_temp**2):.3f}  (expected ~1.0 per dim)")
        print(f"  [debug] spat_ll per-frame:   min={spat_ll.min():.1f}  max={spat_ll.max():.1f}  mean={spat_ll.mean():.1f}")
        print(f"  [debug] temp_ll per-pair:    min={temp_ll.min():.1f}  max={temp_ll.max():.1f}  mean={temp_ll.mean():.1f}")
        print(f"  [debug] spat_ll_agg ({self.spat_agg}):  {spat_agg}  |  calib range [{self.calib_spat_sorted[0]:.1f}, {self.calib_spat_sorted[-1]:.1f}]")
        print(f"  [debug] temp_ll_agg ({self.temp_agg}):  {temp_agg}  |  calib range [{self.calib_temp_sorted[0]:.1f}, {self.calib_temp_sorted[-1]:.1f}]")
        print(f"  [debug] spat_percentile:     {spat_pct}")
        print(f"  [debug] temp_percentile:     {temp_pct}")
        print(f"  [debug] final_score:         {0.5 * (spat_pct + temp_pct)}")
