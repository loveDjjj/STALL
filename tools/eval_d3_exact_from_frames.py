#!/usr/bin/env python3
"""从已抽取的 D3 frame folders 评测 D3 second-order score。

该脚本实现 D3 官方核心推理逻辑的本项目版本：

1. 读取 real/fake CSV 中的 `content_path`；
2. 每个视频按数字文件名读取最多 16 帧，少于 16 但不少于 8 帧时读取 8 帧；
3. 对每帧执行 D3 的长边中心 10% 裁剪、resize 224 和 ImageNet normalize；
4. 用 CLIP/XCLIP/DINOv2 vision encoder 或本地 DINOv3 提取帧级 embedding；
5. 计算相邻帧特征 L2/cos 一阶距离；
6. 对一阶距离做差分并取标准差作为 D3 score。

默认不下载模型；`--local-files-only` 默认开启。若本机没有对应 HuggingFace
cache，需要先手动下载模型或取消该选项。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_PROTOCOL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol"
)

ENCODER_TO_MODEL_ID = {
    "XCLIP-16": "microsoft/xclip-base-patch16",
    "XCLIP-32": "microsoft/xclip-base-patch32",
    "CLIP-16": "openai/clip-vit-base-patch16",
    "CLIP-32": "openai/clip-vit-base-patch32",
    "DINOv2-B": "facebook/dinov2-base",
    "DINOv2-L": "facebook/dinov2-large",
    "DINOv3-L-local": "local:dinov3_vitl16",
}


class LocalDINOv3VisionWrapper(torch.nn.Module):
    """把本地 DINOv3 backbone 包装成 transformers-like vision model 输出。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False):
        del output_hidden_states
        emb = self.model(pixel_values)
        return SimpleNamespace(pooler_output=emb)


def numeric_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def crop_center_by_percentage(image: Image.Image, percentage: float = 0.1) -> Image.Image:
    width, height = image.size
    if width > height:
        trim = int(width * percentage)
        return image.crop((trim, 0, width - trim, height))
    trim = int(height * percentage)
    return image.crop((0, trim, width, height - trim))


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def read_d3_frames(content_path: Path, transform: transforms.Compose) -> torch.Tensor:
    files = sorted(
        [p for p in content_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=numeric_sort_key,
    )
    total = len(files)
    if total < 8:
        raise ValueError(f"{content_path} 帧数不足 8: {total}")
    max_frames = 8 if total < 16 else 16
    tensors = []
    for path in files[:max_frames]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = crop_center_by_percentage(image, 0.1)
            tensors.append(transform(image))
    return torch.stack(tensors, dim=0)


def check_dependencies(encoder: str, local_files_only: bool) -> int:
    rows = []
    required = ["torch", "torchvision", "PIL", "sklearn"]
    if encoder != "DINOv3-L-local":
        required.append("transformers")
    for module in required:
        spec = importlib.util.find_spec(module)
        rows.append((module, bool(spec), spec.origin if spec else ""))
    for module, ok, origin in rows:
        print(f"{module}: {'OK' if ok else 'MISSING'} {origin}")
    if encoder != "DINOv3-L-local" and not importlib.util.find_spec("transformers"):
        print("缺少 transformers；无法加载 CLIP/XCLIP/DINOv2 encoder。")
        return 1
    try:
        load_encoder(encoder, "cpu", local_files_only=local_files_only)
    except Exception as exc:
        print(f"模型加载检查失败: {type(exc).__name__}: {exc}")
        return 2
    print(f"模型加载检查通过: {encoder}")
    return 0


def load_encoder(encoder: str, device: str, local_files_only: bool):
    if encoder not in ENCODER_TO_MODEL_ID:
        raise ValueError(f"未知 encoder: {encoder}. 可选: {sorted(ENCODER_TO_MODEL_ID)}")
    model_id = ENCODER_TO_MODEL_ID[encoder]
    if encoder == "DINOv3-L-local":
        from stall import load_dinov3_model

        model, _ = load_dinov3_model(device)
        return LocalDINOv3VisionWrapper(model).to(device).eval()

    from transformers import AutoModel, CLIPVisionModel, XCLIPVisionModel

    kwargs = {"local_files_only": local_files_only}
    if encoder.startswith("XCLIP"):
        model = XCLIPVisionModel.from_pretrained(model_id, **kwargs)
    elif encoder.startswith("CLIP"):
        model = CLIPVisionModel.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def embed_frames(model, frames: torch.Tensor, device: str, frame_batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(frames), frame_batch_size):
        batch = frames[start : start + frame_batch_size].to(device)
        outputs = model(batch, output_hidden_states=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("encoder output 中未找到 pooler_output 或 CLS token")
        chunks.append(emb.detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def d3_score_from_embeddings(emb: np.ndarray, loss: str) -> float:
    if len(emb) < 3:
        return float("nan")
    left = emb[:-1]
    right = emb[1:]
    if loss == "l2":
        first = np.linalg.norm(left - right, axis=1)
    elif loss == "cos":
        left_n = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
        right_n = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        first = np.sum(left_n * right_n, axis=1)
    else:
        raise ValueError(f"未知 loss: {loss}")
    second = first[1:] - first[:-1]
    return float(np.std(second, ddof=1)) if len(second) > 1 else float("nan")


def load_eval_csv(real_csv: Path, fake_csv: Path, max_real: int | None, max_fake: int | None) -> pd.DataFrame:
    real = pd.read_csv(real_csv)
    fake = pd.read_csv(fake_csv)
    if max_real is not None:
        real = real.head(max_real)
    if max_fake is not None:
        fake = fake.head(max_fake)
    df = pd.concat([real, fake], ignore_index=True)
    required = {"content_path", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"输入 CSV 缺少列: {sorted(missing)}")
    return df


def evaluate_rows(
    df: pd.DataFrame,
    encoder: str,
    loss: str,
    device: str,
    frame_batch_size: int,
    local_files_only: bool,
) -> pd.DataFrame:
    model = load_encoder(encoder, device, local_files_only=local_files_only)
    transform = make_transform()
    rows = []
    for idx, row in tqdm(list(df.iterrows()), desc=f"D3 {encoder}", unit="video"):
        content_path = Path(str(row["content_path"]))
        try:
            frames = read_d3_frames(content_path, transform)
            emb = embed_frames(model, frames, device, frame_batch_size)
            score = d3_score_from_embeddings(emb, loss)
            status = "ok" if np.isfinite(score) else "nan_score"
            error = ""
            n_frames = len(frames)
        except Exception as exc:
            score = float("nan")
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            n_frames = 0
        rows.append(
            {
                "row_index": idx,
                "content_path": str(content_path),
                "label": int(row["label"]),
                "type_id": row.get("type_id", ""),
                "source_model": row.get("source_model", ""),
                "filename": row.get("filename", ""),
                "n_read_frames": n_frames,
                "encoder": encoder,
                "loss": loss,
                "d3_second_order_std": score,
                "status": status,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def metrics_from_scores(scores: pd.DataFrame) -> dict[str, float]:
    ok = scores[scores["status"] == "ok"].dropna(subset=["d3_second_order_std"]).copy()
    if ok["label"].nunique() != 2:
        raise ValueError("有效结果必须同时包含 real(label=0) 和 fake(label=1)")
    y_true_label = ok["label"].to_numpy(dtype=np.uint8)
    y_real = 1 - y_true_label
    score = ok["d3_second_order_std"].to_numpy(dtype=np.float64)
    return {
        "n_total": int(len(scores)),
        "n_ok": int(len(ok)),
        "n_failed": int((scores["status"] != "ok").sum()),
        "d3_official_real_ap": float(average_precision_score(y_real, score)),
        "real_auc": float(roc_auc_score(y_real, score)),
        "fake_ap_with_neg_score": float(average_precision_score(y_true_label, -score)),
        "fake_auc_with_neg_score": float(roc_auc_score(y_true_label, -score)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 D3 frame folders 评测 XCLIP/CLIP/DINOv2/DINOv3 D3 score。")
    parser.add_argument("--real-csv", type=Path, default=DEFAULT_PROTOCOL_DIR / "csv/real_MSRVTT_head1000.csv")
    parser.add_argument("--fake-csv", type=Path, required=False)
    parser.add_argument("--encoder", default="XCLIP-16", choices=sorted(ENCODER_TO_MODEL_ID))
    parser.add_argument("--loss", default="l2", choices=["l2", "cos"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-fake", type=int, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--output-scores-csv", type=Path)
    parser.add_argument("--output-metrics-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_deps:
        raise SystemExit(check_dependencies(args.encoder, args.local_files_only))
    if args.fake_csv is None:
        raise ValueError("--fake-csv 是必需参数；每次评测一个 fake generator。")

    df = load_eval_csv(args.real_csv, args.fake_csv, args.max_real, args.max_fake)
    scores = evaluate_rows(
        df,
        args.encoder,
        args.loss,
        args.device,
        args.frame_batch_size,
        args.local_files_only,
    )
    metrics = pd.DataFrame([metrics_from_scores(scores)])
    metrics.insert(0, "fake_csv", str(args.fake_csv))
    metrics.insert(0, "real_csv", str(args.real_csv))
    metrics.insert(0, "loss", args.loss)
    metrics.insert(0, "encoder", args.encoder)

    if args.output_scores_csv is None:
        stem = Path(args.fake_csv).stem
        args.output_scores_csv = DEFAULT_PROTOCOL_DIR / f"d3_exact_{args.encoder}_{args.loss}_{stem}_scores.csv"
    if args.output_metrics_csv is None:
        stem = Path(args.fake_csv).stem
        args.output_metrics_csv = DEFAULT_PROTOCOL_DIR / f"d3_exact_{args.encoder}_{args.loss}_{stem}_metrics.csv"
    args.output_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_scores_csv, index=False)
    metrics.to_csv(args.output_metrics_csv, index=False)
    print(f"已保存 scores: {args.output_scores_csv}")
    print(f"已保存 metrics: {args.output_metrics_csv}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
