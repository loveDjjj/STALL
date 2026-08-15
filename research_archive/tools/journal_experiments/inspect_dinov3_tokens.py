import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alpha_stalled.backbone import (  # noqa: E402
    create_dinov3_transform,
    load_dinov3_model,
)


DEFAULT_VIDEO = REPO_ROOT / "datasets" / "demo_dataset" / "real" / "MSVD" / "9671.mp4"


def load_one_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取视频第一帧: {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="检查 DINOv3 global 输出和 patch token 输出。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="用于抽取一帧的视频路径。")
    args = parser.parse_args()

    frame = load_one_frame(args.video)
    transform = create_dinov3_transform()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x = transform(image).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_dinov3_model(device)
    x = x.to(device)

    print(f"video={args.video}")
    print(f"frame_shape={frame.shape}")
    print(f"input_tensor_shape={tuple(x.shape)}")

    with torch.no_grad():
        global_out = model(x)
    print(f"global_output_shape={tuple(global_out.shape)}")

    if hasattr(model, "module"):
        core_model = model.module
    else:
        core_model = model

    if not hasattr(core_model, "forward_features"):
        print("模型上没有 forward_features")
        return

    with torch.no_grad():
        features = core_model.forward_features(x)

    print(f"forward_features_type={type(features).__name__}")

    if isinstance(features, dict):
        print("forward_features_keys:")
        for key, value in features.items():
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            print(f"  {key}: {shape}")
    else:
        shape = tuple(features.shape) if hasattr(features, "shape") else type(features).__name__
        print(f"forward_features_shape={shape}")


if __name__ == "__main__":
    main()
