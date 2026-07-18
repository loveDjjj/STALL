"""整理 VideoFeedback 数据目录。

把 VideoFeedback zip 解压后的扁平文件复制/移动为 video_index.py / eval.py 期望的
按生成器标注目录结构。

用法：
    python src/organize_videofeedback.py \
        --tmp-dir datasets/videofeedback_tmp \
        --output  datasets/videofeedback

--tmp-dir 是 clone 下来的 TIGER-Lab/VideoFeedback repo 根目录，其中应包含四个
zip 解压得到的 extracted/ 子目录。脚本读取四个 JSON metadata 文件，把每个
video ID 映射到来源模型，再把视频放到：
--output/{real|fake}/{source_model}/{id}.mp4。
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

SOURCE_MODEL_MAP = {
    "vidprom_pika":         "Pika",
    "vidprom_t2vz":         "Text2Video-Zero",
    "vidprom_vc2":          "VideoCrafter2",
    "vidprom_ms":           "ModelScope",
    "vidprom_lavie_base":   "LaVie-base",
    "vidprom_anidiff":      "AnimateDiff",
    "vidprom_lvdm":         "LVDM",
    "vidprom_hotshot":      "Hotshot-XL",
    "vidprom_zs_576w":      "ZeroScope-576w",
    "fastsvd":              "Fast-SVD",
    "sora":                 "SoRA-Clip",
    "real_didemo_high_res": "DiDeMo",
    "real_pd70m_high_res":  "Panda70M",
}


def organize(tmp_dir: Path, output: Path) -> None:
    splits = ["train", "test"]
    subsets = ["real", "annotated"]

    counts: Counter = Counter()
    unmapped: Counter = Counter()
    missing: list[str] = []

    # 先加载所有条目，使 tqdm 预先知道总数。
    all_items: list[tuple] = []  # (item, top_dir, extracted_dir)
    for split in splits:
        for subset in subsets:
            json_path = tmp_dir / split / f"data_{subset}.json"
            if not json_path.exists():
                print(f"[warn] 缺少 JSON: {json_path}")
                continue

            with open(json_path) as f:
                entries = json.load(f)

            top_dir = "real" if subset == "real" else "fake"
            extracted_dir = tmp_dir / "extracted" / f"{subset}_{split}"
            for item in entries:
                all_items.append((item, top_dir, extracted_dir))

    for item, top_dir, extracted_dir in tqdm(all_items, desc="移动视频", unit="video"):
        vid_id = item.get("id", "")
        video_url = item.get("video link", "")
        path_parts = urlparse(video_url).path.strip("/").split("/")
        raw_model = path_parts[-2] if len(path_parts) >= 2 else ""
        source_model = SOURCE_MODEL_MAP.get(raw_model)

        if source_model is None:
            unmapped[raw_model] += 1
            continue

        src = extracted_dir / f"{vid_id}.mp4"
        dest = output / top_dir / source_model / f"{vid_id}.mp4"
        if dest.exists():
            counts[f"{top_dir}/{source_model}"] += 1
            continue
        if not src.exists():
            missing.append(str(src))
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        counts[f"{top_dir}/{source_model}"] += 1

    print("\n各生成器已移动视频数:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"\n总数: {sum(counts.values())}")

    if unmapped:
        print("\n[warn] 存在未映射 raw_source_model 值（请扩展 SOURCE_MODEL_MAP）:")
        for raw, n in sorted(unmapped.items()):
            print(f"  {raw!r}: {n} 个视频")

    if missing:
        print(f"\n[warn] {len(missing)} 个源文件未找到（前 10 个）:")
        for p in missing[:10]:
            print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="把 VideoFeedback 整理为生成器子目录。")
    parser.add_argument("--tmp-dir", required=True, type=Path,
                        help="clone 后的 TIGER-Lab/VideoFeedback repo 根目录（含 extracted/ 子目录）")
    parser.add_argument("--output", required=True, type=Path,
                        help="目标数据集目录（例如 datasets/videofeedback）")
    args = parser.parse_args()

    organize(args.tmp_dir, args.output)


if __name__ == "__main__":
    main()
