"""原文默认batch32官方对照；只覆盖两个完整协议共用的评价身份并保留旧结果。"""

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from artifacts import paper_json
from reference import file_digest
from evaluation.official_baseline import run_official


def prepare(root):
    root = Path(root)
    source = root / "results/runs/complete23_coverage_only/evaluation.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    out = root / "data/manifests/native_stall"
    out.mkdir(parents=True, exist_ok=True)
    short = frame.video_id.str.endswith(":duration1")
    for length, part in [(16, frame[~short]), (8, frame[short])]:
        p = out / f"evaluation_{length}.csv"
        text = part.to_csv(index=False)
        if p.exists():
            if p.read_text() != text:
                raise ValueError("原生STALL对照身份改变")
        else:
            p.write_text(text)
    paper_json(
        out / "manifest.json",
        dict(
            source_sha256=file_digest(source),
            rows=len(frame),
            scope="coverage_only包含paper_filter全部身份；不重新采样",
            files={p.name: file_digest(p) for p in out.glob("*.csv")},
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--length", type=int, choices=[8, 16])
    a = p.parse_args()
    root = Path.cwd()
    if a.prepare:
        return prepare(root)
    cfg = yaml.safe_load((root / "configs/paper.yaml").read_text())
    cfg["runtime"]["device"] = f"cuda:{a.rank % 2}"
    cfg["encoder"].update(batch_size=32, pad_tail=False)
    cfg["protocol"]["id"] = "official_native_batch32_complete23"
    for length in [a.length] if a.length else (8, 16):
        cfg["selection"]["frames"] = length
        run_official(
            root,
            cfg,
            "complete23",
            root / "results/runs" / f"complete23_native_stall_{length}",
            rank=a.rank,
            world_size=a.world_size,
            manifest=root / "data/manifests/native_stall" / f"evaluation_{length}.csv",
            window_frames=length,
            frame_batch_size=32,
            pad_tail=False,
        )


if __name__ == "__main__":
    main()
