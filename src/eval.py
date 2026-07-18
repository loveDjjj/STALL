"""STALL 的 Phase 4 eval CLI。

用法：
    # HuggingFace dataset：使用预计算 embeddings，无需 DINOv3
    python eval.py --hf-dataset OmerXYZ/comgenvid

    # 包含 video_path、subset、source_model 列的 CSV
    python eval.py --csv my_benchmark.csv

    # video_index.py 生成的 enriched CSV：使用 embedding cache
    python src/eval.py --csv cache/indexes/genvideo.csv --emb-cache cache/embeddings/genvideo/ --output-csv results.csv

    # 两个包含 <model>/*.mp4 子目录的视频目录
    python src/eval.py --real-dir datasets/demo_dataset/real/ --fake-dir datasets/demo_dataset/fake/

    # 保存逐视频分数到 CSV
    python src/eval.py --hf-dataset OmerXYZ/comgenvid --output-csv results.csv

    # 覆盖 DINOv3 路径（仅 CSV/dir 模式需要）
    python src/eval.py --csv bench.csv --dino-repo ~/dinov3 --dino-weights ~/dinov3/weights/...pth

    # Debug：每个来源 5 个视频
    python src/eval.py --csv cache/indexes/genvideo.csv --emb-cache cache/embeddings/genvideo/ --debug-n 5 --output-csv dbg.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from metrics import Score, ScoreDirection, get_results_df, print_results
from stall import STALL

STALL_PARAMS_DEFAULT = "precomputed/stall_params_vatex_dino_v3.npz"


# ─────────────────────────────────────────────────────────────────────────────
# STALL factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_stall(args, load_dino: bool) -> STALL:
    data = np.load(args.params)
    device = "cpu"
    if load_dino:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
    return STALL(
        device=device,
        data_dict=data,
        load_dino=load_dino,
        dino_repo=getattr(args, "dino_repo", None),
        dino_weights=getattr(args, "dino_weights", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-mode runners
# ─────────────────────────────────────────────────────────────────────────────

def _print_debug_pair(model, samples: dict):
    """为一个真实样本和一个生成样本打印逐步 debug 输出。"""
    SEP = "=" * 60
    for label, subset_key in [("REAL", "real"), ("FAKE", "annotated")]:
        sample = samples.get(subset_key)
        if sample is None:
            print(f"\n{SEP}\n  {label}: (未找到样本)\n{SEP}")
            continue
        print(f"\n{SEP}")
        print(f"  {label}: {sample.get('filename', '?')}  [{sample.get('source_model', '?')}]")
        print(SEP)
        result = model._scores_from_embs(sample["embs"])
        model.print_score_debug(result)
    print(f"\n{SEP}")
    print("  正确 HIGHER_IS_REAL 行为的预期：")
    print("    real final_score  >  fake final_score")
    print(f"{SEP}\n")


def run_hf(args) -> pd.DataFrame:
    from dataset_utils import load_hf_dataset
    from tqdm import tqdm

    print(f"加载 HuggingFace dataset: {args.hf_dataset}")
    model = _make_stall(args, load_dino=False)

    debug = getattr(args, "debug", False)
    _debug_samples = {}  # subset -> sample, populated on first occurrence of each

    rows = []
    for sample in tqdm(load_hf_dataset(args.hf_dataset, split=args.split, duration=getattr(args, "duration", 2), verbose=args.debug), desc="Scoring", unit="video", dynamic_ncols=True):
        result = model._scores_from_embs(sample["embs"])
        rows.append({
            "subset": sample["subset"],
            "source_model": sample["source_model"],
            "filename": sample["filename"],
            "final_score": float(result["final_score"][0]),
        })
        if debug and sample["subset"] not in _debug_samples:
            _debug_samples[sample["subset"]] = sample

    if debug:
        _print_debug_pair(model, _debug_samples)

    return pd.DataFrame(rows)


def run_csv(args) -> pd.DataFrame:
    from dataset_utils import load_csv

    # Peek at the CSV to decide which path to take
    df_peek = load_csv(args.csv)
    emb_cache = getattr(args, "emb_cache", None)
    duration_sec = getattr(args, "duration", 2)
    debug_n = getattr(args, "debug_n", None)
    debug = getattr(args, "debug", False)

    use_cache_path = emb_cache is not None or f"{duration_sec}_sec_idxs" in df_peek.columns

    if use_cache_path:
        # Enriched CSV path: per-video embedding cache + frame-index slicing
        from dataset_utils import count_cache_misses, load_csv_with_emb_cache, prefill_emb_cache
        from tqdm import tqdm

        model = _make_stall(args, load_dino=True)
        _csv_df = pd.read_csv(args.csv)
        if debug_n is not None:
            _total = _csv_df.groupby(["subset", "source_model"]).size().clip(upper=debug_n).sum()
        else:
            _total = len(_csv_df)

        # Phase 1: extract DINOv3 embeddings (parallel decode + cross-video GPU batching)
        _n_misses = count_cache_misses(
            args.csv,
            emb_cache_dir=emb_cache or ".",
            duration_sec=duration_sec,
            debug_n=debug_n,
            compact=args.compact,
        )
        print(f"Phase 1/2 — Extracting DINOv3 embeddings ({_n_misses} cache misses, {_total - _n_misses} cached)")
        for _ in tqdm(
            prefill_emb_cache(
                args.csv,
                emb_cache_dir=emb_cache or ".",
                model=model,
                duration_sec=duration_sec,
                debug_n=debug_n,
                compact=args.compact,
                num_workers=args.workers,
                video_batch=args.video_batch,
            ),
            desc="Extracting", unit=" video", dynamic_ncols=True, total=_n_misses,
        ):
            pass

        # Phase 2: score all videos from cache
        print("Phase 2/2 — Scoring")
        _debug_samples = {}
        rows = []
        for sample in tqdm(
            load_csv_with_emb_cache(
                args.csv,
                emb_cache_dir=emb_cache or ".",
                model=model,
                duration_sec=duration_sec,
                debug_n=debug_n,
                compact=args.compact,
            ),
            desc="Scoring", unit="video", dynamic_ncols=True, total=_total,
        ):
            result = model._scores_from_embs(sample["embs"])
            rows.append({
                "subset": sample["subset"],
                "source_model": sample["source_model"],
                "filename": sample["filename"],
                "final_score": float(result["final_score"][0]),
            })
            if debug and sample["subset"] not in _debug_samples:
                _debug_samples[sample["subset"]] = sample

        if debug:
            _print_debug_pair(model, _debug_samples)

        return pd.DataFrame(rows)

    else:
        # Backward-compatible path: plain CSV, on-the-fly batch inference
        df = df_peek
        if debug_n is not None:
            df = (
                df.groupby(["subset", "source_model"], group_keys=False)
                .apply(lambda g: g.head(debug_n))
                .reset_index(drop=True)
            )
        print(f"Loaded {len(df)} videos from {args.csv}")
        model = _make_stall(args, load_dino=True)
        results = model.batch_inference(df["video_path"].tolist())
        df = df.copy()
        df["final_score"] = [float(r["final_score"][0]) for r in results]

        if debug:
            _debug_samples = {}
            for i, (_, row) in enumerate(df.iterrows()):
                subset = row["subset"]
                if subset not in _debug_samples:
                    _debug_samples[subset] = {
                        "embs": results[i]["embs"],
                        "filename": row.get("filename", row["video_path"]),
                        "source_model": row["source_model"],
                    }
            _print_debug_pair(model, _debug_samples)

        return df


def run_dirs(args) -> pd.DataFrame:
    from pathlib import Path

    records = []
    for dir_path, subset_val in [(args.real_dir, "real"), (args.fake_dir, "annotated")]:
        root = Path(dir_path)
        if not root.exists():
            raise ValueError(f"目录不存在: {dir_path}")
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            for video_file in sorted(model_dir.glob("*.mp4")):
                records.append({
                    "video_path": str(video_file),
                    "subset": subset_val,
                    "source_model": model_dir.name,
                })

    if not records:
        raise ValueError("指定目录中未找到 .mp4 文件。")

    df = pd.DataFrame(records)
    n_real = (df["subset"] == "real").sum()
    n_fake = (df["subset"] == "annotated").sum()
    print(f"找到 {len(df)} 个视频: {n_real} real, {n_fake} fake")

    model = _make_stall(args, load_dino=True)
    results = model.batch_inference(df["video_path"].tolist())
    df["final_score"] = [float(r["final_score"][0]) for r in results]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="运行 STALL 检测器并打印逐生成器 AUC/AP 结果。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf-dataset", metavar="REPO_ID",
        help="HuggingFace repo ID（使用预计算 embedding，无需 DINOv3）",
    )
    src.add_argument(
        "--csv", metavar="CSV_PATH",
        help="CSV，包含列：video_path, subset ('real'/'annotated'), source_model",
    )
    src.add_argument(
        "--real-dir", metavar="DIR",
        help="真实视频目录：<model>/*.mp4 子目录（与 --fake-dir 配合）",
    )

    parser.add_argument(
        "--fake-dir", metavar="DIR",
        help="生成视频目录：<model>/*.mp4 子目录（与 --real-dir 配合）",
    )
    parser.add_argument(
        "--dino-repo", metavar="PATH", default=None,
        help="覆盖本地 DINOv3 repo clone 路径（仅 CSV/dir 模式）",
    )
    parser.add_argument(
        "--dino-weights", metavar="PATH", default=None,
        help="覆盖 DINOv3 .pth 权重路径（仅 CSV/dir 模式）",
    )
    parser.add_argument(
        "--params", default=STALL_PARAMS_DEFAULT,
        help=f"STALL params .npz（默认: {STALL_PARAMS_DEFAULT}）",
    )
    parser.add_argument(
        "--output-csv", metavar="PATH", default=None,
        help="保存逐视频分数（subset, source_model, final_score, …）到 CSV",
    )
    parser.add_argument(
        "--split", default="train",
        help="HuggingFace dataset split（默认: train）",
    )
    parser.add_argument(
        "--debug", nargs="?", const=True, default=None, metavar="N",
        help=(
            "启用 debug 输出：打印一个真实样本和一个生成样本的逐步分数。"
            "可选传入 N，将每个 (subset, source_model) 的打分限制为 N 个视频，"
            "例如 --debug 5。"
        ),
    )
    parser.add_argument(
        "--emb-cache", metavar="PATH", default=None,
        help=(
            "DINOv3 embedding cache 目录（每个视频一个 .pt 文件）。"
            "省略时即时计算 embedding 且不缓存。"
            "与 video_index.py 生成的 enriched CSV 配合使用。"
        ),
    )
    parser.add_argument(
        "--duration", type=int, default=2, choices=[1, 2, 3, 4],
        help="使用几秒窗口打分（默认: 2）。需要 enriched CSV。",
    )
    parser.add_argument(
        "--compact", action="store_true", default=False,
        help=(
            "只抽取 --duration 秒窗口帧，而不是 8 fps 下的完整视频。"
            "cache 文件名为 {stem}_{duration}s.pt。对长视频可显著加速。"
            "若后续要用不同 --duration 重新打分，请不带 --compact 重新构建完整 cache。"
            "（仅 CSV + --emb-cache 模式）"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Phase 1 并行视频加载的 CPU decode 线程数（默认: 4）",
    )
    parser.add_argument(
        "--video-batch", type=int, default=8, metavar="N",
        help="Phase 1 单次 GPU pass 合并处理的视频数（默认: 8）",
    )

    args = parser.parse_args()

    # 将 --debug [N] 拆成独立 bool 和 int，供后续使用。
    _debug_raw = args.debug
    args.debug = _debug_raw is not None
    args.debug_n = int(_debug_raw) if isinstance(_debug_raw, str) else None

    if args.real_dir and not args.fake_dir:
        parser.error("--real-dir 需要同时提供 --fake-dir")
    if args.fake_dir and not args.real_dir:
        parser.error("--fake-dir 需要同时提供 --real-dir")

    # 运行对应 loader。
    if args.hf_dataset:
        df = run_hf(args)
    elif args.csv:
        df = run_csv(args)
    else:
        df = run_dirs(args)

    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"已保存逐视频分数 → {args.output_csv}")

    scores_d = {
        "final_score": Score(
            value=df["final_score"].to_numpy(),
            direction=ScoreDirection.HIGHER_IS_REAL,
        )
    }
    results_df = get_results_df(df[["subset", "source_model"]], scores_d)
    print_results(results_df)


if __name__ == "__main__":
    main()
