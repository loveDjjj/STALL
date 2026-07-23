#!/usr/bin/env python3
"""批量运行或生成 GenVideo D3-exact 评测命令。

默认 dry-run，只根据 `d3_exact_csv_manifest.csv` 生成每个 fake generator 的
`eval_d3_exact_from_frames.py` 命令清单。显式传入 `--execute` 后才会顺序运行。

该脚本不安装依赖、不下载模型、不抽帧；它只负责把已经准备好的 D3 CSV
协议转换成可复现评测批处理。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol"
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_manifest(path: Path) -> tuple[Path, pd.DataFrame]:
    manifest = pd.read_csv(path)
    required = {"csv_role", "source_model", "csv_path", "rows"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    real_rows = manifest[manifest["csv_role"] == "real"]
    if len(real_rows) != 1:
        raise ValueError(f"{path} 应包含且只包含一个 real CSV，当前 {len(real_rows)}")
    real_csv = Path(str(real_rows.iloc[0]["csv_path"]))
    fake_rows = manifest[manifest["csv_role"] == "fake"].copy()
    if fake_rows.empty:
        raise ValueError(f"{path} 未包含 fake CSV")
    return real_csv, fake_rows


def build_command(
    real_csv: Path,
    fake_csv: Path,
    output_dir: Path,
    encoder: str,
    loss: str,
    device: str,
    frame_batch_size: int,
    max_real: int | None,
    max_fake: int | None,
    local_files_only: bool,
) -> list[str]:
    stem = fake_csv.stem
    command = [
        sys.executable,
        "tools/eval_d3_exact_from_frames.py",
        "--encoder",
        encoder,
        "--loss",
        loss,
        "--device",
        device,
        "--frame-batch-size",
        str(frame_batch_size),
        "--real-csv",
        rel(real_csv),
        "--fake-csv",
        rel(fake_csv),
        "--output-scores-csv",
        rel(output_dir / f"{stem}_{encoder}_{loss}_scores.csv"),
        "--output-metrics-csv",
        rel(output_dir / f"{stem}_{encoder}_{loss}_metrics.csv"),
    ]
    if max_real is not None:
        command.extend(["--max-real", str(max_real)])
    if max_fake is not None:
        command.extend(["--max-fake", str(max_fake)])
    command.append("--local-files-only" if local_files_only else "--no-local-files-only")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成或运行 GenVideo D3-exact 批量评测命令。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR / "d3_exact_csv_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR / "d3_exact_eval",
    )
    parser.add_argument("--encoder", default="XCLIP-16")
    parser.add_argument("--loss", default="l2", choices=["l2", "cos"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-fake", type=int, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--command-manifest-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_csv, fake_rows = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command_manifest_csv is None:
        suffix = "execute" if args.execute else "dryrun"
        args.command_manifest_csv = args.output_dir / f"d3_exact_batch_{args.encoder}_{args.loss}_{suffix}_commands.csv"

    records = []
    for row in fake_rows.itertuples(index=False):
        fake_csv = Path(str(row.csv_path))
        command = build_command(
            real_csv,
            fake_csv,
            args.output_dir,
            args.encoder,
            args.loss,
            args.device,
            args.frame_batch_size,
            args.max_real,
            args.max_fake,
            args.local_files_only,
        )
        started = time.time()
        status = "dry_run"
        return_code = None
        stdout_tail = ""
        stderr_tail = ""
        if args.execute:
            proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            return_code = int(proc.returncode)
            stdout_tail = (proc.stdout or "")[-1000:]
            stderr_tail = (proc.stderr or "")[-1000:]
            status = "ok" if proc.returncode == 0 else "failed"
            if proc.returncode != 0 and args.stop_on_failure:
                records.append(
                    {
                        "source_model": row.source_model,
                        "fake_csv": str(fake_csv),
                        "rows": int(row.rows),
                        "status": status,
                        "return_code": return_code,
                        "elapsed_sec": time.time() - started,
                        "command": " ".join(shlex.quote(part) for part in command),
                        "stdout_tail": stdout_tail,
                        "stderr_tail": stderr_tail,
                    }
                )
                break
        records.append(
            {
                "source_model": row.source_model,
                "fake_csv": str(fake_csv),
                "rows": int(row.rows),
                "status": status,
                "return_code": return_code,
                "elapsed_sec": time.time() - started,
                "command": " ".join(shlex.quote(part) for part in command),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(args.command_manifest_csv, index=False)
    print(f"execute={args.execute} commands={len(out)}")
    print(out[["source_model", "rows", "status", "return_code"]].to_string(index=False))
    print(f"command_manifest={args.command_manifest_csv}")
    if not args.execute:
        print("dry-run only; add --execute to run D3 evaluation")


if __name__ == "__main__":
    main()
