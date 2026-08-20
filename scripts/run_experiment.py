#!/usr/bin/env python3
"""从唯一基础配置和显式覆盖参数启动一次 Global+Local 方法实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import apply_overrides, load_config
from runner import resume_bootstrap, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="覆盖 YAML 字段，例如 sampling.num_windows=1",
    )
    parser.add_argument("--scores-csv", type=Path)
    parser.add_argument(
        "--resume-bootstrap", action="store_true",
        help="只为同配置的 interrupted run 补跑 bootstrap，不重新读取缓存或评分",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_bootstrap:
        if args.dry_run or args.overwrite or args.scores_csv is not None:
            raise ValueError("--resume-bootstrap 不能与 --dry-run、--overwrite 或 --scores-csv 同时使用")
        # 恢复必须复用当时落盘的完整配置，包括设备等运行时覆盖，不能退回基础 YAML。
        resolved_path = ROOT / "results" / "runs" / args.run_name / "resolved_config.yaml"
        if not resolved_path.is_file():
            raise FileNotFoundError("--resume-bootstrap 需要已有的 resolved_config.yaml")
        config = apply_overrides(load_config(resolved_path), args.overrides)
        output_dir = resume_bootstrap(
            ROOT, args.run_name, config, command=[sys.executable, *sys.argv]
        )
        print(output_dir.relative_to(ROOT))
        return
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = apply_overrides(load_config(config_path), args.overrides)
    scores_csv = args.scores_csv
    if scores_csv is not None and not scores_csv.is_absolute():
        scores_csv = ROOT / scores_csv
    output_dir = run(
        ROOT,
        args.run_name,
        config,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        scores_csv=scores_csv,
        command=[sys.executable, *sys.argv],
    )
    print(output_dir.relative_to(ROOT))


if __name__ == "__main__":
    main()
