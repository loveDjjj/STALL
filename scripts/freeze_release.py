#!/usr/bin/env python3
"""从一个已完成 run 创建新的冻结发布目录。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artifacts import freeze_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="results/runs/ 下已完成的运行名称")
    parser.add_argument("release_name", help="release/ 下的新冻结版本名称")
    args = parser.parse_args()
    print(freeze_run(ROOT, args.run_name, args.release_name).relative_to(ROOT))


if __name__ == "__main__":
    main()
