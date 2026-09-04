#!/usr/bin/env bash
set -euo pipefail

# 从已完成的Local D2 likelihood field缓存评估三种selector的
# Mean、worst 5%、CVaR 10%、CVaR 20%，并同时比较standard与5-fold OOF null。
#
# 用法：
#   bash scripts/run_caes_tail.sh
#   bash scripts/run_caes_tail.sh --field-dir results/caes/tail_fields_smoke --output-dir results/runs/caes_tail_smoke --overwrite
#
# 可传：--field-dir、--output-dir、--source-stage-fs、--overwrite。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/evaluate_caes_tail.py" "$@"
