#!/usr/bin/env bash
set -euo pipefail

# 冻结发布：将一个已完成 run 的配置、逐视频分数和指标复制为不可覆盖的正式版本。
# 用法示例：bash scripts/freeze_release.sh alpha_stall alpha_stall_v1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/freeze_release.py" "$@"
