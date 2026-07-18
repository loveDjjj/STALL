#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$ROOT/results/ste_stall_tail_universal}"

"$PYTHON_BIN" "$ROOT/tools/apply_ste_stall_tail_universal.py" \
  --root "$ROOT" \
  --output-dir "$OUT_DIR"
