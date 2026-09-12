#!/usr/bin/env bash
# 编译当前中文论文；不运行检测实验，缓存与中间文件保留在/data工作区。
# 示例：bash paper/ieee_alpha_stalled/build.sh
# 可显式设置TECTONIC_BIN；也支持系统已安装的XeLaTeX。
set -euo pipefail
paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$paper_dir/../.." && pwd)"
workspace_dir="$(cd "$repo_dir/.." && pwd)"
cd "$paper_dir"
if [[ -d "$repo_dir/results/paper_complete" ]]; then
  python3 export_tables.py --check
fi
mkdir -p build
paper_tectonic="${TECTONIC_BIN:-$workspace_dir/.paper_tools/bin/tectonic}"
if [[ -x "$paper_tectonic" ]]; then
  paper_cache="${PAPER_TEX_CACHE:-$workspace_dir/.paper_tools/cache}"
  mkdir -p "$paper_cache"
  XDG_CACHE_HOME="$paper_cache" "$paper_tectonic" main.tex \
    --outdir build --keep-logs --keep-intermediates
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex --outdir build --keep-logs --keep-intermediates
elif command -v xelatex >/dev/null 2>&1; then
  xelatex -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
  (cd build && BIBINPUTS=..: bibtex main)
  xelatex -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
  xelatex -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
else
  printf '%s\n' '需要Tectonic或XeLaTeX；参见本目录README。' >&2
  exit 1
fi
if rg -q 'Overfull \\[hv]box|Citation .* undefined|Reference .* undefined|Missing character:|Float too large|^!' build/main.log; then
  printf '%s\n' '编译存在需处理的溢出、缺字或引用问题，查看build/main.log。' >&2
  exit 1
fi
cp build/main.pdf main.pdf
printf '%s\n' "已更新：$paper_dir/main.pdf"
