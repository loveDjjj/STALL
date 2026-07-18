#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <dataset> <mode> <region>" >&2
  exit 2
fi

dataset="$1"
mode="$2"
region="$3"

cd /data/OneDay/STALL_project

export PYTHONPATH=
export PYTHONNOUSERSITE=1
export TORCH_HOME=/tmp/torch_cache
export XDG_CACHE_HOME=/tmp

csv="STALL/cache/indexes/${dataset}.csv"
cache="STALL/cache/patch_embeddings/${dataset}"
params_pattern="STALL/precomputed/patch_params_${dataset}_real_${mode}_region${region}_{agg}_v2.npz"
result_pattern="STALL/results/patch_cross_dataset_ablation/${dataset}_patch_${mode}_region${region}_{agg}_spat10_temp90.csv"

mkdir -p STALL/precomputed STALL/results/patch_cross_dataset_ablation

missing_params=()
missing_results=()
for agg in bottomk0p2 bottomk0p5 mean; do
  params="${params_pattern/\{agg\}/$agg}"
  result="${result_pattern/\{agg\}/$agg}"
  [[ -s "$params" ]] || missing_params+=("$agg")
  [[ -s "$result" ]] || missing_results+=("$agg")
done

if [[ ${#missing_params[@]} -gt 0 ]]; then
  echo "== Fit shared params: dataset=$dataset mode=$mode region=$region aggs=${missing_params[*]} =="
  conda run --no-capture-output -n stall \
    python STALL/src/create_patch_params_multiagg.py \
      --csv "$csv" \
      --patch-emb-cache "$cache" \
      --output-pattern "$params_pattern" \
      --duration 2 \
      --compact \
      --real-only \
      --max-patches-for-fit 300000 \
      --patch-temp-mode "$mode" \
      --patch-region-size "$region" \
      --aggs "${missing_params[@]}"
else
  echo "== Skip params: all present for dataset=$dataset mode=$mode region=$region =="
fi

if [[ ${#missing_results[@]} -gt 0 ]]; then
  echo "== Eval shared cache pass: dataset=$dataset mode=$mode region=$region aggs=${missing_results[*]} =="
  conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_multiagg_fast.py \
      --csv "$csv" \
      --patch-emb-cache "$cache" \
      --patch-params-pattern "$params_pattern" \
      --output-csv-pattern "$result_pattern" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 64 \
      --patch-temp-mode "$mode" \
      --patch-region-size "$region" \
      --patch-spat-weight 0.10 \
      --patch-temp-weight 0.90 \
      --aggs "${missing_results[@]}"
else
  echo "== Skip eval: all present for dataset=$dataset mode=$mode region=$region =="
fi

echo "== Group complete: dataset=$dataset mode=$mode region=$region =="
