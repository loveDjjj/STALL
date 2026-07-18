#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <dataset> <mode> <region> <agg>" >&2
  exit 2
fi

dataset="$1"
mode="$2"
region="$3"
agg="$4"

cd /data/OneDay/STALL_project

# The legacy queue calls this script once per aggregation. For large datasets
# that rereads the same patch-cache files three times per dataset/mode/region.
# Delegate to the grouped runner so the first missing aggregation produces all
# aggregation CSVs in a single cache pass; later aggregation jobs will skip.
if [[ -x "STALL/scripts/run_one_patch_ablation_group.sh" ]]; then
  bash STALL/scripts/run_one_patch_ablation_group.sh "$dataset" "$mode" "$region"
  exit 0
fi

export PYTHONPATH=
export PYTHONNOUSERSITE=1
export TORCH_HOME=/tmp/torch_cache
export XDG_CACHE_HOME=/tmp

csv="STALL/cache/indexes/${dataset}.csv"
cache="STALL/cache/patch_embeddings/${dataset}"
params="STALL/precomputed/patch_params_${dataset}_real_${mode}_region${region}_${agg}_v2.npz"
result="STALL/results/patch_cross_dataset_ablation/${dataset}_patch_${mode}_region${region}_${agg}_spat10_temp90.csv"

mkdir -p STALL/precomputed STALL/results/patch_cross_dataset_ablation

case "$agg" in
  bottomk0p2)
    agg_args=(--aggregation bottomk_mean --bottomk-ratio 0.20)
    ;;
  bottomk0p5)
    agg_args=(--aggregation bottomk_mean --bottomk-ratio 0.50)
    ;;
  mean)
    agg_args=(--aggregation mean --bottomk-ratio 0.50)
    ;;
  *)
    echo "Unknown agg: $agg" >&2
    exit 2
    ;;
esac

if [[ ! -s "$params" ]]; then
  echo "== Fit params: dataset=$dataset mode=$mode region=$region agg=$agg =="
  conda run --no-capture-output -n stall \
    python STALL/src/create_patch_params.py \
      --csv "$csv" \
      --patch-emb-cache "$cache" \
      --output "$params" \
      --duration 2 \
      --compact \
      --real-only \
      --max-patches-for-fit 300000 \
      --patch-temp-mode "$mode" \
      --patch-region-size "$region" \
      "${agg_args[@]}"
else
  echo "== Skip params: $params =="
fi

if [[ ! -s "$result" ]]; then
  echo "== Eval: dataset=$dataset mode=$mode region=$region agg=$agg =="
  conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_fast.py \
      --csv "$csv" \
      --patch-emb-cache "$cache" \
      --patch-params "$params" \
      --output-csv "$result" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 64 \
      --patch-temp-mode "$mode" \
      --patch-spat-weight 0.10 \
      --patch-temp-weight 0.90
else
  echo "== Skip eval: $result =="
fi

echo "== Job complete: dataset=$dataset mode=$mode region=$region agg=$agg =="
