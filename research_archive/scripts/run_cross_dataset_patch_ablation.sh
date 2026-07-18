#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

export PYTHONPATH=
export PYTHONNOUSERSITE=1
export TORCH_HOME=/tmp/torch_cache
export XDG_CACHE_HOME=/tmp

mkdir -p STALL/logs STALL/results/patch_cross_dataset_ablation STALL/precomputed

DATASETS=(videofeedback genvideo)
MODES=(same_grid_lag1 same_grid_second_order)
REGIONS=(1 2 3)
AGGS=(bottomk0p2 bottomk0p5 mean)

csv_for_dataset() {
  echo "STALL/cache/indexes/$1.csv"
}

cache_for_dataset() {
  echo "STALL/cache/patch_embeddings/$1"
}

params_path() {
  local dataset="$1"
  local mode="$2"
  local region="$3"
  local agg="$4"
  echo "STALL/precomputed/patch_params_${dataset}_real_${mode}_region${region}_${agg}_v2.npz"
}

result_path() {
  local dataset="$1"
  local mode="$2"
  local region="$3"
  local agg="$4"
  echo "STALL/results/patch_cross_dataset_ablation/${dataset}_patch_${mode}_region${region}_${agg}_spat10_temp90.csv"
}

create_params() {
  local dataset="$1"
  local mode="$2"
  local region="$3"
  local agg="$4"
  local csv cache params
  csv="$(csv_for_dataset "$dataset")"
  cache="$(cache_for_dataset "$dataset")"
  params="$(params_path "$dataset" "$mode" "$region" "$agg")"

  if [[ -s "$params" ]]; then
    echo "[skip params] $params"
    return
  fi

  local agg_args=()
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

  echo
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
}

eval_result() {
  local dataset="$1"
  local mode="$2"
  local region="$3"
  local agg="$4"
  local csv cache params result
  csv="$(csv_for_dataset "$dataset")"
  cache="$(cache_for_dataset "$dataset")"
  params="$(params_path "$dataset" "$mode" "$region" "$agg")"
  result="$(result_path "$dataset" "$mode" "$region" "$agg")"

  if [[ -s "$result" ]]; then
    echo "[skip eval] $result"
    return
  fi

  echo
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
}

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for region in "${REGIONS[@]}"; do
      for agg in "${AGGS[@]}"; do
        create_params "$dataset" "$mode" "$region" "$agg"
        eval_result "$dataset" "$mode" "$region" "$agg"
      done
    done
  done
done

echo
echo "All patch ablation jobs finished."
