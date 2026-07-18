#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

export PYTHONPATH=
export PYTHONNOUSERSITE=1
export TORCH_HOME=/tmp/torch_cache
export XDG_CACHE_HOME=/tmp

datasets=(videofeedback genvideo)
modes=(same_grid_lag1 same_grid_second_order)
regions=(1 2 3)
aggs=(bottomk0p2 bottomk0p5 mean)
score_device="${SCORE_DEVICE:-cpu}"
score_batch_size="${SCORE_BATCH_SIZE:-32}"
cache_workers="${CACHE_WORKERS:-1}"

mkdir -p STALL/precomputed STALL/results/patch_cross_dataset_ablation STALL/logs

for dataset in "${datasets[@]}"; do
  csv="STALL/cache/indexes/${dataset}.csv"
  cache="STALL/cache/patch_embeddings/${dataset}"

  for mode in "${modes[@]}"; do
    echo "============================================================"
    echo "Dataset=${dataset} Mode=${mode}"
    echo "============================================================"

    for region in "${regions[@]}"; do
      echo "== Fit shared params: dataset=${dataset} mode=${mode} region=${region} =="
      conda run --no-capture-output -n stall \
        python STALL/src/create_patch_params_multiagg.py \
          --csv "$csv" \
          --patch-emb-cache "$cache" \
          --output-template "STALL/precomputed/patch_params_${dataset}_real_${mode}_region${region}_{agg}_v2.npz" \
          --duration 2 \
          --compact \
          --real-only \
          --max-patches-for-fit 300000 \
          --patch-temp-mode "$mode" \
          --patch-region-size "$region" \
          --agg "${aggs[@]}" \
          --skip-existing
    done

    for region in "${regions[@]}"; do
      echo "== Eval multi: dataset=${dataset} mode=${mode} region=${region} =="
      echo "   score_device=${score_device} score_batch_size=${score_batch_size} cache_workers=${cache_workers}"
      conda run --no-capture-output -n stall \
        python STALL/src/eval_patch_fast_multi.py \
          --csv "$csv" \
          --patch-emb-cache "$cache" \
          --params-template "STALL/precomputed/patch_params_${dataset}_real_${mode}_region{region}_{agg}_v2.npz" \
          --output-template "STALL/results/patch_cross_dataset_ablation/${dataset}_patch_${mode}_region{region}_{agg}_spat10_temp90.csv" \
          --duration 2 \
          --compact \
          --score-device "$score_device" \
          --score-batch-size "$score_batch_size" \
          --cache-workers "$cache_workers" \
          --patch-temp-mode "$mode" \
          --patch-spat-weight 0.10 \
          --patch-temp-weight 0.90 \
          --regions "$region" \
          --agg "${aggs[@]}" \
          --skip-existing
    done
  done
done

echo "All fast cross-dataset patch ablations complete."
