#!/usr/bin/env bash
set -euo pipefail

# 期刊版 patch 超参数敏感性续跑脚本。
#
# 设计原则：
# - 复用已有 patch cache 和 precomputed patch params；
# - 已存在的 score/metrics 文件默认跳过，方便中断后续跑；
# - 不提交 cache、临时 patch params 或原始视频，只提交轻量 CSV/Markdown 汇总。

DATASET="${1:-comgenvid}"
EXPERIMENT="${2:-bottomk}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
SCORE_DEVICE="${SCORE_DEVICE:-cuda}"

ensure_patch_params() {
  local dataset="$1"
  local params="$2"
  local aggregation="$3"
  local bottomk="$4"
  local region="$5"

  if [[ -f "${params}" ]]; then
    echo "[skip] ${params}"
    return 0
  fi

  mkdir -p "$(dirname "${params}")"
  conda run --no-capture-output -n stall python src/create_patch_params.py \
    --csv "cache/indexes/${dataset}.csv" \
    --patch-emb-cache "cache/patch_embeddings/${dataset}" \
    --compact \
    --real-only \
    --patch-temp-mode same_grid_second_order \
    --patch-region-size "${region}" \
    --aggregation "${aggregation}" \
    --bottomk-ratio "${bottomk}" \
    --output "${params}"
}

run_metric() {
  local score_csv="$1"
  local metrics_csv="$2"
  if [[ -f "${metrics_csv}" ]]; then
    echo "[skip] ${metrics_csv}"
    return 0
  fi
  conda run --no-capture-output -n stall python tools/eval_score_csv.py \
    --csv "${score_csv}" \
    --score-col patch_final_score \
    --output-csv "${metrics_csv}"
}

run_patch_eval() {
  local dataset="$1"
  local params="$2"
  local output_csv="$3"
  local aggregation="$4"
  local bottomk="$5"
  local region="$6"

  if [[ -f "${output_csv}" ]]; then
    echo "[skip] ${output_csv}"
    return 0
  fi

  conda run --no-capture-output -n stall python src/eval_patch_fast.py \
    --csv "cache/indexes/${dataset}.csv" \
    --patch-emb-cache "cache/patch_embeddings/${dataset}" \
    --compact \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --score-device "${SCORE_DEVICE}" \
    --patch-params "${params}" \
    --patch-temp-mode same_grid_second_order \
    --patch-region-size "${region}" \
    --aggregation "${aggregation}" \
    --bottomk-ratio "${bottomk}" \
    --patch-spat-weight 0.10 \
    --patch-temp-weight 0.90 \
    --output-csv "${output_csv}"
}

bottomk_tag() {
  local bottomk="$1"
  echo "${bottomk/./p}"
}

legacy_bottomk_tag() {
  local bottomk="$1"
  local tag
  tag="$(bottomk_tag "${bottomk}")"
  echo "${tag%0}"
}

resolve_bottomk_params() {
  local dataset="$1"
  local region="$2"
  local bottomk="$3"
  local tag
  local legacy_tag
  tag="$(bottomk_tag "${bottomk}")"
  legacy_tag="$(legacy_bottomk_tag "${bottomk}")"

  local params="precomputed/patch_params_${dataset}_real_same_grid_second_order_region${region}_bottomk${tag}_v2.npz"
  local legacy_params="precomputed/patch_params_${dataset}_real_same_grid_second_order_region${region}_bottomk${legacy_tag}_v2.npz"
  if [[ -f "${params}" ]]; then
    echo "${params}"
  elif [[ -f "${legacy_params}" ]]; then
    echo "${legacy_params}"
  else
    echo "${params}"
  fi
}

if [[ "${DATASET}" == "comgenvid" && "${EXPERIMENT}" == "bottomk" ]]; then
  mkdir -p results/journal_experiments/bottomk_sensitivity
  declare -A PARAMS=(
    ["0.10"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p10_v2.npz"
    ["0.15"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p15_v2.npz"
    ["0.20"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz"
    ["0.25"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p25_v2.npz"
    ["0.30"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p30_v2.npz"
    ["0.35"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p35_v2.npz"
    ["0.40"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p40_v2.npz"
    ["0.50"]="precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p50_v2.npz"
  )
  for bottomk in 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50; do
    tag="${bottomk/./p}"
    score_csv="results/journal_experiments/bottomk_sensitivity/comgenvid_region3_bottomk${tag}_patch.csv"
    metrics_csv="results/journal_experiments/bottomk_sensitivity/comgenvid_region3_bottomk${tag}_metrics.csv"
    run_patch_eval "comgenvid" "${PARAMS[${bottomk}]}" "${score_csv}" "bottomk_mean" "${bottomk}" 3
    run_metric "${score_csv}" "${metrics_csv}"
  done
  conda run --no-capture-output -n stall python tools/summarize_journal_experiments.py --kind bottomk
  echo "完成 ComGenVid bottom-k sensitivity。"
  exit 0
fi

if [[ "${EXPERIMENT}" == "region_mean" ]]; then
  mkdir -p results/journal_experiments/region_sensitivity
  for region in 1 2 3; do
    params="precomputed/patch_params_${DATASET}_real_same_grid_second_order_region${region}_mean_v2.npz"
    score_csv="results/journal_experiments/region_sensitivity/${DATASET}_region${region}_mean_patch.csv"
    metrics_csv="results/journal_experiments/region_sensitivity/${DATASET}_region${region}_mean_metrics.csv"
    ensure_patch_params "${DATASET}" "${params}" "mean" 0.50 "${region}"
    run_patch_eval "${DATASET}" "${params}" "${score_csv}" "mean" 0.50 "${region}"
    run_metric "${score_csv}" "${metrics_csv}"
  done
  conda run --no-capture-output -n stall python tools/summarize_journal_experiments.py --kind region
  echo "完成 ${DATASET} region mean sensitivity。"
  exit 0
fi

if [[ "${EXPERIMENT}" == "aggregation" ]]; then
  mkdir -p results/journal_experiments/aggregation_sensitivity
  case "${DATASET}" in
    videofeedback)
      main_region=1
      ;;
    genvideo)
      main_region=2
      ;;
    *)
      echo "aggregation sensitivity 当前支持 videofeedback/genvideo，实际为 ${DATASET}" >&2
      exit 2
      ;;
  esac

  for bottomk in 0.20 0.50; do
    tag="$(bottomk_tag "${bottomk}")"
    params="$(resolve_bottomk_params "${DATASET}" "${main_region}" "${bottomk}")"
    score_csv="results/journal_experiments/aggregation_sensitivity/${DATASET}_region${main_region}_bottomk${tag}_patch.csv"
    metrics_csv="results/journal_experiments/aggregation_sensitivity/${DATASET}_region${main_region}_bottomk${tag}_metrics.csv"
    ensure_patch_params "${DATASET}" "${params}" "bottomk_mean" "${bottomk}" "${main_region}"
    run_patch_eval "${DATASET}" "${params}" "${score_csv}" "bottomk_mean" "${bottomk}" "${main_region}"
    run_metric "${score_csv}" "${metrics_csv}"
  done
  conda run --no-capture-output -n stall python tools/summarize_journal_experiments.py --kind aggregation
  echo "完成 ${DATASET} aggregation sensitivity。"
  exit 0
fi

cat >&2 <<EOF
未知组合: DATASET=${DATASET}, EXPERIMENT=${EXPERIMENT}

支持：
  bash scripts/experiments/run_journal_patch_sensitivity.sh comgenvid bottomk
  bash scripts/experiments/run_journal_patch_sensitivity.sh videofeedback region_mean
  bash scripts/experiments/run_journal_patch_sensitivity.sh genvideo region_mean
  bash scripts/experiments/run_journal_patch_sensitivity.sh videofeedback aggregation
  bash scripts/experiments/run_journal_patch_sensitivity.sh genvideo aggregation
EOF
exit 2
