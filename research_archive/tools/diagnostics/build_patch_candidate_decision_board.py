#!/usr/bin/env python3
"""Build a compact decision board for the current patch/fallback candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _fmt(value: float) -> str:
    return f"{value:+.6f}"


def build(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    promotion = _read(args.strict_promotion_decision)
    source_audit = _read(args.source_audit_summary)
    all_source = source_audit[source_audit["dataset"] == "ALL"].iloc[0]
    plus_hotshot = _read(args.plus_hotshot_source_audit_summary) if args.plus_hotshot_source_audit_summary.exists() else None
    if plus_hotshot is not None:
        plus_all = plus_hotshot[plus_hotshot["dataset"] == "ALL"].iloc[0]
        source_risk_result = (
            f"source negatives AUC/AP={int(all_source['n_negative_auc'])}/{int(all_source['n_negative_ap'])}, "
            f"min delta AUC/AP={_fmt(float(all_source['min_delta_auc']))}/{_fmt(float(all_source['min_delta_ap']))}; "
            f"+Hotshot local 31-source negatives={int(plus_all['n_negative_auc'])}/{int(plus_all['n_negative_ap'])}, "
            f"min delta={_fmt(float(plus_all['min_delta_auc']))}/{_fmt(float(plus_all['min_delta_ap']))}"
        )
        source_evidence = f"{args.strict_promotion_decision}; {args.source_audit_summary}; {args.plus_hotshot_source_audit_summary}"
    else:
        source_risk_result = (
            f"source negatives AUC/AP={int(all_source['n_negative_auc'])}/{int(all_source['n_negative_ap'])}, "
            f"min delta AUC/AP={_fmt(float(all_source['min_delta_auc']))}/{_fmt(float(all_source['min_delta_ap']))}"
        )
        source_evidence = f"{args.strict_promotion_decision}; {args.source_audit_summary}"
    decision = str(promotion.iloc[0]["decision"])
    rows.append(
        {
            "branch": "frozen_rawbase_splitminus_candidate",
            "status": "PROMOTE_INTERNAL_DEFAULT" if decision == "PASS" else "HOLD",
            "evidence": source_evidence,
            "primary_result": f"strict_gate={decision}, checks={int(promotion.iloc[0]['n_passed'])}/{int(promotion.iloc[0]['n_checks'])}",
            "risk_result": source_risk_result,
            "next_action": "Use as frozen internal default; require fresh validation before external promotion.",
        }
    )

    residual = _read(args.residual_summary)
    all_residual = residual[residual["dataset"] == "ALL"].iloc[0]
    rows.append(
        {
            "branch": "primitive_action_headroom",
            "status": "REJECT_MORE_SOURCE_LEVEL_THREE_WAY_ROUTING",
            "evidence": str(args.residual_summary),
            "primary_result": (
                f"sources_with_primitive_headroom={int(all_residual['n_sources_with_primitive_headroom'])}, "
                f"top_priority={all_residual['top_priority_source']}"
            ),
            "risk_result": (
                f"min final AUC/AP={float(all_residual['min_final_auc']):.6f}/{float(all_residual['min_final_ap']):.6f}"
            ),
            "next_action": "Do not keep tuning source-level base/universal/split selection.",
        }
    )

    hard_tail = _read(args.hard_tail_aggregate)
    deployable = hard_tail[hard_tail["apply_mode"] == "all_samples"].sort_values(
        ["passes_source_nonregression", "mean_delta_avg_auc", "mean_delta_avg_ap"],
        ascending=[False, False, False],
    ).iloc[0]
    oracle = hard_tail[hard_tail["apply_mode"] == "fake_only_oracle"].sort_values(
        ["mean_delta_avg_auc", "mean_delta_avg_ap"], ascending=False
    ).iloc[0]
    rows.append(
        {
            "branch": "posthoc_hard_tail_score_cap",
            "status": "REJECT",
            "evidence": str(args.hard_tail_aggregate),
            "primary_result": (
                f"best deployable mean delta AUC/AP="
                f"{_fmt(float(deployable['mean_delta_avg_auc']))}/{_fmt(float(deployable['mean_delta_avg_ap']))}"
            ),
            "risk_result": (
                f"oracle upper bound only {_fmt(float(oracle['mean_delta_avg_auc']))}/"
                f"{_fmt(float(oracle['mean_delta_avg_ap']))}; features too weak"
            ),
            "next_action": "Do not promote post-hoc cap; use richer residual-tail features only as diagnostics.",
        }
    )

    probe = _read(args.residual_probe_summary)
    best_probe = probe.sort_values(["best_global_auc", "best_source_balanced_auc"], ascending=False).iloc[0]
    rows.append(
        {
            "branch": "residual_tail_feature_signal",
            "status": "DIAGNOSTIC_SIGNAL_CONFIRMED",
            "evidence": str(args.residual_probe_summary),
            "primary_result": (
                f"best={best_probe['dataset']}/{best_probe['feature_family']}/"
                f"{best_probe['best_global_feature']} AUC={float(best_probe['best_global_auc']):.6f}"
            ),
            "risk_result": "Tail labels are proxy labels derived from frozen-score tails.",
            "next_action": "Use only to design held-out/fresh real-safe gates.",
        }
    )

    heldout = _read(args.heldout_gate_summary)
    all_samples = heldout[heldout["apply_mode"] == "all_samples"]
    oracle_rows = heldout[heldout["apply_mode"] == "fake_only_oracle"]
    worst_all = all_samples.sort_values(["min_heldout_delta_ap", "min_heldout_delta_auc"]).iloc[0]
    best_oracle = oracle_rows.sort_values(["mean_heldout_delta_auc", "mean_heldout_delta_ap"], ascending=False).iloc[0]
    rows.append(
        {
            "branch": "residual_tail_feature_gate",
            "status": "HOLD_DIAGNOSTIC_ONLY",
            "evidence": str(args.heldout_gate_summary),
            "primary_result": (
                f"best oracle mean heldout delta AUC/AP="
                f"{_fmt(float(best_oracle['mean_heldout_delta_auc']))}/"
                f"{_fmt(float(best_oracle['mean_heldout_delta_ap']))}"
            ),
            "risk_result": (
                f"all_samples has negative heldout sources; worst min delta AUC/AP="
                f"{_fmt(float(worst_all['min_heldout_delta_auc']))}/"
                f"{_fmt(float(worst_all['min_heldout_delta_ap']))}"
            ),
            "next_action": "Need fresh/held-out real-safe calibration before any scoring correction.",
        }
    )

    scaffold = _read(args.fresh_scaffold_verification)
    fresh_gap = _read(args.fresh_promotion_readiness_gap)
    fresh_gap_row = fresh_gap.iloc[0]
    n_passed = int(scaffold["passed"].sum())
    rows.append(
        {
            "branch": "fresh_validation_pipeline",
            "status": "READY_NOT_RUN",
            "evidence": f"{args.fresh_scaffold_verification}; {args.fresh_promotion_readiness_gap}",
            "primary_result": (
                f"scaffold verification PASS {n_passed}/{len(scaffold)}; "
                f"fresh readiness={fresh_gap_row['status']}"
            ),
            "risk_result": (
                f"fresh_like_ready_count={int(fresh_gap_row['fresh_like_ready_count'])}; "
                f"hotshot_status={fresh_gap_row['hotshot_status']}"
            ),
            "next_action": str(fresh_gap_row["next_action"]),
        }
    )

    hotshot_gap = _read(args.hotshot_gap_decision)
    hotshot_runbook = _read(args.hotshot_completion_summary)
    hotshot_prefill = _read(args.hotshot_cache_prefill_summary)
    hotshot_duration = _read(args.hotshot_duration_gate)
    hotshot_merge = _read(args.hotshot_merge_decision)
    hotshot_result = _read(args.hotshot_local_variant_summary) if args.hotshot_local_variant_summary.exists() else None
    if hotshot_result is not None:
        hotshot_status = "COMPLETED_LOCAL_VARIANT_WITH_CAVEAT"
        hotshot_primary = (
            f"local delta AUC/AP={_fmt(float(hotshot_result.iloc[0]['delta_avg_auc']))}/"
            f"{_fmt(float(hotshot_result.iloc[0]['delta_avg_ap']))}; "
            f"Hotshot-XL AUC/AP={float(hotshot_result.iloc[0]['hotshot_xl_auc']):.6f}/"
            f"{float(hotshot_result.iloc[0]['hotshot_xl_ap']):.6f}"
        )
        hotshot_next = (
            "Use as completed local stress-test evidence only; do not use as fresh external or protocol-compatible promotion evidence."
        )
        hotshot_evidence = (
            f"{args.hotshot_gap_decision}; {args.hotshot_completion_summary}; "
            f"{args.hotshot_cache_prefill_summary}; {args.hotshot_duration_gate}; "
            f"{args.hotshot_merge_decision}; {args.hotshot_local_variant_summary}"
        )
    else:
        hotshot_status = "READY_FOR_LONG_LOCAL_VARIANT_JOB_WITH_CAVEAT"
        hotshot_primary = (
            f"missing source={hotshot_gap.iloc[0]['blocking_source_models']}, "
            f"target rows={int(hotshot_runbook.iloc[0]['target_rows'])}, "
            f"cache misses={int(hotshot_prefill.iloc[0]['misses_before'])}"
        )
        hotshot_next = (
            "Run Hotshot-XL cache/scoring as local variant only; do not use as fresh external or protocol-compatible promotion evidence."
        )
        hotshot_evidence = (
            f"{args.hotshot_gap_decision}; {args.hotshot_completion_summary}; "
            f"{args.hotshot_cache_prefill_summary}; {args.hotshot_duration_gate}; "
            f"{args.hotshot_merge_decision}"
        )
    rows.append(
        {
            "branch": "hotshot_local_variant_pipeline",
            "status": hotshot_status,
            "evidence": hotshot_evidence,
            "primary_result": hotshot_primary,
            "risk_result": (
                f"{hotshot_duration.iloc[0]['decision']}; "
                f"runbook duration={int(hotshot_duration.iloc[0]['runbook_duration'])}, "
                f"params duration={int(hotshot_duration.iloc[0]['params_duration'])}; "
                f"merge state={hotshot_merge.iloc[0]['decision']}"
            ),
            "next_action": hotshot_next,
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-promotion-decision", type=Path, default=Path("results/patch_calibrated_persistence/sample_fallback_full_scope_alpha060_rawbase_splitminus_strict_promotion_decision.csv"))
    parser.add_argument("--source-audit-summary", type=Path, default=Path("results/patch_calibrated_persistence/rawbase_splitminus_vs_default_source_audit_summary.csv"))
    parser.add_argument("--plus-hotshot-source-audit-summary", type=Path, default=Path("results/patch_calibrated_persistence/rawbase_splitminus_vs_default_plus_hotshot_local_source_audit_summary.csv"))
    parser.add_argument("--residual-summary", type=Path, default=Path("results/patch_calibrated_persistence/frozen_candidate_residual_summary.csv"))
    parser.add_argument("--hard-tail-aggregate", type=Path, default=Path("results/patch_calibrated_persistence/hard_tail_suppression_sweep_aggregate.csv"))
    parser.add_argument("--residual-probe-summary", type=Path, default=Path("results/patch_calibrated_persistence/residual_tail_feature_probe_summary.csv"))
    parser.add_argument("--heldout-gate-summary", type=Path, default=Path("results/patch_calibrated_persistence/heldout_residual_tail_gate_summary.csv"))
    parser.add_argument("--fresh-scaffold-verification", type=Path, default=Path("results/patch_calibrated_persistence/fresh_validation_scaffold_template_verification.csv"))
    parser.add_argument("--fresh-promotion-readiness-gap", type=Path, default=Path("results/patch_calibrated_persistence/fresh_promotion_readiness_gap.csv"))
    parser.add_argument("--hotshot-gap-decision", type=Path, default=Path("results/patch_calibrated_persistence/hotshot_scaffold_gap_decision.csv"))
    parser.add_argument("--hotshot-completion-summary", type=Path, default=Path("results/patch_calibrated_persistence/hotshot_completion_runbook_summary.csv"))
    parser.add_argument("--hotshot-cache-prefill-summary", type=Path, default=Path("results/patch_calibrated_persistence/hotshot_xl_patch_cache_prefill_dryrun_summary.csv"))
    parser.add_argument("--hotshot-duration-gate", type=Path, default=Path("results/patch_calibrated_persistence/hotshot_duration_compatibility_gate.csv"))
    parser.add_argument("--hotshot-merge-decision", type=Path, default=Path("results/patch_calibrated_persistence/hotshot_merge_preparation_decision.csv"))
    parser.add_argument("--hotshot-local-variant-summary", type=Path, default=Path("results/patch_calibrated_persistence/videofeedback_hotshot_local_variant_summary.csv"))
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    board = build(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.output_csv, index=False)
    print(board.to_string(index=False))
    print(f"Saved decision board -> {args.output_csv}")


if __name__ == "__main__":
    main()
