#!/usr/bin/env python3
"""Held-out validation for a source-level split-vs-universal fallback.

For each held-out fake source, choose a label-free rule on the remaining
sources to decide whether a source should use split policy or fall back to the
universal sample gate. Then evaluate that rule on the held-out source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df["validation_mode"] == "fixed_threshold"].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _source_stats(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        dataset = path.name.replace("heldout_source_", "").replace("_stats.csv", "")
        # Normalize to the dataset keys used by validation CSVs.
        dataset_map = {
            "comgenvid": "comgenvid_full",
            "videofeedback": "videofeedback_full",
            "genvideo": "genvideo_balanced300",
        }
        df["dataset"] = dataset_map.get(dataset, dataset)
        df = df.rename(columns={"n_fake": "source_n_fake"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _action_gains(rows: pd.DataFrame, use_split: np.ndarray, fallback_action: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fallback_action == "universal":
        auc_vs_universal = np.where(use_split, rows["delta_vs_universal_auc"], 0.0)
        ap_vs_universal = np.where(use_split, rows["delta_vs_universal_ap"], 0.0)
        auc_vs_base = np.where(use_split, rows["delta_auc"], rows["universal_auc"] - rows["base_auc"])
        ap_vs_base = np.where(use_split, rows["delta_ap"], rows["universal_ap"] - rows["base_ap"])
    elif fallback_action == "base":
        auc_vs_universal = np.where(use_split, rows["delta_vs_universal_auc"], rows["base_auc"] - rows["universal_auc"])
        ap_vs_universal = np.where(use_split, rows["delta_vs_universal_ap"], rows["base_ap"] - rows["universal_ap"])
        auc_vs_base = np.where(use_split, rows["delta_auc"], 0.0)
        ap_vs_base = np.where(use_split, rows["delta_ap"], 0.0)
    else:
        raise ValueError(f"Unknown fallback action: {fallback_action}")
    return auc_vs_universal, ap_vs_universal, auc_vs_base, ap_vs_base


def _candidate_rules(
    train: pd.DataFrame,
    feature_names: list[str],
    thresholds: list[float],
    fallback_actions: list[str],
) -> list[dict]:
    rules = []
    for feature in feature_names:
        for threshold in thresholds:
            for direction in ["ge", "lt"]:
                if direction == "ge":
                    use_split = train[feature] >= threshold
                else:
                    use_split = train[feature] < threshold
                # If the rule never chooses either side on training data, keep it
                # out. Such rules are just aliases for always-split/universal.
                if use_split.nunique() < 2:
                    continue
                for fallback_action in fallback_actions:
                    auc_u, ap_u, auc_b, ap_b = _action_gains(train, use_split.to_numpy(bool), fallback_action)
                    rules.append(
                        {
                            "rule_type": "threshold",
                            "feature": feature,
                            "threshold": threshold,
                            "direction": direction,
                            "fallback_action": fallback_action,
                            "train_mean_auc_gain": float(np.mean(auc_u)),
                            "train_min_auc_gain": float(np.min(auc_u)),
                            "train_mean_ap_gain": float(np.mean(ap_u)),
                            "train_min_ap_gain": float(np.min(ap_u)),
                            "train_mean_base_auc_gain": float(np.mean(auc_b)),
                            "train_min_base_auc_gain": float(np.min(auc_b)),
                            "train_mean_base_ap_gain": float(np.mean(ap_b)),
                            "train_min_base_ap_gain": float(np.min(ap_b)),
                        }
                    )
    # Baselines: always split, always universal, or always base.
    rules.append(
        {
            "rule_type": "always_split",
            "feature": "",
            "threshold": np.nan,
            "direction": "",
            "fallback_action": "split",
            "train_mean_auc_gain": float(train["delta_vs_universal_auc"].mean()),
            "train_min_auc_gain": float(train["delta_vs_universal_auc"].min()),
            "train_mean_ap_gain": float(train["delta_vs_universal_ap"].mean()),
            "train_min_ap_gain": float(train["delta_vs_universal_ap"].min()),
            "train_mean_base_auc_gain": float(train["delta_auc"].mean()),
            "train_min_base_auc_gain": float(train["delta_auc"].min()),
            "train_mean_base_ap_gain": float(train["delta_ap"].mean()),
            "train_min_base_ap_gain": float(train["delta_ap"].min()),
        }
    )
    if "universal" in fallback_actions:
        rules.append(
            {
                "rule_type": "always_universal",
                "feature": "",
                "threshold": np.nan,
                "direction": "",
                "fallback_action": "universal",
                "train_mean_auc_gain": 0.0,
                "train_min_auc_gain": 0.0,
                "train_mean_ap_gain": 0.0,
                "train_min_ap_gain": 0.0,
                "train_mean_base_auc_gain": float((train["universal_auc"] - train["base_auc"]).mean()),
                "train_min_base_auc_gain": float((train["universal_auc"] - train["base_auc"]).min()),
                "train_mean_base_ap_gain": float((train["universal_ap"] - train["base_ap"]).mean()),
                "train_min_base_ap_gain": float((train["universal_ap"] - train["base_ap"]).min()),
            }
        )
    if "base" in fallback_actions:
        rules.append(
            {
                "rule_type": "always_base",
                "feature": "",
                "threshold": np.nan,
                "direction": "",
                "fallback_action": "base",
                "train_mean_auc_gain": float((train["base_auc"] - train["universal_auc"]).mean()),
                "train_min_auc_gain": float((train["base_auc"] - train["universal_auc"]).min()),
                "train_mean_ap_gain": float((train["base_ap"] - train["universal_ap"]).mean()),
                "train_min_ap_gain": float((train["base_ap"] - train["universal_ap"]).min()),
                "train_mean_base_auc_gain": 0.0,
                "train_min_base_auc_gain": 0.0,
                "train_mean_base_ap_gain": 0.0,
                "train_min_base_ap_gain": 0.0,
            }
        )
    return rules


def _choose_rule(
    train: pd.DataFrame,
    feature_names: list[str],
    thresholds: list[float],
    fallback_actions: list[str],
    objective: str,
) -> dict:
    rules = _candidate_rules(train, feature_names, thresholds, fallback_actions)
    if objective == "mean_auc":
        key = lambda r: (r["train_mean_auc_gain"], r["train_min_auc_gain"], r["train_mean_ap_gain"])
    elif objective == "min_auc":
        key = lambda r: (r["train_min_auc_gain"], r["train_mean_auc_gain"], r["train_min_ap_gain"])
    elif objective == "mean_ap":
        key = lambda r: (r["train_mean_ap_gain"], r["train_min_ap_gain"], r["train_mean_auc_gain"])
    elif objective == "min_base_auc":
        key = lambda r: (r["train_min_base_auc_gain"], r["train_mean_base_auc_gain"], r["train_min_base_ap_gain"])
    elif objective == "mean_base_auc":
        key = lambda r: (r["train_mean_base_auc_gain"], r["train_min_base_auc_gain"], r["train_mean_base_ap_gain"])
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return max(rules, key=key)


def _apply_rule(row: pd.Series, rule: dict) -> bool:
    if rule["rule_type"] == "always_split":
        return True
    if rule["rule_type"] == "always_universal":
        return False
    if rule["direction"] == "ge":
        return bool(row[rule["feature"]] >= rule["threshold"])
    if rule["direction"] == "lt":
        return bool(row[rule["feature"]] < rule["threshold"])
    raise ValueError(f"Unknown rule: {rule}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = _load_inputs(args.validation_csvs)
    stats = _source_stats(args.source_stats_csvs)
    df = validation.merge(
        stats,
        left_on=["dataset", "heldout_source"],
        right_on=["dataset", "source_model"],
        how="left",
        validate="one_to_one",
    )
    feature_names = [x.strip() for x in args.features.split(",") if x.strip()]
    thresholds = _parse_floats(args.thresholds)
    fallback_actions = [x.strip() for x in args.fallback_actions.split(",") if x.strip()]
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    rows = []
    rule_rows = []
    for objective in [x.strip() for x in args.objectives.split(",") if x.strip()]:
        for idx, heldout in df.iterrows():
            if args.scope == "within_dataset":
                train = df[(df["dataset"] == heldout["dataset"]) & (df.index != idx)]
            elif args.scope == "all_other_sources":
                train = df[df.index != idx]
            else:
                raise ValueError(f"Unknown scope: {args.scope}")
            if len(train) == 0:
                continue
            rule = _choose_rule(train, feature_names, thresholds, fallback_actions, objective)
            use_split = _apply_rule(heldout, rule)
            if use_split:
                chosen_action = "split"
                auc_gain_vs_universal = float(heldout["delta_vs_universal_auc"])
                ap_gain_vs_universal = float(heldout["delta_vs_universal_ap"])
                auc_gain_vs_base = float(heldout["delta_auc"])
                ap_gain_vs_base = float(heldout["delta_ap"])
            elif rule["fallback_action"] == "base":
                chosen_action = "base"
                auc_gain_vs_universal = float(heldout["base_auc"] - heldout["universal_auc"])
                ap_gain_vs_universal = float(heldout["base_ap"] - heldout["universal_ap"])
                auc_gain_vs_base = 0.0
                ap_gain_vs_base = 0.0
            else:
                chosen_action = "universal"
                auc_gain_vs_universal = 0.0
                ap_gain_vs_universal = 0.0
                auc_gain_vs_base = float(heldout["universal_auc"] - heldout["base_auc"])
                ap_gain_vs_base = float(heldout["universal_ap"] - heldout["base_ap"])
            rows.append(
                {
                    "dataset": heldout["dataset"],
                    "heldout_source": heldout["heldout_source"],
                    "objective": objective,
                    "scope": args.scope,
                    "use_split": use_split,
                    "chosen_action": chosen_action,
                    "rule_type": rule["rule_type"],
                    "feature": rule["feature"],
                    "threshold": rule["threshold"],
                    "direction": rule["direction"],
                    "fallback_action": rule["fallback_action"],
                    "train_mean_auc_gain": rule["train_mean_auc_gain"],
                    "train_min_auc_gain": rule["train_min_auc_gain"],
                    "train_mean_ap_gain": rule["train_mean_ap_gain"],
                    "train_min_ap_gain": rule["train_min_ap_gain"],
                    "train_mean_base_auc_gain": rule["train_mean_base_auc_gain"],
                    "train_min_base_auc_gain": rule["train_min_base_auc_gain"],
                    "train_mean_base_ap_gain": rule["train_mean_base_ap_gain"],
                    "train_min_base_ap_gain": rule["train_min_base_ap_gain"],
                    "fallback_delta_vs_universal_auc": auc_gain_vs_universal,
                    "fallback_delta_vs_universal_ap": ap_gain_vs_universal,
                    "fallback_delta_vs_base_auc": auc_gain_vs_base,
                    "fallback_delta_vs_base_ap": ap_gain_vs_base,
                    "split_delta_vs_universal_auc": float(heldout["delta_vs_universal_auc"]),
                    "split_delta_vs_universal_ap": float(heldout["delta_vs_universal_ap"]),
                    "split_delta_vs_base_auc": float(heldout["delta_auc"]),
                    "split_delta_vs_base_ap": float(heldout["delta_ap"]),
                    "universal_delta_vs_base_auc": float(heldout["universal_auc"] - heldout["base_auc"]),
                    "universal_delta_vs_base_ap": float(heldout["universal_ap"] - heldout["base_ap"]),
                    "fake_gate_mean": float(heldout["fake_gate_mean"]),
                    "fake_disagreement_mean": float(heldout["fake_disagreement_mean"]),
                    "fake_conf_mean": float(heldout["fake_conf_mean"]),
                    "n_fake": int(heldout["source_n_fake"]),
                }
            )
            rule_rows.append(
                {
                    "dataset": heldout["dataset"],
                    "heldout_source": heldout["heldout_source"],
                    "objective": objective,
                    "scope": args.scope,
                    **rule,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(rule_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--source-stats-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rules-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--features", default="fake_gate_mean,fake_disagreement_mean,fake_conf_mean,source_n_fake")
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,100,300,1000,2000,4000")
    parser.add_argument("--objectives", default="mean_auc,min_auc,mean_ap,min_base_auc,mean_base_auc")
    parser.add_argument("--fallback-actions", default="universal")
    parser.add_argument("--scope", choices=["within_dataset", "all_other_sources"], default="all_other_sources")
    args = parser.parse_args()

    out, rules = run(args)
    summary = out.groupby(["objective", "scope"]).agg(
        mean_delta_vs_universal_auc=("fallback_delta_vs_universal_auc", "mean"),
        min_delta_vs_universal_auc=("fallback_delta_vs_universal_auc", "min"),
        mean_delta_vs_universal_ap=("fallback_delta_vs_universal_ap", "mean"),
        min_delta_vs_universal_ap=("fallback_delta_vs_universal_ap", "min"),
        mean_delta_vs_base_auc=("fallback_delta_vs_base_auc", "mean"),
        min_delta_vs_base_auc=("fallback_delta_vs_base_auc", "min"),
        mean_delta_vs_base_ap=("fallback_delta_vs_base_ap", "mean"),
        min_delta_vs_base_ap=("fallback_delta_vs_base_ap", "min"),
        split_rate=("use_split", "mean"),
        base_rate=("chosen_action", lambda s: float((s == "base").mean())),
        universal_rate=("chosen_action", lambda s: float((s == "universal").mean())),
        n_sources=("heldout_source", "count"),
    ).reset_index()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    rules.to_csv(args.rules_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    print("SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved fallback validation -> {args.output_csv}")
    print(f"Saved fallback rules -> {args.rules_csv}")
    print(f"Saved fallback summary -> {args.summary_csv}")


if __name__ == "__main__":
    main()
