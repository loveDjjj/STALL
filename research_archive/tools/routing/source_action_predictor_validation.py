#!/usr/bin/env python3
"""Leave-one-source validation for source-level split/universal predictors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


def _load_validation(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df[df["validation_mode"] == "fixed_threshold"].copy())
    return pd.concat(frames, ignore_index=True)


def _load_stats(paths: list[Path]) -> pd.DataFrame:
    frames = []
    dataset_map = {
        "comgenvid": "comgenvid_full",
        "videofeedback": "videofeedback_full",
        "genvideo": "genvideo_balanced300",
    }
    for path in paths:
        dataset = path.name.replace("heldout_source_", "").replace("_stats.csv", "")
        df = pd.read_csv(path).rename(columns={"source_model": "heldout_source"})
        df["dataset"] = dataset_map.get(dataset, dataset)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _make_table(validation_paths: list[Path], stats_paths: list[Path]) -> pd.DataFrame:
    validation = _load_validation(validation_paths)
    stats = _load_stats(stats_paths)
    df = validation.merge(stats, on=["dataset", "heldout_source"], how="left", validate="one_to_one")
    # Prefer split when it is at least tied with universal on AUC, with AP as tie-breaker.
    df["oracle_action"] = np.where(
        (df["delta_vs_universal_auc"] > 0)
        | ((df["delta_vs_universal_auc"].abs() <= 1e-12) & (df["delta_vs_universal_ap"] >= 0)),
        "split",
        "universal",
    )
    df["label_split"] = (df["oracle_action"] == "split").astype(int)
    df["log_n_fake"] = np.log1p(df["n_fake_y"].astype(float) if "n_fake_y" in df else df["n_fake"].astype(float))
    df["gate_conf_product"] = df["fake_gate_mean"] * df["fake_conf_mean"]
    df["gate_disagreement_product"] = df["fake_gate_mean"] * df["fake_disagreement_mean"]
    df["conf_minus_gate"] = df["fake_conf_mean"] - df["fake_gate_mean"]
    return df


def _model(name: str, random_state: int):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.5, solver="liblinear", random_state=random_state),
        )
    if name == "tree1":
        return DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight="balanced", random_state=random_state)
    if name == "tree2":
        return DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, class_weight="balanced", random_state=random_state)
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model: {name}")


def _action_gains(row: pd.Series, action: str) -> dict[str, float]:
    if action == "split":
        return {
            "delta_vs_universal_auc": float(row["delta_vs_universal_auc"]),
            "delta_vs_universal_ap": float(row["delta_vs_universal_ap"]),
            "delta_vs_base_auc": float(row["delta_auc"]),
            "delta_vs_base_ap": float(row["delta_ap"]),
        }
    if action == "universal":
        return {
            "delta_vs_universal_auc": 0.0,
            "delta_vs_universal_ap": 0.0,
            "delta_vs_base_auc": float(row["universal_auc"] - row["base_auc"]),
            "delta_vs_base_ap": float(row["universal_ap"] - row["base_ap"]),
        }
    raise ValueError(f"Unknown action: {action}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _make_table(args.validation_csvs, args.source_stats_csvs)
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    rows = []
    rule_rows = []
    X_all = df[features].to_numpy(float)
    y_all = df["label_split"].to_numpy(int)
    for model_name in models:
        for i, heldout in df.iterrows():
            train_mask = np.ones(len(df), dtype=bool)
            train_mask[i] = False
            clf = _model(model_name, args.random_state)
            clf.fit(X_all[train_mask], y_all[train_mask])
            pred = int(clf.predict(X_all[[i]])[0])
            pred_action = "split" if pred == 1 else "universal"
            gains = _action_gains(heldout, pred_action)
            oracle_gains = _action_gains(heldout, heldout["oracle_action"])
            rows.append(
                {
                    "model": model_name,
                    "dataset": heldout["dataset"],
                    "heldout_source": heldout["heldout_source"],
                    "pred_action": pred_action,
                    "oracle_action": heldout["oracle_action"],
                    "correct": pred_action == heldout["oracle_action"],
                    "delta_vs_universal_auc": gains["delta_vs_universal_auc"],
                    "delta_vs_universal_ap": gains["delta_vs_universal_ap"],
                    "delta_vs_base_auc": gains["delta_vs_base_auc"],
                    "delta_vs_base_ap": gains["delta_vs_base_ap"],
                    "oracle_delta_vs_universal_auc": oracle_gains["delta_vs_universal_auc"],
                    "oracle_delta_vs_universal_ap": oracle_gains["delta_vs_universal_ap"],
                    "oracle_delta_vs_base_auc": oracle_gains["delta_vs_base_auc"],
                    "oracle_delta_vs_base_ap": oracle_gains["delta_vs_base_ap"],
                    "regret_vs_oracle_auc": oracle_gains["delta_vs_universal_auc"]
                    - gains["delta_vs_universal_auc"],
                    "regret_vs_oracle_ap": oracle_gains["delta_vs_universal_ap"] - gains["delta_vs_universal_ap"],
                    **{f: float(heldout[f]) for f in features},
                }
            )
            if model_name.startswith("tree"):
                fitted = clf
                rule_rows.append(
                    {
                        "model": model_name,
                        "heldout_source": heldout["heldout_source"],
                        "dataset": heldout["dataset"],
                        "tree": export_text(fitted, feature_names=features),
                    }
                )

    pred = pd.DataFrame(rows)
    summary = pred.groupby("model").agg(
        accuracy=("correct", "mean"),
        split_rate=("pred_action", lambda s: float((s == "split").mean())),
        mean_delta_vs_universal_auc=("delta_vs_universal_auc", "mean"),
        min_delta_vs_universal_auc=("delta_vs_universal_auc", "min"),
        mean_delta_vs_universal_ap=("delta_vs_universal_ap", "mean"),
        min_delta_vs_universal_ap=("delta_vs_universal_ap", "min"),
        mean_delta_vs_base_auc=("delta_vs_base_auc", "mean"),
        min_delta_vs_base_auc=("delta_vs_base_auc", "min"),
        mean_delta_vs_base_ap=("delta_vs_base_ap", "mean"),
        min_delta_vs_base_ap=("delta_vs_base_ap", "min"),
        mean_regret_auc=("regret_vs_oracle_auc", "mean"),
        max_regret_auc=("regret_vs_oracle_auc", "max"),
        mean_regret_ap=("regret_vs_oracle_ap", "mean"),
        max_regret_ap=("regret_vs_oracle_ap", "max"),
        n=("heldout_source", "count"),
    ).reset_index()
    oracle_summary = pd.DataFrame(
        [
            {
                "model": "oracle",
                "accuracy": 1.0,
                "split_rate": float((df["oracle_action"] == "split").mean()),
                "mean_delta_vs_universal_auc": pred[pred["model"] == models[0]][
                    "oracle_delta_vs_universal_auc"
                ].mean(),
                "min_delta_vs_universal_auc": pred[pred["model"] == models[0]][
                    "oracle_delta_vs_universal_auc"
                ].min(),
                "mean_delta_vs_universal_ap": pred[pred["model"] == models[0]][
                    "oracle_delta_vs_universal_ap"
                ].mean(),
                "min_delta_vs_universal_ap": pred[pred["model"] == models[0]][
                    "oracle_delta_vs_universal_ap"
                ].min(),
                "mean_delta_vs_base_auc": pred[pred["model"] == models[0]]["oracle_delta_vs_base_auc"].mean(),
                "min_delta_vs_base_auc": pred[pred["model"] == models[0]]["oracle_delta_vs_base_auc"].min(),
                "mean_delta_vs_base_ap": pred[pred["model"] == models[0]]["oracle_delta_vs_base_ap"].mean(),
                "min_delta_vs_base_ap": pred[pred["model"] == models[0]]["oracle_delta_vs_base_ap"].min(),
                "mean_regret_auc": 0.0,
                "max_regret_auc": 0.0,
                "mean_regret_ap": 0.0,
                "max_regret_ap": 0.0,
                "n": len(df),
            }
        ]
    )
    summary = pd.concat([summary, oracle_summary], ignore_index=True)
    rules = pd.DataFrame(rule_rows)
    return pred, summary, rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--source-stats-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--rules-csv", type=Path, required=True)
    parser.add_argument(
        "--features",
        default=(
            "fake_gate_mean,fake_disagreement_mean,fake_conf_mean,log_n_fake,"
            "gate_conf_product,gate_disagreement_product,conf_minus_gate"
        ),
    )
    parser.add_argument("--models", default="majority,logistic,tree1,tree2")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    pred, summary, rules = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    rules.to_csv(args.rules_csv, index=False)
    print("SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved source action predictions -> {args.output_csv}")
    print(f"Saved source action summary -> {args.summary_csv}")
    print(f"Saved source action rules -> {args.rules_csv}")


if __name__ == "__main__":
    main()
