#!/usr/bin/env python3
"""Build a compact comparison table for default and fallback protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_ORDER = ["comgenvid", "genvideo", "videofeedback"]
FALLBACK_NAME_MAP = {
    "comgenvid_full": "comgenvid",
    "comgenvid_full_alpha060": "comgenvid",
    "genvideo_balanced300": "genvideo",
    "genvideo_full_alpha060": "genvideo",
    "comgenvid_full_alpha060_rawbase": "comgenvid",
    "genvideo_full_alpha060_rawbase": "genvideo",
    "videofeedback_full_alpha060_rawbase": "videofeedback",
    "comgenvid_full_alpha060_rawbase_splitminus": "comgenvid",
    "genvideo_full_alpha060_rawbase_splitminus": "genvideo",
    "videofeedback_full_alpha060_rawbase_splitminus": "videofeedback",
    "videofeedback_full": "videofeedback",
}
DEFAULT_SCOPE = {
    "comgenvid": "full",
    "genvideo": "full",
    "videofeedback": "full",
}
FALLBACK_SCOPE = {
    "comgenvid_full": "full",
    "comgenvid_full_alpha060": "full",
    "genvideo_balanced300": "balanced300",
    "genvideo_full_alpha060": "full",
    "comgenvid_full_alpha060_rawbase": "full",
    "genvideo_full_alpha060_rawbase": "full",
    "videofeedback_full_alpha060_rawbase": "full",
    "comgenvid_full_alpha060_rawbase_splitminus": "full",
    "genvideo_full_alpha060_rawbase_splitminus": "full",
    "videofeedback_full_alpha060_rawbase_splitminus": "full",
    "videofeedback_full": "full",
}


def _default_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = df[df["method"] == "fixed060"].copy()
    rows["protocol"] = "alpha060_default"
    rows["usage"] = "deployable_default"
    rows["dataset_key"] = rows["dataset"]
    rows["split_scope"] = rows["dataset_key"].map(DEFAULT_SCOPE)
    rows["base_alpha"] = 0.60
    return rows.rename(columns={"avg_auc": "auc", "avg_ap": "ap"})[
        ["protocol", "usage", "dataset_key", "split_scope", "base_alpha", "auc", "ap"]
    ]


def _fallback_rows(path: Path, protocol: str, usage: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dataset"] != "Mean"].copy()
    df["protocol"] = protocol
    df["usage"] = usage
    df["dataset_key"] = df["dataset"].map(FALLBACK_NAME_MAP)
    if df["dataset_key"].isna().any():
        missing = df.loc[df["dataset_key"].isna(), "dataset"].tolist()
        raise ValueError(f"Unmapped fallback datasets: {missing}")
    df["split_scope"] = df["dataset"].map(FALLBACK_SCOPE)
    if protocol == "dataset_best_fallback":
        df["base_alpha"] = df["dataset_key"].map({"comgenvid": 0.20, "genvideo": 0.60, "videofeedback": 0.60})
    else:
        df["base_alpha"] = 0.60
    return df.rename(columns={"avg_auc": "auc", "avg_ap": "ap"})[
        ["protocol", "usage", "dataset_key", "split_scope", "base_alpha", "auc", "ap"]
    ]


def _add_means(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for protocol, group in rows.groupby("protocol", sort=False):
        group = group.copy()
        out.append(group)
        first = group.iloc[0]
        out.append(
            pd.DataFrame(
                [
                    {
                        "protocol": protocol,
                        "usage": first["usage"],
                        "dataset_key": "Mean",
                        "split_scope": "mixed",
                        "base_alpha": float("nan"),
                        "auc": group["auc"].mean(),
                        "ap": group["ap"].mean(),
                    }
                ]
            )
        )
    return pd.concat(out, ignore_index=True)


def _with_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    default = rows[(rows["protocol"] == "alpha060_default") & (rows["dataset_key"] != "Mean")][
        ["dataset_key", "split_scope", "auc", "ap"]
    ].rename(columns={"split_scope": "default_split_scope", "auc": "default_auc", "ap": "default_ap"})
    out = rows.merge(default, on="dataset_key", how="left")
    out["comparable_to_alpha060_default"] = out["split_scope"] == out["default_split_scope"]
    out["delta_vs_alpha060_auc"] = out["auc"] - out["default_auc"]
    out["delta_vs_alpha060_ap"] = out["ap"] - out["default_ap"]
    out.loc[~out["comparable_to_alpha060_default"], ["delta_vs_alpha060_auc", "delta_vs_alpha060_ap"]] = pd.NA
    mean_mask = out["dataset_key"] == "Mean"
    default_mean = out[(out["protocol"] == "alpha060_default") & mean_mask].iloc[0]
    out.loc[mean_mask, "default_auc"] = default_mean["auc"]
    out.loc[mean_mask, "default_ap"] = default_mean["ap"]
    out.loc[mean_mask, "default_split_scope"] = "mixed"
    out.loc[mean_mask, "comparable_to_alpha060_default"] = False
    out.loc[mean_mask, ["delta_vs_alpha060_auc", "delta_vs_alpha060_ap"]] = pd.NA
    return out


def _markdown(rows: pd.DataFrame) -> str:
    ordered = rows.copy()
    protocol_order = {
        "alpha060_default": 0,
        "dataset_best_fallback": 1,
        "deployable_alpha060_fallback": 2,
        "deployable_alpha060_rawbase_fallback": 3,
        "deployable_alpha060_rawbase_splitminus_fallback": 4,
    }
    dataset_order = {name: i for i, name in enumerate([*DATASET_ORDER, "Mean"])}
    ordered["_p"] = ordered["protocol"].map(protocol_order)
    ordered["_d"] = ordered["dataset_key"].map(dataset_order)
    ordered = ordered.sort_values(["_p", "_d"])
    lines = [
        "# Sample Fallback Protocol Comparison",
        "",
        "| protocol | usage | dataset | split | alpha | AUC/AP | delta vs alpha=0.60 default |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered.itertuples(index=False):
        alpha = "" if pd.isna(row.base_alpha) else f"{row.base_alpha:.2f}"
        delta = ""
        if row.protocol != "alpha060_default" and row.comparable_to_alpha060_default:
            delta = f"{row.delta_vs_alpha060_auc:+.4f} / {row.delta_vs_alpha060_ap:+.4f}"
        elif row.protocol != "alpha060_default":
            delta = "NA: split mismatch"
        lines.append(
            "| "
            f"{row.protocol} | {row.usage} | {row.dataset_key} | {row.split_scope} | {alpha} | "
            f"{row.auc:.4f} / {row.ap:.4f} | {delta} |"
        )
    lines.extend(
        [
            "",
            "Use `dataset_best_fallback` only for current benchmark-score reporting.",
            "Use `deployable_alpha060_fallback` for default-replacement audits.",
            "Delta is shown only when the default and fallback rows use the same split scope.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-summary-csv", type=Path, required=True)
    parser.add_argument("--dataset-best-summary-csv", type=Path, required=True)
    parser.add_argument("--deployable-alpha060-summary-csv", type=Path, required=True)
    parser.add_argument("--deployable-alpha060-rawbase-summary-csv", type=Path, default=None)
    parser.add_argument("--deployable-alpha060-rawbase-splitminus-summary-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    frames = [
        _default_rows(args.default_summary_csv),
        _fallback_rows(args.dataset_best_summary_csv, "dataset_best_fallback", "research_best"),
        _fallback_rows(
            args.deployable_alpha060_summary_csv,
            "deployable_alpha060_fallback",
            "default_replacement_audit_rank_base",
        ),
    ]
    if args.deployable_alpha060_rawbase_summary_csv is not None:
        frames.append(
            _fallback_rows(
                args.deployable_alpha060_rawbase_summary_csv,
                "deployable_alpha060_rawbase_fallback",
                "default_replacement_audit_raw_base",
            )
        )
    if args.deployable_alpha060_rawbase_splitminus_summary_csv is not None:
        frames.append(
            _fallback_rows(
                args.deployable_alpha060_rawbase_splitminus_summary_csv,
                "deployable_alpha060_rawbase_splitminus_fallback",
                "default_replacement_audit_raw_base_validated_selector",
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    rows["dataset_key"] = pd.Categorical(rows["dataset_key"], [*DATASET_ORDER], ordered=True)
    rows = rows.sort_values(["protocol", "dataset_key"]).reset_index(drop=True)
    rows["dataset_key"] = rows["dataset_key"].astype(str)
    rows = _with_deltas(_add_means(rows))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_csv, index=False)
    args.output_md.write_text(_markdown(rows), encoding="utf-8")
    print(rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved comparison CSV -> {args.output_csv}")
    print(f"Saved comparison Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
