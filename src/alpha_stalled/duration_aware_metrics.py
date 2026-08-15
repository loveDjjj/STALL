"""Cohort construction and prevalence-stable metrics for full-23 coverage."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = ("comgenvid", "videofeedback", "genvideo")
COVERAGE_BRANCHES = ("G", "S")
BRANCHES = COVERAGE_BRANCHES
PAPER_FAKE_COUNTS = {
    "comgenvid": {"Sora": 1700, "VEO3": 1700},
    "videofeedback": {
        "AnimateDiff": 992,
        "Fast-SVD": 959,
        "Hotshot-XL": 2736,
        "LVDM": 2973,
        "LaVie-base": 2789,
        "ModelScope": 3722,
        "Pika": 1906,
        "SoRA-Clip": 898,
        "Text2Video-Zero": 3722,
        "VideoCrafter2": 3543,
        "ZeroScope-576w": 2022,
    },
    "genvideo": {
        "Crafter": 188,
        "Gen2": 1380,
        "HotShot": 700,
        "Lavie": 1400,
        "ModelScope": 700,
        "MoonValley": 626,
        "MorphStudio": 700,
        "Show_1": 700,
        "Sora": 56,
        "WildScrape": 529,
    },
}


def stable_rank(namespace: str, value: str) -> str:
    """Return a deterministic SHA-256 ordering within a named cohort."""

    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


def metric(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    """Compute AUC, raw AP, and fixed-50% real/fake-positive AP."""

    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if not len(real) or not len(fake):
        raise ValueError("coverage metrics require non-empty real and fake cohorts")
    labels = np.concatenate(
        [np.ones(len(real), dtype=np.uint8), np.zeros(len(fake), dtype=np.uint8)]
    )
    scores = np.concatenate([real, fake])
    weights = np.concatenate(
        [
            np.full(len(real), 0.5 / len(real), dtype=np.float64),
            np.full(len(fake), 0.5 / len(fake), dtype=np.float64),
        ]
    )
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap_raw": float(average_precision_score(labels, scores)),
        "ap_std50": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "fake_ap_std50": float(
            average_precision_score(
                1 - labels, 1.0 - scores, sample_weight=weights
            )
        ),
    }


def proportional_allocation(counts: pd.Series, total: int) -> dict[int, int]:
    """Allocate an exact total by largest remainder with stable tie breaks."""

    counts = counts.sort_index().astype(int)
    exact = counts.astype(float) * (float(total) / float(counts.sum()))
    allocated = np.floor(exact).astype(int)
    remainder = total - int(allocated.sum())
    order = sorted(
        counts.index,
        key=lambda key: (-(exact[key] - allocated[key]), key),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    return {int(key): int(value) for key, value in allocated.items()}


def duration_matched_real(
    real: pd.DataFrame, fake: pd.DataFrame, namespace: str
) -> pd.DataFrame:
    """Use every physical real once while matching the fake duration mixture."""

    metadata = real[["video_id", "source_model"]].drop_duplicates("video_id")
    allocations = proportional_allocation(
        fake.groupby("protocol_duration_sec").size(), len(metadata)
    )
    ranked = metadata.copy()
    ranked["_rank"] = ranked["video_id"].map(
        lambda value: stable_rank(namespace, str(value))
    )
    ranked = ranked.sort_values("_rank").reset_index(drop=True)
    selected = []
    start = 0
    for duration, count in sorted(allocations.items()):
        block = ranked.iloc[start : start + count].copy()
        block["protocol_duration_sec"] = duration
        selected.append(block)
        start += count
    assignment = pd.concat(selected, ignore_index=True)
    output = assignment.merge(
        real,
        on=["video_id", "source_model", "protocol_duration_sec"],
        validate="one_to_one",
    )
    if len(output) != metadata["video_id"].nunique():
        raise ValueError("duration-matched real selection lost physical identities")
    return output


def balanced_real_ids(real: pd.DataFrame, n: int, namespace: str) -> list[str]:
    """Select exactly ``n`` real IDs, balanced across real source groups."""

    metadata = real[["video_id", "source_model"]].drop_duplicates("video_id")
    sources = sorted(metadata["source_model"].unique())
    base, remainder = divmod(n, len(sources))
    selected: list[str] = []
    for index, source in enumerate(sources):
        count = base + int(index < remainder)
        group = metadata[metadata["source_model"].eq(source)].copy()
        group["_rank"] = group["video_id"].map(
            lambda value: stable_rank(namespace, str(value))
        )
        selected.extend(
            group.sort_values("_rank").head(count)["video_id"].tolist()
        )
    if len(selected) != n or len(set(selected)) != n:
        raise ValueError(f"could not select {n} balanced real identities")
    return sorted(
        selected,
        key=lambda value: stable_rank(namespace + ":merge", value),
    )


def balanced_pair_frame(
    real: pd.DataFrame, fake: pd.DataFrame, namespace: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic equally sized real/fake cohorts for one generator."""

    real_metadata = real[["video_id", "source_model"]].drop_duplicates(
        "video_id"
    )
    real_counts = real_metadata.groupby("source_model")["video_id"].nunique()
    maximum = int(real_counts.min() * len(real_counts))
    n = min(len(fake), maximum)
    chosen_fake = fake.copy()
    chosen_fake["_rank"] = chosen_fake["video_id"].map(
        lambda value: stable_rank(namespace + ":fake", str(value))
    )
    chosen_fake = chosen_fake.sort_values("_rank").head(n).reset_index(drop=True)
    real_ids = balanced_real_ids(real, n, namespace + ":real")
    assignment = pd.DataFrame(
        {
            "video_id": real_ids,
            "protocol_duration_sec": chosen_fake[
                "protocol_duration_sec"
            ].astype(int),
        }
    )
    chosen_real = assignment.merge(
        real,
        on=["video_id", "protocol_duration_sec"],
        validate="one_to_one",
    )
    return chosen_real, chosen_fake


def select_paper_fake_cohort(fake: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the generated-video counts declared by STALL Table 1."""

    pieces = []
    for dataset, generators in PAPER_FAKE_COUNTS.items():
        for source_model, count in generators.items():
            group = fake[
                fake["dataset"].eq(dataset)
                & fake["source_model"].eq(source_model)
            ].copy()
            if len(group) < count:
                raise ValueError(
                    f"paper cohort needs {count} {dataset}/{source_model}, "
                    f"found {len(group)}"
                )
            namespace = f"paper-table-count-v1:{dataset}:{source_model}"
            group["paper_rank"] = group["video_id"].map(
                lambda value: stable_rank(namespace, str(value))
            )
            pieces.append(group.sort_values("paper_rank").head(count))
    output = pd.concat(pieces, ignore_index=True)
    expected = sum(sum(items.values()) for items in PAPER_FAKE_COUNTS.values())
    if len(output) != expected or output["video_id"].duplicated().any():
        raise ValueError("paper fake cohort has invalid size or duplicate identities")
    return output


__all__ = [
    "BRANCHES",
    "COVERAGE_BRANCHES",
    "DATASETS",
    "PAPER_FAKE_COUNTS",
    "balanced_pair_frame",
    "balanced_real_ids",
    "duration_matched_real",
    "metric",
    "proportional_allocation",
    "select_paper_fake_cohort",
    "stable_rank",
]
