"""Dataset contracts for the historical Local-D2 residual experiment family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .release_io import REPOSITORY_ROOT


KEY_COLUMNS = ["subset", "source_model", "filename"]


@dataclass(frozen=True)
class LocalDatasetSpec:
    name: str
    calib_index: Path
    eval_index: Path
    patch_cache: Path
    aggregation: str
    bottomk_ratio: float
    patch_region_size: int


def dataset_specs(root: Path = REPOSITORY_ROOT) -> tuple[LocalDatasetSpec, ...]:
    """Return the frozen three-dataset Local-D2 cache/index specification."""

    return (
        LocalDatasetSpec(
            "comgenvid",
            root / "cache/indexes/comgenvid_calib_real200.csv",
            root / "cache/indexes/comgenvid_eval_holdout_real900_all_fake.csv",
            root / "cache/patch_embeddings/comgenvid",
            "bottomk_mean",
            0.2,
            3,
        ),
        LocalDatasetSpec(
            "videofeedback",
            root / "cache/indexes/videofeedback_small_calib_real200.csv",
            root
            / "cache/indexes/videofeedback_small_eval_holdout_real500_fake300permodel.csv",
            root / "cache/patch_embeddings/videofeedback",
            "mean",
            0.5,
            1,
        ),
        LocalDatasetSpec(
            "genvideo",
            root / "cache/indexes/genvideo_calib_real200.csv",
            root / "cache/indexes/genvideo_eval_holdout_real7984_all_fake.csv",
            root / "cache/patch_embeddings/genvideo",
            "mean",
            0.5,
            2,
        ),
    )


def _with_filename(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["filename"] = output["video_path"].map(
        lambda value: Path(str(value)).name
    )
    for column in KEY_COLUMNS:
        output[column] = output[column].astype(str)
    return output


def build_strict_eval_index(
    spec: LocalDatasetSpec, stage1_scores: pd.DataFrame
) -> pd.DataFrame:
    """Recover exactly the historical Stage-1 evaluation identity intersection."""

    index = _with_filename(pd.read_csv(spec.eval_index))
    keys = stage1_scores[stage1_scores["dataset"].eq(spec.name)][KEY_COLUMNS]
    merged = index.merge(keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(keys):
        raise ValueError(
            f"{spec.name}: failed to recover every Stage-1 protocol row"
        )
    return merged.drop(columns=["filename"])


__all__ = [
    "KEY_COLUMNS",
    "LocalDatasetSpec",
    "build_strict_eval_index",
    "dataset_specs",
]
