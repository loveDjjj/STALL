"""CAES固定Local D2检测参考与selector-specific matched calibration。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from math_utils import StableGaussianParams, stable_sorted


@dataclass(frozen=True)
class FrozenLocalD2Reference:
    """由FS0 calibration窗口拟合、供全部selector共享的Local D2参数。"""

    dataset: str
    params: StableGaussianParams
    calibration_ids: tuple[str, ...]
    source_run: str
    source_config_hash: str
    schema_version: str = "caes_frozen_local_d2_v1"

    def validate(self) -> None:
        if self.schema_version != "caes_frozen_local_d2_v1" or not self.dataset:
            raise ValueError("FrozenLocalD2Reference identity无效")
        if self.params.mean.shape != (1024,):
            raise ValueError("Frozen Local D2 mean必须为1024维")
        if self.params.whitening.ndim != 2 or self.params.whitening.shape[0] != 1024:
            raise ValueError("Frozen Local D2 whitening shape无效")
        stable_sorted(self.params.calibration_raw)
        if len(self.calibration_ids) == 0 or len(set(self.calibration_ids)) != len(self.calibration_ids):
            raise ValueError("Frozen Local D2 calibration IDs无效")
        if len(self.source_config_hash) != 64:
            raise ValueError("Frozen Local D2缺少source config hash")

    def digest(self) -> str:
        self.validate()
        payload = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "source_run": self.source_run,
            "source_config_hash": self.source_config_hash,
            "calibration_ids": list(self.calibration_ids),
            "mean": hashlib.sha256(self.params.mean.tobytes()).hexdigest(),
            "whitening": hashlib.sha256(self.params.whitening.tobytes()).hexdigest(),
            "calibration_raw": hashlib.sha256(
                self.params.calibration_raw.tobytes()
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def save_frozen_local_reference(
    path: Path, reference: FrozenLocalD2Reference
) -> str:
    reference.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray([reference.schema_version]),
        dataset=np.asarray([reference.dataset]),
        mean=reference.params.mean,
        whitening=reference.params.whitening,
        calibration_raw=reference.params.calibration_raw,
        calibration_ids=np.asarray(reference.calibration_ids),
        source_run=np.asarray([reference.source_run]),
        source_config_hash=np.asarray([reference.source_config_hash]),
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_local_reference(path: Path) -> FrozenLocalD2Reference:
    with np.load(path, allow_pickle=False) as data:
        reference = FrozenLocalD2Reference(
            dataset=str(data["dataset"][0]),
            params=StableGaussianParams(
                mean=np.asarray(data["mean"], dtype=np.float64),
                whitening=np.asarray(data["whitening"], dtype=np.float64),
                calibration_raw=stable_sorted(
                    np.asarray(data["calibration_raw"], dtype=np.float64)
                ),
            ),
            calibration_ids=tuple(str(item) for item in data["calibration_ids"]),
            source_run=str(data["source_run"][0]),
            source_config_hash=str(data["source_config_hash"][0]),
            schema_version=str(data["schema_version"][0]),
        )
    reference.validate()
    return reference


__all__ = [
    "FrozenLocalD2Reference",
    "load_frozen_local_reference",
    "save_frozen_local_reference",
]
