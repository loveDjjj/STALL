"""读取唯一基础配置，并应用命令行显式覆盖参数。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


# 主线配置使用封闭字段集合；未知键和改变冻结数值合同的覆盖均拒绝。
PAPER_FIELDS = {
    "protocol.id": str,
    "encoder.repo": str,
    "encoder.weights": str,
    "encoder.weights_sha256": str,
    "encoder.input_size": int,
    "encoder.batch_size": int,
    "encoder.pad_tail": bool,
    "selection.name": str,
    "selection.k": int,
    "selection.frames": int,
    "selection.fps": int,
    "selection.coarse_stride": int,
    "selection.candidate_stride": int,
    "method.global_enabled": bool,
    "method.local_enabled": bool,
    "method.spatial_weight": (int, float),
    "method.global_weight": (int, float),
    "reference.directory": str,
    "reference.manifest": str,
    "data.manifests": str,
    "runtime.device": str,
    "runtime.score_dtype": str,
    "output.root": str,
    "evaluation.score_direction": str,
    "evaluation.real_fpr_levels": list,
    "evaluation.bootstrap_seed": int,
    "evaluation.bootstrap_iterations": int,
    "fit.feature_source": str,
    "runtime.decode_workers": int,
    "runtime.prefetch_depth": int,
    "runtime.prefetch_memory_mb": int,
    "runtime.devices": list,
    "runtime.prefetch_enabled": bool,
}
PAPER_OPTIONAL = {
    "fit.feature_source",
    "runtime.decode_workers",
    "runtime.prefetch_depth",
    "runtime.prefetch_memory_mb",
    "runtime.devices",
    "runtime.prefetch_enabled",
}


def _paper_leaves(value, prefix=""):
    if not isinstance(value, dict):
        raise ValueError("配置必须为映射")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("配置键必须为字符串")
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            if not any(field.startswith(name + ".") for field in PAPER_FIELDS):
                raise ValueError(f"未知配置段：{name}")
            result.update(_paper_leaves(item, name))
        else:
            result[name] = item
    return result


def validate_paper_config(config):
    import math
    import re

    leaves = _paper_leaves(config)
    missing = set(PAPER_FIELDS) - PAPER_OPTIONAL - set(leaves)
    unknown = set(leaves) - set(PAPER_FIELDS)
    if missing or unknown:
        raise ValueError(f"新主线配置字段不符，缺少{sorted(missing)}，未知{sorted(unknown)}")
    for name, value in leaves.items():
        types = PAPER_FIELDS[name]
        types = types if isinstance(types, tuple) else (types,)
        if type(value) not in types:
            raise ValueError(f"配置类型错误：{name}")
        if isinstance(value, str) and not value:
            raise ValueError(f"配置不能为空：{name}")
    fixed = {
        "encoder.input_size": 224,
        "encoder.batch_size": 8,
        "encoder.pad_tail": True,
        "selection.frames": 16,
        "selection.fps": 8,
        "selection.coarse_stride": 8,
        "selection.candidate_stride": 4,
        "runtime.score_dtype": "float64",
        "evaluation.score_direction": "higher_is_real",
    }
    for name, value in fixed.items():
        if leaves[name] != value:
            raise ValueError(f"冻结数值合同不允许改变{name}")
    if not re.fullmatch(r"[0-9a-f]{64}", leaves["encoder.weights_sha256"]):
        raise ValueError("权重hash无效")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", leaves["protocol.id"]):
        raise ValueError("协议ID包含非法路径字符")
    if leaves["selection.name"] not in ("uniform", "feature_change") or leaves[
        "selection.k"
    ] not in (1, 2, 3):
        raise ValueError("选择器或窗口预算无效")
    if leaves["runtime.device"] not in ("cpu", "cuda:0", "cuda:1"):
        raise ValueError("设备无效")
    devices = leaves.get("runtime.devices", [])
    if any(type(d) is not str or d not in ("cpu", "cuda:0", "cuda:1") for d in devices) or len(
        set(devices)
    ) != len(devices):
        raise ValueError("设备列表无效或重复")
    if len(devices) > 1 and "cpu" in devices:
        raise ValueError("多设备执行只支持独立CUDA卡，不混用CPU数值路径")
    if not leaves["method.global_enabled"] and not leaves["method.local_enabled"]:
        raise ValueError("至少保留一个分支")
    for name in ("method.spatial_weight", "method.global_weight"):
        if not math.isfinite(leaves[name]) or not 0 <= leaves[name] <= 1:
            raise ValueError("权重必须在[0,1]")
    levels = leaves["evaluation.real_fpr_levels"]
    if not levels or any(
        type(x) not in (int, float) or not math.isfinite(x) or not 0 < x < 1 for x in levels
    ):
        raise ValueError("FPR列表无效")
    if leaves["evaluation.bootstrap_iterations"] < 1000:
        raise ValueError("正式bootstrap至少1000次")
    if leaves["evaluation.bootstrap_seed"] < 0:
        raise ValueError("seed必须非负")
    if leaves.get("fit.feature_source", "cache") not in ("cache", "video"):
        raise ValueError("拟合特征源只能为cache/video")
    for name in ("runtime.decode_workers", "runtime.prefetch_depth", "runtime.prefetch_memory_mb"):
        if name in leaves and leaves[name] < 1:
            raise ValueError("预取线程、深度和内存预算必须为正")


def load_paper_config(path, overrides=()):
    config = load_config(Path(path))
    validate_paper_config(config)
    result = copy.deepcopy(config)
    for value in overrides:
        name, item = parse_override(value)
        if name not in PAPER_FIELDS:
            raise ValueError(f"未知覆盖字段：{name}")
        target = result
        parts = name.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = item
    validate_paper_config(result)
    return result


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("实验配置必须是 YAML 映射对象")
    return payload


def parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"覆盖参数必须使用 key=value 格式：{value!r}")
    key, raw_value = value.split("=", 1)
    if not key or any(not part for part in key.split(".")):
        raise ValueError(f"覆盖参数键必须是点分路径：{key!r}")
    return key, yaml.safe_load(raw_value)


def config_digest(config: dict[str, Any]) -> str:
    import hashlib

    canonical = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dump_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
