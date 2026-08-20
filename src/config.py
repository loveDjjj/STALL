"""读取唯一基础配置，并应用命令行显式覆盖参数。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


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


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """返回应用点分路径覆盖后的深拷贝配置，绝不修改基础 YAML 文件。"""

    resolved = copy.deepcopy(config)
    for item in overrides:
        key, value = parse_override(item)
        target: dict[str, Any] = resolved
        parts = key.split(".")
        for part in parts[:-1]:
            current = target.get(part)
            if current is None:
                target[part] = {}
                current = target[part]
            if not isinstance(current, dict):
                raise ValueError(f"不能在标量配置下继续覆盖嵌套键：{key!r}")
            target = current
        target[parts[-1]] = value
    validate_config(resolved)
    return resolved


def config_digest(config: dict[str, Any]) -> str:
    import hashlib

    canonical = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    required = {"method", "sampling", "calibration", "data", "runtime", "metrics"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"实验配置缺少必要段：{sorted(missing)}")
    method = config["method"]
    if method.get("name") != "alpha_stall":
        raise ValueError("method.name 只能是 alpha_stall")
    windows = config["sampling"].get("num_windows")
    if not isinstance(windows, int) or windows < 1:
        raise ValueError("sampling.num_windows 必须是正整数")
    local = method.get("local", {})
    global_branch = method.get("global", {})
    if not global_branch.get("enabled", False) and not local.get("enabled", False):
        raise ValueError("至少必须启用 Global 或 Local 证据分支")
    if local.get("enabled", False) and local.get("temporal_enabled", False):
        if local.get("temporal_order") not in {1, 2}:
            raise ValueError("method.local.temporal_order 只能是 1 或 2")
    if config["calibration"].get("real_videos_per_dataset", 0) < 1:
        raise ValueError("calibration.real_videos_per_dataset 必须为正数")
    if config["data"].get("short_video_policy") not in {"error", "exclude"}:
        raise ValueError("data.short_video_policy 只能是 error 或 exclude")
    runtime = config["runtime"]
    if not isinstance(runtime.get("cache_dir"), str) or not runtime["cache_dir"]:
        raise ValueError("runtime.cache_dir 必须是非空路径")
    if runtime.get("cache_policy") != "strict":
        raise ValueError("主实验只能使用 runtime.cache_policy=strict")
    if not isinstance(runtime.get("max_features_for_fit"), int) or runtime["max_features_for_fit"] < 2:
        raise ValueError("runtime.max_features_for_fit 必须至少为 2")
    if not isinstance(runtime.get("minimum_free_gib"), (int, float)) or runtime["minimum_free_gib"] <= 0:
        raise ValueError("runtime.minimum_free_gib 必须为正数")
    for key in ("score_batch_size", "cache_io_workers"):
        if not isinstance(runtime.get(key), int) or runtime[key] < 1:
            raise ValueError(f"runtime.{key} 必须是正整数")
    devices = runtime.get("devices", [])
    if not isinstance(devices, list) or any(not isinstance(item, str) or not item for item in devices):
        raise ValueError("runtime.devices 必须是由非空设备名组成的列表")
    if len(devices) > 2:
        raise ValueError("当前运行器最多支持两张评分卡")


def dump_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
