"""Alpha STALL 的视频 manifest 读取与基本校验。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_active_groups(root, manifest_root, scores):
    """按已登记活跃manifest读取源组，不从实验脚本或运行名猜测分组。"""
    import json
    from pathlib import Path
    from reference import file_digest

    root = Path(root)
    manifest_root = Path(manifest_root)
    registry = json.loads((manifest_root / "manifest.json").read_text())
    if registry["status"] != "completed":
        raise ValueError("活跃数据清单未完成")
    frames = []
    provenance = {}
    for domain in sorted(scores.dataset.unique()):
        path = manifest_root / domain / "evaluation.csv"
        key = str(path.relative_to(root))
        if key not in registry["files"] or file_digest(path) != registry["files"][key]["sha256"]:
            raise ValueError("源组清单未登记或哈希改变")
        frame = pd.read_csv(path)
        required = {"video_id", "dataset", "subset", "source_model", "source_group"}
        if required - set(frame) or frame.video_id.duplicated().any():
            raise ValueError("源组清单字段/身份错误")
        frames.append(frame[list(required)])
        provenance[key] = registry["files"][key]["sha256"]
    all_groups = pd.concat(frames, ignore_index=True).set_index("video_id", verify_integrity=True)
    if not set(scores.video_id).issubset(all_groups.index):
        raise ValueError("评分视频缺失源组身份")
    aligned = all_groups.loc[scores.video_id]
    for column in ("dataset", "subset", "source_model"):
        if not (aligned[column].to_numpy() == scores[column].to_numpy()).all():
            raise ValueError("评分与源组清单标签/来源不一致")
    if aligned.source_group.isna().any() or aligned.source_group.astype(str).str.len().eq(0).any():
        raise ValueError("源组为空")
    return aligned.source_group, provenance
