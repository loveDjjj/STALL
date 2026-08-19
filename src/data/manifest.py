"""Alpha STALL 的视频 manifest 读取与基本校验。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_missing_window(value: object) -> bool:
    """判断窗口索引是否为空。"""

    return value is None or (isinstance(value, float) and np.isnan(value))


def load_manifest(path: str) -> pd.DataFrame:
    """读取包含视频路径、类别和来源模型的 manifest。"""

    frame = pd.read_csv(path)
    required = {"video_path", "subset", "source_model"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest '{path}' 缺少字段：{sorted(missing)}")
    if not set(frame["subset"].dropna().unique()).issubset({"real", "annotated"}):
        raise ValueError("manifest 的 subset 只能是 real 或 annotated")
    return frame
