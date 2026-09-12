"""固定窗必须来自官方RNG流程，不能偷换成首窗。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from evaluation.official_baseline import official_window, official_numpy_score


def test_official_window_uses_second_rng_draw():
    ids = list(range(0, 240, 3))
    rng = np.random.RandomState(42)
    starts = [rng.randint(0, len(ids) - n + 1) for n in (8, 16, 24, 32)]
    assert official_window(ids) == ids[starts[1] : starts[1] + 16]
    assert official_window(list(range(16))) == list(range(16))


def test_zero_temporal_excluded_and_right_ties():
    p = dict(
        mu_spat=np.zeros(2),
        W_spat=np.eye(2),
        mu_temp=np.zeros(2),
        W_temp=np.eye(2),
        calib_ll_spat=np.array([[-5.0, -4.0], [-3.0, -2.0]]),
        calib_ll_temp=np.array([[-5.0, -4.0], [-3.0, -2.0]]),
    )
    result = official_numpy_score(np.zeros((1, 16, 2), dtype="float32"), p)
    assert np.isposinf(result["global_temporal_raw"][0])
    assert result["final_score"][0] == 1.0
