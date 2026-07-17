"""新版路线图 RF01c1：相邻 U 坡度标准差测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS
from src.rf01c_std_features import (
    ALL_RF01C_STD_FEATURE_COLUMNS,
    RF01C_STD_FEATURE_COLUMNS,
    build_rf01c_std_lgbm_rows,
    calculate_adjacent_u_slope_std,
)
from tests.test_rf01a_features import make_linear_u_well


def make_piecewise_adjacent_slope_well() -> pd.DataFrame:
    """构造相邻 U 坡度依次为 1、-1、2、-2 的可见前缀。"""

    md = np.arange(7, dtype=np.float64)
    u = np.array([1200.0, 1201.0, 1200.0, 1202.0, 1200.0, 1198.0, 1196.0])
    z = 1000.0 + 0.2 * md
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[5:] = np.nan
    return pd.DataFrame(
        {
            "MD": md,
            "X": md,
            "Y": 0.5 * md,
            "Z": z,
            "GR": 50.0 + md,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_adjacent_u_slope_std_matches_hand_calculation() -> None:
    """四个已知相邻坡度的总体标准差应等于 sqrt(2.5)。"""

    slope_std = calculate_adjacent_u_slope_std(
        make_piecewise_adjacent_slope_well(),
        window_ft=50.0,
    )
    np.testing.assert_allclose(slope_std, np.sqrt(2.5))


def test_rf01c_std_adds_only_five_window_statistics() -> None:
    """线性 U 的五个相邻坡度标准差都应为零。"""

    rows = build_rf01c_std_lgbm_rows(make_linear_u_well(), "linear", 0)
    expected_features = [
        "u_adjacent_slope_std_50",
        "u_adjacent_slope_std_100",
        "u_adjacent_slope_std_200",
        "u_adjacent_slope_std_500",
        "u_adjacent_slope_std_1000",
    ]

    assert RF01C_STD_FEATURE_COLUMNS == expected_features
    assert ALL_RF01C_STD_FEATURE_COLUMNS == (
        ALL_RF01A_FEATURE_COLUMNS + expected_features
    )
    assert len(ALL_RF01C_STD_FEATURE_COLUMNS) == 22

    for feature_name in expected_features:
        np.testing.assert_allclose(rows[feature_name], [0.0, 0.0], atol=1e-12)

