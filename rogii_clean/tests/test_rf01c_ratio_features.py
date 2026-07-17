"""新版路线图 RF01c2：相邻 U 坡度正值比例测试。"""

import sys
from pathlib import Path

import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS
from src.rf01c_ratio_features import (
    ALL_RF01C_RATIO_FEATURE_COLUMNS,
    RF01C_RATIO_FEATURE_COLUMNS,
    build_rf01c_ratio_lgbm_rows,
    calculate_positive_adjacent_slope_ratio,
)
from tests.test_rf01a_features import make_linear_u_well
from tests.test_rf01c_std_features import make_piecewise_adjacent_slope_well


def test_positive_slope_ratio_matches_hand_calculation() -> None:
    """四个相邻坡度中两个为正，正坡比例应为 0.5。"""

    ratio = calculate_positive_adjacent_slope_ratio(
        make_piecewise_adjacent_slope_well(),
        window_ft=50.0,
    )
    np.testing.assert_allclose(ratio, 0.5)


def test_rf01c_ratio_adds_only_five_window_ratios() -> None:
    """线性上升 U 的五个窗口正坡比例都应为 1。"""

    rows = build_rf01c_ratio_lgbm_rows(make_linear_u_well(), "linear", 0)
    expected_features = [
        "u_positive_adjacent_slope_ratio_50",
        "u_positive_adjacent_slope_ratio_100",
        "u_positive_adjacent_slope_ratio_200",
        "u_positive_adjacent_slope_ratio_500",
        "u_positive_adjacent_slope_ratio_1000",
    ]

    assert RF01C_RATIO_FEATURE_COLUMNS == expected_features
    assert ALL_RF01C_RATIO_FEATURE_COLUMNS == (
        ALL_RF01A_FEATURE_COLUMNS + expected_features
    )
    assert len(ALL_RF01C_RATIO_FEATURE_COLUMNS) == 22

    for feature_name in expected_features:
        np.testing.assert_allclose(rows[feature_name], [1.0, 1.0])

