"""新版路线图 RF01b：相邻尺度 Huber 坡差测试。"""

import sys
from pathlib import Path

import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS
from src.rf01b_features import (
    ALL_RF01B_FEATURE_COLUMNS,
    RF01B_DIFFERENCE_COLUMNS,
    build_rf01b_lgbm_rows,
)
from tests.test_rf01a_features import make_linear_u_well


def test_rf01b_adds_only_four_adjacent_scale_differences() -> None:
    """线性 U 的五尺度坡度相同，因此四个相邻坡差都应严格为零。"""

    rows = build_rf01b_lgbm_rows(make_linear_u_well(), "linear", 0)
    expected_differences = [
        "u_huber_slope_50_minus_100",
        "u_huber_slope_100_minus_200",
        "u_huber_slope_200_minus_500",
        "u_huber_slope_500_minus_1000",
    ]

    assert RF01B_DIFFERENCE_COLUMNS == expected_differences
    assert ALL_RF01B_FEATURE_COLUMNS == (
        ALL_RF01A_FEATURE_COLUMNS + expected_differences
    )
    assert len(ALL_RF01B_FEATURE_COLUMNS) == 21

    for feature_name in expected_differences:
        np.testing.assert_allclose(rows[feature_name], [0.0, 0.0], atol=1e-12)

