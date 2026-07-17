"""新版路线图 RF01c3：U 曲率特征测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS
from src.rf01c_curvature_features import (
    ALL_RF01C_CURVATURE_FEATURE_COLUMNS,
    RF01C_CURVATURE_FEATURE_COLUMNS,
    build_rf01c_curvature_lgbm_rows,
    calculate_u_curvature,
)
from tests.test_rf01a_features import make_linear_u_well


def make_quadratic_u_well() -> pd.DataFrame:
    """构造 U=1200+MD²，因此相邻坡度对中点 MD 的斜率严格为 2。"""

    md = np.arange(8, dtype=np.float64)
    u = 1200.0 + md * md
    z = 1000.0 + 0.2 * md
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan
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


def test_u_curvature_matches_quadratic_hand_calculation() -> None:
    """二次 U 的局部坡度随 MD 线性增加，Huber curvature 应为 2。"""

    curvature = calculate_u_curvature(
        make_quadratic_u_well(),
        window_ft=50.0,
    )
    np.testing.assert_allclose(curvature, 2.0, atol=1e-12)


def test_rf01c_curvature_adds_only_five_window_curvatures() -> None:
    """线性 U 的五个窗口 curvature 都应为零。"""

    rows = build_rf01c_curvature_lgbm_rows(make_linear_u_well(), "linear", 0)
    expected_features = [
        "u_curvature_50",
        "u_curvature_100",
        "u_curvature_200",
        "u_curvature_500",
        "u_curvature_1000",
    ]

    assert RF01C_CURVATURE_FEATURE_COLUMNS == expected_features
    assert ALL_RF01C_CURVATURE_FEATURE_COLUMNS == (
        ALL_RF01A_FEATURE_COLUMNS + expected_features
    )
    assert len(ALL_RF01C_CURVATURE_FEATURE_COLUMNS) == 22

    for feature_name in expected_features:
        np.testing.assert_allclose(rows[feature_name], [0.0, 0.0], atol=1e-12)

