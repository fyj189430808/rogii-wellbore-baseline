"""F01b 特征消融测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 把 rogii_clean 加入模块搜索路径，测试真实的项目代码。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f01b_features import (
    ALL_F01B_FEATURE_COLUMNS,
    F01B_FEATURE_COLUMNS,
    build_f01b_lgbm_rows,
)
from src.lgbm_features import FEATURE_COLUMNS


def make_linear_u_well() -> pd.DataFrame:
    """构造一口 U 斜率为 0.5、后两行自然隐藏的小井。"""

    md = np.arange(6, dtype=np.float64)
    z = 1000.0 + 0.2 * md
    tvt = 200.0 + 0.3 * md
    tvt_input = tvt.copy()
    tvt_input[4:] = np.nan

    return pd.DataFrame(
        {
            "MD": md,
            "X": 10.0 + md,
            "Y": 20.0 + 2.0 * md,
            "Z": z,
            "GR": 50.0 + md,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_f01b_keeps_statistics_and_removes_unbounded_projection() -> None:
    """F01b 应只删除三列线性外推，保留其余七个前缀 U 统计。"""

    rows = build_f01b_lgbm_rows(make_linear_u_well(), "linear", 0)

    expected_f01b_features = [
        "u_slope_200",
        "u_fit_rmse_200",
        "u_slope_500",
        "u_fit_rmse_500",
        "u_slope_1000",
        "u_fit_rmse_1000",
        "u_slope_200_minus_1000",
    ]

    assert F01B_FEATURE_COLUMNS == expected_f01b_features
    assert ALL_F01B_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_f01b_features
    assert len(ALL_F01B_FEATURE_COLUMNS) == 19

    for removed_feature in [
        "u_linear_delta_200",
        "u_linear_delta_500",
        "u_linear_delta_1000",
    ]:
        assert removed_feature not in rows.columns

    np.testing.assert_allclose(rows["u_slope_200"], [0.5, 0.5])
    np.testing.assert_allclose(rows["u_fit_rmse_1000"], [0.0, 0.0], atol=1e-12)

