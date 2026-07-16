"""F01c 短距离门控线性外推测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 把 rogii_clean 加入模块路径，确保测试调用项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f01c_features import (
    ALL_F01C_FEATURE_COLUMNS,
    F01C_GATED_FEATURE_COLUMNS,
    build_f01c_lgbm_rows,
)
from src.f01b_features import ALL_F01B_FEATURE_COLUMNS


def make_long_linear_u_well() -> pd.DataFrame:
    """构造隐藏距离分别为 400、800、1400 ft 的严格线性小井。"""

    md = np.array([0.0, 200.0, 400.0, 800.0, 1200.0, 1800.0])
    z = 1000.0 + 0.2 * md
    tvt = 200.0 + 0.3 * md
    tvt_input = tvt.copy()
    tvt_input[3:] = np.nan

    return pd.DataFrame(
        {
            "MD": md,
            "X": 10.0 + md,
            "Y": 20.0 + 0.5 * md,
            "Z": z,
            "GR": 50.0 + 0.01 * md,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_f01c_keeps_projection_only_within_first_1000_ft() -> None:
    """门控外推在 1000 ft 内等于真实公式，超过阈值后必须为缺失。"""

    rows = build_f01c_lgbm_rows(make_long_linear_u_well(), "linear", 0)

    expected_gated_features = [
        "u_linear_delta_200_within_1000",
        "u_linear_delta_500_within_1000",
        "u_linear_delta_1000_within_1000",
    ]

    assert F01C_GATED_FEATURE_COLUMNS == expected_gated_features
    assert ALL_F01C_FEATURE_COLUMNS == ALL_F01B_FEATURE_COLUMNS + expected_gated_features
    assert len(ALL_F01C_FEATURE_COLUMNS) == 22

    for feature_name in expected_gated_features:
        np.testing.assert_allclose(rows[feature_name].iloc[:2], [120.0, 240.0])
        assert np.isnan(rows[feature_name].iloc[2])

    for raw_feature_name in [
        "u_linear_delta_200",
        "u_linear_delta_500",
        "u_linear_delta_1000",
    ]:
        assert raw_feature_name not in rows.columns

