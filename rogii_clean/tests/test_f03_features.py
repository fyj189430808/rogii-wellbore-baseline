"""F03 累计 XY 水平距离斜率特征测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f03_features import (
    ALL_F03_FEATURE_COLUMNS,
    F03_FEATURE_COLUMNS,
    build_f03_lgbm_rows,
    calculate_visible_cumulative_xy_distance,
)
from src.lgbm_features import FEATURE_COLUMNS


def make_linear_xy_u_well() -> pd.DataFrame:
    """构造 U 对 XY 距离斜率为 0.5、对 MD 斜率为 1.0 的小井。"""

    row_number = np.arange(8, dtype=np.float64)
    md = row_number.copy()
    x = 2.0 * row_number
    y = np.zeros(len(row_number), dtype=np.float64)
    cumulative_xy_distance = 2.0 * row_number
    u = 1200.0 + 0.5 * cumulative_xy_distance
    z = 1000.0 + 0.1 * md
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan

    return pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": 50.0 + row_number,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_f03_xy_features_match_known_horizontal_slope() -> None:
    """三个窗口都应恢复已知的 0.5 ft/ft XY 斜率和零拟合误差。"""

    horizontal_df = make_linear_xy_u_well()
    rows = build_f03_lgbm_rows(horizontal_df, "linear_xy", 0)
    visible_xy_distance = calculate_visible_cumulative_xy_distance(horizontal_df)

    expected_features = [
        "u_xy_slope_200",
        "u_xy_fit_rmse_200",
        "u_xy_slope_500",
        "u_xy_fit_rmse_500",
        "u_xy_slope_1000",
        "u_xy_fit_rmse_1000",
        "u_xy_slope_200_minus_1000",
    ]

    assert F03_FEATURE_COLUMNS == expected_features
    assert ALL_F03_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_F03_FEATURE_COLUMNS) == 19
    np.testing.assert_allclose(visible_xy_distance, [0, 2, 4, 6, 8, 10])

    for window_ft in [200, 500, 1000]:
        np.testing.assert_allclose(rows[f"u_xy_slope_{window_ft}"], [0.5, 0.5])
        np.testing.assert_allclose(
            rows[f"u_xy_fit_rmse_{window_ft}"],
            [0.0, 0.0],
            atol=1e-12,
        )

    np.testing.assert_allclose(rows["u_xy_slope_200_minus_1000"], [0.0, 0.0])


def test_hidden_tvt_does_not_change_f03_features() -> None:
    """修改隐藏 TVT 真值时，十九个模型输入必须逐位不变。"""

    original = make_linear_xy_u_well()
    changed = make_linear_xy_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_f03_lgbm_rows(original, "linear_xy", 0)
    changed_rows = build_f03_lgbm_rows(changed, "linear_xy", 0)

    assert_frame_equal(
        original_rows[ALL_F03_FEATURE_COLUMNS],
        changed_rows[ALL_F03_FEATURE_COLUMNS],
        check_exact=True,
    )

