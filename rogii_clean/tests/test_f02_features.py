"""F02 可见前缀斜率稳定性特征测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试真实项目实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f01b_features import ALL_F01B_FEATURE_COLUMNS
from src.f02_features import (
    ALL_F02_FEATURE_COLUMNS,
    F02_FEATURE_COLUMNS,
    build_f02_lgbm_rows,
    calculate_visible_block_slopes,
)


def make_piecewise_u_well() -> pd.DataFrame:
    """构造四个 50 ft 可见分块，真实 U 斜率依次为 1、-1、2、-2。"""

    md = np.arange(203, dtype=np.float64)
    u = np.full(len(md), 1200.0, dtype=np.float64)

    # U 在每个整数 MD 间隔上连续变化，便于手算每个 50 ft 分块的斜率。
    for row_index in range(1, len(md)):
        current_md = md[row_index]
        if current_md <= 50.0:
            interval_slope = 1.0
        elif current_md <= 100.0:
            interval_slope = -1.0
        elif current_md <= 150.0:
            interval_slope = 2.0
        else:
            interval_slope = -2.0
        u[row_index] = u[row_index - 1] + interval_slope

    z = 1000.0 + 0.1 * md
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[201:] = np.nan

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


def test_f02_block_statistics_match_piecewise_slopes() -> None:
    """F02 应从可见前缀恢复分块斜率及其稳定性统计。"""

    horizontal_df = make_piecewise_u_well()
    rows = build_f02_lgbm_rows(horizontal_df, "piecewise", 0)
    block_slopes = calculate_visible_block_slopes(
        horizontal_df,
        window_ft=500.0,
        block_size_ft=50.0,
    )

    expected_features = [
        "u_slope_100_minus_500",
        "u_block_slope_std_500",
        "u_block_slope_std_1000",
        "u_positive_slope_ratio_500",
        "u_positive_slope_ratio_1000",
        "u_max_block_slope_jump_1000",
    ]

    assert F02_FEATURE_COLUMNS == expected_features
    assert ALL_F02_FEATURE_COLUMNS == ALL_F01B_FEATURE_COLUMNS + expected_features
    assert len(ALL_F02_FEATURE_COLUMNS) == 25

    np.testing.assert_allclose(block_slopes, [1.0, -1.0, 2.0, -2.0])
    np.testing.assert_allclose(rows["u_block_slope_std_500"], np.sqrt(2.5))
    np.testing.assert_allclose(rows["u_block_slope_std_1000"], np.sqrt(2.5))
    np.testing.assert_allclose(rows["u_positive_slope_ratio_500"], 0.5)
    np.testing.assert_allclose(rows["u_positive_slope_ratio_1000"], 0.5)
    np.testing.assert_allclose(rows["u_max_block_slope_jump_1000"], 4.0)

    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    visible_md = visible_df["MD"].to_numpy(dtype=np.float64)
    visible_u = (
        visible_df["TVT_input"].to_numpy(dtype=np.float64)
        + visible_df["Z"].to_numpy(dtype=np.float64)
    )
    slope_100 = np.polyfit(visible_md[visible_md >= 100.0], visible_u[visible_md >= 100.0], 1)[0]
    slope_500 = np.polyfit(visible_md, visible_u, 1)[0]
    np.testing.assert_allclose(rows["u_slope_100_minus_500"], slope_100 - slope_500)


def test_hidden_tvt_does_not_change_f02_features() -> None:
    """修改隐藏 TVT 真值时，全部二十五个模型输入必须逐位不变。"""

    original = make_piecewise_u_well()
    changed = make_piecewise_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_f02_lgbm_rows(original, "piecewise", 0)
    changed_rows = build_f02_lgbm_rows(changed, "piecewise", 0)

    assert_frame_equal(
        original_rows[ALL_F02_FEATURE_COLUMNS],
        changed_rows[ALL_F02_FEATURE_COLUMNS],
        check_exact=True,
    )

