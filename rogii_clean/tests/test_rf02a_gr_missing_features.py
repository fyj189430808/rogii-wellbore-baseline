"""RF02a：原始 GR 缺失几何特征测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_features import FEATURE_COLUMNS
from src.rf02a_gr_missing_features import (
    ALL_RF02A_FEATURE_COLUMNS,
    RF02A_MISSING_FEATURE_COLUMNS,
    build_rf02a_lgbm_rows,
    calculate_gr_missing_geometry,
)


def make_missing_gr_well() -> pd.DataFrame:
    """构造两个内部 GR 缺口、前两行 TVT 可见的 1 ft 采样小井。"""

    md = np.arange(10, dtype=np.float64)
    tvt = 200.0 + 0.1 * md
    tvt_input = tvt.copy()
    tvt_input[2:] = np.nan
    gr = np.array(
        [10.0, 11.0, 20.0, np.nan, np.nan, 23.0, 24.0, np.nan, 26.0, 27.0],
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "MD": md,
            "X": 100.0 + md,
            "Y": 200.0 + md,
            "Z": 1000.0 + 0.2 * md,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_missing_run_geometry_matches_hand_calculation() -> None:
    """长度 2 的内部缺口应得到正确的左右距离、长度和相对位置。"""

    geometry = calculate_gr_missing_geometry(
        make_missing_gr_well(),
        windows_ft=[4],
    )

    np.testing.assert_allclose(geometry.loc[3, "gr_gap_length_ft"], 2.0)
    np.testing.assert_allclose(geometry.loc[4, "gr_gap_length_ft"], 2.0)
    np.testing.assert_allclose(geometry.loc[3, "gr_distance_to_left_observed_ft"], 1.0)
    np.testing.assert_allclose(geometry.loc[3, "gr_distance_to_right_observed_ft"], 2.0)
    np.testing.assert_allclose(geometry.loc[3, "gr_relative_position_inside_gap"], 1.0 / 3.0)
    np.testing.assert_allclose(geometry.loc[4, "gr_relative_position_inside_gap"], 2.0 / 3.0)

    # MD=4 的 ±2 ft 窗口含 5 行，其中 3 行有原始 GR。
    np.testing.assert_allclose(geometry.loc[4, "gr_valid_fraction_4"], 3.0 / 5.0)

    # 实测行没有缺口几何，长度和左右距离固定为 0，相对位置保留 NaN。
    np.testing.assert_allclose(geometry.loc[5, "gr_gap_length_ft"], 0.0)
    np.testing.assert_allclose(geometry.loc[5, "gr_distance_to_left_observed_ft"], 0.0)
    np.testing.assert_allclose(geometry.loc[5, "gr_distance_to_right_observed_ft"], 0.0)
    assert np.isnan(geometry.loc[5, "gr_relative_position_inside_gap"])


def test_rf02a_feature_list_and_hidden_well_statistics_are_fixed() -> None:
    """RF02a 必须严格新增九列，且隐藏区有效率与最长缺口计算正确。"""

    rows = build_rf02a_lgbm_rows(make_missing_gr_well(), "missing", 0)
    expected_features = [
        "gr_gap_length_ft",
        "gr_distance_to_left_observed_ft",
        "gr_distance_to_right_observed_ft",
        "gr_relative_position_inside_gap",
        "gr_valid_fraction_50",
        "gr_valid_fraction_100",
        "gr_valid_fraction_200",
        "well_hidden_gr_valid_fraction",
        "well_hidden_longest_gr_gap_ft",
    ]

    assert RF02A_MISSING_FEATURE_COLUMNS == expected_features
    assert ALL_RF02A_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_RF02A_FEATURE_COLUMNS) == 21

    # 隐藏位置 2..9 共 8 行，其中 5 行有 GR；最长缺口含 2 行，即 2 ft。
    np.testing.assert_allclose(rows["well_hidden_gr_valid_fraction"], 5.0 / 8.0)
    np.testing.assert_allclose(rows["well_hidden_longest_gr_gap_ft"], 2.0)


def test_geometry_ignores_gr_values_and_hidden_tvt() -> None:
    """保持 GR mask 时改变数值，或改变隐藏 TVT，都不能改变九个缺失几何特征。"""

    original = make_missing_gr_well()
    changed = make_missing_gr_well()
    changed.loc[changed["GR"].notna(), "GR"] += 1000.0
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_rf02a_lgbm_rows(original, "missing", 0)
    changed_rows = build_rf02a_lgbm_rows(changed, "missing", 0)

    assert_frame_equal(
        original_rows[RF02A_MISSING_FEATURE_COLUMNS],
        changed_rows[RF02A_MISSING_FEATURE_COLUMNS],
        check_exact=True,
    )
