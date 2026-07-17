"""RF02a 归因消融：井级两列与局部有效率三列测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_features import FEATURE_COLUMNS
from src.rf02a_ablation_features import (
    ALL_RF02A_LOCAL_FEATURE_COLUMNS,
    ALL_RF02A_WELL_FEATURE_COLUMNS,
    RF02A_LOCAL_FEATURE_COLUMNS,
    RF02A_WELL_FEATURE_COLUMNS,
    build_rf02a_local_null_rows,
    build_rf02a_local_rows,
    build_rf02a_well_null_rows,
    build_rf02a_well_rows,
)
from tests.test_rf02a_gr_missing_features import make_missing_gr_well


def make_long_missing_gr_well() -> pd.DataFrame:
    """构造 300 行小井，使 50/100/200 ft 窗口小于井长并含局部缺口变化。"""

    md = np.arange(300, dtype=np.float64)
    tvt = 200.0 + 0.1 * md
    tvt_input = tvt.copy()
    tvt_input[50:] = np.nan
    gr = 50.0 + 0.01 * md
    gr[60:80] = np.nan
    gr[160:200] = np.nan
    gr[250:260] = np.nan

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


def test_attribution_feature_lists_are_exact_and_disjoint() -> None:
    """well 组只能有两列，local 组只能有三个窗口有效率，二者不得混入。"""

    assert RF02A_WELL_FEATURE_COLUMNS == [
        "well_hidden_gr_valid_fraction",
        "well_hidden_longest_gr_gap_ft",
    ]
    assert RF02A_LOCAL_FEATURE_COLUMNS == [
        "gr_valid_fraction_50_centered",
        "gr_valid_fraction_100_centered",
        "gr_valid_fraction_200_centered",
    ]
    assert set(RF02A_WELL_FEATURE_COLUMNS).isdisjoint(RF02A_LOCAL_FEATURE_COLUMNS)
    assert ALL_RF02A_WELL_FEATURE_COLUMNS == FEATURE_COLUMNS + RF02A_WELL_FEATURE_COLUMNS
    assert ALL_RF02A_LOCAL_FEATURE_COLUMNS == FEATURE_COLUMNS + RF02A_LOCAL_FEATURE_COLUMNS
    assert len(ALL_RF02A_WELL_FEATURE_COLUMNS) == 14
    assert len(ALL_RF02A_LOCAL_FEATURE_COLUMNS) == 15


def test_real_and_null_keep_b00_equal_but_move_new_features() -> None:
    """同维 real/null 的 B00 输入相同，而错误 mask 至少改变一个新增统计。"""

    short_well_df = make_missing_gr_well()
    long_well_df = make_long_missing_gr_well()
    real_well_rows = build_rf02a_well_rows(short_well_df, "missing", 0)
    null_well_rows = build_rf02a_well_null_rows(short_well_df, "missing", 0)
    real_local_rows = build_rf02a_local_rows(long_well_df, "long", 0)
    null_local_rows = build_rf02a_local_null_rows(long_well_df, "long", 0)

    assert_frame_equal(
        real_well_rows[FEATURE_COLUMNS],
        null_well_rows[FEATURE_COLUMNS],
        check_exact=True,
    )
    assert_frame_equal(
        real_local_rows[FEATURE_COLUMNS],
        null_local_rows[FEATURE_COLUMNS],
        check_exact=True,
    )
    assert not np.array_equal(
        real_well_rows[RF02A_WELL_FEATURE_COLUMNS].to_numpy(),
        null_well_rows[RF02A_WELL_FEATURE_COLUMNS].to_numpy(),
    )
    assert not np.array_equal(
        real_local_rows[RF02A_LOCAL_FEATURE_COLUMNS].to_numpy(),
        null_local_rows[RF02A_LOCAL_FEATURE_COLUMNS].to_numpy(),
    )

    # centered local 必须逐行等于原始局部有效率减去本井隐藏段总体有效率。
    for window_ft in [50, 100, 200]:
        expected = (
            real_local_rows[f"gr_valid_fraction_{window_ft}"]
            - real_local_rows["well_hidden_gr_valid_fraction"]
        )
        np.testing.assert_allclose(
            real_local_rows[f"gr_valid_fraction_{window_ft}_centered"],
            expected,
        )


def test_attribution_subsets_ignore_hidden_tvt() -> None:
    """四个归因 builder 的模型输入都不能随隐藏 TVT 真值变化。"""

    original = make_missing_gr_well()
    changed = make_missing_gr_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    builders_and_features = [
        (build_rf02a_well_rows, ALL_RF02A_WELL_FEATURE_COLUMNS),
        (build_rf02a_well_null_rows, ALL_RF02A_WELL_FEATURE_COLUMNS),
        (build_rf02a_local_rows, ALL_RF02A_LOCAL_FEATURE_COLUMNS),
        (build_rf02a_local_null_rows, ALL_RF02A_LOCAL_FEATURE_COLUMNS),
    ]
    for builder, feature_columns in builders_and_features:
        original_rows = builder(original, "missing", 0)
        changed_rows = builder(changed, "missing", 0)
        assert_frame_equal(
            original_rows[feature_columns],
            changed_rows[feature_columns],
            check_exact=True,
        )
