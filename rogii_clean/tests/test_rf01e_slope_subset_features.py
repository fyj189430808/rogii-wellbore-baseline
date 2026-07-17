"""RF01e/f：Huber U 倾角窗口子集测试。"""

import sys
from pathlib import Path

import numpy as np
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_features import FEATURE_COLUMNS
from src.rf01e_slope_subset_features import (
    ALL_RF01E_FEATURE_COLUMNS,
    ALL_RF01F_FEATURE_COLUMNS,
    RF01E_SLOPE_FEATURE_COLUMNS,
    RF01F_SLOPE_FEATURE_COLUMNS,
    build_rf01e_lgbm_rows,
    build_rf01f_lgbm_rows,
)
from tests.test_rf01a_features import make_linear_u_well


def test_rf01e_keeps_only_50_500_1000_slopes() -> None:
    """三窗口版必须只在 B00 后加入 50、500、1000 ft Huber 倾角。"""

    expected_features = [
        "u_huber_slope_50",
        "u_huber_slope_500",
        "u_huber_slope_1000",
    ]
    rows = build_rf01e_lgbm_rows(make_linear_u_well(), "linear", 0)

    assert RF01E_SLOPE_FEATURE_COLUMNS == expected_features
    assert ALL_RF01E_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_RF01E_FEATURE_COLUMNS) == 15
    for feature_name in expected_features:
        np.testing.assert_allclose(rows[feature_name], [0.5, 0.5], atol=1e-12)


def test_rf01f_keeps_only_500_slope() -> None:
    """最终正对照必须只在 B00 后加入一个 500 ft Huber 倾角。"""

    expected_features = ["u_huber_slope_500"]
    rows = build_rf01f_lgbm_rows(make_linear_u_well(), "linear", 0)

    assert RF01F_SLOPE_FEATURE_COLUMNS == expected_features
    assert ALL_RF01F_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_RF01F_FEATURE_COLUMNS) == 13
    np.testing.assert_allclose(rows["u_huber_slope_500"], [0.5, 0.5], atol=1e-12)


def test_hidden_tvt_does_not_change_slope_subset_features() -> None:
    """三窗口和单窗口版都不能读取隐藏 TVT 真值。"""

    original = make_linear_u_well()
    changed = make_linear_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    for builder, feature_columns in [
        (build_rf01e_lgbm_rows, ALL_RF01E_FEATURE_COLUMNS),
        (build_rf01f_lgbm_rows, ALL_RF01F_FEATURE_COLUMNS),
    ]:
        original_rows = builder(original, "linear", 0)
        changed_rows = builder(changed, "linear", 0)
        assert_frame_equal(
            original_rows[feature_columns],
            changed_rows[feature_columns],
            check_exact=True,
        )
