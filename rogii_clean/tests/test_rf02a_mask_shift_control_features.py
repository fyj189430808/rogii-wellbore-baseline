"""RF02a-null：井内循环平移 GR mask 的同维负对照测试。"""

import sys
from pathlib import Path

import numpy as np
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_features import build_simple_lgbm_rows
from src.rf02a_gr_missing_features import (
    ALL_RF02A_FEATURE_COLUMNS,
    RF02A_MISSING_FEATURE_COLUMNS,
)
from src.rf02a_mask_shift_control_features import (
    ALL_RF02A_CONTROL_FEATURE_COLUMNS,
    build_rf02a_mask_shift_control_rows,
    choose_mask_shift_rows,
    circular_shift_observed_mask,
)
from tests.test_rf02a_gr_missing_features import make_missing_gr_well


def test_control_shift_preserves_rate_and_moves_mask() -> None:
    """真实规模井的固定平移至少 201 行，并严格保持 observed 总数。"""

    observed_mask = np.zeros(1000, dtype=bool)
    observed_mask[[0, 5, 20, 400, 999]] = True
    shift_rows = choose_mask_shift_rows(len(observed_mask))
    shifted_mask = circular_shift_observed_mask(observed_mask)

    assert shift_rows >= 201
    assert int(shifted_mask.sum()) == int(observed_mask.sum())
    assert not np.array_equal(shifted_mask, observed_mask)


def test_control_keeps_true_b00_gr_and_same_feature_dimension() -> None:
    """null 只扰动新增九列，B00 的真实 gr_raw/gr_missing 必须逐位不变。"""

    horizontal_df = make_missing_gr_well()
    base_rows = build_simple_lgbm_rows(horizontal_df, "missing", 0)
    control_rows = build_rf02a_mask_shift_control_rows(horizontal_df, "missing", 0)

    assert ALL_RF02A_CONTROL_FEATURE_COLUMNS == ALL_RF02A_FEATURE_COLUMNS
    assert len(ALL_RF02A_CONTROL_FEATURE_COLUMNS) == 21
    np.testing.assert_allclose(
        control_rows["gr_raw"],
        base_rows["gr_raw"],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        control_rows["gr_missing"],
        base_rows["gr_missing"],
    )


def test_control_features_ignore_hidden_tvt() -> None:
    """修改隐藏 TVT 真值时，null 的九个扰动特征必须逐位不变。"""

    original = make_missing_gr_well()
    changed = make_missing_gr_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_rf02a_mask_shift_control_rows(original, "missing", 0)
    changed_rows = build_rf02a_mask_shift_control_rows(changed, "missing", 0)

    assert_frame_equal(
        original_rows[RF02A_MISSING_FEATURE_COLUMNS],
        changed_rows[RF02A_MISSING_FEATURE_COLUMNS],
        check_exact=True,
    )
