"""F01 可见前缀 U 倾角特征测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f01_features import ALL_FEATURE_COLUMNS, F01_FEATURE_COLUMNS, build_f01_lgbm_rows
from src.lgbm_features import FEATURE_COLUMNS


def make_linear_u_well() -> pd.DataFrame:
    """构造 U 斜率恒为 0.5、TVT 斜率恒为 0.3 的小井。"""

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


def test_f01_linear_u_features_match_known_slope() -> None:
    """线性 U 的三个窗口斜率和外推 delta 应精确等于手算值。"""

    rows = build_f01_lgbm_rows(make_linear_u_well(), "linear", 0)

    assert len(FEATURE_COLUMNS) == 12
    assert len(F01_FEATURE_COLUMNS) == 10
    assert len(ALL_FEATURE_COLUMNS) == 22

    for window in [200, 500, 1000]:
        np.testing.assert_allclose(rows[f"u_slope_{window}"], [0.5, 0.5])
        np.testing.assert_allclose(rows[f"u_fit_rmse_{window}"], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            rows[f"u_linear_delta_{window}"],
            [0.3, 0.6],
            atol=1e-12,
        )

    np.testing.assert_allclose(rows["u_slope_200_minus_1000"], [0.0, 0.0])


def test_hidden_tvt_does_not_change_f01_features() -> None:
    """隐藏真值改变时，基础和 F01 特征都必须逐位不变。"""

    original = make_linear_u_well()
    changed = make_linear_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_f01_lgbm_rows(original, "linear", 0)
    changed_rows = build_f01_lgbm_rows(changed, "linear", 0)

    assert_frame_equal(
        original_rows[ALL_FEATURE_COLUMNS],
        changed_rows[ALL_FEATURE_COLUMNS],
        check_exact=True,
    )
