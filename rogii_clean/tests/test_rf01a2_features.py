"""RF01a2：短窗 OLS、长窗 Huber 的合法混合倾角测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f01_features import fit_visible_u_trend
from src.lgbm_features import FEATURE_COLUMNS
from src.rf01a_features import fit_visible_u_huber_slope
from src.rf01a2_features import (
    ALL_RF01A2_FEATURE_COLUMNS,
    RF01A2_FEATURE_COLUMNS,
    build_rf01a2_lgbm_rows,
)


def make_outlier_u_well() -> pd.DataFrame:
    """构造带一个可见 U 异常点且具有自然隐藏后缀的小井。"""

    md = np.arange(30, dtype=np.float64)
    z = 1000.0 + 0.2 * md
    u = 1200.0 + 0.5 * md
    u[20] += 20.0
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[25:] = np.nan
    return pd.DataFrame(
        {
            "MD": md,
            "X": 10.0 + md,
            "Y": 20.0 + 0.5 * md,
            "Z": z,
            "GR": 50.0 + md,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_rf01a2_uses_ols_for_short_windows_and_huber_for_long_windows() -> None:
    """五列必须严格采用伪 holdout 预选的方法，而不是隐式混用。"""

    horizontal_df = make_outlier_u_well()
    rows = build_rf01a2_lgbm_rows(horizontal_df, "outlier", 0)
    expected_features = [
        "u_ols_slope_50",
        "u_ols_slope_100",
        "u_ols_slope_200",
        "u_huber_slope_500",
        "u_huber_slope_1000",
    ]

    assert RF01A2_FEATURE_COLUMNS == expected_features
    assert ALL_RF01A2_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_RF01A2_FEATURE_COLUMNS) == 17

    for window_ft in [50, 100, 200]:
        expected_slope, _ = fit_visible_u_trend(
            horizontal_df,
            window_ft=float(window_ft),
        )
        np.testing.assert_allclose(rows[f"u_ols_slope_{window_ft}"], expected_slope)

    for window_ft in [500, 1000]:
        expected_slope = fit_visible_u_huber_slope(
            horizontal_df,
            window_ft=float(window_ft),
        )
        np.testing.assert_allclose(
            rows[f"u_huber_slope_{window_ft}"],
            expected_slope,
        )


def test_hidden_tvt_does_not_change_rf01a2_features() -> None:
    """修改隐藏 TVT 时，混合倾角的十七个模型输入必须逐位不变。"""

    original = make_outlier_u_well()
    changed = make_outlier_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_rf01a2_lgbm_rows(original, "outlier", 0)
    changed_rows = build_rf01a2_lgbm_rows(changed, "outlier", 0)
    assert_frame_equal(
        original_rows[ALL_RF01A2_FEATURE_COLUMNS],
        changed_rows[ALL_RF01A2_FEATURE_COLUMNS],
        check_exact=True,
    )

