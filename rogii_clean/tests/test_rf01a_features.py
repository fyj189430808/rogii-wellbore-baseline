"""新版路线图 RF01a：Huber 多窗口 U 倾角测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_features import FEATURE_COLUMNS
from src.rf01a_features import (
    ALL_RF01A_FEATURE_COLUMNS,
    RF01A_FEATURE_COLUMNS,
    build_rf01a_lgbm_rows,
    fit_huber_slope,
)
from src.rf01_diagnostics import build_prefix_holdout_predictions


def make_linear_u_well() -> pd.DataFrame:
    """构造 U 对 MD 斜率为 0.5、最后两行自然隐藏的小井。"""

    md = np.arange(8, dtype=np.float64)
    z = 1000.0 + 0.2 * md
    u = 1200.0 + 0.5 * md
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan

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


def test_huber_slope_resists_single_large_outlier() -> None:
    """单个极端 U 异常值不应像 OLS 一样明显拖动 Huber 斜率。"""

    md = np.arange(20, dtype=np.float64)
    clean_u = 100.0 + 2.0 * md
    contaminated_u = clean_u.copy()
    contaminated_u[-1] += 100.0

    huber_slope = fit_huber_slope(md, contaminated_u)
    ols_slope = float(np.polyfit(md, contaminated_u, 1)[0])

    assert abs(huber_slope - 2.0) < 0.15
    assert abs(huber_slope - 2.0) < abs(ols_slope - 2.0)


def test_rf01a_five_windows_match_known_linear_slope() -> None:
    """五个窗口都应恢复 0.5 ft/ft，并形成 B00 加五列的固定输入。"""

    rows = build_rf01a_lgbm_rows(make_linear_u_well(), "linear", 0)
    expected_features = [
        "u_huber_slope_50",
        "u_huber_slope_100",
        "u_huber_slope_200",
        "u_huber_slope_500",
        "u_huber_slope_1000",
    ]

    assert RF01A_FEATURE_COLUMNS == expected_features
    assert ALL_RF01A_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_RF01A_FEATURE_COLUMNS) == 17

    for feature_name in expected_features:
        np.testing.assert_allclose(rows[feature_name], [0.5, 0.5], atol=1e-12)


def test_hidden_tvt_does_not_change_rf01a_features() -> None:
    """修改隐藏 TVT 真值时，十七个模型输入必须逐位不变。"""

    original = make_linear_u_well()
    changed = make_linear_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_rf01a_lgbm_rows(original, "linear", 0)
    changed_rows = build_rf01a_lgbm_rows(changed, "linear", 0)

    assert_frame_equal(
        original_rows[ALL_RF01A_FEATURE_COLUMNS],
        changed_rows[ALL_RF01A_FEATURE_COLUMNS],
        check_exact=True,
    )


def test_prefix_holdout_predictions_are_exact_for_linear_u() -> None:
    """在线性 U 小井上，OLS/Huber 回放应精确命中可见前缀后段。"""

    predictions = build_prefix_holdout_predictions(
        make_linear_u_well(),
        cut_fraction=0.75,
        windows_ft=[50],
    )

    np.testing.assert_allclose(
        predictions["huber_50_tvt"],
        predictions["target_tvt"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        predictions["ols_50_tvt"],
        predictions["target_tvt"],
        atol=1e-12,
    )
    assert np.sqrt(
        np.mean(
            (predictions["carry_tvt"] - predictions["target_tvt"]) ** 2
        )
    ) > 0.0
