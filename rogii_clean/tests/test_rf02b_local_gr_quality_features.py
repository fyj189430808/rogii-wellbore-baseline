"""RF02b：局部 GR MAD、总体方差和观测支持量测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_features import FEATURE_COLUMNS
from src.rf02b_local_gr_quality_features import (
    ALL_RF02B_FEATURE_COLUMNS,
    RF02B_QUALITY_FEATURE_COLUMNS,
    build_rf02b_lgbm_rows,
    calculate_local_gr_quality,
)
from tests.test_rf02a_ablation_features import make_long_missing_gr_well


def test_local_mad_and_variance_match_hand_calculation() -> None:
    """中心 5 行的四个实测值应得到 MAD=1.5、总体方差=7.1875。"""

    horizontal_df = pd.DataFrame(
        {
            "MD": np.arange(5, dtype=np.float64),
            "GR": np.array([1.0, 2.0, np.nan, 4.0, 8.0]),
        }
    )
    quality = calculate_local_gr_quality(
        horizontal_df,
        windows_ft=[4],
        minimum_observed_count=1,
    )

    np.testing.assert_allclose(quality.loc[2, "gr_local_observed_count_4"], 4.0)
    np.testing.assert_allclose(quality.loc[2, "gr_local_mad_4"], 1.5)
    np.testing.assert_allclose(quality.loc[2, "gr_local_variance_4"], 7.1875)


def test_local_quality_is_nan_below_minimum_support() -> None:
    """窗口只有四个实测点而门槛为五时，MAD/方差必须为 NaN，count 仍保留。"""

    horizontal_df = pd.DataFrame(
        {
            "MD": np.arange(5, dtype=np.float64),
            "GR": np.array([1.0, 2.0, np.nan, 4.0, 8.0]),
        }
    )
    quality = calculate_local_gr_quality(
        horizontal_df,
        windows_ft=[4],
        minimum_observed_count=5,
    )

    np.testing.assert_allclose(quality.loc[2, "gr_local_observed_count_4"], 4.0)
    assert np.isnan(quality.loc[2, "gr_local_mad_4"])
    assert np.isnan(quality.loc[2, "gr_local_variance_4"])


def test_rf02b_model_uses_only_six_quality_features() -> None:
    """observed count 只用于诊断；模型严格使用三个窗口的 MAD/方差六列。"""

    rows = build_rf02b_lgbm_rows(make_long_missing_gr_well(), "long", 0)
    expected_features = [
        "gr_local_mad_50",
        "gr_local_variance_50",
        "gr_local_mad_100",
        "gr_local_variance_100",
        "gr_local_mad_200",
        "gr_local_variance_200",
    ]

    assert RF02B_QUALITY_FEATURE_COLUMNS == expected_features
    assert ALL_RF02B_FEATURE_COLUMNS == FEATURE_COLUMNS + expected_features
    assert len(ALL_RF02B_FEATURE_COLUMNS) == 18
    for window_ft in [50, 100, 200]:
        assert f"gr_local_observed_count_{window_ft}" in rows.columns


def test_rf02b_features_ignore_hidden_tvt() -> None:
    """修改隐藏 TVT 真值不能改变 B00 加六个局部 GR 数值特征。"""

    original = make_long_missing_gr_well()
    changed = make_long_missing_gr_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_rf02b_lgbm_rows(original, "long", 0)
    changed_rows = build_rf02b_lgbm_rows(changed, "long", 0)
    assert_frame_equal(
        original_rows[ALL_RF02B_FEATURE_COLUMNS],
        changed_rows[ALL_RF02B_FEATURE_COLUMNS],
        check_exact=True,
    )
