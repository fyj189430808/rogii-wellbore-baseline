"""新版路线图 RF01d：逐行 U 倾角投影增量测试。"""

import sys
from pathlib import Path

import numpy as np
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS
from src.rf01d_projection_features import (
    ALL_RF01D_FEATURE_COLUMNS,
    RF01D_PROJECTION_FEATURE_COLUMNS,
    build_rf01d_lgbm_rows,
)
from tests.test_rf01a_features import make_linear_u_well


def test_rf01d_projection_matches_linear_u_geometry() -> None:
    """线性 U 小井的投影应精确等于隐藏 TVT 相对可见末值的变化。"""

    horizontal_df = make_linear_u_well()
    rows = build_rf01d_lgbm_rows(horizontal_df, "linear", 0)
    expected_projection = np.array([0.3, 0.6], dtype=np.float64)

    for feature_name in RF01D_PROJECTION_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            rows[feature_name].to_numpy(dtype=np.float64),
            expected_projection,
            atol=1e-12,
        )


def test_rf01d_adds_only_five_projection_columns() -> None:
    """RF01d 必须严格等于 RF01a 十七列加五个投影增量。"""

    expected_features = [
        "u_projection_delta_50",
        "u_projection_delta_100",
        "u_projection_delta_200",
        "u_projection_delta_500",
        "u_projection_delta_1000",
    ]

    assert RF01D_PROJECTION_FEATURE_COLUMNS == expected_features
    assert ALL_RF01D_FEATURE_COLUMNS == ALL_RF01A_FEATURE_COLUMNS + expected_features
    assert len(ALL_RF01D_FEATURE_COLUMNS) == 22


def test_hidden_tvt_does_not_change_rf01d_features() -> None:
    """修改隐藏 TVT 真值时，RF01d 的全部模型输入必须逐位不变。"""

    original = make_linear_u_well()
    changed = make_linear_u_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_rows = build_rf01d_lgbm_rows(original, "linear", 0)
    changed_rows = build_rf01d_lgbm_rows(changed, "linear", 0)

    assert_frame_equal(
        original_rows[ALL_RF01D_FEATURE_COLUMNS],
        changed_rows[ALL_RF01D_FEATURE_COLUMNS],
        check_exact=True,
    )
