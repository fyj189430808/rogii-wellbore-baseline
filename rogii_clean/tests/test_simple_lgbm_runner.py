"""最简 LGBM runner 的两个核心数学测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# 把 rogii_clean 加入模块路径，便于直接导入脚本中的纯函数。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (
    fold_indices,
    restore_absolute_tvt,
    select_feature_definition,
    validate_feature_table,
)
from src.f01b_features import ALL_F01B_FEATURE_COLUMNS, build_f01b_lgbm_rows
from src.f01c_features import ALL_F01C_FEATURE_COLUMNS, build_f01c_lgbm_rows
from src.f02_features import ALL_F02_FEATURE_COLUMNS, build_f02_lgbm_rows
from src.f02a_features import ALL_F02A_FEATURE_COLUMNS, build_f02a_lgbm_rows
from src.f03_features import ALL_F03_FEATURE_COLUMNS, build_f03_lgbm_rows
from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS, build_rf01a_lgbm_rows
from src.rf01a2_features import ALL_RF01A2_FEATURE_COLUMNS, build_rf01a2_lgbm_rows
from src.rf01b_features import ALL_RF01B_FEATURE_COLUMNS, build_rf01b_lgbm_rows
from src.rf01c_std_features import (
    ALL_RF01C_STD_FEATURE_COLUMNS,
    build_rf01c_std_lgbm_rows,
)
from src.rf01c_ratio_features import (
    ALL_RF01C_RATIO_FEATURE_COLUMNS,
    build_rf01c_ratio_lgbm_rows,
)
from src.rf01c_curvature_features import (
    ALL_RF01C_CURVATURE_FEATURE_COLUMNS,
    build_rf01c_curvature_lgbm_rows,
)
from src.rf01d_projection_features import (
    ALL_RF01D_FEATURE_COLUMNS,
    build_rf01d_lgbm_rows,
)
from src.rf01e_slope_subset_features import (
    ALL_RF01E_FEATURE_COLUMNS,
    ALL_RF01F_FEATURE_COLUMNS,
    build_rf01e_lgbm_rows,
    build_rf01f_lgbm_rows,
)
from src.rf02a_gr_missing_features import (
    ALL_RF02A_FEATURE_COLUMNS,
    build_rf02a_lgbm_rows,
)
from src.rf02a_mask_shift_control_features import (
    ALL_RF02A_CONTROL_FEATURE_COLUMNS,
    build_rf02a_mask_shift_control_rows,
)
from src.rf02a_ablation_features import (
    ALL_RF02A_LOCAL_FEATURE_COLUMNS,
    ALL_RF02A_WELL_FEATURE_COLUMNS,
    build_rf02a_local_null_rows,
    build_rf02a_local_rows,
    build_rf02a_well_null_rows,
    build_rf02a_well_rows,
)
from src.rf02b_local_gr_quality_features import (
    ALL_RF02B_FEATURE_COLUMNS,
    build_rf02b_lgbm_rows,
)


def test_fold_indices_separates_train_and_validation() -> None:
    """验证指定 fold 的行只进入验证集，不会同时进入训练集。"""

    feature_table = pd.DataFrame({"fold": [0, 0, 1, 1, 2]})
    train_indices, validation_indices = fold_indices(feature_table, validation_fold=1)

    assert train_indices.tolist() == [0, 1, 4]
    assert validation_indices.tolist() == [2, 3]
    assert set(train_indices).isdisjoint(set(validation_indices))


def test_restore_absolute_tvt_adds_visible_anchor() -> None:
    """验证模型输出的相对 TVT 会逐行加回本井可见末值。"""

    anchor = np.array([100.0, 100.0, 200.0])
    predicted_delta = np.array([1.5, -2.0, 3.0])

    prediction = restore_absolute_tvt(anchor, predicted_delta)

    np.testing.assert_allclose(prediction, [101.5, 98.0, 203.0])


def test_select_feature_definition_returns_f01b_ablation() -> None:
    """F01b 配置必须显式选择十九列特征及对应的逐井构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "prefix_u_slope_19_v1"}
    )

    assert feature_columns == ALL_F01B_FEATURE_COLUMNS
    assert row_builder is build_f01b_lgbm_rows


def test_select_feature_definition_returns_f01c_gated_projection() -> None:
    """F01c 配置必须显式选择二十二列特征及对应构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "prefix_u_gated_projection_22_v1"}
    )

    assert feature_columns == ALL_F01C_FEATURE_COLUMNS
    assert row_builder is build_f01c_lgbm_rows


def test_select_feature_definition_returns_f02_stability_features() -> None:
    """F02 配置必须显式选择二十五列特征及对应构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "prefix_u_stability_25_v1"}
    )

    assert feature_columns == ALL_F02_FEATURE_COLUMNS
    assert row_builder is build_f02_lgbm_rows


def test_select_feature_definition_returns_f02a_single_std_feature() -> None:
    """F02a 配置必须显式选择二十列特征及对应构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "prefix_u_std500_20_v1"}
    )

    assert feature_columns == ALL_F02A_FEATURE_COLUMNS
    assert row_builder is build_f02a_lgbm_rows


def test_select_feature_definition_returns_f03_xy_slope_features() -> None:
    """F03 配置必须显式选择十九列特征及对应构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "prefix_u_xy_slope_19_v1"}
    )

    assert feature_columns == ALL_F03_FEATURE_COLUMNS
    assert row_builder is build_f03_lgbm_rows


def test_select_feature_definition_returns_rf01a_huber_slopes() -> None:
    """RF01a 配置必须显式选择十七列及五窗口 Huber 构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_slopes_17_v1"}
    )

    assert feature_columns == ALL_RF01A_FEATURE_COLUMNS
    assert row_builder is build_rf01a_lgbm_rows


def test_select_feature_definition_returns_rf01a2_hybrid_slopes() -> None:
    """RF01a2 配置必须选择十七列及预注册的混合倾角构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_hybrid_slopes_17_v1"}
    )

    assert feature_columns == ALL_RF01A2_FEATURE_COLUMNS
    assert row_builder is build_rf01a2_lgbm_rows


def test_select_feature_definition_returns_rf01b_slope_differences() -> None:
    """RF01b 配置必须选择二十一列及相邻尺度坡差构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_slope_differences_21_v1"}
    )

    assert feature_columns == ALL_RF01B_FEATURE_COLUMNS
    assert row_builder is build_rf01b_lgbm_rows


def test_select_feature_definition_returns_rf01c_slope_std() -> None:
    """RF01c1 配置必须选择二十二列及相邻坡度标准差构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_plus_adjacent_std_22_v1"}
    )

    assert feature_columns == ALL_RF01C_STD_FEATURE_COLUMNS
    assert row_builder is build_rf01c_std_lgbm_rows


def test_select_feature_definition_returns_rf01c_positive_ratio() -> None:
    """RF01c2 配置必须选择二十二列及正坡比例构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_plus_positive_ratio_22_v1"}
    )

    assert feature_columns == ALL_RF01C_RATIO_FEATURE_COLUMNS
    assert row_builder is build_rf01c_ratio_lgbm_rows


def test_select_feature_definition_returns_rf01c_curvature() -> None:
    """RF01c3 配置必须选择二十二列及 robust curvature 构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_plus_curvature_22_v1"}
    )

    assert feature_columns == ALL_RF01C_CURVATURE_FEATURE_COLUMNS
    assert row_builder is build_rf01c_curvature_lgbm_rows


def test_select_feature_definition_returns_rf01d_projection() -> None:
    """RF01d 配置必须选择二十二列及逐行 U 投影增量构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_plus_projection_22_v1"}
    )

    assert feature_columns == ALL_RF01D_FEATURE_COLUMNS
    assert row_builder is build_rf01d_lgbm_rows


def test_select_feature_definition_returns_rf01e_three_slopes() -> None:
    """RF01e 配置必须选择 B00 加 50/500/1000 ft 三个倾角。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_slopes_50_500_1000_15_v1"}
    )

    assert feature_columns == ALL_RF01E_FEATURE_COLUMNS
    assert row_builder is build_rf01e_lgbm_rows


def test_select_feature_definition_returns_rf01f_single_500_slope() -> None:
    """RF01f 配置必须选择 B00 加单个 500 ft Huber 倾角。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf01_huber_slope_500_13_v1"}
    )

    assert feature_columns == ALL_RF01F_FEATURE_COLUMNS
    assert row_builder is build_rf01f_lgbm_rows


def test_select_feature_definition_returns_rf02a_gr_missing_geometry() -> None:
    """RF02a 配置必须选择 B00 加九个原始 GR 缺失几何特征。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_gr_missing_geometry_21_v1"}
    )

    assert feature_columns == ALL_RF02A_FEATURE_COLUMNS
    assert row_builder is build_rf02a_lgbm_rows


def test_select_feature_definition_returns_rf02a_mask_shift_control() -> None:
    """RF02a-null 必须选择与真实版同维的错误位置 mask 构造函数。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_gr_missing_mask_shift_control_21_v1"}
    )

    assert feature_columns == ALL_RF02A_CONTROL_FEATURE_COLUMNS
    assert row_builder is build_rf02a_mask_shift_control_rows


def test_select_feature_definition_returns_rf02a_well_only() -> None:
    """RF02a-well 必须只选择 B00 加两个真实井级缺失统计。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_gr_missing_well_only_14_v1"}
    )
    assert feature_columns == ALL_RF02A_WELL_FEATURE_COLUMNS
    assert row_builder is build_rf02a_well_rows


def test_select_feature_definition_returns_rf02a_well_null() -> None:
    """RF02a-well-null 必须选择同维的平移 mask 井级统计。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_gr_missing_well_null_14_v1"}
    )
    assert feature_columns == ALL_RF02A_WELL_FEATURE_COLUMNS
    assert row_builder is build_rf02a_well_null_rows


def test_select_feature_definition_returns_rf02a_centered_local() -> None:
    """RF02a-local 必须只选择 B00 加三个中心化局部有效率。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_gr_missing_centered_local_15_v1"}
    )
    assert feature_columns == ALL_RF02A_LOCAL_FEATURE_COLUMNS
    assert row_builder is build_rf02a_local_rows


def test_select_feature_definition_returns_rf02a_centered_local_null() -> None:
    """RF02a-local-null 必须选择同维的平移 mask 中心化局部有效率。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_gr_missing_centered_local_null_15_v1"}
    )
    assert feature_columns == ALL_RF02A_LOCAL_FEATURE_COLUMNS
    assert row_builder is build_rf02a_local_null_rows


def test_select_feature_definition_returns_rf02b_local_gr_quality() -> None:
    """RF02b 配置必须选择 B00 加六个局部 GR MAD/方差特征。"""

    feature_columns, row_builder = select_feature_definition(
        {"feature_version": "rf02_local_gr_quality_18_v1"}
    )
    assert feature_columns == ALL_RF02B_FEATURE_COLUMNS
    assert row_builder is build_rf02b_lgbm_rows


def make_minimal_validation_tables(gated_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造覆盖五个 fold 的最小特征表和对应注册表。"""

    feature_table = pd.DataFrame(
        {
            "well_id": [f"well_{fold}" for fold in range(5)],
            "row_index": [0, 0, 0, 0, 0],
            "fold": [0, 1, 2, 3, 4],
            "gr_raw": [np.nan, 10.0, 20.0, 30.0, 40.0],
            "gated_projection": gated_values,
            "target_tvt": [100.0] * 5,
            "target_delta": [0.0] * 5,
            "carry_tvt": [100.0] * 5,
        }
    )
    registry = pd.DataFrame(
        {
            "fold": [0, 1, 2, 3, 4],
            "hidden_rows": [1, 1, 1, 1, 1],
        }
    )
    return feature_table, registry


def test_validate_feature_table_allows_configured_nan_feature() -> None:
    """显式声明为门控特征的列可以含 NaN，其他完整性检查仍然执行。"""

    feature_table, registry = make_minimal_validation_tables(
        [1.0, 2.0, np.nan, np.nan, 5.0]
    )

    validate_feature_table(
        feature_table,
        registry,
        expected_rows=5,
        feature_columns=["gr_raw", "gated_projection"],
        allow_nan_features=["gated_projection"],
    )


def test_validate_feature_table_rejects_unconfigured_nan_feature() -> None:
    """普通特征中的 NaN 仍然必须触发错误，不能放松原有数据检查。"""

    feature_table, registry = make_minimal_validation_tables(
        [1.0, 2.0, np.nan, np.nan, 5.0]
    )

    with pytest.raises(ValueError, match="NaN"):
        validate_feature_table(
            feature_table,
            registry,
            expected_rows=5,
            feature_columns=["gr_raw", "gated_projection"],
            allow_nan_features=[],
        )
