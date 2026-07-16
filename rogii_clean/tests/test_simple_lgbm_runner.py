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
