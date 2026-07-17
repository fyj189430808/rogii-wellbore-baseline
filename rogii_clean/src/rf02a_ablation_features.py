"""RF02a 归因消融：把井级缺失分布与逐行局部有效率分开输入模型。"""

from __future__ import annotations

import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS
from src.rf02a_gr_missing_features import build_rf02a_lgbm_rows
from src.rf02a_mask_shift_control_features import (
    build_rf02a_mask_shift_control_rows,
)


# well 组只保留两个对整口隐藏后缀汇总的 GR 可观测性统计。
RF02A_WELL_FEATURE_COLUMNS = [
    "well_hidden_gr_valid_fraction",
    "well_hidden_longest_gr_gap_ft",
]

# well 组模型输入为 B00 十二列加两个井级统计，共十四列。
ALL_RF02A_WELL_FEATURE_COLUMNS = FEATURE_COLUMNS + RF02A_WELL_FEATURE_COLUMNS

# local 组减去本井隐藏段总体有效率，只保留当前行附近相对全井的偏离。
RF02A_LOCAL_FEATURE_COLUMNS = [
    "gr_valid_fraction_50_centered",
    "gr_valid_fraction_100_centered",
    "gr_valid_fraction_200_centered",
]

# local 组模型输入为 B00 十二列加三个逐行统计，共十五列。
ALL_RF02A_LOCAL_FEATURE_COLUMNS = FEATURE_COLUMNS + RF02A_LOCAL_FEATURE_COLUMNS


def add_centered_local_features(result: pd.DataFrame) -> pd.DataFrame:
    """追加三个局部有效率减井级有效率的中心化列，并返回同一张隐藏行表。"""

    # well_fraction 在同一口井的所有隐藏行相同，表示隐藏段总体 GR 可观测率。
    well_fraction = result["well_hidden_gr_valid_fraction"]

    # 中心化会删除几乎完全相关的井级均值，只留下沿 MD 变化的局部稀疏程度。
    for window_ft in [50, 100, 200]:
        local_fraction = result[f"gr_valid_fraction_{window_ft}"]
        centered_fraction = local_fraction - well_fraction
        result[f"gr_valid_fraction_{window_ft}_centered"] = centered_fraction

    return result


def build_rf02a_well_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """计算真实 RF02a 表；runner 只选择其中 B00 加两个井级统计。"""

    # 复用已通过公式和防泄漏测试的真实 mask builder，不复制计算逻辑。
    return build_rf02a_lgbm_rows(horizontal_df, well_id, fold)


def build_rf02a_well_null_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """计算 mask-shift null 表；runner 只选择 B00 加两个错误位置井级统计。"""

    # 复用同维负对照 builder，确保真实 gr_raw/gr_missing 保持不变。
    return build_rf02a_mask_shift_control_rows(horizontal_df, well_id, fold)


def build_rf02a_local_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """计算真实 RF02a 表；runner 只选择 B00 加三个逐行局部有效率。"""

    # 复用真实 mask 统计，再显式去掉三个局部窗口共同携带的井级均值。
    result = build_rf02a_lgbm_rows(horizontal_df, well_id, fold)
    return add_centered_local_features(result)


def build_rf02a_local_null_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """计算 mask-shift null 表；runner 只选择 B00 加三个错误位置有效率。"""

    # null 也减去自身平移后隐藏段有效率，保证 real/null 都只剩局部偏离。
    result = build_rf02a_mask_shift_control_rows(horizontal_df, well_id, fold)
    return add_centered_local_features(result)
