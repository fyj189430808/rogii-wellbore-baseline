"""F01b：保留前缀 U 统计，但删除会随隐藏距离无限增长的外推特征。"""

from __future__ import annotations

import pandas as pd

from src.f01_features import build_f01_lgbm_rows
from src.lgbm_features import FEATURE_COLUMNS


# 这七列全部来自可见前缀，每口井只计算一次，不使用隐藏段 TVT 真值。
F01B_FEATURE_COLUMNS = [
    "u_slope_200",
    "u_fit_rmse_200",
    "u_slope_500",
    "u_fit_rmse_500",
    "u_slope_1000",
    "u_fit_rmse_1000",
    "u_slope_200_minus_1000",
]

# 单模 LightGBM 的输入严格等于 B00 的 12 列加上面七列，共 19 列。
ALL_F01B_FEATURE_COLUMNS = FEATURE_COLUMNS + F01B_FEATURE_COLUMNS

# 这三列是 F01 中被消融的唯一内容；它们会把局部斜率乘以完整隐藏距离。
REMOVED_LINEAR_PROJECTION_COLUMNS = [
    "u_linear_delta_200",
    "u_linear_delta_500",
    "u_linear_delta_1000",
]


def build_f01b_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """构造 F01 的合法隐藏行，然后只删除三列无界线性外推特征。"""

    # 复用已经测试过的 F01 公式，确保斜率和拟合误差与上一实验完全一致。
    f01_rows = build_f01_lgbm_rows(horizontal_df, well_id, fold)

    # 返回新表而不修改 f01_rows；其余元数据、标签和十九个输入特征保持不变。
    f01b_rows = f01_rows.drop(columns=REMOVED_LINEAR_PROJECTION_COLUMNS)

    return f01b_rows

