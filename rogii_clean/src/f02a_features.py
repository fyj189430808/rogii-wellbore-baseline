"""F02a：只保留 500 ft 窗口内的 50 ft 分块 U 斜率标准差。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.f01b_features import ALL_F01B_FEATURE_COLUMNS, build_f01b_lgbm_rows
from src.f02_features import calculate_visible_block_slopes


# Oracle 审计中只有这一列与未来斜率误差显示出可辨认的相关性。
F02A_FEATURE_COLUMNS = ["u_block_slope_std_500"]

# F02a 严格使用 F01b 十九列加一列，共二十列。
ALL_F02A_FEATURE_COLUMNS = ALL_F01B_FEATURE_COLUMNS + F02A_FEATURE_COLUMNS


def build_f02a_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """构造 F01b 隐藏行，并追加一个可见前缀斜率离散程度特征。"""

    # 基础十九列、标签和隐藏行顺序与 F01b 完全相同。
    result = build_f01b_lgbm_rows(horizontal_df, well_id, fold)

    # 最近 500 ft 最多形成十个 50 ft 块，每块只使用 TVT_input 可见前缀。
    slopes_500 = calculate_visible_block_slopes(
        horizontal_df,
        window_ft=500.0,
    )

    # 使用总体标准差；异常短前缀没有有效块时回退为有限的零值。
    if len(slopes_500) > 0:
        slope_std_500 = float(np.std(slopes_500))
    else:
        slope_std_500 = 0.0

    # 这是井级前缀统计，对该井全部隐藏行重复，不读取隐藏 TVT 真值。
    result["u_block_slope_std_500"] = slope_std_500
    return result

