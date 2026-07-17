"""新版路线图 RF01b：在五窗口 Huber 倾角上增加相邻尺度坡差。"""

from __future__ import annotations

import pandas as pd

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS, build_rf01a_lgbm_rows


# 只比较相邻尺度，避免同时制造大量高度重复的任意窗口组合。
RF01B_DIFFERENCE_COLUMNS = [
    "u_huber_slope_50_minus_100",
    "u_huber_slope_100_minus_200",
    "u_huber_slope_200_minus_500",
    "u_huber_slope_500_minus_1000",
]

# RF01b 输入为 RF01a 十七列加四个坡差，共二十一列。
ALL_RF01B_FEATURE_COLUMNS = ALL_RF01A_FEATURE_COLUMNS + RF01B_DIFFERENCE_COLUMNS


def build_rf01b_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """复用 RF01a 五个 Huber 倾角，并追加四个相邻窗口差值。"""

    result = build_rf01a_lgbm_rows(horizontal_df, well_id, fold)
    adjacent_windows = [(50, 100), (100, 200), (200, 500), (500, 1000)]

    for short_window, long_window in adjacent_windows:
        short_slope = result[f"u_huber_slope_{short_window}"]
        long_slope = result[f"u_huber_slope_{long_window}"]
        result[
            f"u_huber_slope_{short_window}_minus_{long_window}"
        ] = short_slope - long_slope

    return result

