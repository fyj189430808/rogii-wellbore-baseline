"""新版路线图 RF01c1：五个窗口内相邻 U 坡度的总体标准差。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS, build_rf01a_lgbm_rows


# 本阶段只新增 slope std，不带上一阶段失败的坡差。
RF01C_STD_FEATURE_COLUMNS = [
    "u_adjacent_slope_std_50",
    "u_adjacent_slope_std_100",
    "u_adjacent_slope_std_200",
    "u_adjacent_slope_std_500",
    "u_adjacent_slope_std_1000",
]

# RF01a 十七列加五个稳定性统计，共二十二列。
ALL_RF01C_STD_FEATURE_COLUMNS = (
    ALL_RF01A_FEATURE_COLUMNS + RF01C_STD_FEATURE_COLUMNS
)


def calculate_adjacent_u_slope_std(
    horizontal_df: pd.DataFrame,
    window_ft: float,
) -> float:
    """返回可见末端窗口内相邻有效点 ΔU/ΔMD 的总体标准差。"""

    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    if visible_df.empty:
        raise ValueError("RF01c slope std 需要可见 TVT_input 前缀")

    last_visible_md = float(visible_df["MD"].iloc[-1])
    window_start_md = last_visible_md - float(window_ft)
    window_df = visible_df.loc[visible_df["MD"] >= window_start_md]

    md_values = window_df["MD"].to_numpy(dtype=np.float64)
    u_values = (
        window_df["TVT_input"].to_numpy(dtype=np.float64)
        + window_df["Z"].to_numpy(dtype=np.float64)
    )
    md_difference = np.diff(md_values)
    u_difference = np.diff(u_values)
    valid_step = md_difference > 1e-12
    if not valid_step.any():
        return float("nan")

    adjacent_slopes = u_difference[valid_step] / md_difference[valid_step]
    return float(np.std(adjacent_slopes, ddof=0))


def build_rf01c_std_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在纯 Huber RF01a 底座上追加五个相邻坡度标准差。"""

    result = build_rf01a_lgbm_rows(horizontal_df, well_id, fold)
    for window_ft in [50, 100, 200, 500, 1000]:
        slope_std = calculate_adjacent_u_slope_std(
            horizontal_df,
            window_ft=float(window_ft),
        )
        result[f"u_adjacent_slope_std_{window_ft}"] = slope_std

    return result

