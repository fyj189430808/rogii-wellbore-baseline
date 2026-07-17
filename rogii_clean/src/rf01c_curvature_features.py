"""新版路线图 RF01c3：五个窗口内 U 倾角随 MD 的 robust curvature。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rf01a_features import (
    ALL_RF01A_FEATURE_COLUMNS,
    build_rf01a_lgbm_rows,
    fit_huber_slope,
)


# 本阶段只新增 curvature，不携带失败的坡差、std 或正坡比例。
RF01C_CURVATURE_FEATURE_COLUMNS = [
    "u_curvature_50",
    "u_curvature_100",
    "u_curvature_200",
    "u_curvature_500",
    "u_curvature_1000",
]

# RF01a 十七列加五个 curvature，共二十二列。
ALL_RF01C_CURVATURE_FEATURE_COLUMNS = (
    ALL_RF01A_FEATURE_COLUMNS + RF01C_CURVATURE_FEATURE_COLUMNS
)


def calculate_u_curvature(
    horizontal_df: pd.DataFrame,
    window_ft: float,
) -> float:
    """用 Huber 拟合相邻 U 坡度对坡段中点 MD 的斜率，单位为 1/ft。"""

    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    if visible_df.empty:
        raise ValueError("RF01c curvature 需要可见 TVT_input 前缀")

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
    if int(np.sum(valid_step)) < 2:
        return float("nan")

    adjacent_slopes = u_difference[valid_step] / md_difference[valid_step]
    segment_midpoint_md = (
        (md_values[:-1] + md_values[1:]) / 2.0
    )[valid_step]
    return fit_huber_slope(segment_midpoint_md, adjacent_slopes)


def build_rf01c_curvature_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在纯 Huber RF01a 底座上追加五个 robust curvature。"""

    result = build_rf01a_lgbm_rows(horizontal_df, well_id, fold)
    for window_ft in [50, 100, 200, 500, 1000]:
        curvature = calculate_u_curvature(
            horizontal_df,
            window_ft=float(window_ft),
        )
        result[f"u_curvature_{window_ft}"] = curvature

    return result

