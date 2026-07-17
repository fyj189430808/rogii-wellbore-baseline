"""F03：用累计 XY 水平距离代替 MD，计算可见前缀的 U 倾角。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


# 三个窗口各产生斜率和拟合误差，再加入短长窗口斜率差，共七列。
F03_FEATURE_COLUMNS = [
    "u_xy_slope_200",
    "u_xy_fit_rmse_200",
    "u_xy_slope_500",
    "u_xy_fit_rmse_500",
    "u_xy_slope_1000",
    "u_xy_fit_rmse_1000",
    "u_xy_slope_200_minus_1000",
]

# F03 严格使用 B00 十二列加七个 XY 前缀统计，共十九列。
ALL_F03_FEATURE_COLUMNS = FEATURE_COLUMNS + F03_FEATURE_COLUMNS


def calculate_visible_cumulative_xy_distance(
    horizontal_df: pd.DataFrame,
) -> np.ndarray:
    """返回可见前缀从首点开始累计的 XY 路径距离，单位沿用原始坐标。"""

    # 只截取 TVT_input 可见行，隐藏段坐标不会改变前缀累计距离。
    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    if visible_df.empty:
        raise ValueError("F03 需要至少一个 TVT_input 可见点")

    visible_x = visible_df["X"].to_numpy(dtype=np.float64)
    visible_y = visible_df["Y"].to_numpy(dtype=np.float64)
    if not np.isfinite(visible_x).all() or not np.isfinite(visible_y).all():
        raise ValueError("F03 可见前缀的 X/Y 含 NaN 或 Inf")

    # 每一步水平位移为 sqrt(dX^2+dY^2)，不把垂向 Z 或测量深度 MD 计入距离。
    x_step = np.diff(visible_x)
    y_step = np.diff(visible_y)
    horizontal_step = np.sqrt(x_step * x_step + y_step * y_step)

    # 第一个可见点距离定义为零，之后逐步累加，结果长度等于可见行数。
    cumulative_distance = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.cumsum(horizontal_step, dtype=np.float64),
        ]
    )
    return cumulative_distance


def fit_visible_u_xy_trend(
    horizontal_df: pd.DataFrame,
    window_ft: float,
) -> tuple[float, float]:
    """在末端指定 XY 距离窗口内拟合 U 对累计水平距离的 OLS 斜率和 RMSE。"""

    # visible_df 与累计距离逐位对应，只含推理时合法的 TVT_input 前缀。
    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    cumulative_xy_distance = calculate_visible_cumulative_xy_distance(horizontal_df)

    last_visible_distance = float(cumulative_xy_distance[-1])
    window_start_distance = last_visible_distance - float(window_ft)
    window_mask = cumulative_xy_distance >= window_start_distance

    window_distance = cumulative_xy_distance[window_mask]
    visible_u = (
        visible_df["TVT_input"].to_numpy(dtype=np.float64)
        + visible_df["Z"].to_numpy(dtype=np.float64)
    )
    window_u = visible_u[window_mask]

    # 中心化后计算 OLS，既提高数值稳定性，也让截距等于窗口 U 均值。
    distance_centered = window_distance - float(np.mean(window_distance))
    denominator = float(np.sum(distance_centered * distance_centered))

    # 水平距离没有变化时无法识别地层倾角，使用零斜率和常数 U 拟合。
    if denominator <= 1e-12:
        slope = 0.0
        fitted_u = np.full_like(window_u, float(np.mean(window_u)))
    else:
        u_centered = window_u - float(np.mean(window_u))
        slope = float(np.sum(distance_centered * u_centered) / denominator)
        fitted_u = float(np.mean(window_u)) + slope * distance_centered

    residual = window_u - fitted_u
    fit_rmse = float(np.sqrt(np.mean(residual * residual)))
    return slope, fit_rmse


def build_f03_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在 B00 十二列上增加七个 XY 水平距离前缀统计。"""

    # B00 builder 负责连续前缀、隐藏后缀、目标和基础十二列的统一定义。
    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    xy_slopes: dict[int, float] = {}
    for window_ft in [200, 500, 1000]:
        slope, fit_rmse = fit_visible_u_xy_trend(
            horizontal_df,
            window_ft=float(window_ft),
        )
        xy_slopes[window_ft] = slope
        result[f"u_xy_slope_{window_ft}"] = slope
        result[f"u_xy_fit_rmse_{window_ft}"] = fit_rmse

    result["u_xy_slope_200_minus_1000"] = xy_slopes[200] - xy_slopes[1000]
    return result

