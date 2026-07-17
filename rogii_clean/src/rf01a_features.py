"""新版路线图 RF01a：五个可见前缀窗口的 Huber robust U 倾角。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


# Huber 的 1.345 阈值在正态噪声下保留较高效率，同时降低极端残差权重。
HUBER_DELTA = 1.345
HUBER_MAX_ITERATIONS = 30
HUBER_TOLERANCE = 1e-10

# 本阶段只测试 robust slope，不混入坡差、曲率、稳定性或外推 delta。
RF01A_FEATURE_COLUMNS = [
    "u_huber_slope_50",
    "u_huber_slope_100",
    "u_huber_slope_200",
    "u_huber_slope_500",
    "u_huber_slope_1000",
]

# 单模输入严格为 B00 十二列加五个 Huber 倾角，共十七列。
ALL_RF01A_FEATURE_COLUMNS = FEATURE_COLUMNS + RF01A_FEATURE_COLUMNS


def fit_weighted_line(
    md_values: np.ndarray,
    u_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    """对 U 与 MD 做带权直线拟合，返回 slope、intercept；不可识别时返回 None。"""

    total_weight = float(np.sum(weights))
    if total_weight <= 1e-12:
        return None

    weighted_md_mean = float(np.sum(weights * md_values) / total_weight)
    weighted_u_mean = float(np.sum(weights * u_values) / total_weight)
    md_centered = md_values - weighted_md_mean
    denominator = float(np.sum(weights * md_centered * md_centered))
    if denominator <= 1e-12:
        return None

    u_centered = u_values - weighted_u_mean
    slope = float(np.sum(weights * md_centered * u_centered) / denominator)
    intercept = weighted_u_mean - slope * weighted_md_mean
    return slope, intercept


def fit_huber_slope(
    md_values: np.ndarray,
    u_values: np.ndarray,
) -> float:
    """使用确定性的 Huber IRLS 拟合 U 对 MD 的 robust slope，单位为 ft/ft。"""

    md_array = np.asarray(md_values, dtype=np.float64)
    u_array = np.asarray(u_values, dtype=np.float64)
    if md_array.shape != u_array.shape:
        raise ValueError("Huber slope 的 MD 与 U shape 不一致")
    if len(md_array) < 2:
        return float("nan")
    if not np.isfinite(md_array).all() or not np.isfinite(u_array).all():
        raise ValueError("Huber slope 输入含 NaN 或 Inf")

    # 第一次使用等权 OLS，后续迭代只改变异常残差对应的权重。
    weights = np.ones(len(md_array), dtype=np.float64)
    fitted_line = fit_weighted_line(md_array, u_array, weights)
    if fitted_line is None:
        return float("nan")
    slope, intercept = fitted_line

    for _ in range(HUBER_MAX_ITERATIONS):
        residual = u_array - (intercept + slope * md_array)
        residual_median = float(np.median(residual))
        absolute_deviation = np.abs(residual - residual_median)
        robust_scale = 1.4826 * float(np.median(absolute_deviation))

        # 完全线性或只剩浮点噪声时已经收敛，不再制造不稳定权重。
        if robust_scale <= 1e-12:
            break

        cutoff = HUBER_DELTA * robust_scale
        centered_residual = np.abs(residual - residual_median)
        new_weights = np.ones(len(md_array), dtype=np.float64)
        outlier_mask = centered_residual > cutoff
        new_weights[outlier_mask] = cutoff / centered_residual[outlier_mask]

        new_fitted_line = fit_weighted_line(md_array, u_array, new_weights)
        if new_fitted_line is None:
            return float("nan")
        new_slope, new_intercept = new_fitted_line

        slope_change = abs(new_slope - slope)
        intercept_change = abs(new_intercept - intercept)
        slope = new_slope
        intercept = new_intercept
        weights = new_weights

        if slope_change <= HUBER_TOLERANCE and intercept_change <= HUBER_TOLERANCE:
            break

    return slope


def fit_visible_u_huber_slope(
    horizontal_df: pd.DataFrame,
    window_ft: float,
) -> float:
    """只在末端指定 MD 窗口的可见前缀上拟合 U=TVT_input+Z 的 Huber 倾角。"""

    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    if visible_df.empty:
        raise ValueError("RF01a 需要至少一个 TVT_input 可见点")

    last_visible_md = float(visible_df["MD"].iloc[-1])
    window_start_md = last_visible_md - float(window_ft)
    window_df = visible_df.loc[visible_df["MD"] >= window_start_md]

    md_values = window_df["MD"].to_numpy(dtype=np.float64)
    u_values = (
        window_df["TVT_input"].to_numpy(dtype=np.float64)
        + window_df["Z"].to_numpy(dtype=np.float64)
    )
    return fit_huber_slope(md_values, u_values)


def build_rf01a_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在 B00 十二列上追加五个可见前缀 Huber U 倾角。"""

    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)
    for window_ft in [50, 100, 200, 500, 1000]:
        slope = fit_visible_u_huber_slope(
            horizontal_df,
            window_ft=float(window_ft),
        )
        result[f"u_huber_slope_{window_ft}"] = slope

    return result

