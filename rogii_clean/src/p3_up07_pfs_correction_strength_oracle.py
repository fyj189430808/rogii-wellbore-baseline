"""P3-UP07 的纯数学函数：固定方向上的修正强度 oracle 诊断。"""

from __future__ import annotations

import numpy as np


def make_alpha_grid(start: float = -0.5, stop: float = 1.0, step: float = 0.05) -> np.ndarray:
    """生成包含两端点的稳定 alpha 网格，避免浮点累积漏掉 1.0。"""

    count = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(count, dtype=np.float64), 10)


def quadratic_sse_curve(
    sum_error_squared: float,
    sum_error_times_direction: float,
    sum_direction_squared: float,
    alphas: np.ndarray,
) -> np.ndarray:
    """由充分统计量计算 SSE(alpha)，其中预测为 base + alpha * direction。"""

    alpha_values = np.asarray(alphas, dtype=np.float64)
    sse = (
        float(sum_error_squared)
        - 2.0 * alpha_values * float(sum_error_times_direction)
        + alpha_values * alpha_values * float(sum_direction_squared)
    )
    # 理论上 SSE 非负；只截断浮点舍入造成的极小负数。
    return np.maximum(sse, 0.0)


def rmse_curve_from_sufficient_statistics(
    row_count: int,
    sum_error_squared: float,
    sum_error_times_direction: float,
    sum_direction_squared: float,
    alphas: np.ndarray,
) -> np.ndarray:
    """把 SSE(alpha) 转成 RMSE(alpha)。"""

    if int(row_count) <= 0:
        raise ValueError("row_count 必须为正数")
    sse = quadratic_sse_curve(
        sum_error_squared,
        sum_error_times_direction,
        sum_direction_squared,
        alphas,
    )
    return np.sqrt(sse / float(row_count))


def grid_oracle(
    row_count: int,
    sum_error_squared: float,
    sum_error_times_direction: float,
    sum_direction_squared: float,
    alphas: np.ndarray,
) -> tuple[float, float, int]:
    """返回网格内 RMSE 最低的 alpha、RMSE 和网格下标。"""

    rmse_values = rmse_curve_from_sufficient_statistics(
        row_count,
        sum_error_squared,
        sum_error_times_direction,
        sum_direction_squared,
        alphas,
    )
    best_index = int(np.argmin(rmse_values))
    return float(alphas[best_index]), float(rmse_values[best_index]), best_index


def linear_slope_per_1000ft(md: np.ndarray, values: np.ndarray) -> float:
    """计算 values 对 MD 的普通最小二乘斜率，并换算成每 1000 ft 的变化。"""

    md_values = np.asarray(md, dtype=np.float64)
    y_values = np.asarray(values, dtype=np.float64)
    if md_values.shape != y_values.shape or md_values.ndim != 1:
        raise ValueError("md 与 values 必须是一维同形数组")
    if len(md_values) == 0:
        raise ValueError("不能对空数组计算斜率")
    centered_md = md_values - float(np.mean(md_values))
    denominator = float(np.dot(centered_md, centered_md))
    if denominator <= 0.0:
        return 0.0
    centered_values = y_values - float(np.mean(y_values))
    slope_per_ft = float(np.dot(centered_md, centered_values) / denominator)
    return 1000.0 * slope_per_ft

