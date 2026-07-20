"""UP06 投影强度 oracle 诊断的纯数学函数。"""

from __future__ import annotations

import numpy as np


def projection_candidate(
    base_prediction: np.ndarray,
    projected_prediction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """沿“原路径到二次投影路径”的方向移动固定比例 alpha。"""

    base = np.asarray(base_prediction, dtype=np.float64)
    projected = np.asarray(projected_prediction, dtype=np.float64)
    if base.shape != projected.shape:
        raise ValueError("原路径和投影路径形状必须一致")
    if not np.isfinite(base).all() or not np.isfinite(projected).all():
        raise ValueError("原路径和投影路径必须全部为有限值")
    if not np.isfinite(float(alpha)):
        raise ValueError("alpha 必须为有限值")
    return base + float(alpha) * (projected - base)


def alpha_rmse_curve(
    target: np.ndarray,
    base_prediction: np.ndarray,
    projected_prediction: np.ndarray,
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """返回每个 alpha 对应的逐行 micro RMSE。"""

    truth = np.asarray(target, dtype=np.float64)
    base = np.asarray(base_prediction, dtype=np.float64)
    projected = np.asarray(projected_prediction, dtype=np.float64)
    grid = np.asarray(alpha_grid, dtype=np.float64)
    if truth.shape != base.shape or truth.shape != projected.shape:
        raise ValueError("真值、原路径和投影路径形状必须一致")
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha_grid 必须是一维非空数组")
    if not np.isfinite(truth).all() or not np.isfinite(grid).all():
        raise ValueError("真值和 alpha 网格必须全部为有限值")

    direction = projected - base
    sum_squared_error = np.empty(grid.size, dtype=np.float64)
    for alpha_index, alpha in enumerate(grid):
        residual = truth - (base + float(alpha) * direction)
        sum_squared_error[alpha_index] = float(np.dot(residual, residual))
    return np.sqrt(sum_squared_error / float(truth.size))


def best_grid_alpha(
    target: np.ndarray,
    base_prediction: np.ndarray,
    projected_prediction: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[float, float]:
    """在固定网格上返回最小 RMSE 的 alpha；并列时取更小 alpha。"""

    grid = np.asarray(alpha_grid, dtype=np.float64)
    curve = alpha_rmse_curve(target, base_prediction, projected_prediction, grid)
    best_index = int(np.argmin(curve))
    return float(grid[best_index]), float(curve[best_index])


def quadratic_sse_terms(
    target: np.ndarray,
    base_prediction: np.ndarray,
    projected_prediction: np.ndarray,
) -> tuple[float, float, float]:
    """返回 SSE(alpha)=A-2*alpha*B+alpha^2*C 的三个系数。"""

    truth = np.asarray(target, dtype=np.float64)
    base = np.asarray(base_prediction, dtype=np.float64)
    projected = np.asarray(projected_prediction, dtype=np.float64)
    if truth.shape != base.shape or truth.shape != projected.shape:
        raise ValueError("真值、原路径和投影路径形状必须一致")
    base_error = truth - base
    direction = projected - base
    return (
        float(np.dot(base_error, base_error)),
        float(np.dot(base_error, direction)),
        float(np.dot(direction, direction)),
    )


def sse_curve_from_terms(
    terms: tuple[float, float, float],
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """利用二次式系数快速计算整条 alpha 网格的 SSE。"""

    grid = np.asarray(alpha_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("alpha_grid 必须是一维、非空且全部有限")
    base_sse, cross_term, direction_sse = (float(value) for value in terms)
    curve = base_sse - 2.0 * grid * cross_term + grid * grid * direction_sse
    return np.maximum(curve, 0.0)
