"""UP01：只根据预测路径本身，对结构坐标 U 做稳健多项式投影。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


DEFAULT_HUBER_K = 1.345
DEFAULT_IRLS_ITERATIONS = 12
DEFAULT_SCALE_FLOOR = 1.0e-6


def _normalized_md(md: np.ndarray) -> np.ndarray | None:
    """把单井隐藏段 MD 线性映射到 [-1, 1]；无跨度时返回 None。"""

    md_values = np.asarray(md, dtype=np.float64)
    md_min = float(np.min(md_values))
    md_max = float(np.max(md_values))
    md_span = md_max - md_min
    if not np.isfinite(md_span) or md_span <= 0.0:
        return None
    return 2.0 * (md_values - md_min) / md_span - 1.0


def robust_polynomial_projection(
    md: np.ndarray,
    predicted_u: np.ndarray,
    degree: int,
    *,
    huber_k: float = DEFAULT_HUBER_K,
    irls_iterations: int = DEFAULT_IRLS_ITERATIONS,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
) -> np.ndarray:
    """用固定 Huber-IRLS 将一条预测 U 路径投影为二次或三次曲线。

    这里的拟合目标是 ``predicted_u`` 本身，不接收也不读取真实 TVT。
    MD 先归一化到 [-1, 1]，避免高次项因为 MD 数值很大而病态。
    """

    md_values = np.asarray(md, dtype=np.float64)
    u_values = np.asarray(predicted_u, dtype=np.float64)
    if md_values.ndim != 1 or u_values.ndim != 1 or len(md_values) != len(u_values):
        raise ValueError("md 和 predicted_u 必须是同长度的一维数组")
    if degree not in (2, 3):
        raise ValueError("UP01 只预登记二次和三次投影")
    if len(u_values) < degree + 1:
        return u_values.copy()
    if not np.isfinite(md_values).all() or not np.isfinite(u_values).all():
        raise ValueError("md 和 predicted_u 不允许包含 NaN 或无穷值")
    if huber_k <= 0.0 or irls_iterations <= 0 or scale_floor <= 0.0:
        raise ValueError("Huber 参数必须为正数")

    normalized_md = _normalized_md(md_values)
    if normalized_md is None:
        return u_values.copy()
    design = np.vander(normalized_md, N=degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(design, u_values, rcond=None)[0]

    for _ in range(irls_iterations):
        fitted = design @ coefficients
        residual = u_values - fitted
        residual_center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_center)))
        robust_scale = max(1.4826 * mad, scale_floor)
        cutoff = huber_k * robust_scale
        absolute_residual = np.abs(residual - residual_center)
        weights = np.ones_like(absolute_residual)
        outside = absolute_residual > cutoff
        weights[outside] = cutoff / absolute_residual[outside]

        square_root_weight = np.sqrt(weights)
        weighted_design = design * square_root_weight[:, None]
        weighted_u = u_values * square_root_weight
        new_coefficients = np.linalg.lstsq(weighted_design, weighted_u, rcond=None)[0]
        if np.max(np.abs(new_coefficients - coefficients)) <= 1.0e-10:
            coefficients = new_coefficients
            break
        coefficients = new_coefficients

    return design @ coefficients


def build_projection_candidates(
    md: np.ndarray,
    predicted_u: np.ndarray,
    *,
    degrees: Iterable[int] = (2, 3),
    blend_fractions: Iterable[float] = (0.25, 0.50, 0.75),
) -> dict[str, np.ndarray]:
    """生成全部预登记候选；不根据真值挑次数或融合比例。"""

    base_u = np.asarray(predicted_u, dtype=np.float64)
    result: dict[str, np.ndarray] = {}
    for degree in degrees:
        projected_u = robust_polynomial_projection(md, base_u, int(degree))
        for blend_fraction in blend_fractions:
            blend = float(blend_fraction)
            if not 0.0 < blend < 1.0:
                raise ValueError("融合比例必须严格位于 0 和 1 之间")
            candidate_name = f"degree{int(degree)}_blend{int(round(100.0 * blend)):02d}"
            result[candidate_name] = base_u + blend * (projected_u - base_u)
    return result
