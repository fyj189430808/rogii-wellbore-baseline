"""P3-UP02：按实际 MD 距离对预测结构坐标 U 做固定局部高斯平滑。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """将输入转成一维 float64，并拒绝缺失或无穷值。"""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def smooth_u_on_md_grid(
    md: np.ndarray,
    predicted_u: np.ndarray,
    smoothing_window_ft: float,
    grid_step_ft: float = 1.0,
    sigma_fraction_of_window: float = 0.25,
) -> np.ndarray:
    """在近似 1 ft 的均匀 MD 网格上反射边界高斯平滑，再插回原采样点。

    ``smoothing_window_ft`` 是预登记的等效窗口；实际高斯标准差固定为
    ``smoothing_window_ft * sigma_fraction_of_window``。不规则 MD 先线性插值到
    均匀网格，因此窗口始终以 ft 为单位，而不是以行数为单位。
    """

    md_values = _as_finite_vector(md, "md")
    u_values = _as_finite_vector(predicted_u, "predicted_u")
    if md_values.size != u_values.size:
        raise ValueError("md and predicted_u must have the same length")
    if md_values.size > 1 and np.any(np.diff(md_values) <= 0.0):
        raise ValueError("md must be strictly increasing within each well")
    if not np.isfinite(smoothing_window_ft) or smoothing_window_ft <= 0.0:
        raise ValueError("smoothing_window_ft must be positive")
    if not np.isfinite(grid_step_ft) or grid_step_ft <= 0.0:
        raise ValueError("grid_step_ft must be positive")
    if not np.isfinite(sigma_fraction_of_window) or sigma_fraction_of_window <= 0.0:
        raise ValueError("sigma_fraction_of_window must be positive")
    if md_values.size == 1:
        return u_values.copy()

    md_span_ft = float(md_values[-1] - md_values[0])
    interval_count = max(1, int(np.ceil(md_span_ft / grid_step_ft)))
    uniform_md = np.linspace(
        md_values[0],
        md_values[-1],
        interval_count + 1,
        dtype=np.float64,
    )
    actual_grid_step_ft = md_span_ft / float(interval_count)
    uniform_u = np.interp(uniform_md, md_values, u_values)

    sigma_ft = float(smoothing_window_ft) * float(sigma_fraction_of_window)
    sigma_grid_points = sigma_ft / actual_grid_step_ft
    smoothed_uniform_u = gaussian_filter1d(
        uniform_u,
        sigma=sigma_grid_points,
        mode="reflect",
        truncate=4.0,
    )
    smoothed_at_original_md = np.interp(md_values, uniform_md, smoothed_uniform_u)
    return np.asarray(smoothed_at_original_md, dtype=np.float64)


def build_local_smoothing_candidates(
    md: np.ndarray,
    predicted_u: np.ndarray,
    smoothing_windows_ft: Iterable[float],
    blend_fractions: Iterable[float],
    grid_step_ft: float = 1.0,
    sigma_fraction_of_window: float = 0.25,
) -> OrderedDict[str, np.ndarray]:
    """生成按“窗口优先、融合比例次之”排序的全部固定候选 U 路径。"""

    md_values = _as_finite_vector(md, "md")
    u_values = _as_finite_vector(predicted_u, "predicted_u")
    windows = tuple(float(value) for value in smoothing_windows_ft)
    blends = tuple(float(value) for value in blend_fractions)
    if not windows:
        raise ValueError("smoothing_windows_ft must not be empty")
    if not blends:
        raise ValueError("blend_fractions must not be empty")
    for blend in blends:
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError("blend fractions must be between 0 and 1")

    candidates: OrderedDict[str, np.ndarray] = OrderedDict()
    for window_ft in windows:
        smoothed_u = smooth_u_on_md_grid(
            md_values,
            u_values,
            smoothing_window_ft=window_ft,
            grid_step_ft=grid_step_ft,
            sigma_fraction_of_window=sigma_fraction_of_window,
        )
        window_label = int(round(window_ft))
        for blend in blends:
            blend_label = int(round(100.0 * blend))
            candidate_name = f"smooth{window_label}_blend{blend_label:02d}"
            candidate_u = (1.0 - blend) * u_values + blend * smoothed_u
            candidates[candidate_name] = np.asarray(candidate_u, dtype=np.float64)
    return candidates
