"""P3-N01b 固定强度邻井残差修正的纯计算函数。"""

from __future__ import annotations

import numpy as np


FROZEN_MULTIPLIER = 0.5
FOLD0_SELECTION_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} 含非有限值")
    return vector


def apply_neighbor_multiplier(
    baseline_tvt: np.ndarray,
    full_neighbor_corrected_tvt: np.ndarray,
    multiplier: float = FROZEN_MULTIPLIER,
) -> np.ndarray:
    """将 N01a 完整邻井修正乘固定强度后加回原 scale8 路径。"""

    baseline = _finite_vector(baseline_tvt, "baseline_tvt")
    full = _finite_vector(full_neighbor_corrected_tvt, "full_neighbor_corrected_tvt")
    if baseline.shape != full.shape:
        raise ValueError("baseline 与完整邻井路径必须同形")
    strength = float(multiplier)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("multiplier 必须是非负有限值")
    return baseline + strength * (full - baseline)


def pooled_multiplier_grid(
    target_chunks: list[np.ndarray],
    baseline_chunks: list[np.ndarray],
    full_neighbor_chunks: list[np.ndarray],
    multipliers: list[float] | tuple[float, ...] = FOLD0_SELECTION_GRID,
) -> list[dict[str, float | int]]:
    """只按总 SSE/总行数计算预设 multiplier 网格的逐行 pooled RMSE。"""

    if not target_chunks or not (
        len(target_chunks) == len(baseline_chunks) == len(full_neighbor_chunks)
    ):
        raise ValueError("三组分块必须非空且数量一致")
    rows = 0
    total_sse = np.zeros(len(multipliers), dtype=np.float64)
    for target_values, baseline_values, full_values in zip(
        target_chunks,
        baseline_chunks,
        full_neighbor_chunks,
        strict=True,
    ):
        target = _finite_vector(target_values, "target")
        baseline = _finite_vector(baseline_values, "baseline")
        full = _finite_vector(full_values, "full_neighbor")
        if target.shape != baseline.shape or target.shape != full.shape:
            raise ValueError("每个分块的 target、baseline 和 full_neighbor 必须同形")
        correction = full - baseline
        for index, multiplier in enumerate(multipliers):
            prediction = baseline + float(multiplier) * correction
            error = target - prediction
            total_sse[index] += float(error @ error)
        rows += len(target)
    return [
        {
            "multiplier": float(multiplier),
            "rows": int(rows),
            "sse": float(sse),
            "rmse": float(np.sqrt(sse / rows)),
        }
        for multiplier, sse in zip(multipliers, total_sse, strict=True)
    ]


def select_grid_multiplier(rows: list[dict[str, float | int]]) -> float:
    """按 RMSE 最低选择；完全并列时固定取较小 multiplier。"""

    if not rows:
        raise ValueError("网格结果不能为空")
    return float(min(rows, key=lambda row: (float(row["rmse"]), float(row["multiplier"])))["multiplier"])

