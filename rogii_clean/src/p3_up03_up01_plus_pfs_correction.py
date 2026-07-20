"""UP01 路径叠加固定 PFS 修正的纯确定性运算。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def build_up03_candidate(
    up01_prediction: np.ndarray,
    p3b00_prediction: np.ndarray,
    pfs_lag1000_absolute: np.ndarray,
    correction_fraction: float,
) -> np.ndarray:
    """按固定公式构造 UP03，不做任何拟合或参数选择。"""

    up01 = np.asarray(up01_prediction, dtype=np.float64)
    p3b00 = np.asarray(p3b00_prediction, dtype=np.float64)
    pfs = np.asarray(pfs_lag1000_absolute, dtype=np.float64)
    if up01.shape != p3b00.shape or up01.shape != pfs.shape:
        raise ValueError("UP01、P3B00 与 PFS 路径形状不一致")
    candidate = up01 + float(correction_fraction) * (pfs - p3b00)
    if not np.isfinite(candidate).all():
        raise ValueError("UP03 候选含 NaN 或无穷值")
    return candidate


def evaluate_confirmation_gate(
    fold_improvements_ft: Iterable[float],
    minimum_mean_improvement_ft: float,
    maximum_any_fold_degradation_ft: float,
) -> dict[str, Any]:
    """按预登记的确认折算术平均改善和最差单折恶化判定。"""

    improvements = np.asarray(list(fold_improvements_ft), dtype=np.float64)
    if improvements.size == 0 or not np.isfinite(improvements).all():
        raise ValueError("确认折改善必须是非空有限数组")
    mean_improvement = float(np.mean(improvements))
    maximum_degradation = float(max(0.0, -float(np.min(improvements))))
    return {
        "arithmetic_mean_fold_improvement_ft": mean_improvement,
        "maximum_fold_degradation_ft": maximum_degradation,
        "minimum_arithmetic_mean_fold_improvement_ft": float(minimum_mean_improvement_ft),
        "maximum_any_fold_degradation_ft": float(maximum_any_fold_degradation_ft),
        "passed": bool(
            mean_improvement >= float(minimum_mean_improvement_ft)
            and maximum_degradation <= float(maximum_any_fold_degradation_ft)
        ),
    }
