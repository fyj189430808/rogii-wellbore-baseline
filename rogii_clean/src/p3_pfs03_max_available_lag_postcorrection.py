"""按逐行剩余 MD 选择仍拥有完整未来证据的最长固定滞后路径。"""

from __future__ import annotations

import numpy as np


def choose_dynamic_pfs_absolute(
    remaining_md: np.ndarray,
    pf_scale8_absolute: np.ndarray,
    pfs_lag250_absolute: np.ndarray,
    pfs_lag500_absolute: np.ndarray,
    pfs_lag1000_absolute: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回动态 PFS 绝对 TVT 路径和每行实际选用的滞后距离。"""

    remaining = np.asarray(remaining_md, dtype=np.float64)
    pf_scale8 = np.asarray(pf_scale8_absolute, dtype=np.float64)
    lag250 = np.asarray(pfs_lag250_absolute, dtype=np.float64)
    lag500 = np.asarray(pfs_lag500_absolute, dtype=np.float64)
    lag1000 = np.asarray(pfs_lag1000_absolute, dtype=np.float64)
    if not (
        remaining.shape == pf_scale8.shape == lag250.shape == lag500.shape == lag1000.shape
    ):
        raise ValueError("动态 PFS 的所有逐行数组形状必须一致")
    if not all(np.isfinite(value).all() for value in (remaining, pf_scale8, lag250, lag500, lag1000)):
        raise ValueError("动态 PFS 输入含 NaN 或无穷值")

    selected = pf_scale8.copy()
    selected_lag = np.zeros(remaining.shape, dtype=np.int16)

    use_250 = remaining >= 250.0
    selected[use_250] = lag250[use_250]
    selected_lag[use_250] = 250

    use_500 = remaining >= 500.0
    selected[use_500] = lag500[use_500]
    selected_lag[use_500] = 500

    use_1000 = remaining >= 1000.0
    selected[use_1000] = lag1000[use_1000]
    selected_lag[use_1000] = 1000
    return selected, selected_lag


def build_fixed_candidate(
    up01_prediction: np.ndarray,
    p3b00_prediction: np.ndarray,
    dynamic_pfs_absolute: np.ndarray,
    correction_fraction: float = 0.25,
) -> np.ndarray:
    """按锁定公式 UP01 + 0.25×(动态 PFS−P3B00) 构造候选。"""

    up01 = np.asarray(up01_prediction, dtype=np.float64)
    p3b00 = np.asarray(p3b00_prediction, dtype=np.float64)
    dynamic = np.asarray(dynamic_pfs_absolute, dtype=np.float64)
    if up01.shape != p3b00.shape or up01.shape != dynamic.shape:
        raise ValueError("UP01、P3B00 和动态 PFS 路径形状必须一致")
    candidate = up01 + float(correction_fraction) * (dynamic - p3b00)
    if not np.isfinite(candidate).all():
        raise ValueError("PFS03 候选含 NaN 或无穷值")
    return candidate

