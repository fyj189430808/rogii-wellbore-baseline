"""P3-UP04：对已锁定的全局路径和局部路径做固定 1:1 融合。"""

from __future__ import annotations

import numpy as np


def equal_blend_paths(global_path: np.ndarray, local_path: np.ndarray) -> np.ndarray:
    """返回两条同行 TVT 路径的算术平均，不拟合任何参数。"""

    global_values = np.asarray(global_path, dtype=np.float64)
    local_values = np.asarray(local_path, dtype=np.float64)
    if global_values.shape != local_values.shape:
        raise ValueError("global_path and local_path must have the same shape")
    if global_values.ndim != 1:
        raise ValueError("both paths must be one-dimensional")
    if not np.isfinite(global_values).all() or not np.isfinite(local_values).all():
        raise ValueError("both paths must contain only finite values")
    return 0.5 * global_values + 0.5 * local_values
