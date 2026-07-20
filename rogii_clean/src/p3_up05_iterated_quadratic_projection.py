"""P3-UP05：对 UP03 绝对路径再做一次固定二次稳健投影。"""

from __future__ import annotations

import numpy as np

from src.p3_up01_robust_u_projection import robust_polynomial_projection


def iterated_quadratic_tvt(
    md: np.ndarray,
    z: np.ndarray,
    up03_pred_tvt: np.ndarray,
) -> np.ndarray:
    """使用 UP01 原参数投影 UP03 的 U，并固定以 50% 回混后还原 TVT。"""

    md_values = np.asarray(md, dtype=np.float64)
    z_values = np.asarray(z, dtype=np.float64)
    base_tvt = np.asarray(up03_pred_tvt, dtype=np.float64)
    if md_values.ndim != 1 or z_values.ndim != 1 or base_tvt.ndim != 1:
        raise ValueError("md, z and up03_pred_tvt must be one-dimensional")
    if not (md_values.shape == z_values.shape == base_tvt.shape):
        raise ValueError("md, z and up03_pred_tvt must have the same shape")
    if not (
        np.isfinite(md_values).all()
        and np.isfinite(z_values).all()
        and np.isfinite(base_tvt).all()
    ):
        raise ValueError("inputs must contain only finite values")

    base_u = base_tvt + z_values
    projected_u = robust_polynomial_projection(md_values, base_u, degree=2)
    blended_u = 0.5 * base_u + 0.5 * projected_u
    return blended_u - z_values
