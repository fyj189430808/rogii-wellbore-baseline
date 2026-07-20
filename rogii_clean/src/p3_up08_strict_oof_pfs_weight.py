"""UP08 严格 OOF PFS 权重的无标签数值核心。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_LAGS = ("lag250", "lag500", "lag1000")


def _as_finite_vector(values: object, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 含 NaN 或无穷值")
    return result


def optimal_alpha(
    target: np.ndarray,
    up01: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float]:
    """计算单井连续最小二乘 PFS 权重及其方向能量。

    训练目标遵循 UP08 预登记范围 ``[-0.5, 1.0]``；零能量方向没有可识别的
    权重，保守地回退到固定 UP03 权重 0.25。
    """

    target_values = _as_finite_vector(target, "target")
    up01_values = _as_finite_vector(up01, "up01")
    direction_values = _as_finite_vector(direction, "direction")
    if not (target_values.shape == up01_values.shape == direction_values.shape):
        raise ValueError("target、up01 与 direction 的形状必须一致")

    energy = float(np.dot(direction_values, direction_values))
    if energy <= 0.0:
        return 0.25, 0.0
    raw = float(np.dot(direction_values, target_values - up01_values) / energy)
    return float(np.clip(raw, -0.5, 1.0)), energy


def _slope_per_1000ft(md: np.ndarray, values: np.ndarray) -> float:
    centered_md = md - float(np.mean(md))
    denominator = float(np.dot(centered_md, centered_md))
    if denominator <= 0.0:
        return 0.0
    centered_values = values - float(np.mean(values))
    return float(1000.0 * np.dot(centered_md, centered_values) / denominator)


def _runtime_fraction(runtime: Mapping[str, object], lag: str) -> float:
    smooth = float(runtime[f"{lag}_smooth_rows"])
    fallback = float(runtime[f"{lag}_fallback_rows"])
    total = smooth + fallback
    if not np.isfinite(total) or smooth < 0.0 or fallback < 0.0 or total <= 0.0:
        raise ValueError(f"{lag} 平滑/回退计数必须为非负且总数为正")
    return fallback / total


def build_well_features(
    md: np.ndarray,
    base_pred: np.ndarray,
    pfs_paths: Mapping[str, np.ndarray],
    runtime: Mapping[str, object],
) -> dict[str, float]:
    """构造 UP08 的合法单井特征，不接收或读取 TVT 真值。

    ``pfs_paths`` 的三个绝对 TVT 路径固定为 ``lag250``、``lag500`` 与
    ``lag1000``。``runtime`` 使用 PFS01 的逐井诊断字段，并额外由调用者提供
    当前井可见 GR 的 ``gr_observed_fraction``。
    """

    md_values = _as_finite_vector(md, "md")
    base_values = _as_finite_vector(base_pred, "base_pred")
    if md_values.shape != base_values.shape:
        raise ValueError("md 与 base_pred 的形状必须一致")
    if np.any(np.diff(md_values) < 0.0):
        raise ValueError("md 必须单调不减")
    if not isinstance(pfs_paths, Mapping):
        raise ValueError("pfs_paths 必须为映射")
    for lag in _LAGS:
        if lag not in pfs_paths:
            raise ValueError(f"pfs_paths 缺少必要 lag: {lag}")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime 必须为映射")
    required_runtime_keys = (
        "hidden_rows",
        "gr_observed_fraction",
        "resample_count",
        *(f"{lag}_{kind}_rows" for lag in _LAGS for kind in ("smooth", "fallback")),
    )
    for key in required_runtime_keys:
        if key not in runtime:
            raise ValueError(f"runtime 缺少必要键: {key}")

    paths = {lag: _as_finite_vector(pfs_paths[lag], f"pfs_paths[{lag}]") for lag in _LAGS}
    if any(values.shape != base_values.shape for values in paths.values()):
        raise ValueError("所有 PFS 路径必须与 base_pred 形状一致")

    hidden_rows = float(len(md_values))
    runtime_rows = float(runtime["hidden_rows"])
    if not np.isfinite(runtime_rows) or runtime_rows != hidden_rows:
        raise ValueError("runtime hidden_rows 必须与路径行数一致")
    gr_observed_fraction = float(runtime["gr_observed_fraction"])
    if not np.isfinite(gr_observed_fraction) or not 0.0 <= gr_observed_fraction <= 1.0:
        raise ValueError("gr_observed_fraction 必须位于 [0, 1]")

    direction = paths["lag1000"] - base_values
    pairwise = {
        "lag250_lag500_rms_ft": paths["lag250"] - paths["lag500"],
        "lag250_lag1000_rms_ft": paths["lag250"] - paths["lag1000"],
        "lag500_lag1000_rms_ft": paths["lag500"] - paths["lag1000"],
    }
    lag_spread = np.max(np.stack(tuple(paths.values())), axis=0) - np.min(
        np.stack(tuple(paths.values())), axis=0
    )
    fallback_fractions = {lag: _runtime_fraction(runtime, lag) for lag in _LAGS}
    particle_rows = float(runtime["lag250_smooth_rows"]) + float(
        runtime["lag250_fallback_rows"]
    )
    resample_count = float(runtime["resample_count"])
    if not np.isfinite(resample_count) or resample_count < 0.0:
        raise ValueError("resample_count 必须为非负有限数")

    return {
        "hidden_rows": hidden_rows,
        "hidden_md_span_ft": float(np.max(md_values) - np.min(md_values)),
        "d_rms_ft": float(np.sqrt(np.mean(direction * direction))),
        "d_mean_ft": float(np.mean(direction)),
        "d_abs_mean_ft": float(np.mean(np.abs(direction))),
        "d_std_ft": float(np.std(direction)),
        "d_end_ft": float(direction[-1]),
        "d_slope_ft_per_1000ft": _slope_per_1000ft(md_values, direction),
        **{
            name: float(np.sqrt(np.mean(values * values)))
            for name, values in pairwise.items()
        },
        "three_lag_spread_rms_ft": float(np.sqrt(np.mean(lag_spread * lag_spread))),
        "gr_observed_fraction": gr_observed_fraction,
        "resample_rate": resample_count / particle_rows,
        **{
            f"{lag}_fallback_fraction": fraction
            for lag, fraction in fallback_fractions.items()
        },
    }


def shrink_alpha(
    raw: np.ndarray,
    center: float = 0.25,
    strength: float = 0.5,
) -> np.ndarray:
    """把预测权重向固定 0.25 收缩后截断到 UP08 提交范围。"""

    raw_values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(raw_values).all():
        raise ValueError("raw 含 NaN 或无穷值")
    center_value = float(center)
    strength_value = float(strength)
    if not np.isfinite(center_value) or not np.isfinite(strength_value):
        raise ValueError("center 与 strength 必须有限")
    shrunk = center_value + strength_value * (raw_values - center_value)
    return np.clip(shrunk, -0.25, 0.75)
