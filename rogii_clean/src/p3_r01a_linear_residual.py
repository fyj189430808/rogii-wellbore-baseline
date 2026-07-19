"""P3-R01a 的小型数学核心：每井线性残差、岭回归与井级重采样。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def normalized_progress(md: np.ndarray) -> np.ndarray:
    """把一口井隐藏段 MD 映射到 0～1；只有一个位置时全部返回 0。"""

    md_values = np.asarray(md, dtype=np.float64)
    if md_values.ndim != 1 or len(md_values) == 0:
        raise ValueError("md 必须是一维非空数组")
    if not np.isfinite(md_values).all():
        raise ValueError("md 含 NaN 或 Inf")
    md_span = float(np.max(md_values) - np.min(md_values))
    if md_span <= 0.0:
        return np.zeros(len(md_values), dtype=np.float64)
    return (md_values - float(np.min(md_values))) / md_span


def fit_linear_residual(
    md: np.ndarray,
    base_tvt: np.ndarray,
    target_tvt: np.ndarray,
) -> np.ndarray:
    """用隐藏段真值仅作训练目标，拟合不锚定的 r(t)=a0+a1*t。"""

    md_values = np.asarray(md, dtype=np.float64)
    base_values = np.asarray(base_tvt, dtype=np.float64)
    target_values = np.asarray(target_tvt, dtype=np.float64)
    if not (md_values.shape == base_values.shape == target_values.shape):
        raise ValueError("md、base_tvt、target_tvt shape 不一致")
    if not np.isfinite(np.column_stack([md_values, base_values, target_values])).all():
        raise ValueError("线性残差拟合输入含 NaN 或 Inf")

    progress = normalized_progress(md_values)
    design = np.column_stack([np.ones(len(progress)), progress])
    residual = target_values - base_values
    coefficients, _, _, _ = np.linalg.lstsq(design, residual, rcond=None)
    return coefficients.astype(np.float64, copy=False)


def apply_linear_residual(
    md: np.ndarray,
    base_tvt: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    """把预测的两个井级系数加到逐行 P2-P02 路径上。"""

    base_values = np.asarray(base_tvt, dtype=np.float64)
    progress = normalized_progress(md)
    if base_values.shape != progress.shape:
        raise ValueError("md 与 base_tvt shape 不一致")
    corrected = base_values + float(intercept) + float(slope) * progress
    if not np.isfinite(corrected).all():
        raise ValueError("修正路径含 NaN 或 Inf")
    return corrected


def fit_predict_ridge_coefficients(
    train_features: pd.DataFrame,
    train_targets: np.ndarray,
    validation_features: pd.DataFrame,
    alpha: float,
) -> tuple[np.ndarray, Pipeline]:
    """只在外层训练井上拟合 StandardScaler + 多输出 Ridge。"""

    if list(train_features.columns) != list(validation_features.columns):
        raise ValueError("训练与验证井级特征列不一致")
    x_train = train_features.to_numpy(dtype=np.float64)
    x_validation = validation_features.to_numpy(dtype=np.float64)
    y_train = np.asarray(train_targets, dtype=np.float64)
    if y_train.shape != (len(train_features), 2):
        raise ValueError("系数目标必须为 [井数, 2]")
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise ValueError("岭回归特征含 NaN 或 Inf")
    if not np.isfinite(y_train).all():
        raise ValueError("岭回归目标含 NaN 或 Inf")

    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_validation), dtype=np.float64)
    return prediction, model


def paired_well_bootstrap_delta(
    per_well: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """以整口井为抽样单位，估计 candidate RMSE - base RMSE 的区间。"""

    required = {"well_id", "rows", "base_sse", "candidate_sse"}
    missing = required.difference(per_well.columns)
    if missing:
        raise ValueError(f"逐井表缺列：{sorted(missing)}")
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    base_sse = per_well["base_sse"].to_numpy(dtype=np.float64)
    candidate_sse = per_well["candidate_sse"].to_numpy(dtype=np.float64)
    if len(rows) == 0 or np.any(rows <= 0):
        raise ValueError("逐井行数必须为正")

    generator = np.random.default_rng(int(seed))
    deltas = np.empty(int(n_resamples), dtype=np.float64)
    for sample_id in range(int(n_resamples)):
        indices = generator.integers(0, len(rows), size=len(rows))
        sampled_rows = float(np.sum(rows[indices]))
        base_rmse = np.sqrt(float(np.sum(base_sse[indices])) / sampled_rows)
        candidate_rmse = np.sqrt(
            float(np.sum(candidate_sse[indices])) / sampled_rows
        )
        deltas[sample_id] = candidate_rmse - base_rmse

    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_better": float(np.mean(deltas < 0.0)),
    }
