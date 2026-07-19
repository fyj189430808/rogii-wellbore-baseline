"""P3-PF01a：只用真实可见 GR 估计观测噪声，其余 PF 完全复用 P2-P01。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.p2_p01_multiseed_pf import (
    FEATURE_COLUMNS,
    _kernel_arguments,
    _likelihood_weighted_path,
    particle_filter_all_seeds_numba,
    prepare_particle_filter_inputs,
)


def observed_only_gr_sigma(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> float:
    """用可见且原始 GR 有限的行计算 residual std，并沿用旧 PF 的 10～60 clip。"""

    visible_mask = horizontal_well["TVT_input"].notna().to_numpy()
    horizontal_gr = horizontal_well["GR"].to_numpy(dtype=np.float64, copy=True)
    observed_mask = visible_mask & np.isfinite(horizontal_gr)
    if not observed_mask.any():
        raise ValueError("可见前缀没有有限 GR，无法计算 observed-only sigma")

    sorted_typewell = typewell.sort_values("TVT", kind="mergesort")
    typewell_tvt = sorted_typewell["TVT"].to_numpy(dtype=np.float64, copy=True)
    typewell_gr_series = sorted_typewell["GR"].astype(np.float64)
    typewell_gr_mean = float(typewell_gr_series.mean(skipna=True))
    if not np.isfinite(typewell_gr_mean):
        raise ValueError("Typewell GR 全部缺失")
    typewell_gr = typewell_gr_series.fillna(typewell_gr_mean).to_numpy(dtype=np.float64)
    if not np.isfinite(typewell_tvt).all() or not np.isfinite(typewell_gr).all():
        raise ValueError("Typewell TVT/GR 含非有限值")

    visible_tvt = horizontal_well.loc[observed_mask, "TVT_input"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    observed_gr = horizontal_gr[observed_mask]
    typewell_gr_at_visible_tvt = np.interp(visible_tvt, typewell_tvt, typewell_gr)
    residual_sigma = float(np.std(observed_gr - typewell_gr_at_visible_tvt))
    sigma = float(
        np.clip(
            residual_sigma,
            float(parameters["gr_sigma_min_api"]),
            float(parameters["gr_sigma_max_api"]),
        )
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("observed-only GR sigma 非法")
    return sigma


def build_observed_only_sigma_pf_features(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """生成与冻结 P2-P01 完全同 schema 的 mean/scale3/5/8/12 PF 路径。"""

    prepared = prepare_particle_filter_inputs(horizontal_well, typewell, parameters)
    legacy_sigma = float(prepared["gr_sigma"])
    new_sigma = observed_only_gr_sigma(horizontal_well, typewell, parameters)
    prepared["gr_sigma"] = new_sigma

    seed_predictions, seed_log_likelihoods = particle_filter_all_seeds_numba(
        **_kernel_arguments(prepared)
    )
    mean_tvt = seed_predictions.mean(axis=0).astype(np.float32)
    seed0_tvt = seed_predictions[0].astype(np.float32)
    seed_std = seed_predictions.std(axis=0).astype(np.float32)
    last_visible_tvt = np.float32(prepared["last_visible_tvt"])

    scale_paths: dict[float, np.ndarray] = {}
    for scale in parameters["likelihood_scales"]:
        numeric_scale = float(scale)
        scale_paths[numeric_scale] = _likelihood_weighted_path(
            seed_predictions,
            seed_log_likelihoods,
            numeric_scale,
        ).astype(np.float32)
    required_scales = (3.0, 5.0, 8.0, 12.0)
    missing = [scale for scale in required_scales if scale not in scale_paths]
    if missing:
        raise ValueError(f"likelihood_scales 缺少冻结值: {missing}")

    hidden_rows = len(prepared["row_index"])
    features = pd.DataFrame(
        {
            "row_index": prepared["row_index"],
            "last_visible_tvt": np.full(hidden_rows, last_visible_tvt, dtype=np.float32),
            "pf128_mean_tvt": mean_tvt,
            "pf128_mean_delta": mean_tvt - last_visible_tvt,
            "pf128_seed0_delta": seed0_tvt - last_visible_tvt,
            "pf128_scale_3_delta": scale_paths[3.0] - last_visible_tvt,
            "pf128_scale_5_delta": scale_paths[5.0] - last_visible_tvt,
            "pf128_scale_8_delta": scale_paths[8.0] - last_visible_tvt,
            "pf128_scale_12_delta": scale_paths[12.0] - last_visible_tvt,
            "pf128_seed_std": seed_std,
        },
        columns=FEATURE_COLUMNS,
    )
    if not np.isfinite(features.drop(columns="row_index").to_numpy(dtype=np.float64)).all():
        raise RuntimeError("P3-PF01a 生成了非有限路径")

    quality = {
        "legacy_gr_sigma": legacy_sigma,
        "observed_only_gr_sigma": new_sigma,
        "visible_observed_gr_rows": int(
            (
                horizontal_well["TVT_input"].notna().to_numpy()
                & np.isfinite(horizontal_well["GR"].to_numpy(dtype=np.float64))
            ).sum()
        ),
        "pf_best_ll_per_row": float(np.max(seed_log_likelihoods)) / hidden_rows,
        "pf_ll_spread": float(np.std(seed_log_likelihoods)),
    }
    return features, quality

