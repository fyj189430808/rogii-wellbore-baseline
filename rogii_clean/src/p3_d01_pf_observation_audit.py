"""P3-D01：PF 种子权重和 GR 观测证据的无标签审计函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.p2_p01_multiseed_pf import (
    _kernel_arguments,
    _likelihood_weighted_path,
    particle_filter_all_seeds_numba,
    prepare_particle_filter_inputs,
)


def _validated_likelihoods(log_likelihoods: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("log_likelihoods 必须是一维数组")
    if values.size == 0:
        raise ValueError("log_likelihoods 不能为空")
    if not np.isfinite(values).all():
        raise ValueError("log_likelihoods 必须全部有限")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须为有限正数")
    return values


def seed_weights(log_likelihoods: np.ndarray, temperature: float) -> np.ndarray:
    """以温度缩放的稳定 softmax 将种子对数似然转为权重。"""
    values = _validated_likelihoods(log_likelihoods, temperature)
    scaled = (values - np.max(values)) / float(temperature)
    unnormalized = np.exp(scaled)
    return unnormalized / np.sum(unnormalized)


def summarize_weight_scale(log_likelihoods: np.ndarray, temperature: float) -> dict[str, float]:
    """返回 PF 种子权重的 ESS、最大权重和归一化熵。"""
    weights = seed_weights(log_likelihoods, temperature)
    entropy = -float(np.sum(np.where(weights > 0.0, weights * np.log(weights), 0.0)))
    normalized_entropy = 1.0 if len(weights) == 1 else entropy / float(np.log(len(weights)))
    return {
        "effective_sample_size": float(1.0 / np.sum(np.square(weights))),
        "normalized_effective_sample_size": float(
            (1.0 / np.sum(np.square(weights))) / len(weights)
        ),
        "max_weight": float(np.max(weights)),
        "entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "perplexity": float(np.exp(entropy)),
        "top4_mass": float(np.sort(weights)[-4:].sum()),
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}")


def _longest_missing_run(md: np.ndarray, observed: np.ndarray) -> tuple[int, float]:
    longest_rows = 0
    longest_ft = 0.0
    position = 0
    sampling_step = 1.0 if len(md) < 2 else float(np.median(np.diff(md)))
    while position < len(observed):
        if observed[position]:
            position += 1
            continue
        start = position
        while position < len(observed) and not observed[position]:
            position += 1
        rows = position - start
        length = float(md[position - 1] - md[start]) + sampling_step
        longest_rows = max(longest_rows, rows)
        longest_ft = max(longest_ft, length)
    return longest_rows, longest_ft


def hidden_gr_evidence_statistics(horizontal_well: pd.DataFrame) -> dict[str, float | int]:
    """按旧 PF 的整井双向插值语义，审计自然隐藏段 GR 证据来源。"""
    _require_columns(horizontal_well, ["MD", "GR", "TVT_input"], "水平井")
    hidden = horizontal_well["TVT_input"].isna().to_numpy()
    if not hidden.any():
        raise ValueError("水平井没有自然隐藏行")
    md = horizontal_well.loc[hidden, "MD"].to_numpy(dtype=np.float64)
    if not np.isfinite(md).all() or (len(md) > 1 and np.any(np.diff(md) <= 0.0)):
        raise ValueError("隐藏段 MD 必须有限且严格递增")
    observed_all = np.isfinite(horizontal_well["GR"].to_numpy(dtype=np.float64))
    observed = observed_all[hidden]
    has_left = np.cumsum(observed_all) > 0
    has_right = np.cumsum(observed_all[::-1])[::-1] > 0
    internal = hidden & ~observed_all & has_left & has_right
    edge = hidden & ~observed_all & ~(has_left & has_right) & (has_left | has_right)
    fallback = hidden & ~observed_all & ~(has_left | has_right)
    hidden_rows = int(hidden.sum())
    observed_rows = int(observed.sum())
    internal_rows = int(internal.sum())
    edge_rows = int(edge.sum())
    fallback_rows = int(fallback.sum())
    longest_rows, longest_ft = _longest_missing_run(md, observed)
    return {
        "hidden_rows": hidden_rows,
        "observed_gr_rows": observed_rows,
        "interpolated_gr_rows": internal_rows + edge_rows,
        "internal_interpolated_gr_rows": internal_rows,
        "edge_filled_gr_rows": edge_rows,
        "typewell_mean_fallback_rows": fallback_rows,
        "observed_gr_fraction": float(observed_rows / hidden_rows),
        "longest_gr_gap_rows": longest_rows,
        "longest_gr_gap_md_ft": longest_ft,
        "effective_observation_count": observed_rows,
        "effective_observation_count_q025": float(observed_rows + 0.25 * (internal_rows + edge_rows)),
        "current_likelihood_update_count": hidden_rows,
        "current_evidence_inflation": float(hidden_rows / max(observed_rows, 1)),
    }


def visible_prefix_gr_diagnostics(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> dict[str, float | int | bool]:
    """在可见且原始 GR 有效的行上拟合 Typewell GR affine 对照。"""
    _require_columns(horizontal_well, ["MD", "Z", "GR", "TVT_input"], "水平井")
    _require_columns(typewell, ["TVT", "GR"], "Typewell")
    reference = typewell.sort_values("TVT", kind="mergesort")
    reference_tvt = reference["TVT"].to_numpy(dtype=np.float64)
    reference_gr = reference["GR"].astype(np.float64)
    mean_gr = float(reference_gr.mean(skipna=True))
    if len(reference_tvt) < 2 or not np.isfinite(reference_tvt).all() or not np.isfinite(mean_gr):
        raise ValueError("Typewell TVT/GR 不足以诊断")
    cleaned_reference_gr = reference_gr.fillna(mean_gr).to_numpy(dtype=np.float64)
    if np.any(np.diff(reference_tvt) <= 0.0):
        raise ValueError("Typewell TVT 必须严格递增")
    visible = horizontal_well["TVT_input"].notna().to_numpy()
    visible_tvt = horizontal_well.loc[visible, "TVT_input"].to_numpy(dtype=np.float64)
    visible_gr_raw = horizontal_well.loc[visible, "GR"].to_numpy(dtype=np.float64)
    if np.isinf(visible_gr_raw).any():
        raise ValueError("可见 GR 含无穷值")
    visible_gr_legacy = pd.Series(visible_gr_raw).fillna(0.0).to_numpy(dtype=np.float64)
    if len(visible_tvt) == 0 or not np.isfinite(visible_tvt).all():
        raise ValueError("没有有限的可见 TVT_input")
    reference_at_visible = np.interp(visible_tvt, reference_tvt, cleaned_reference_gr)
    sigma_min = float(parameters["gr_sigma_min_api"])
    sigma_max = float(parameters["gr_sigma_max_api"])
    gr_sigma = float(np.clip(np.nanstd(visible_gr_legacy - reference_at_visible), sigma_min, sigma_max))

    original_gr = visible_gr_raw
    observed = np.isfinite(original_gr)
    observed_count = int(observed.sum())
    fallback = observed_count < 2 or np.ptp(reference_at_visible[observed]) == 0.0
    if fallback:
        slope, intercept = 1.0, 0.0
    else:
        slope, intercept = np.polyfit(reference_at_visible[observed], original_gr[observed], deg=1)
        slope, intercept = float(slope), float(intercept)
    if observed_count == 0:
        observed_only_sigma = gr_sigma
    else:
        observed_residual = original_gr[observed] - reference_at_visible[observed]
        observed_only_sigma = float(np.clip(np.nanstd(observed_residual), sigma_min, sigma_max))
    return {
        "visible_rows": int(visible.sum()),
        "visible_observed_gr_rows": observed_count,
        "affine_gr_slope": slope,
        "affine_gr_intercept": intercept,
        "affine_fallback_used": bool(fallback),
        "gr_sigma": gr_sigma,
        "observed_only_gr_sigma": observed_only_sigma,
    }


def run_pf_likelihood_audit(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Directly replay the frozen PF for one well without reading hidden labels."""
    prepared = prepare_particle_filter_inputs(horizontal_well, typewell, parameters)
    seed_predictions, seed_log_likelihoods = particle_filter_all_seeds_numba(
        **_kernel_arguments(prepared)
    )
    seed_log_likelihoods = np.asarray(seed_log_likelihoods, dtype=np.float64)
    if seed_log_likelihoods.shape != (prepared["number_of_seeds"],):
        raise RuntimeError("PF seed log likelihood shape is invalid")
    if not np.isfinite(seed_log_likelihoods).all():
        raise RuntimeError("PF seed log likelihood contains NaN or infinity")

    last_visible_tvt = np.float32(prepared["last_visible_tvt"])
    mean_tvt = seed_predictions.mean(axis=0).astype(np.float32)
    path_features: dict[str, np.ndarray] = {
        "row_index": prepared["row_index"].copy(),
        "last_visible_tvt": np.full(
            len(prepared["row_index"]), last_visible_tvt, dtype=np.float32
        ),
        "pf128_mean_tvt": mean_tvt,
        "pf128_mean_delta": mean_tvt - last_visible_tvt,
    }
    scale_summaries: dict[float, dict[str, float]] = {}
    for scale in (3.0, 5.0, 8.0, 12.0):
        scale_tvt = _likelihood_weighted_path(
            seed_predictions, seed_log_likelihoods, scale
        ).astype(np.float32)
        path_features[f"pf128_scale_{int(scale)}_delta"] = (
            scale_tvt - last_visible_tvt
        )
        scale_summaries[scale] = summarize_weight_scale(seed_log_likelihoods, scale)

    report: dict[str, Any] = {
        **hidden_gr_evidence_statistics(horizontal_well),
        **visible_prefix_gr_diagnostics(horizontal_well, typewell, parameters),
        "seed_ll_min": float(np.min(seed_log_likelihoods)),
        "seed_ll_max": float(np.max(seed_log_likelihoods)),
        "seed_ll_mean": float(np.mean(seed_log_likelihoods)),
        "seed_ll_median": float(np.median(seed_log_likelihoods)),
        "seed_ll_std": float(np.std(seed_log_likelihoods)),
        "seed_ll_range": float(np.ptp(seed_log_likelihoods)),
        "seed_ll_iqr": float(np.subtract(*np.percentile(seed_log_likelihoods, [75, 25]))),
        "seed_ll_mean_per_hidden_row": float(np.mean(seed_log_likelihoods) / len(prepared["row_index"])),
        "ll_per_observed_row": float(
            np.mean(seed_log_likelihoods)
            / max(int(hidden_gr_evidence_statistics(horizontal_well)["observed_gr_rows"]), 1)
        ),
        "best_seed_id": int(np.argmax(seed_log_likelihoods)),
        "pf_best_ll_per_row": float(np.max(seed_log_likelihoods) / len(prepared["row_index"])),
        "pf_ll_spread": float(np.std(seed_log_likelihoods)),
    }
    for scale, summary in scale_summaries.items():
        prefix = f"scale_{int(scale)}"
        report.update({f"{prefix}_{key}": value for key, value in summary.items()})
    return seed_log_likelihoods, report, path_features


def write_likelihood_cache(
    path: Path,
    seed_log_likelihoods: np.ndarray,
    fingerprint: str,
) -> None:
    """Atomically store only finite per-seed likelihoods and their fingerprint."""
    values = np.asarray(seed_log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("seed_log_likelihoods must be a non-empty finite 1D array")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("fingerprint must be a non-empty string")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                seed_log_likelihoods=values,
                cache_fingerprint=np.asarray(fingerprint),
            )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def read_likelihood_cache(
    path: Path,
    expected_fingerprint: str,
    expected_number_of_seeds: int,
) -> np.ndarray:
    """Load a cache only when its schema, fingerprint and finite values match."""
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("expected_fingerprint must be a non-empty string")
    if not isinstance(expected_number_of_seeds, int) or expected_number_of_seeds <= 0:
        raise ValueError("expected_number_of_seeds must be a positive integer")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"seed_log_likelihoods", "cache_fingerprint"}:
                raise ValueError("likelihood cache has an invalid schema")
            values = np.asarray(archive["seed_log_likelihoods"], dtype=np.float64)
            fingerprint_value = str(np.asarray(archive["cache_fingerprint"]).item())
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("likelihood cache cannot be read") from error
    if fingerprint_value != expected_fingerprint:
        raise ValueError("likelihood cache fingerprint mismatch")
    if values.ndim != 1 or values.shape[0] != expected_number_of_seeds:
        raise ValueError("likelihood cache seed count mismatch")
    if not np.isfinite(values).all():
        raise ValueError("likelihood cache contains NaN or infinity")
    return values
