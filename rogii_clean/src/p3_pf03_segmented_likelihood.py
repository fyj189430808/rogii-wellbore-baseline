"""P3-PF03：128 条冻结 PF seed 路径的沿 MD 分段似然软加权。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from src.p2_p01_multiseed_pf import (
    _interpolate_regular_grid,
    _kernel_arguments,
    particle_filter_all_seeds_numba,
    prepare_particle_filter_inputs,
)
from src.p3_pf02_target_ess_paths import (
    require_exact_log_likelihood_match,
    solve_temperature_for_target_ess,
)


FORMAL_WINDOW_SIZES_FT = (250, 500, 1000)
FORMAL_TARGET_ESS = 16.0
FORMAL_CENTER_STEP_FT = 100.0
LOCAL_PATH_COLUMNS = [
    "pf128_local250_delta",
    "pf128_local500_delta",
    "pf128_local1000_delta",
]
SHARED_CACHE_FORMAT_VERSION = 1


def write_pf03_shared_cache_atomic(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    fingerprint: str,
) -> None:
    """原子写入供 PF03、MODE01 共用的逐井紧凑 NPZ，不做慢速压缩。"""

    cache_path = Path(path)
    if not fingerprint:
        raise ValueError("共享缓存指纹不能为空")
    required = {
        "seed_delta",
        "row_index",
        "hidden_md",
        "last_tvt",
        "final_ll",
        "seed_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"共享缓存缺少数组: {missing}")
    seed_predictions = np.asarray(arrays["seed_delta"])
    row_index = np.asarray(arrays["row_index"])
    md = np.asarray(arrays["hidden_md"])
    final_likelihoods = np.asarray(arrays["final_ll"])
    seed_ids = np.asarray(arrays["seed_ids"])
    if seed_predictions.ndim != 2:
        raise ValueError("seed_predictions 必须是二维数组")
    if row_index.shape != (seed_predictions.shape[1],) or md.shape != row_index.shape:
        raise ValueError("共享缓存的 row_index、MD 与路径行数不一致")
    if final_likelihoods.shape != (seed_predictions.shape[0],):
        raise ValueError("final_ll 与 seed 数不一致")
    if seed_ids.shape != (seed_predictions.shape[0],):
        raise ValueError("seed_ids 与 seed 数不一致")

    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["_cache_fingerprint"] = np.asarray([fingerprint])
    payload["_format_version"] = np.asarray(
        [SHARED_CACHE_FORMAT_VERSION], dtype=np.int64
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    with temporary_path.open("wb") as output_file:
        np.savez(output_file, **payload)
    temporary_path.replace(cache_path)


def read_pf03_shared_cache(
    path: str | Path,
    expected_fingerprint: str,
) -> dict[str, np.ndarray]:
    """读取逐井共享缓存，并在返回数据前强制核对版本和配置指纹。"""

    with np.load(Path(path), allow_pickle=False) as cache:
        fingerprint = str(cache["_cache_fingerprint"][0])
        if fingerprint != expected_fingerprint:
            raise ValueError("PF03 共享缓存指纹与当前配置不一致")
        version = int(cache["_format_version"][0])
        if version != SHARED_CACHE_FORMAT_VERSION:
            raise ValueError("PF03 共享缓存格式版本不一致")
        return {
            name: np.asarray(cache[name]).copy()
            for name in cache.files
            if not name.startswith("_")
        }


@njit(cache=True, nogil=True)
def particle_filter_with_row_likelihoods_numba(
    md: np.ndarray,
    z: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell_gr_grid: np.ndarray,
    typewell_min_tvt: float,
    typewell_step_ft: float,
    gr_sigma: float,
    initial_u: float,
    initial_rate: float,
    number_of_particles: int,
    number_of_seeds: int,
    seed_base: int,
    rate_momentum: float,
    rate_noise: float,
    position_noise_ft: float,
    resample_position_noise_ft: float,
    resample_rate_noise: float,
    resample_effective_fraction: float,
    initial_position_spread_ft: float,
    position_limit_beyond_typewell_ft: float,
    initial_rate_std: float,
    minimum_md_step_ft: float,
    squared_gr_residual_cap: float,
    likelihood_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐位复刻 P2-P01，并额外记录每个 seed、每一行的观测 log-LL。"""

    number_of_rows = len(md)
    predictions = np.empty((number_of_seeds, number_of_rows), dtype=np.float64)
    log_likelihoods = np.empty(number_of_seeds, dtype=np.float64)
    row_log_likelihoods = np.empty(
        (number_of_seeds, number_of_rows), dtype=np.float64
    )

    typewell_limit_maximum = (
        typewell_min_tvt + len(typewell_gr_grid) * typewell_step_ft
    )

    for seed_offset in range(number_of_seeds):
        np.random.seed(seed_base + seed_offset)

        particle_u_positions = np.empty(number_of_particles, dtype=np.float64)
        particle_rates = np.empty(number_of_particles, dtype=np.float64)
        particle_weights = (
            np.ones(number_of_particles, dtype=np.float64) / number_of_particles
        )

        for particle_index in range(number_of_particles):
            particle_u_positions[particle_index] = (
                initial_u + initial_position_spread_ft * np.random.randn()
            )
            particle_rates[particle_index] = (
                initial_rate + initial_rate_std * np.random.randn()
            )

        path_log_likelihood = 0.0
        previous_md = md[0] - minimum_md_step_ft

        for row_index in range(number_of_rows):
            md_change = md[row_index] - previous_md
            if md_change < minimum_md_step_ft:
                md_change = minimum_md_step_ft

            for particle_index in range(number_of_particles):
                particle_rates[particle_index] = (
                    rate_momentum * particle_rates[particle_index]
                    + rate_noise * np.random.randn()
                )
                particle_u_positions[particle_index] += (
                    particle_rates[particle_index] * md_change
                    + position_noise_ft * np.random.randn()
                )

                particle_tvt = particle_u_positions[particle_index] - z[row_index]
                lower_limit = typewell_min_tvt - position_limit_beyond_typewell_ft
                upper_limit = (
                    typewell_limit_maximum + position_limit_beyond_typewell_ft
                )
                if particle_tvt < lower_limit:
                    particle_tvt = lower_limit
                if particle_tvt > upper_limit:
                    particle_tvt = upper_limit
                particle_u_positions[particle_index] = particle_tvt + z[row_index]

            average_likelihood = 0.0
            for particle_index in range(number_of_particles):
                particle_tvt = particle_u_positions[particle_index] - z[row_index]
                expected_gr = _interpolate_regular_grid(
                    typewell_gr_grid,
                    particle_tvt,
                    typewell_min_tvt,
                    typewell_step_ft,
                )
                standardized_residual = (
                    horizontal_gr[row_index] - expected_gr
                ) / gr_sigma
                squared_residual = standardized_residual * standardized_residual
                if squared_residual > squared_gr_residual_cap:
                    squared_residual = squared_gr_residual_cap

                observation_likelihood = np.exp(-0.5 * squared_residual)
                if observation_likelihood < likelihood_floor:
                    observation_likelihood = likelihood_floor

                average_likelihood += (
                    particle_weights[particle_index] * observation_likelihood
                )
                particle_weights[particle_index] *= observation_likelihood

            if average_likelihood < likelihood_floor:
                average_likelihood = likelihood_floor
            row_log_likelihood = np.log(average_likelihood)
            row_log_likelihoods[seed_offset, row_index] = row_log_likelihood
            path_log_likelihood += row_log_likelihood

            weight_sum = 0.0
            for particle_index in range(number_of_particles):
                weight_sum += particle_weights[particle_index]
            if weight_sum > 0.0:
                for particle_index in range(number_of_particles):
                    particle_weights[particle_index] /= weight_sum
            else:
                for particle_index in range(number_of_particles):
                    particle_weights[particle_index] = 1.0 / number_of_particles

            inverse_effective_count = 0.0
            for particle_index in range(number_of_particles):
                inverse_effective_count += (
                    particle_weights[particle_index]
                    * particle_weights[particle_index]
                )
            effective_particle_count = 1.0 / inverse_effective_count

            if (
                effective_particle_count
                < resample_effective_fraction * number_of_particles
            ):
                cumulative_weights = np.empty(number_of_particles, dtype=np.float64)
                cumulative_weight = 0.0
                for particle_index in range(number_of_particles):
                    cumulative_weight += particle_weights[particle_index]
                    cumulative_weights[particle_index] = cumulative_weight

                first_systematic_position = np.random.uniform(
                    0.0, 1.0 / number_of_particles
                )
                new_positions = np.empty(number_of_particles, dtype=np.float64)
                new_rates = np.empty(number_of_particles, dtype=np.float64)
                source_index = 0

                for particle_index in range(number_of_particles):
                    systematic_position = (
                        first_systematic_position
                        + particle_index / number_of_particles
                    )
                    while (
                        source_index < number_of_particles - 1
                        and cumulative_weights[source_index] < systematic_position
                    ):
                        source_index += 1
                    new_positions[particle_index] = (
                        particle_u_positions[source_index]
                        + resample_position_noise_ft * np.random.randn()
                    )
                    new_rates[particle_index] = (
                        particle_rates[source_index]
                        + resample_rate_noise * np.random.randn()
                    )

                for particle_index in range(number_of_particles):
                    particle_u_positions[particle_index] = new_positions[particle_index]
                    particle_rates[particle_index] = new_rates[particle_index]
                    particle_weights[particle_index] = 1.0 / number_of_particles

            estimated_tvt = 0.0
            for particle_index in range(number_of_particles):
                estimated_tvt += particle_weights[particle_index] * (
                    particle_u_positions[particle_index] - z[row_index]
                )
            predictions[seed_offset, row_index] = estimated_tvt
            previous_md = md[row_index]

        log_likelihoods[seed_offset] = path_log_likelihood

    return predictions, log_likelihoods, row_log_likelihoods


def path_observation_row_log_likelihoods(
    seed_predictions: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell_gr_grid: np.ndarray,
    typewell_min_tvt: float,
    typewell_step_ft: float,
    gr_sigma: float,
    squared_gr_residual_cap: float,
    likelihood_floor: float,
) -> np.ndarray:
    """按冻结 GR 残差公式计算每条 seed 路径在每个隐藏行的观测 log-LL。"""

    predictions = np.asarray(seed_predictions, dtype=np.float64)
    observed_gr = np.asarray(horizontal_gr, dtype=np.float64)
    reference_gr = np.asarray(typewell_gr_grid, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != len(observed_gr):
        raise ValueError("seed_predictions 必须为 [seed, 行] 且与 GR 行数一致")
    if len(reference_gr) < 2:
        raise ValueError("Typewell GR 网格至少需要两个点")
    if not np.isfinite(predictions).all() or not np.isfinite(observed_gr).all():
        raise ValueError("seed 路径和隐藏 GR 必须全部有限")
    if not np.isfinite(reference_gr).all():
        raise ValueError("Typewell GR 网格必须全部有限")
    if typewell_step_ft <= 0.0 or gr_sigma <= 0.0:
        raise ValueError("Typewell 步长和 GR sigma 必须为正数")
    if squared_gr_residual_cap <= 0.0 or likelihood_floor <= 0.0:
        raise ValueError("残差上限和 likelihood floor 必须为正数")

    scaled_positions = (predictions - typewell_min_tvt) / typewell_step_ft
    left_indices = scaled_positions.astype(np.int64)
    final_index = len(reference_gr) - 1
    below = left_indices < 0
    above = left_indices >= final_index
    middle = ~(below | above)
    expected_gr = np.empty_like(predictions)
    expected_gr[below] = reference_gr[0]
    expected_gr[above] = reference_gr[final_index]
    middle_left = left_indices[middle]
    fractional = scaled_positions[middle] - middle_left
    expected_gr[middle] = (
        reference_gr[middle_left] * (1.0 - fractional)
        + reference_gr[middle_left + 1] * fractional
    )

    standardized_residual = (observed_gr[None, :] - expected_gr) / gr_sigma
    squared_residual = np.minimum(
        np.square(standardized_residual), squared_gr_residual_cap
    )
    observation_likelihood = np.maximum(
        np.exp(-0.5 * squared_residual), likelihood_floor
    )
    row_log_likelihoods = np.log(observation_likelihood)
    if not np.isfinite(row_log_likelihoods).all():
        raise RuntimeError("路径观测 LL 含 NaN 或无穷值")
    return row_log_likelihoods


def _validate_local_inputs(
    seed_predictions: np.ndarray,
    row_log_likelihoods: np.ndarray,
    md: np.ndarray,
    observed_gr_mask: np.ndarray,
    last_visible_tvt: float,
    window_ft: float,
    target_ess: float,
    center_step_ft: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """统一检查局部路径聚合所需的四个数组和固定数值参数。"""

    predictions = np.asarray(seed_predictions, dtype=np.float64)
    row_likelihoods = np.asarray(row_log_likelihoods, dtype=np.float64)
    hidden_md = np.asarray(md, dtype=np.float64)
    observed = np.asarray(observed_gr_mask, dtype=bool)
    if predictions.ndim != 2 or predictions.shape[1] == 0:
        raise ValueError("seed_predictions 必须是非空 [seed, 隐藏行] 二维数组")
    if row_likelihoods.shape != predictions.shape:
        raise ValueError("row_log_likelihoods 必须与 seed_predictions 同形")
    if hidden_md.shape != (predictions.shape[1],):
        raise ValueError("md 长度必须等于隐藏行数")
    if observed.shape != hidden_md.shape:
        raise ValueError("observed_gr_mask 长度必须等于隐藏行数")
    if not np.isfinite(predictions).all() or not np.isfinite(row_likelihoods).all():
        raise ValueError("预测和逐行 LL 必须全部有限")
    if not np.isfinite(hidden_md).all() or np.any(np.diff(hidden_md) < 0.0):
        raise ValueError("隐藏 MD 必须有限且单调不减")
    if not np.isfinite(last_visible_tvt):
        raise ValueError("last_visible_tvt 必须有限")
    if not np.isfinite(window_ft) or window_ft <= 0.0:
        raise ValueError("window_ft 必须为正数")
    if not np.isfinite(center_step_ft) or center_step_ft <= 0.0:
        raise ValueError("center_step_ft 必须为正数")
    if not np.isfinite(target_ess) or not 1.0 < target_ess < predictions.shape[0]:
        raise ValueError("target_ess 必须严格位于 1 与 seed 数之间")
    return predictions, row_likelihoods, hidden_md, observed


def _window_centers(md: np.ndarray, center_step_ft: float) -> np.ndarray:
    """从隐藏段首点开始，每隔固定 MD 距离放置一个窗口中心。"""

    span = max(0.0, float(md[-1] - md[0]))
    center_count = int(math.floor(span / center_step_ft)) + 1
    return md[0] + np.arange(center_count, dtype=np.float64) * center_step_ft


def _segment_log_likelihoods(
    row_log_likelihoods: np.ndarray,
    md: np.ndarray,
    observed_gr_mask: np.ndarray,
    window_ft: float,
    center_step_ft: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用前缀和计算每个重叠窗口的 128 条局部累计 LL。"""

    centers = _window_centers(md, center_step_ft)
    observed_contribution = np.where(
        observed_gr_mask[None, :], row_log_likelihoods, 0.0
    )
    cumulative = np.zeros(
        (row_log_likelihoods.shape[0], row_log_likelihoods.shape[1] + 1),
        dtype=np.float64,
    )
    np.cumsum(observed_contribution, axis=1, out=cumulative[:, 1:])
    observed_cumulative = np.zeros(len(md) + 1, dtype=np.int64)
    np.cumsum(observed_gr_mask.astype(np.int64), out=observed_cumulative[1:])

    segment_likelihoods = np.empty(
        (row_log_likelihoods.shape[0], len(centers)), dtype=np.float64
    )
    observed_counts = np.empty(len(centers), dtype=np.int64)
    half_window = window_ft / 2.0
    for center_index, center_md in enumerate(centers):
        left = int(np.searchsorted(md, center_md - half_window, side="left"))
        right = int(np.searchsorted(md, center_md + half_window, side="right"))
        segment_likelihoods[:, center_index] = (
            cumulative[:, right] - cumulative[:, left]
        )
        observed_counts[center_index] = (
            observed_cumulative[right] - observed_cumulative[left]
        )
    return centers, segment_likelihoods, observed_counts


def _log_weights_for_segments(
    segment_log_likelihoods: np.ndarray,
    observed_counts: np.ndarray,
    target_ess: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """逐窗口二分温度到目标 ESS；无真实 GR 或不可辨识时使用均匀权重。"""

    seed_count, segment_count = segment_log_likelihoods.shape
    uniform_log_weight = -math.log(seed_count)
    log_weights = np.full(
        (seed_count, segment_count), uniform_log_weight, dtype=np.float64
    )
    achieved_ess = np.full(segment_count, float(seed_count), dtype=np.float64)
    unidentifiable_count = 0
    for segment_index in range(segment_count):
        if observed_counts[segment_index] == 0:
            continue
        local_likelihoods = segment_log_likelihoods[:, segment_index]
        maximum_count = int(
            np.count_nonzero(local_likelihoods == np.max(local_likelihoods))
        )
        if maximum_count > target_ess or np.ptp(local_likelihoods) == 0.0:
            unidentifiable_count += 1
            continue
        # 旧 D01 的熵审计会先计算 log(0) 再用 where 丢弃；这里只屏蔽该无害告警。
        with np.errstate(divide="ignore", invalid="ignore"):
            solution = solve_temperature_for_target_ess(
                local_likelihoods, target_ess=target_ess
            )
        temperature = solution["temperature"]
        centered = (local_likelihoods - np.max(local_likelihoods)) / temperature
        normalizer = float(np.log(np.exp(centered).sum()))
        log_weights[:, segment_index] = centered - normalizer
        achieved_ess[segment_index] = solution["achieved_ess"]
    return log_weights, achieved_ess, unidentifiable_count


def build_local_likelihood_path(
    seed_predictions: np.ndarray,
    row_log_likelihoods: np.ndarray,
    md: np.ndarray,
    observed_gr_mask: np.ndarray,
    last_visible_tvt: float,
    window_ft: float,
    target_ess: float = FORMAL_TARGET_ESS,
    center_step_ft: float = FORMAL_CENTER_STEP_FT,
) -> tuple[np.ndarray, dict[str, float | int], dict[str, np.ndarray]]:
    """生成一条局部软加权路径，并返回质量统计及可复用的分段 LL。"""

    predictions, row_likelihoods, hidden_md, observed = _validate_local_inputs(
        seed_predictions,
        row_log_likelihoods,
        md,
        observed_gr_mask,
        last_visible_tvt,
        window_ft,
        target_ess,
        center_step_ft,
    )
    centers, segment_likelihoods, observed_counts = _segment_log_likelihoods(
        row_likelihoods,
        hidden_md,
        observed,
        window_ft,
        center_step_ft,
    )
    segment_log_weights, achieved_ess, unidentifiable_count = (
        _log_weights_for_segments(
            segment_likelihoods,
            observed_counts,
            target_ess,
        )
    )

    interpolated_log_weights = np.empty_like(predictions)
    for seed_index in range(predictions.shape[0]):
        interpolated_log_weights[seed_index] = np.interp(
            hidden_md,
            centers,
            segment_log_weights[seed_index],
        )
    interpolated_log_weights -= np.max(interpolated_log_weights, axis=0)
    row_weights = np.exp(interpolated_log_weights)
    row_weights /= row_weights.sum(axis=0, keepdims=True)
    absolute_path = np.sum(row_weights * predictions, axis=0).astype(np.float32)
    delta = absolute_path - np.float32(last_visible_tvt)

    uniform_weight = 1.0 / predictions.shape[0]
    uniform_rows = np.max(
        np.abs(row_weights - uniform_weight), axis=0
    ) <= 1e-12
    dominant_seed = np.argmax(row_weights, axis=0)
    quality: dict[str, float | int] = {
        "window_count": int(len(centers)),
        "no_observation_window_count": int(np.count_nonzero(observed_counts == 0)),
        "unidentifiable_window_count": int(unidentifiable_count),
        "uniform_fallback_row_count": int(np.count_nonzero(uniform_rows)),
        "median_achieved_ess": float(np.median(achieved_ess)),
        "minimum_achieved_ess": float(np.min(achieved_ess)),
        "maximum_achieved_ess": float(np.max(achieved_ess)),
        "dominant_seed_switch_count": int(
            np.count_nonzero(np.diff(dominant_seed) != 0)
        ),
    }
    segment_cache = {
        "center_md": centers,
        "segment_log_likelihoods": segment_likelihoods,
        "segment_observed_counts": observed_counts,
    }
    return delta, quality, segment_cache


def aggregate_local_likelihood_path(
    seed_predictions: np.ndarray,
    row_log_likelihoods: np.ndarray,
    md: np.ndarray,
    observed_gr_mask: np.ndarray,
    last_visible_tvt: float,
    window_ft: float,
    target_ess: float = FORMAL_TARGET_ESS,
    center_step_ft: float = FORMAL_CENTER_STEP_FT,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """便于测试和复用的两返回值接口。"""

    delta, quality, _ = build_local_likelihood_path(
        seed_predictions=seed_predictions,
        row_log_likelihoods=row_log_likelihoods,
        md=md,
        observed_gr_mask=observed_gr_mask,
        last_visible_tvt=last_visible_tvt,
        window_ft=window_ft,
        target_ess=target_ess,
        center_step_ft=center_step_ft,
    )
    return delta, quality


def build_segmented_pf_features(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
    d01_log_likelihoods: np.ndarray,
    window_sizes_ft: Sequence[int] = FORMAL_WINDOW_SIZES_FT,
    target_ess: float = FORMAL_TARGET_ESS,
    center_step_ft: float = FORMAL_CENTER_STEP_FT,
) -> tuple[pd.DataFrame, dict[str, float | int | bool], dict[str, np.ndarray]]:
    """一次正式冻结回放，生成三条局部路径和供 MODE01 复用的紧凑缓存。"""

    windows = tuple(int(value) for value in window_sizes_ft)
    if windows != FORMAL_WINDOW_SIZES_FT:
        raise ValueError("正式 PF03 窗口必须固定为 250/500/1000 ft")
    prepared = prepare_particle_filter_inputs(horizontal_well, typewell, parameters)
    seed_predictions, full_likelihoods = particle_filter_all_seeds_numba(
        **_kernel_arguments(prepared)
    )
    require_exact_log_likelihood_match(full_likelihoods, d01_log_likelihoods)
    row_likelihoods = path_observation_row_log_likelihoods(
        seed_predictions=seed_predictions,
        horizontal_gr=prepared["horizontal_gr"],
        typewell_gr_grid=prepared["typewell_gr_grid"],
        typewell_min_tvt=float(prepared["typewell_min_tvt"]),
        typewell_step_ft=float(prepared["typewell_step_ft"]),
        gr_sigma=float(prepared["gr_sigma"]),
        squared_gr_residual_cap=float(prepared["squared_gr_residual_cap"]),
        likelihood_floor=float(prepared["likelihood_floor"]),
    )

    hidden_mask = horizontal_well["TVT_input"].isna().to_numpy()
    original_hidden_gr = pd.to_numeric(
        horizontal_well.loc[hidden_mask, "GR"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    observed_gr_mask = np.isfinite(original_hidden_gr)
    output_values: dict[str, np.ndarray] = {}
    quality: dict[str, float | int | bool] = {
        "d01_log_likelihood_bitwise_match": True,
        "legacy_gr_sigma": float(prepared["gr_sigma"]),
        "observed_gr_row_count": int(np.count_nonzero(observed_gr_mask)),
        "observed_gr_fraction": float(np.mean(observed_gr_mask)),
    }
    last_visible_tvt = np.float32(prepared["last_visible_tvt"])
    shared_cache: dict[str, np.ndarray] = {
        "seed_delta": seed_predictions.astype(np.float32) - last_visible_tvt,
        "row_index": np.asarray(prepared["row_index"], dtype=np.int32),
        "hidden_md": np.asarray(prepared["md"], dtype=np.float32),
        "last_tvt": np.asarray([prepared["last_visible_tvt"]], dtype=np.float64),
        "final_ll": np.asarray(full_likelihoods, dtype=np.float64),
        "seed_ids": (
            int(prepared["seed_base"])
            + np.arange(int(prepared["number_of_seeds"]), dtype=np.int32)
        ),
    }
    for window_ft, column_name in zip(
        windows, LOCAL_PATH_COLUMNS, strict=True
    ):
        delta, window_quality, _ = build_local_likelihood_path(
            seed_predictions=seed_predictions,
            row_log_likelihoods=row_likelihoods,
            md=prepared["md"],
            observed_gr_mask=observed_gr_mask,
            last_visible_tvt=float(prepared["last_visible_tvt"]),
            window_ft=float(window_ft),
            target_ess=float(target_ess),
            center_step_ft=float(center_step_ft),
        )
        output_values[column_name] = delta
        for key, value in window_quality.items():
            quality[f"local{window_ft}_{key}"] = value

    hidden_rows = len(prepared["row_index"])
    features = pd.DataFrame(
        {
            "row_index": prepared["row_index"],
            "last_visible_tvt": np.full(
                hidden_rows,
                np.float32(prepared["last_visible_tvt"]),
                dtype=np.float32,
            ),
            **output_values,
        }
    )
    expected_columns = ["row_index", "last_visible_tvt", *LOCAL_PATH_COLUMNS]
    if list(features.columns) != expected_columns:
        raise RuntimeError("PF03 输出 schema 不正确")
    if not np.isfinite(
        features.drop(columns="row_index").to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("PF03 路径含 NaN 或无穷值")
    return features, quality, shared_cache
