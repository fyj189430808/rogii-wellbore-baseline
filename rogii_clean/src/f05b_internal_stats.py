"""F05b：从固定 ANCC 粒子滤波中读取归一化的逐行内部统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from src.f05a_candidate_reproduction import (
    ANCC_ALPHA,
    ANCC_INITIAL_SPREAD,
    ANCC_N,
    ANCC_POSITION_NOISE,
    ANCC_RATE_NOISE,
    ANCC_ROUGH_POSITION,
    ANCC_ROUGH_RATE,
    PF_RESAMPLE_THRESHOLD,
    _gr_sigma,
    _interp_regular_grid,
    _prepare_inputs,
    _regular_typewell_grid,
    _systematic_resample,
)


F05B_FEATURE_COLUMNS = (
    "f05b_ess_ratio",
    "f05b_mean_valid_ess_ratio",
    "f05b_min_ess_ratio",
    "f05b_low_ess_fraction",
    "f05b_resample_fraction",
    "f05b_weight_collapse_fraction",
    "f05b_mean_loglik_gap",
    "f05b_entropy_ratio",
    "f05b_steps_since_valid_update_ratio",
    "f05b_valid_update_fraction",
)


@njit(cache=False)
def _particle_filter_ancc_with_diagnostics(
    hidden_md: np.ndarray,
    hidden_z: np.ndarray,
    hidden_gr: np.ndarray,
    typewell_gr_grid: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_sigma: float,
    initial_structure_position: float,
    initial_structure_rate: float,
    particle_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """运行原 ANCC PF，同时在重采样前只读权重并记录 10 项诊断。"""

    # 从这里到路径预测部分保持与 F05a 的 ANCC PF 相同的随机数和计算顺序。
    np.random.seed(seed)
    positions = np.empty(particle_count)
    rates = np.empty(particle_count)
    weights = np.ones(particle_count) / particle_count
    for particle_index in range(particle_count):
        positions[particle_index] = (
            initial_structure_position + ANCC_INITIAL_SPREAD * np.random.randn()
        )
        rates[particle_index] = initial_structure_rate + 0.01 * np.random.randn()

    row_count = len(hidden_md)
    predictions = np.empty(row_count)
    standard_deviations = np.empty(row_count)
    diagnostics = np.empty((row_count, len(F05B_FEATURE_COLUMNS)))
    previous_md = hidden_md[0] - 1.0
    maximum_tvt = grid_minimum + len(typewell_gr_grid) * grid_step + 50.0
    minimum_tvt = grid_minimum - 50.0

    valid_observation_count = 0
    normalized_ess_sum_at_valid_updates = 0.0
    minimum_normalized_ess = 1.0
    low_ess_count = 0
    resampling_count = 0
    weight_collapse_count = 0
    cumulative_loglik_gap = 0.0
    steps_since_valid_update = 0
    log_particle_count = np.log(float(particle_count))

    for row_index in range(row_count):
        md_step = max(hidden_md[row_index] - previous_md, 1.0)
        for particle_index in range(particle_count):
            rates[particle_index] = (
                ANCC_ALPHA * rates[particle_index]
                + ANCC_RATE_NOISE * np.random.randn()
            )
            positions[particle_index] += (
                rates[particle_index] * md_step
                + ANCC_POSITION_NOISE * np.random.randn()
            )
            particle_tvt = positions[particle_index] - hidden_z[row_index]
            particle_tvt = max(particle_tvt, minimum_tvt)
            particle_tvt = min(particle_tvt, maximum_tvt)
            positions[particle_index] = particle_tvt + hidden_z[row_index]

        has_valid_gr = not np.isnan(hidden_gr[row_index])
        largest_log_likelihood = -1e308
        second_largest_log_likelihood = -1e308
        if has_valid_gr:
            weight_sum = 0.0
            for particle_index in range(particle_count):
                expected_gr = _interp_regular_grid(
                    typewell_gr_grid,
                    positions[particle_index] - hidden_z[row_index],
                    grid_minimum,
                    grid_step,
                )
                normalized_error = (hidden_gr[row_index] - expected_gr) / gr_sigma
                squared_error = normalized_error * normalized_error
                likelihood = (
                    np.exp(-0.5 * squared_error) if squared_error < 600.0 else 0.0
                )
                likelihood = max(likelihood, 1e-300)
                weights[particle_index] *= likelihood
                weight_sum += weights[particle_index]

                log_likelihood = np.log(likelihood)
                if log_likelihood > largest_log_likelihood:
                    second_largest_log_likelihood = largest_log_likelihood
                    largest_log_likelihood = log_likelihood
                elif log_likelihood > second_largest_log_likelihood:
                    second_largest_log_likelihood = log_likelihood

            if weight_sum > 0.0:
                for particle_index in range(particle_count):
                    weights[particle_index] /= weight_sum
            else:
                weight_collapse_count += 1
                for particle_index in range(particle_count):
                    weights[particle_index] = 1.0 / particle_count

            valid_observation_count += 1
            cumulative_loglik_gap += (
                largest_log_likelihood - second_largest_log_likelihood
            )
            steps_since_valid_update = 0
        else:
            steps_since_valid_update += 1

        squared_weight_sum = 0.0
        entropy = 0.0
        for particle_index in range(particle_count):
            particle_weight = weights[particle_index]
            squared_weight_sum += particle_weight * particle_weight
            if particle_weight > 0.0:
                entropy -= particle_weight * np.log(particle_weight)

        effective_particle_count = 1.0 / squared_weight_sum
        normalized_ess = effective_particle_count / particle_count
        if normalized_ess < minimum_normalized_ess:
            minimum_normalized_ess = normalized_ess
        if has_valid_gr:
            normalized_ess_sum_at_valid_updates += normalized_ess
            if normalized_ess < 0.1:
                low_ess_count += 1

        needs_resampling = (
            effective_particle_count < PF_RESAMPLE_THRESHOLD * particle_count
        )
        if needs_resampling:
            resampling_count += 1

        trajectory_length = row_index + 1
        if valid_observation_count > 0:
            mean_valid_ess_ratio = (
                normalized_ess_sum_at_valid_updates / valid_observation_count
            )
            low_ess_fraction = low_ess_count / valid_observation_count
            collapse_fraction = weight_collapse_count / valid_observation_count
            mean_loglik_gap = cumulative_loglik_gap / valid_observation_count
        else:
            mean_valid_ess_ratio = 1.0
            low_ess_fraction = 0.0
            collapse_fraction = 0.0
            mean_loglik_gap = 0.0

        entropy_ratio = entropy / log_particle_count
        entropy_ratio = max(0.0, min(entropy_ratio, 1.0))
        diagnostics[row_index, 0] = max(0.0, min(normalized_ess, 1.0))
        diagnostics[row_index, 1] = max(0.0, min(mean_valid_ess_ratio, 1.0))
        diagnostics[row_index, 2] = max(0.0, min(minimum_normalized_ess, 1.0))
        diagnostics[row_index, 3] = max(0.0, min(low_ess_fraction, 1.0))
        diagnostics[row_index, 4] = resampling_count / trajectory_length
        diagnostics[row_index, 5] = max(0.0, min(collapse_fraction, 1.0))
        diagnostics[row_index, 6] = max(0.0, mean_loglik_gap)
        diagnostics[row_index, 7] = entropy_ratio
        diagnostics[row_index, 8] = steps_since_valid_update / trajectory_length
        diagnostics[row_index, 9] = valid_observation_count / trajectory_length

        if needs_resampling:
            positions, rates = _systematic_resample(
                positions,
                rates,
                weights,
                particle_count,
                ANCC_ROUGH_POSITION,
                ANCC_ROUGH_RATE,
            )
            for particle_index in range(particle_count):
                weights[particle_index] = 1.0 / particle_count

        predicted_tvt = 0.0
        for particle_index in range(particle_count):
            predicted_tvt += weights[particle_index] * (
                positions[particle_index] - hidden_z[row_index]
            )
        predictions[row_index] = predicted_tvt
        variance = 0.0
        for particle_index in range(particle_count):
            difference = (
                positions[particle_index] - hidden_z[row_index] - predicted_tvt
            )
            variance += weights[particle_index] * difference * difference
        standard_deviations[row_index] = variance**0.5
        previous_md = hidden_md[row_index]

    return predictions, standard_deviations, diagnostics


def build_internal_stats_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """返回隐藏行的 ``row_index`` 和固定顺序的 10 个 F05b 特征。"""

    horizontal, typewell, visible_mask, hidden_mask = _prepare_inputs(
        horizontal_df,
        typewell_df,
    )
    normalized_seed = int(seed)
    if normalized_seed < 0 or normalized_seed >= 2**31 - 1:
        raise ValueError("seed 必须位于 [0, 2**31-1) 内")

    visible = horizontal.loc[visible_mask]
    hidden = horizontal.loc[hidden_mask]
    typewell_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    typewell_gr = typewell["GR"].to_numpy(dtype=np.float64)
    gr_sigma = _gr_sigma(horizontal, typewell_tvt, typewell_gr)

    visible_tail = visible.tail(30)
    tail_tvt_change = np.diff(visible_tail["TVT_input"].to_numpy(dtype=np.float64))
    tail_z_change = np.diff(visible_tail["Z"].to_numpy(dtype=np.float64))
    tail_md_change = np.diff(visible_tail["MD"].to_numpy(dtype=np.float64))
    valid_tail_steps = tail_md_change > 0
    if int(valid_tail_steps.sum()) >= 3:
        initial_structure_rate = float(
            np.median(
                (tail_tvt_change + tail_z_change)[valid_tail_steps]
                / tail_md_change[valid_tail_steps]
            )
        )
    else:
        initial_structure_rate = 0.0

    typewell_gr_grid, grid_minimum, grid_step = _regular_typewell_grid(
        typewell_tvt,
        typewell_gr,
    )
    hidden_md = hidden["MD"].to_numpy(dtype=np.float64)
    hidden_z = hidden["Z"].to_numpy(dtype=np.float64)
    hidden_gr = hidden["GR"].to_numpy(dtype=np.float64)
    initial_structure_position = float(visible["TVT_input"].iloc[-1]) + float(
        visible["Z"].iloc[-1]
    )

    _, _, diagnostics = _particle_filter_ancc_with_diagnostics(
        hidden_md,
        hidden_z,
        hidden_gr,
        typewell_gr_grid,
        grid_minimum,
        grid_step,
        gr_sigma,
        initial_structure_position,
        initial_structure_rate,
        ANCC_N,
        normalized_seed,
    )
    if not np.isfinite(diagnostics).all():
        raise ValueError("F05b 内部统计产生了 NaN 或 Inf")

    feature_data = {
        column: diagnostics[:, column_index].astype(np.float32)
        for column_index, column in enumerate(F05B_FEATURE_COLUMNS)
    }
    result = pd.DataFrame(
        {
            "row_index": hidden.index.to_numpy(dtype=np.int64),
            **feature_data,
        }
    )
    return result[["row_index", *F05B_FEATURE_COLUMNS]]
