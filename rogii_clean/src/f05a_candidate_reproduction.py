"""从原始单井数据复现 notebook 的 24 个 PF、Beam 与 NCC 候选特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS


PF_N = 600
ANCC_N = 600
PF_MOM = 0.993
PF_VN = 0.005
PF_PN = 0.01
PF_GR_SIG_MIN = 10.0
PF_GR_SIG_MAX = 60.0
PF_GR_SIG_DEFAULT = 30.0
PF_RESAMPLE_THRESHOLD = 0.5
PF_ROUGH_POSITION = 0.2
PF_ROUGH_VELOCITY = 0.003
PF_GR_WINDOW = 5
PF_SMOOTH_GR_WEIGHT = 0.3
ANCC_ALPHA = 0.998
ANCC_RATE_NOISE = 0.002
ANCC_POSITION_NOISE = 0.005
ANCC_INITIAL_SPREAD = 0.3
ANCC_ROUGH_POSITION = 0.1
ANCC_ROUGH_RATE = 0.001

BEAM_CONFIGS = (
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
)


@njit(cache=False)
def _interp_regular_grid(
    grid: np.ndarray,
    value: float,
    minimum: float,
    step: float,
) -> float:
    """在等间隔 Typewell 网格上做线性插值。"""

    left_index = int((value - minimum) / step)
    if left_index < 0:
        return grid[0]
    final_index = len(grid) - 1
    if left_index >= final_index:
        return grid[final_index]
    fraction = (value - minimum) / step - left_index
    return grid[left_index] * (1.0 - fraction) + grid[left_index + 1] * fraction


@njit(cache=False)
def _systematic_resample(
    positions: np.ndarray,
    auxiliary_state: np.ndarray,
    weights: np.ndarray,
    particle_count: int,
    position_roughness: float,
    auxiliary_roughness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按权重系统重采样，并给复制出的状态加入 notebook 原始粗化噪声。"""

    cumulative = np.zeros(particle_count + 1)
    for particle_index in range(particle_count):
        cumulative[particle_index + 1] = (
            cumulative[particle_index] + weights[particle_index]
        )
    first_sample = np.random.uniform(0.0, 1.0 / particle_count)
    new_positions = np.empty(particle_count)
    new_auxiliary = np.empty(particle_count)
    source_index = 0
    for particle_index in range(particle_count):
        sample_position = first_sample + particle_index / particle_count
        while (
            source_index < particle_count - 1
            and cumulative[source_index + 1] < sample_position
        ):
            source_index += 1
        new_positions[particle_index] = (
            positions[source_index] + position_roughness * np.random.randn()
        )
        new_auxiliary[particle_index] = (
            auxiliary_state[source_index] + auxiliary_roughness * np.random.randn()
        )
    return new_positions, new_auxiliary


@njit(cache=False)
def _particle_filter_ancc(
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
) -> tuple[np.ndarray, np.ndarray]:
    """追踪 U=TVT+Z；显式 seed 是相对原 notebook 的可复现性修正。"""

    np.random.seed(seed)
    positions = np.empty(particle_count)
    rates = np.empty(particle_count)
    weights = np.ones(particle_count) / particle_count
    for particle_index in range(particle_count):
        positions[particle_index] = (
            initial_structure_position + ANCC_INITIAL_SPREAD * np.random.randn()
        )
        rates[particle_index] = initial_structure_rate + 0.01 * np.random.randn()

    predictions = np.empty(len(hidden_md))
    standard_deviations = np.empty(len(hidden_md))
    previous_md = hidden_md[0] - 1.0
    maximum_tvt = grid_minimum + len(typewell_gr_grid) * grid_step + 50.0
    minimum_tvt = grid_minimum - 50.0

    for row_index in range(len(hidden_md)):
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

        if not np.isnan(hidden_gr[row_index]):
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
            if weight_sum > 0.0:
                for particle_index in range(particle_count):
                    weights[particle_index] /= weight_sum
            else:
                for particle_index in range(particle_count):
                    weights[particle_index] = 1.0 / particle_count

        squared_weight_sum = 0.0
        for particle_index in range(particle_count):
            squared_weight_sum += weights[particle_index] * weights[particle_index]
        if 1.0 / squared_weight_sum < PF_RESAMPLE_THRESHOLD * particle_count:
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

    return predictions, standard_deviations


@njit(cache=False)
def _particle_filter_z(
    hidden_md: np.ndarray,
    hidden_z: np.ndarray,
    hidden_gr: np.ndarray,
    hidden_smoothed_gr: np.ndarray,
    typewell_gr_grid: np.ndarray,
    typewell_smoothed_gr_grid: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_sigma: float,
    initial_tvt: float,
    initial_velocity: float,
    z_velocity_coefficient: float,
    z_velocity_intercept: float,
    z_velocity_sigma: float,
    particle_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """追踪 TVT 和 dTVT/dMD；显式 seed 保证同值重跑一致。"""

    np.random.seed(seed)
    positions = np.empty(particle_count)
    velocities = np.empty(particle_count)
    weights = np.ones(particle_count) / particle_count
    for particle_index in range(particle_count):
        positions[particle_index] = initial_tvt + 0.5 * np.random.randn()
        velocities[particle_index] = initial_velocity + 0.02 * np.random.randn()

    predictions = np.empty(len(hidden_md))
    standard_deviations = np.empty(len(hidden_md))
    previous_md = hidden_md[0] - 1.0
    previous_z = hidden_z[0] - 1.0
    maximum_tvt = grid_minimum + len(typewell_gr_grid) * grid_step + 50.0
    minimum_tvt = grid_minimum - 50.0

    for row_index in range(len(hidden_md)):
        md_step = max(hidden_md[row_index] - previous_md, 1.0)
        z_change_per_md = (hidden_z[row_index] - previous_z) / md_step
        expected_velocity = (
            z_velocity_coefficient * z_change_per_md + z_velocity_intercept
        )
        for particle_index in range(particle_count):
            velocities[particle_index] = (
                PF_MOM * velocities[particle_index] + PF_VN * np.random.randn()
            )
            positions[particle_index] += (
                velocities[particle_index] * md_step + PF_PN * np.random.randn()
            )
            positions[particle_index] = max(positions[particle_index], minimum_tvt)
            positions[particle_index] = min(positions[particle_index], maximum_tvt)

        if not np.isnan(hidden_gr[row_index]):
            weight_sum = 0.0
            for particle_index in range(particle_count):
                expected_raw_gr = _interp_regular_grid(
                    typewell_gr_grid,
                    positions[particle_index],
                    grid_minimum,
                    grid_step,
                )
                raw_error = (hidden_gr[row_index] - expected_raw_gr) / gr_sigma
                raw_squared_error = raw_error * raw_error
                raw_likelihood = (
                    np.exp(-0.5 * raw_squared_error)
                    if raw_squared_error < 600.0
                    else 0.0
                )
                raw_likelihood = max(raw_likelihood, 1e-300)
                if not np.isnan(hidden_smoothed_gr[row_index]):
                    expected_smoothed_gr = _interp_regular_grid(
                        typewell_smoothed_gr_grid,
                        positions[particle_index],
                        grid_minimum,
                        grid_step,
                    )
                    smooth_error = (
                        hidden_smoothed_gr[row_index] - expected_smoothed_gr
                    ) / (gr_sigma * 1.5)
                    smooth_squared_error = smooth_error * smooth_error
                    smooth_likelihood = (
                        np.exp(-0.5 * smooth_squared_error)
                        if smooth_squared_error < 600.0
                        else 0.0
                    )
                    smooth_likelihood = max(smooth_likelihood, 1e-300)
                    likelihood = (
                        (1.0 - PF_SMOOTH_GR_WEIGHT) * raw_likelihood
                        + PF_SMOOTH_GR_WEIGHT * smooth_likelihood
                    )
                else:
                    likelihood = raw_likelihood
                weights[particle_index] *= max(likelihood, 1e-300)
                weight_sum += weights[particle_index]
            if weight_sum > 0.0:
                for particle_index in range(particle_count):
                    weights[particle_index] /= weight_sum
            else:
                for particle_index in range(particle_count):
                    weights[particle_index] = 1.0 / particle_count

        velocity_weight_sum = 0.0
        velocity_scale = max(z_velocity_sigma * 2.0, 0.005)
        for particle_index in range(particle_count):
            velocity_error = (
                velocities[particle_index] - expected_velocity
            ) / velocity_scale
            velocity_squared_error = velocity_error * velocity_error
            velocity_likelihood = (
                np.exp(-0.5 * velocity_squared_error)
                if velocity_squared_error < 600.0
                else 0.0
            )
            velocity_likelihood = max(velocity_likelihood, 1e-300)
            weights[particle_index] *= velocity_likelihood
            velocity_weight_sum += weights[particle_index]
        if velocity_weight_sum > 0.0:
            for particle_index in range(particle_count):
                weights[particle_index] /= velocity_weight_sum
        else:
            for particle_index in range(particle_count):
                weights[particle_index] = 1.0 / particle_count

        squared_weight_sum = 0.0
        for particle_index in range(particle_count):
            squared_weight_sum += weights[particle_index] * weights[particle_index]
        if 1.0 / squared_weight_sum < PF_RESAMPLE_THRESHOLD * particle_count:
            positions, velocities = _systematic_resample(
                positions,
                velocities,
                weights,
                particle_count,
                PF_ROUGH_POSITION,
                PF_ROUGH_VELOCITY,
            )
            for particle_index in range(particle_count):
                weights[particle_index] = 1.0 / particle_count

        predicted_tvt = 0.0
        for particle_index in range(particle_count):
            predicted_tvt += weights[particle_index] * positions[particle_index]
        predictions[row_index] = predicted_tvt
        variance = 0.0
        for particle_index in range(particle_count):
            difference = positions[particle_index] - predicted_tvt
            variance += weights[particle_index] * difference * difference
        standard_deviations[row_index] = variance**0.5
        previous_md = hidden_md[row_index]
        previous_z = hidden_z[row_index]

    return predictions, standard_deviations


@njit(cache=False)
def _beam_search_indices(
    smoothed_gr: np.ndarray,
    typewell_gr: np.ndarray,
    start_index: int,
    beam_size: int,
    movement_cost: float,
    error_scale: float,
) -> np.ndarray:
    """复现 notebook 的允许 -2 到 +2 移动并最终回溯的 Beam Search。"""

    row_count = len(smoothed_gr)
    typewell_count = len(typewell_gr)
    maximum_candidates = beam_size * 6
    beam_indices = np.zeros(beam_size, np.int64)
    beam_indices[0] = start_index
    beam_costs = np.full(beam_size, 1e30)
    beam_costs[0] = 0.0
    active_beams = np.int64(1)
    history_indices = np.zeros((row_count, beam_size), np.int64)
    history_parents = np.zeros((row_count, beam_size), np.int64)
    candidate_indices = np.zeros(maximum_candidates, np.int64)
    candidate_costs = np.full(maximum_candidates, 1e30)
    candidate_parents = np.zeros(maximum_candidates, np.int64)

    for row_index in range(row_count):
        observed_gr = smoothed_gr[row_index]
        candidate_count = np.int64(0)
        for beam_index in range(active_beams):
            current_index = beam_indices[beam_index]
            current_cost = beam_costs[beam_index]
            for movement in range(-2, 3):
                next_index = current_index + movement
                if next_index < 0 or next_index >= typewell_count:
                    continue
                absolute_movement = movement if movement >= 0 else -movement
                total_cost = (
                    current_cost
                    + (observed_gr - typewell_gr[next_index]) ** 2 / error_scale
                    + movement_cost * absolute_movement
                )
                found_index = np.int64(-1)
                for candidate_index in range(candidate_count):
                    if candidate_indices[candidate_index] == next_index:
                        found_index = candidate_index
                        break
                if found_index >= 0:
                    if total_cost < candidate_costs[found_index]:
                        candidate_costs[found_index] = total_cost
                        candidate_parents[found_index] = beam_index
                elif candidate_count < maximum_candidates:
                    candidate_indices[candidate_count] = next_index
                    candidate_costs[candidate_count] = total_cost
                    candidate_parents[candidate_count] = beam_index
                    candidate_count += 1

        kept_count = min(beam_size, candidate_count)
        for kept_index in range(kept_count):
            minimum_index = kept_index
            for candidate_index in range(kept_index + 1, candidate_count):
                if candidate_costs[candidate_index] < candidate_costs[minimum_index]:
                    minimum_index = candidate_index
            if minimum_index != kept_index:
                candidate_indices[kept_index], candidate_indices[minimum_index] = (
                    candidate_indices[minimum_index],
                    candidate_indices[kept_index],
                )
                candidate_costs[kept_index], candidate_costs[minimum_index] = (
                    candidate_costs[minimum_index],
                    candidate_costs[kept_index],
                )
                candidate_parents[kept_index], candidate_parents[minimum_index] = (
                    candidate_parents[minimum_index],
                    candidate_parents[kept_index],
                )
        history_indices[row_index, :kept_count] = candidate_indices[:kept_count]
        history_parents[row_index, :kept_count] = candidate_parents[:kept_count]
        beam_indices[:kept_count] = candidate_indices[:kept_count]
        beam_costs[:kept_count] = candidate_costs[:kept_count]
        active_beams = kept_count

    best_beam = np.int64(0)
    for beam_index in range(1, active_beams):
        if beam_costs[beam_index] < beam_costs[best_beam]:
            best_beam = beam_index
    path = np.zeros(row_count, np.int64)
    current_beam = best_beam
    for row_index in range(row_count - 1, -1, -1):
        path[row_index] = history_indices[row_index, current_beam]
        current_beam = history_parents[row_index, current_beam]
    return path


def _regular_typewell_grid(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    step: float = 0.2,
) -> tuple[np.ndarray, float, float]:
    minimum = float(typewell_tvt.min())
    maximum = float(typewell_tvt.max())
    tvt_grid = np.arange(minimum, maximum + step, step)
    gr_grid = np.interp(tvt_grid, typewell_tvt, typewell_gr).astype(np.float64)
    return gr_grid, minimum, float(step)


def _gr_sigma(
    horizontal_df: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> float:
    visible = horizontal_df[
        horizontal_df["TVT_input"].notna() & horizontal_df["GR"].notna()
    ]
    if len(visible) < 20:
        return PF_GR_SIG_DEFAULT
    expected_gr = np.interp(
        visible["TVT_input"].to_numpy(dtype=np.float64),
        typewell_tvt,
        typewell_gr,
    )
    residual = visible["GR"].to_numpy(dtype=np.float64) - expected_gr
    return float(np.clip(np.std(residual), PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _nearest_index(sorted_values: np.ndarray, value: float) -> int:
    insertion_index = int(np.searchsorted(sorted_values, value, side="left"))
    if insertion_index >= len(sorted_values):
        return len(sorted_values) - 1
    if insertion_index > 0:
        left_distance = abs(sorted_values[insertion_index - 1] - value)
        right_distance = abs(sorted_values[insertion_index] - value)
        if left_distance <= right_distance:
            return insertion_index - 1
    return insertion_index


def _smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    series = pd.Series(values, dtype="float32")
    series = series.interpolate(limit_direction="both").fillna(fallback)
    if radius > 0:
        series = series.rolling(
            radius * 2 + 1,
            center=True,
            min_periods=1,
        ).mean()
    return series.to_numpy(dtype=np.float32)


def _beam_search(
    horizontal_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    start_tvt: float,
    beam_size: int,
    movement_cost: float,
    error_scale: float,
    smoothing_radius: int,
) -> np.ndarray:
    start_index = _nearest_index(typewell_tvt, start_tvt)
    smoothed_gr = _smooth_gr(
        horizontal_gr,
        float(np.nanmean(typewell_gr)),
        smoothing_radius,
    ).astype(np.float64)
    path_indices = _beam_search_indices(
        smoothed_gr,
        typewell_gr.astype(np.float64),
        start_index,
        beam_size,
        float(movement_cost),
        float(error_scale),
    )
    return typewell_tvt[path_indices].astype(np.float32)


def _multi_scale_ncc(
    visible_gr: np.ndarray,
    visible_tvt: np.ndarray,
    hidden_gr: np.ndarray,
    half_windows: tuple[int, ...] = (8, 15, 25),
    stride: int = 3,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for half_window in half_windows:
        window = 2 * half_window + 1
        visible_count = len(visible_gr)
        hidden_count = len(hidden_gr)
        if visible_count < window + 1 or hidden_count == 0:
            fallback_tvt = np.full(hidden_count, visible_tvt[-1], dtype=np.float32)
            fallback_score = np.zeros(hidden_count, dtype=np.float32)
            results.append((fallback_tvt, fallback_score))
            continue

        smoothed_visible = (
            pd.Series(visible_gr)
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float32)
        )
        smoothed_hidden = (
            pd.Series(hidden_gr)
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float32)
        )
        starts = np.arange(0, visible_count - window + 1, stride, dtype=np.int32)
        if len(starts) == 0:
            fallback_tvt = np.full(hidden_count, visible_tvt[-1], dtype=np.float32)
            fallback_score = np.zeros(hidden_count, dtype=np.float32)
            results.append((fallback_tvt, fallback_score))
            continue

        candidate_windows = smoothed_visible[
            starts[:, None] + np.arange(window, dtype=np.int32)[None, :]
        ].astype(np.float32)
        normalized_candidates = (
            candidate_windows - candidate_windows.mean(axis=1, keepdims=True)
        ) / (candidate_windows.std(axis=1, keepdims=True) + 1e-6)
        padded_hidden = np.pad(smoothed_hidden, half_window, mode="edge")
        hidden_windows = padded_hidden[
            np.arange(hidden_count)[:, None] + np.arange(window)[None, :]
        ].astype(np.float32)
        normalized_hidden = (
            hidden_windows - hidden_windows.mean(axis=1, keepdims=True)
        ) / (hidden_windows.std(axis=1, keepdims=True) + 1e-6)
        correlations = normalized_hidden @ normalized_candidates.T / window
        best_indices = correlations.argmax(axis=1)
        best_scores = correlations.max(axis=1).astype(np.float32)
        best_tvt_indices = np.clip(
            starts[best_indices] + half_window,
            0,
            visible_count - 1,
        )
        results.append((visible_tvt[best_tvt_indices].astype(np.float32), best_scores))

    candidate_tvts = np.stack([result[0] for result in results], axis=1)
    candidate_scores = np.stack([result[1] for result in results], axis=1)
    score_weights = np.exp(3.0 * candidate_scores)
    score_weights /= score_weights.sum(axis=1, keepdims=True) + 1e-9
    ensemble = (candidate_tvts * score_weights).sum(axis=1).astype(np.float32)
    return results, ensemble


def _beam_delta_summaries(
    beam_paths: dict[str, np.ndarray],
    last_known_tvt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """先转成相对最后可见 TVT 的路径，再按 notebook 顺序做汇总。"""

    anchor = np.float32(last_known_tvt)
    delta_matrix = np.stack(
        [(path - anchor).astype(np.float32) for path in beam_paths.values()],
        axis=1,
    )
    mean_delta = delta_matrix.mean(axis=1).astype(np.float32)
    std_delta = delta_matrix.std(axis=1).astype(np.float32)
    median_delta = np.median(delta_matrix, axis=1).astype(np.float32)
    return delta_matrix, mean_delta, std_delta, median_delta


def _prepare_inputs(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    missing_horizontal = required_horizontal - set(horizontal_df.columns)
    missing_typewell = required_typewell - set(typewell_df.columns)
    if missing_horizontal:
        raise ValueError(f"水平井缺少列：{sorted(missing_horizontal)}")
    if missing_typewell:
        raise ValueError(f"Typewell 缺少列：{sorted(missing_typewell)}")

    horizontal = horizontal_df.copy()
    for column in required_horizontal:
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    visible_mask = horizontal["TVT_input"].notna().to_numpy()
    hidden_mask = ~visible_mask
    if int(visible_mask.sum()) < 10:
        raise ValueError("可见前缀少于 10 行，无法复现 notebook 候选")
    if not bool(hidden_mask.any()):
        raise ValueError("水平井没有 TVT_input 为空的隐藏行")
    first_hidden_position = int(np.flatnonzero(hidden_mask)[0])
    if visible_mask[first_hidden_position:].any():
        raise ValueError("TVT_input 必须是连续可见前缀，隐藏行必须位于井尾")
    if not np.isfinite(horizontal.loc[visible_mask, ["MD", "Z", "TVT_input"]]).all().all():
        raise ValueError("可见前缀的 MD、Z 或 TVT_input 含非有限值")
    if not np.isfinite(horizontal.loc[hidden_mask, ["MD", "Z"]]).all().all():
        raise ValueError("隐藏段的 MD 或 Z 含非有限值")

    typewell = typewell_df[["TVT", "GR"]].copy()
    typewell["TVT"] = pd.to_numeric(typewell["TVT"], errors="coerce")
    typewell["GR"] = pd.to_numeric(typewell["GR"], errors="coerce")
    typewell = typewell.dropna(subset=["TVT"]).sort_values("TVT").reset_index(drop=True)
    if len(typewell) < 3:
        raise ValueError("Typewell 有效 TVT 少于 3 行")
    typewell["GR"] = typewell["GR"].interpolate(limit_direction="both")
    if not np.isfinite(typewell[["TVT", "GR"]]).all().all():
        raise ValueError("Typewell 的 TVT 或 GR 无法得到有限值")
    return horizontal, typewell, visible_mask, hidden_mask


def build_candidate_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """返回隐藏行的 ``row_index`` 与固定顺序的 24 个合法候选特征。"""

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
    last_known_tvt = float(visible["TVT_input"].iloc[-1])

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
    hidden_raw_gr = hidden["GR"].to_numpy(dtype=np.float64)
    initial_structure_position = last_known_tvt + float(visible["Z"].iloc[-1])
    pf_ancc, pf_ancc_std = _particle_filter_ancc(
        hidden_md,
        hidden_z,
        hidden_raw_gr,
        typewell_gr_grid,
        grid_minimum,
        grid_step,
        gr_sigma,
        initial_structure_position,
        initial_structure_rate,
        ANCC_N,
        normalized_seed,
    )

    visible_z_change = np.diff(visible["Z"].to_numpy(dtype=np.float64))
    visible_tvt_change = np.diff(visible["TVT_input"].to_numpy(dtype=np.float64))
    visible_md_change = np.diff(visible["MD"].to_numpy(dtype=np.float64))
    valid_visible_steps = visible_md_change > 0
    if int(valid_visible_steps.sum()) >= 10:
        z_velocity = visible_z_change[valid_visible_steps] / visible_md_change[valid_visible_steps]
        tvt_velocity = (
            visible_tvt_change[valid_visible_steps] / visible_md_change[valid_visible_steps]
        )
        design = np.column_stack([z_velocity, np.ones_like(z_velocity)])
        coefficients = np.linalg.lstsq(design, tvt_velocity, rcond=None)[0]
        z_velocity_coefficient = float(coefficients[0])
        z_velocity_intercept = float(coefficients[1])
        z_velocity_residual = tvt_velocity - (
            z_velocity_coefficient * z_velocity + z_velocity_intercept
        )
        z_velocity_sigma = max(float(np.std(z_velocity_residual)), 0.001)
    else:
        z_velocity_coefficient = -1.0
        z_velocity_intercept = 0.0
        z_velocity_sigma = 0.1

    velocity_tail = visible.tail(20)
    velocity_tvt_change = np.diff(
        velocity_tail["TVT_input"].to_numpy(dtype=np.float64)
    )
    velocity_md_change = np.diff(velocity_tail["MD"].to_numpy(dtype=np.float64))
    valid_velocity_steps = velocity_md_change > 0
    if int(valid_velocity_steps.sum()) >= 3:
        initial_tvt_velocity = float(
            np.median(
                velocity_tvt_change[valid_velocity_steps]
                / velocity_md_change[valid_velocity_steps]
            )
        )
    else:
        initial_tvt_velocity = 0.0

    typewell_smoothed_gr = (
        pd.Series(typewell_gr)
        .rolling(PF_GR_WINDOW, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    typewell_smoothed_gr_grid, _, _ = _regular_typewell_grid(
        typewell_tvt,
        typewell_smoothed_gr,
    )
    horizontal_smoothed_gr = (
        horizontal["GR"]
        .rolling(PF_GR_WINDOW, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    hidden_smoothed_gr = horizontal_smoothed_gr[hidden_mask]
    pf_z, _pf_z_std = _particle_filter_z(
        hidden_md,
        hidden_z,
        hidden_raw_gr,
        hidden_smoothed_gr,
        typewell_gr_grid,
        typewell_smoothed_gr_grid,
        grid_minimum,
        grid_step,
        gr_sigma,
        last_known_tvt,
        initial_tvt_velocity,
        z_velocity_coefficient,
        z_velocity_intercept,
        z_velocity_sigma,
        PF_N,
        normalized_seed + 1,
    )

    gr_full = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(dtype=np.float32)
    )
    hidden_gr = gr_full[hidden_mask]
    visible_gr = gr_full[visible_mask]
    visible_tvt = visible["TVT_input"].to_numpy(dtype=np.float32)

    beam_paths: dict[str, np.ndarray] = {}
    for beam_size, movement_cost, error_scale, smoothing_radius, tag in BEAM_CONFIGS:
        beam_paths[tag] = _beam_search(
            hidden_gr,
            typewell_tvt,
            typewell_gr,
            last_known_tvt,
            beam_size,
            movement_cost,
            error_scale,
            smoothing_radius,
        )
    (
        beam_delta_matrix,
        beam_mean_delta,
        beam_std_delta,
        beam_median_delta,
    ) = _beam_delta_summaries(beam_paths, last_known_tvt)
    beam_reference = (beam_paths["cons"] + beam_paths["sm5"]) / 2.0

    ncc_results, ncc_ensemble = _multi_scale_ncc(
        visible_gr,
        visible_tvt,
        hidden_gr,
    )
    sc8, sc8_score = ncc_results[0]
    sc15, sc15_score = ncc_results[1]
    sc25, sc25_score = ncc_results[2]
    ncc_consensus = (sc8 + sc15 + sc25) / 3.0
    ncc_trust = float(np.clip(len(visible) / 200.0, 0.0, 0.6))
    hybrid_reference = (
        (1.0 - ncc_trust) * beam_reference + ncc_trust * ncc_ensemble
    )

    hidden_count = len(hidden)
    candidate_data: dict[str, np.ndarray] = {
        "pf_ancc_delta": (pf_ancc - last_known_tvt).astype(np.float32),
        "pf_ancc_std": pf_ancc_std.astype(np.float32),
        "pf_z_delta": (pf_z - last_known_tvt).astype(np.float32),
        "pf_vs_z": (pf_ancc - pf_z).astype(np.float32),
        "beam_cons_d": (beam_paths["cons"] - last_known_tvt).astype(np.float32),
        "beam_loose_d": (beam_paths["loose"] - last_known_tvt).astype(np.float32),
        "beam_vcons_d": (beam_paths["vcons"] - last_known_tvt).astype(np.float32),
        "beam_sm5_d": (beam_paths["sm5"] - last_known_tvt).astype(np.float32),
        "beam_vloose_d": (beam_paths["vloose"] - last_known_tvt).astype(np.float32),
        "beam_mid_d": (beam_paths["mid"] - last_known_tvt).astype(np.float32),
        "beam_stiff_d": (beam_paths["stiff"] - last_known_tvt).astype(np.float32),
        "beam_mean_d": beam_mean_delta,
        "beam_std_d": beam_std_delta,
        "beam_med_d": beam_median_delta,
        "sc8_d": (sc8 - last_known_tvt).astype(np.float32),
        "sc8_sc": sc8_score.astype(np.float32),
        "sc15_d": (sc15 - last_known_tvt).astype(np.float32),
        "sc15_sc": sc15_score.astype(np.float32),
        "sc25_d": (sc25 - last_known_tvt).astype(np.float32),
        "sc25_sc": sc25_score.astype(np.float32),
        "sc_cons_d": (ncc_consensus - last_known_tvt).astype(np.float32),
        "sc_ens_d": (ncc_ensemble - last_known_tvt).astype(np.float32),
        "sc_trust": np.full(hidden_count, ncc_trust, dtype=np.float32),
        "hyb_d": (hybrid_reference - last_known_tvt).astype(np.float32),
    }
    result = pd.DataFrame(
        {"row_index": hidden.index.to_numpy(dtype=np.int64), **candidate_data}
    )
    values = result[DIRECT_CANDIDATE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("候选复现产生了 NaN 或 Inf")
    return result[["row_index", *DIRECT_CANDIDATE_COLUMNS]]
