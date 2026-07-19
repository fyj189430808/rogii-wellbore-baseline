"""P2-P01：复刻旧 Notebook 的多随机种子粒子滤波路径。

本模块只使用当前井在测试期可获得的信息：MD、Z、GR、TVT_input，
以及配对 Typewell 的 TVT/GR。隐藏 TVT 不会被读取。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


FEATURE_COLUMNS = [
    "row_index",
    "last_visible_tvt",
    "pf128_mean_tvt",
    "pf128_mean_delta",
    "pf128_seed0_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
    "pf128_seed_std",
]


@njit(cache=True, nogil=True)
def _interpolate_regular_grid(
    grid: np.ndarray,
    value: float,
    grid_minimum: float,
    grid_step: float,
) -> float:
    """按 Notebook `_interp1` 的规则在线性等距网格上插值。"""

    left_index = int((value - grid_minimum) / grid_step)
    if left_index < 0:
        return grid[0]

    final_index = len(grid) - 1
    if left_index >= final_index:
        return grid[final_index]

    fractional_position = (value - grid_minimum) / grid_step - left_index
    return (
        grid[left_index] * (1.0 - fractional_position)
        + grid[left_index + 1] * fractional_position
    )


@njit(cache=True, nogil=True)
def particle_filter_all_seeds_numba(
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
) -> tuple[np.ndarray, np.ndarray]:
    """运行所有固定 seed；随机数调用顺序严格复刻 Notebook cell 37。

    返回：
    - `predictions`：形状 [seed 数, 隐藏行数]，每项是粒子加权 TVT；
    - `log_likelihoods`：形状 [seed 数]，每条完整路径的累计对数似然。
    """

    number_of_rows = len(md)
    predictions = np.empty((number_of_seeds, number_of_rows), dtype=np.float64)
    log_likelihoods = np.empty(number_of_seeds, dtype=np.float64)

    # 这里故意保留 Notebook 的定义：上限基准是 min + len(grid) * step。
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

        # 必须逐粒子交替抽位置和倾角，不能向量化，否则随机序列会改变。
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

            # 状态转移：先更新 U 沿 MD 的变化率，再把变化率积分到 U。
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
                lower_limit = (
                    typewell_min_tvt - position_limit_beyond_typewell_ft
                )
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
            path_log_likelihood += np.log(average_likelihood)

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

    return predictions, log_likelihoods


def _interpolate_regular_grid_reference(
    grid: np.ndarray,
    value: float,
    grid_minimum: float,
    grid_step: float,
) -> float:
    """Python 参考版插值；保留向零截断的下标规则。"""

    left_index = int((value - grid_minimum) / grid_step)
    if left_index < 0:
        return float(grid[0])
    final_index = len(grid) - 1
    if left_index >= final_index:
        return float(grid[final_index])
    fractional_position = (value - grid_minimum) / grid_step - left_index
    return float(
        grid[left_index] * (1.0 - fractional_position)
        + grid[left_index + 1] * fractional_position
    )


def particle_filter_all_seeds_reference(
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
) -> tuple[np.ndarray, np.ndarray]:
    """便于单元测试的直接循环版；公式和随机数调用顺序与生产内核相同。"""

    number_of_rows = len(md)
    predictions = np.empty((number_of_seeds, number_of_rows), dtype=np.float64)
    log_likelihoods = np.empty(number_of_seeds, dtype=np.float64)
    typewell_limit_maximum = (
        typewell_min_tvt + len(typewell_gr_grid) * typewell_step_ft
    )

    for seed_offset in range(number_of_seeds):
        random_state = np.random.RandomState(seed_base + seed_offset)
        particle_u_positions = np.empty(number_of_particles, dtype=np.float64)
        particle_rates = np.empty(number_of_particles, dtype=np.float64)
        particle_weights = (
            np.ones(number_of_particles, dtype=np.float64) / number_of_particles
        )

        for particle_index in range(number_of_particles):
            particle_u_positions[particle_index] = (
                initial_u + initial_position_spread_ft * random_state.randn()
            )
            particle_rates[particle_index] = (
                initial_rate + initial_rate_std * random_state.randn()
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
                    + rate_noise * random_state.randn()
                )
                particle_u_positions[particle_index] += (
                    particle_rates[particle_index] * md_change
                    + position_noise_ft * random_state.randn()
                )

                particle_tvt = particle_u_positions[particle_index] - z[row_index]
                lower_limit = (
                    typewell_min_tvt - position_limit_beyond_typewell_ft
                )
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
                expected_gr = _interpolate_regular_grid_reference(
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
                observation_likelihood = math.exp(-0.5 * squared_residual)
                if observation_likelihood < likelihood_floor:
                    observation_likelihood = likelihood_floor
                average_likelihood += (
                    particle_weights[particle_index] * observation_likelihood
                )
                particle_weights[particle_index] *= observation_likelihood

            if average_likelihood < likelihood_floor:
                average_likelihood = likelihood_floor
            path_log_likelihood += math.log(average_likelihood)

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
                first_systematic_position = random_state.uniform(
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
                        + resample_position_noise_ft * random_state.randn()
                    )
                    new_rates[particle_index] = (
                        particle_rates[source_index]
                        + resample_rate_noise * random_state.randn()
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

    return predictions, log_likelihoods


def _require_columns(table: pd.DataFrame, columns: list[str], table_name: str) -> None:
    """在进入数值代码前给出清楚的缺列错误。"""

    missing_columns = [column for column in columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"{table_name} 缺少列: {missing_columns}")


def _finite_float_array(values: Any, name: str) -> np.ndarray:
    """统一转为连续 float64，并阻止 NaN/无穷进入 PF 内核。"""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含 NaN 或无穷值")
    return array


def prepare_particle_filter_inputs(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """把一口井整理成 PF 输入；返回字典刻意不含隐藏 `TVT`。"""

    _require_columns(horizontal_well, ["MD", "Z", "GR", "TVT_input"], "水平井")
    _require_columns(typewell, ["TVT", "GR"], "Typewell")

    visible_mask = horizontal_well["TVT_input"].notna().to_numpy()
    hidden_mask = ~visible_mask
    visible_positions = np.flatnonzero(visible_mask)
    hidden_positions = np.flatnonzero(hidden_mask)
    if len(visible_positions) == 0:
        raise ValueError("水平井没有可见 TVT_input，无法初始化粒子")
    if len(hidden_positions) == 0:
        raise ValueError("水平井没有自然隐藏行，无法生成路径特征")

    # 比赛数据的可见段必须是前缀；若中间有洞，Notebook 的 last 语义会含糊。
    if visible_positions[-1] >= hidden_positions[0]:
        raise ValueError("TVT_input 可见行不是连续前缀")

    sorted_typewell = typewell.sort_values("TVT", kind="mergesort")
    typewell_tvt = sorted_typewell["TVT"].to_numpy(dtype=np.float64, copy=True)
    typewell_gr_series = sorted_typewell["GR"].astype(np.float64)
    typewell_gr_mean = float(typewell_gr_series.mean(skipna=True))
    if not np.isfinite(typewell_gr_mean):
        raise ValueError("Typewell GR 全部缺失")
    typewell_gr = typewell_gr_series.fillna(typewell_gr_mean).to_numpy(copy=True)
    typewell_tvt = _finite_float_array(typewell_tvt, "Typewell TVT")
    typewell_gr = _finite_float_array(typewell_gr, "Typewell GR")
    if len(typewell_tvt) < 2:
        raise ValueError("Typewell 至少需要两个 TVT/GR 点")
    if np.any(np.diff(typewell_tvt) <= 0.0):
        raise ValueError("Typewell TVT 必须严格递增且不能重复")

    grid_step = float(parameters["typewell_grid_step_ft"])
    if not np.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("typewell_grid_step_ft 必须为正数")
    typewell_min_tvt = float(typewell_tvt[0])
    typewell_grid_tvt = np.arange(
        typewell_min_tvt,
        float(typewell_tvt[-1]) + grid_step,
        grid_step,
        dtype=np.float64,
    )
    typewell_gr_grid = _finite_float_array(
        np.interp(typewell_grid_tvt, typewell_tvt, typewell_gr),
        "Typewell 等距 GR 网格",
    )

    visible_tvt = _finite_float_array(
        horizontal_well.loc[visible_mask, "TVT_input"].to_numpy(),
        "可见 TVT_input",
    )
    visible_z = _finite_float_array(
        horizontal_well.loc[visible_mask, "Z"].to_numpy(),
        "可见 Z",
    )
    visible_md = _finite_float_array(
        horizontal_well.loc[visible_mask, "MD"].to_numpy(),
        "可见 MD",
    )

    # 严格复刻 lik_pf：可见 GR 缺失填 0，再与 Typewell GR 算总体标准差。
    visible_gr = (
        horizontal_well.loc[visible_mask, "GR"]
        .astype(np.float64)
        .fillna(0.0)
        .to_numpy()
    )
    typewell_gr_at_visible_tvt = np.interp(
        visible_tvt, typewell_tvt, typewell_gr
    )
    residual_sigma = float(
        np.nanstd(visible_gr - typewell_gr_at_visible_tvt)
    )
    gr_sigma = float(
        np.clip(
            residual_sigma,
            float(parameters["gr_sigma_min_api"]),
            float(parameters["gr_sigma_max_api"]),
        )
    )
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        raise ValueError("由可见前缀得到的 GR sigma 非法")

    tail_rows = int(parameters["initial_rate_visible_tail_rows"])
    tail_start = max(0, len(visible_tvt) - tail_rows)
    tail_tvt = visible_tvt[tail_start:]
    tail_z = visible_z[tail_start:]
    tail_md = visible_md[tail_start:]
    tvt_change = np.diff(tail_tvt)
    z_change = np.diff(tail_z)
    md_change = np.diff(tail_md)
    valid_rate_change = md_change > 0.0
    if int(valid_rate_change.sum()) >= 3:
        initial_rate = float(
            np.median(
                (tvt_change[valid_rate_change] + z_change[valid_rate_change])
                / md_change[valid_rate_change]
            )
        )
    else:
        initial_rate = 0.0

    last_visible_position = int(visible_positions[-1])
    last_visible_tvt = float(visible_tvt[-1])
    initial_u = last_visible_tvt + float(visible_z[-1])

    # 和 Notebook 一样，先在整井行序上双向线性插值水平井 GR。
    interpolated_horizontal_gr = (
        horizontal_well["GR"]
        .astype(np.float64)
        .interpolate(limit_direction="both")
        .fillna(typewell_gr_mean)
        .to_numpy()
    )

    prepared: dict[str, Any] = {
        "md": _finite_float_array(
            horizontal_well.loc[hidden_mask, "MD"].to_numpy(), "隐藏段 MD"
        ),
        "z": _finite_float_array(
            horizontal_well.loc[hidden_mask, "Z"].to_numpy(), "隐藏段 Z"
        ),
        "horizontal_gr": _finite_float_array(
            interpolated_horizontal_gr[hidden_positions], "隐藏段插值 GR"
        ),
        "typewell_gr_grid": typewell_gr_grid,
        "typewell_min_tvt": typewell_min_tvt,
        "typewell_step_ft": grid_step,
        "gr_sigma": gr_sigma,
        "initial_u": initial_u,
        "initial_rate": initial_rate,
        "number_of_particles": int(parameters["number_of_particles"]),
        "number_of_seeds": int(parameters["number_of_seeds"]),
        "seed_base": int(parameters["seed_base"]),
        "rate_momentum": float(parameters["rate_momentum"]),
        "rate_noise": float(parameters["rate_noise"]),
        "position_noise_ft": float(parameters["position_noise_ft"]),
        "resample_position_noise_ft": float(
            parameters["resample_position_noise_ft"]
        ),
        "resample_rate_noise": float(parameters["resample_rate_noise"]),
        "resample_effective_fraction": float(
            parameters["resample_effective_fraction"]
        ),
        "initial_position_spread_ft": float(
            parameters["initial_position_spread_ft"]
        ),
        "position_limit_beyond_typewell_ft": float(
            parameters["position_limit_beyond_typewell_ft"]
        ),
        "initial_rate_std": float(parameters["initial_rate_std"]),
        "minimum_md_step_ft": float(parameters["minimum_md_step_ft"]),
        "squared_gr_residual_cap": float(
            parameters["squared_gr_residual_cap"]
        ),
        "likelihood_floor": float(parameters["likelihood_floor"]),
        "row_index": hidden_positions.astype(np.int64, copy=False),
        "last_visible_tvt": last_visible_tvt,
        "last_visible_position": last_visible_position,
    }
    return prepared


def _kernel_arguments(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """从准备结果中只取生产内核声明的参数。"""

    return {
        "md": prepared["md"],
        "z": prepared["z"],
        "horizontal_gr": prepared["horizontal_gr"],
        "typewell_gr_grid": prepared["typewell_gr_grid"],
        "typewell_min_tvt": prepared["typewell_min_tvt"],
        "typewell_step_ft": prepared["typewell_step_ft"],
        "gr_sigma": prepared["gr_sigma"],
        "initial_u": prepared["initial_u"],
        "initial_rate": prepared["initial_rate"],
        "number_of_particles": prepared["number_of_particles"],
        "number_of_seeds": prepared["number_of_seeds"],
        "seed_base": prepared["seed_base"],
        "rate_momentum": prepared["rate_momentum"],
        "rate_noise": prepared["rate_noise"],
        "position_noise_ft": prepared["position_noise_ft"],
        "resample_position_noise_ft": prepared["resample_position_noise_ft"],
        "resample_rate_noise": prepared["resample_rate_noise"],
        "resample_effective_fraction": prepared["resample_effective_fraction"],
        "initial_position_spread_ft": prepared["initial_position_spread_ft"],
        "position_limit_beyond_typewell_ft": prepared[
            "position_limit_beyond_typewell_ft"
        ],
        "initial_rate_std": prepared["initial_rate_std"],
        "minimum_md_step_ft": prepared["minimum_md_step_ft"],
        "squared_gr_residual_cap": prepared["squared_gr_residual_cap"],
        "likelihood_floor": prepared["likelihood_floor"],
    }


def _likelihood_weighted_path(
    seed_predictions: np.ndarray,
    seed_log_likelihoods: np.ndarray,
    scale: float,
) -> np.ndarray:
    """按 Notebook 的温度缩放公式，将多 seed 路径变成一条加权路径。"""

    centered_log_likelihoods = seed_log_likelihoods - np.max(
        seed_log_likelihoods
    )
    seed_weights = np.exp(centered_log_likelihoods / float(scale))
    seed_weights /= seed_weights.sum()
    return (seed_weights[:, None] * seed_predictions).sum(axis=0)


def build_multiseed_pf_features(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """生成正式均值特征、四条审计 scale 路径和三个井级质量量。"""

    prepared = prepare_particle_filter_inputs(horizontal_well, typewell, parameters)
    seed_predictions, seed_log_likelihoods = particle_filter_all_seeds_numba(
        **_kernel_arguments(prepared)
    )

    # 旧特征表先把绝对路径压成 float32，再用 float32 的末值计算差值。
    mean_tvt = seed_predictions.mean(axis=0).astype(np.float32)
    seed0_tvt = seed_predictions[0].astype(np.float32)
    seed_std = seed_predictions.std(axis=0).astype(np.float32)
    last_visible_tvt = np.float32(prepared["last_visible_tvt"])
    number_of_hidden_rows = len(prepared["row_index"])

    scale_paths: dict[float, np.ndarray] = {}
    for scale_value in parameters["likelihood_scales"]:
        numeric_scale = float(scale_value)
        scale_paths[numeric_scale] = _likelihood_weighted_path(
            seed_predictions, seed_log_likelihoods, numeric_scale
        ).astype(np.float32)

    required_scales = (3.0, 5.0, 8.0, 12.0)
    missing_scales = [scale for scale in required_scales if scale not in scale_paths]
    if missing_scales:
        raise ValueError(f"likelihood_scales 缺少冻结值: {missing_scales}")

    features = pd.DataFrame(
        {
            "row_index": prepared["row_index"],
            "last_visible_tvt": np.full(
                number_of_hidden_rows, last_visible_tvt, dtype=np.float32
            ),
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

    if not np.isfinite(features.drop(columns="row_index").to_numpy()).all():
        raise RuntimeError("PF 生成了 NaN 或无穷特征")

    quality = {
        "pf_best_ll_per_row": float(np.max(seed_log_likelihoods))
        / number_of_hidden_rows,
        "pf_ll_spread": float(np.std(seed_log_likelihoods)),
        "pf_gr_sigma": float(prepared["gr_sigma"]),
    }
    if not np.isfinite(np.asarray(list(quality.values()), dtype=np.float64)).all():
        raise RuntimeError("PF 生成了 NaN 或无穷质量指标")

    return features, quality
