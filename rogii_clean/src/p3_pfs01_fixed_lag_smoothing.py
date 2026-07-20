"""P3-PFS01：不修改旧 PF 的固定滞后粒子祖先平滑内核。

这里的“平滑”不是对一条均值曲线做移动平均，而是保留每个当前粒子的
祖先 ``U = TVT + Z``。等未来走过指定 MD 距离后，再用当前后验粒子权重
回看旧位置，因此未来 GR 证据能够修正较早位置。

本模块刻意不接收隐藏 TVT，也不读取文件；它只处理已经由
``prepare_particle_filter_inputs`` 整理好的测试期合法数组。
"""

from __future__ import annotations

import numpy as np
from numba import njit

from src.p2_p01_multiseed_pf import _interpolate_regular_grid


# 诊断矩阵固定列 0：每个 seed 实际发生系统重采样的次数。
DIAGNOSTIC_RESAMPLE_COUNT = 0

# 诊断矩阵固定列 1：重采样时被同步重排的活动历史“行”总数。
DIAGNOSTIC_HISTORY_ROWS_REORDERED = 1

# 诊断矩阵固定列 2：运行中同时保留的最大历史行数。
DIAGNOSTIC_MAX_ACTIVE_ROWS = 2

# 诊断矩阵固定列 3：至少复制过一个重复祖先的重采样次数。
DIAGNOSTIC_REPEATED_SOURCE_RESAMPLES = 3

# 固定诊断列之后依次是每个 lag 的平滑行数，再依次是每个 lag 的回退行数。
DIAGNOSTIC_FIXED_COLUMN_COUNT = 4


@njit(cache=True, nogil=True)
def _maximum_active_history_rows(
    md: np.ndarray,
    maximum_lag_ft: float,
) -> int:
    """按真实 MD 双指针计算环形历史缓冲区所需的最大行数。

    输入 ``md`` 的形状为 ``[H]``，单位 ft。返回值包含“当前刚写入、但尚未
    释放”的那一行，因此即使一次 MD 跳跃让很多旧行同时到期，也不会覆盖
    尚未用于平滑的祖先状态。
    """

    # ``oldest_active_row`` 是当前仍未达到最大 lag 的最早行号。
    oldest_active_row = 0

    # ``active_row_count`` 是处理完上一行后仍需保留的历史行数。
    active_row_count = 0

    # ``maximum_active_rows`` 是环形缓冲区最终容量，至少会被更新为 1。
    maximum_active_rows = 0

    for current_row in range(len(md)):
        # 当前行必须先进入历史，之后才能用它的后验粒子输出 lag=0。
        active_row_count += 1
        if active_row_count > maximum_active_rows:
            maximum_active_rows = active_row_count

        # 到达最大 lag 的旧行在本行完成平滑后即可释放；大步长可一次释放多行。
        while (
            oldest_active_row <= current_row
            and md[current_row] - md[oldest_active_row] >= maximum_lag_ft
        ):
            active_row_count -= 1
            oldest_active_row += 1

    return maximum_active_rows


@njit(cache=True, nogil=True)
def _particle_filter_fixed_lag_core_numba(
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
    lag_distances_ft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """运行冻结 PF，并同时生成各 lag 的精确祖先平滑路径与性能计数。"""

    # ``number_of_rows`` 记为 H，是当前井自然隐藏段的行数。
    number_of_rows = len(md)

    # ``number_of_lags`` 记为 L，是本次同时计算的固定滞后数量。
    number_of_lags = len(lag_distances_ft)

    if number_of_rows <= 0:
        raise ValueError("md 不能为空")
    if len(z) != number_of_rows or len(horizontal_gr) != number_of_rows:
        raise ValueError("md、z、horizontal_gr 的长度必须相同")
    if len(typewell_gr_grid) <= 0:
        raise ValueError("typewell_gr_grid 不能为空")
    if number_of_particles <= 0 or number_of_seeds <= 0:
        raise ValueError("粒子数和 seed 数必须为正数")
    if number_of_lags <= 0:
        raise ValueError("lag_distances_ft 不能为空")

    # MD 必须单调不减，否则“未来多少 ft”没有唯一含义，双指针也不再成立。
    for row_index in range(number_of_rows):
        if not np.isfinite(md[row_index]):
            raise ValueError("md 含 NaN 或无穷值")
        if row_index > 0 and md[row_index] < md[row_index - 1]:
            raise ValueError("md 必须单调不减")

    # ``maximum_lag_ft`` 决定祖先历史需要保留多远，单位 ft。
    maximum_lag_ft = 0.0
    for lag_index in range(number_of_lags):
        lag_distance_ft = lag_distances_ft[lag_index]
        if not np.isfinite(lag_distance_ft) or lag_distance_ft < 0.0:
            raise ValueError("lag 必须是有限且非负的 ft 距离")
        if lag_distance_ft > maximum_lag_ft:
            maximum_lag_ft = lag_distance_ft

    # ``history_capacity`` 记为 R，由 MD 而非固定行数计算。
    history_capacity = _maximum_active_history_rows(md, maximum_lag_ft)

    # ``filtered_paths`` 的形状为 [S,H]，保存旧 PF 的逐 seed 过滤输出，单位 ft TVT。
    filtered_paths = np.empty(
        (number_of_seeds, number_of_rows),
        dtype=np.float64,
    )

    # ``smoothed_paths`` 的形状为 [L,S,H]，保存每个 lag、seed、行的平滑 TVT。
    smoothed_paths = np.empty(
        (number_of_lags, number_of_seeds, number_of_rows),
        dtype=np.float64,
    )

    # ``final_log_likelihoods`` 的形状为 [S]，必须与旧 PF 的累计值逐位一致。
    final_log_likelihoods = np.empty(number_of_seeds, dtype=np.float64)

    # 诊断形状为 [S,4+2L]；后 2L 列分别记录平滑定稿数和尾段回退数。
    diagnostics = np.zeros(
        (
            number_of_seeds,
            DIAGNOSTIC_FIXED_COLUMN_COUNT + 2 * number_of_lags,
        ),
        dtype=np.int64,
    )

    # 保留旧 Notebook 的上界定义：min + len(grid) * step，而非末格坐标。
    typewell_limit_maximum = (
        typewell_min_tvt + len(typewell_gr_grid) * typewell_step_ft
    )

    for seed_offset in range(number_of_seeds):
        # 每个 seed 在入口重新播种；后续随机调用顺序与旧内核完全一致。
        np.random.seed(seed_base + seed_offset)

        # 当前粒子 U，形状 [P]，单位 ft；U = TVT + 当前行 Z。
        particle_u_positions = np.empty(number_of_particles, dtype=np.float64)

        # 当前粒子 dU/dMD，形状 [P]，单位 ft/ft。
        particle_rates = np.empty(number_of_particles, dtype=np.float64)

        # 当前归一化后验权重，形状 [P]，无量纲。
        particle_weights = (
            np.ones(number_of_particles, dtype=np.float64)
            / number_of_particles
        )

        # 必须逐粒子交替抽 U 和 rate，不能向量化或分成两轮抽样。
        for particle_index in range(number_of_particles):
            particle_u_positions[particle_index] = (
                initial_u + initial_position_spread_ft * np.random.randn()
            )
            particle_rates[particle_index] = (
                initial_rate + initial_rate_std * np.random.randn()
            )

        # ``history_u`` 的形状为 [R,P]，每个活动槽保存所有当前粒子的祖先 U。
        history_u = np.empty(
            (history_capacity, number_of_particles),
            dtype=np.float64,
        )

        # ``history_scratch`` 是重采样专用双缓冲，防止重复 source 时原地覆盖祖先。
        history_scratch = np.empty(
            (history_capacity, number_of_particles),
            dtype=np.float64,
        )

        # ``history_row_ids`` 将每个环形槽绑定到原始隐藏行号，避免槽复用错位。
        history_row_ids = np.full(history_capacity, -1, dtype=np.int64)

        # ``active_start_slot`` 指向环形缓冲区最早仍活动的历史槽。
        active_start_slot = 0

        # ``active_row_count`` 是尚未完成最大 lag 平滑、不可覆盖的连续历史行数。
        active_row_count = 0

        # 每个 lag 独立维护下一条尚未定稿的行号，形状 [L]。
        next_output_rows = np.zeros(number_of_lags, dtype=np.int64)

        # 以下四个计数只做性能与正确性审计，不参与路径数值。
        resample_count = 0
        history_rows_reordered_total = 0
        maximum_active_rows_observed = 0
        repeated_source_resample_count = 0

        # 这两个标量与旧 PF 同义、同初值，累计路径观测对数似然。
        path_log_likelihood = 0.0
        previous_md = md[0] - minimum_md_step_ft

        for row_index in range(number_of_rows):
            # 传播使用的 MD 步长单位为 ft；过小步长沿用旧内核下限。
            md_change = md[row_index] - previous_md
            if md_change < minimum_md_step_ft:
                md_change = minimum_md_step_ft

            # 1) 先传播。随机数调用顺序逐粒子保持 rate noise → position noise。
            for particle_index in range(number_of_particles):
                particle_rates[particle_index] = (
                    rate_momentum * particle_rates[particle_index]
                    + rate_noise * np.random.randn()
                )
                particle_u_positions[particle_index] += (
                    particle_rates[particle_index] * md_change
                    + position_noise_ft * np.random.randn()
                )

                # 传播后按旧规则把 TVT 截在 Typewell 边界外允许范围内。
                particle_tvt = (
                    particle_u_positions[particle_index] - z[row_index]
                )
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
                particle_u_positions[particle_index] = (
                    particle_tvt + z[row_index]
                )

            # 2) 再用当前 GR 更新似然；运算顺序与旧内核逐项一致。
            average_likelihood = 0.0
            for particle_index in range(number_of_particles):
                particle_tvt = (
                    particle_u_positions[particle_index] - z[row_index]
                )
                expected_gr = _interpolate_regular_grid(
                    typewell_gr_grid,
                    particle_tvt,
                    typewell_min_tvt,
                    typewell_step_ft,
                )
                standardized_residual = (
                    horizontal_gr[row_index] - expected_gr
                ) / gr_sigma
                squared_residual = (
                    standardized_residual * standardized_residual
                )
                if squared_residual > squared_gr_residual_cap:
                    squared_residual = squared_gr_residual_cap

                observation_likelihood = np.exp(-0.5 * squared_residual)
                if observation_likelihood < likelihood_floor:
                    observation_likelihood = likelihood_floor

                average_likelihood += (
                    particle_weights[particle_index]
                    * observation_likelihood
                )
                particle_weights[particle_index] *= observation_likelihood

            if average_likelihood < likelihood_floor:
                average_likelihood = likelihood_floor
            path_log_likelihood += np.log(average_likelihood)

            # 归一化权重；极端下溢时按旧规则恢复均匀权重。
            weight_sum = 0.0
            for particle_index in range(number_of_particles):
                weight_sum += particle_weights[particle_index]
            if weight_sum > 0.0:
                for particle_index in range(number_of_particles):
                    particle_weights[particle_index] /= weight_sum
            else:
                for particle_index in range(number_of_particles):
                    particle_weights[particle_index] = (
                        1.0 / number_of_particles
                    )

            # 按旧公式计算有效粒子数，决定是否系统重采样。
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
                resample_count += 1

                # 累计权重与系统起点的计算完全复刻旧 PF。
                cumulative_weights = np.empty(
                    number_of_particles,
                    dtype=np.float64,
                )
                cumulative_weight = 0.0
                for particle_index in range(number_of_particles):
                    cumulative_weight += particle_weights[particle_index]
                    cumulative_weights[particle_index] = cumulative_weight

                first_systematic_position = np.random.uniform(
                    0.0,
                    1.0 / number_of_particles,
                )

                # 新粒子数组和 source 索引形状都为 [P]。
                new_positions = np.empty(
                    number_of_particles,
                    dtype=np.float64,
                )
                new_rates = np.empty(
                    number_of_particles,
                    dtype=np.float64,
                )
                source_indices = np.empty(
                    number_of_particles,
                    dtype=np.int64,
                )
                source_index = 0
                has_repeated_source = False

                for particle_index in range(number_of_particles):
                    systematic_position = (
                        first_systematic_position
                        + particle_index / number_of_particles
                    )
                    while (
                        source_index < number_of_particles - 1
                        and cumulative_weights[source_index]
                        < systematic_position
                    ):
                        source_index += 1

                    source_indices[particle_index] = source_index
                    if (
                        particle_index > 0
                        and source_indices[particle_index]
                        == source_indices[particle_index - 1]
                    ):
                        has_repeated_source = True

                    # 两次 jitter 随机数仍按旧顺序紧邻抽取。
                    new_positions[particle_index] = (
                        particle_u_positions[source_index]
                        + resample_position_noise_ft * np.random.randn()
                    )
                    new_rates[particle_index] = (
                        particle_rates[source_index]
                        + resample_rate_noise * np.random.randn()
                    )

                if has_repeated_source:
                    repeated_source_resample_count += 1

                # 3) 用完整 source_indices 双缓冲重排“所有旧活动槽”。
                # 当前 row 尚未写入，所以不会错误保留 pre-resample 当前状态。
                for active_offset in range(active_row_count):
                    history_slot = (
                        active_start_slot + active_offset
                    ) % history_capacity
                    for particle_index in range(number_of_particles):
                        source_particle_index = source_indices[particle_index]
                        history_scratch[history_slot, particle_index] = (
                            history_u[
                                history_slot,
                                source_particle_index,
                            ]
                        )

                # 第二轮再写回，禁止重复 source 导致的原地覆盖污染。
                for active_offset in range(active_row_count):
                    history_slot = (
                        active_start_slot + active_offset
                    ) % history_capacity
                    for particle_index in range(number_of_particles):
                        history_u[history_slot, particle_index] = (
                            history_scratch[history_slot, particle_index]
                        )
                history_rows_reordered_total += active_row_count

                # 最后才替换当前粒子并重置权重，与旧 PF 顺序一致。
                for particle_index in range(number_of_particles):
                    particle_u_positions[particle_index] = (
                        new_positions[particle_index]
                    )
                    particle_rates[particle_index] = new_rates[particle_index]
                    particle_weights[particle_index] = (
                        1.0 / number_of_particles
                    )

            # 4) 当前槽写入 post-resample U；lag=0 因而与旧过滤输出一致。
            if active_row_count >= history_capacity:
                raise RuntimeError("活动历史超过预计算环形容量")
            current_history_slot = (
                active_start_slot + active_row_count
            ) % history_capacity
            history_row_ids[current_history_slot] = row_index
            for particle_index in range(number_of_particles):
                history_u[current_history_slot, particle_index] = (
                    particle_u_positions[particle_index]
                )
            active_row_count += 1
            if active_row_count > maximum_active_rows_observed:
                maximum_active_rows_observed = active_row_count

            # 旧 PF 的过滤 TVT：post-resample 粒子位置乘当前权重。
            estimated_tvt = 0.0
            for particle_index in range(number_of_particles):
                estimated_tvt += particle_weights[particle_index] * (
                    particle_u_positions[particle_index] - z[row_index]
                )
            filtered_paths[seed_offset, row_index] = estimated_tvt

            # 5) 每个 lag 可在一次大 MD 跨距中连续定稿多条旧行。
            for lag_index in range(number_of_lags):
                lag_distance_ft = lag_distances_ft[lag_index]
                output_row = next_output_rows[lag_index]
                while (
                    output_row <= row_index
                    and md[row_index] - md[output_row] >= lag_distance_ft
                ):
                    oldest_row_id = history_row_ids[active_start_slot]
                    history_offset = output_row - oldest_row_id
                    if history_offset < 0 or history_offset >= active_row_count:
                        raise RuntimeError("待输出祖先已离开活动历史")
                    output_history_slot = (
                        active_start_slot + history_offset
                    ) % history_capacity
                    if history_row_ids[output_history_slot] != output_row:
                        raise RuntimeError("环形历史槽与行号不一致")

                    smoothed_tvt = 0.0
                    for particle_index in range(number_of_particles):
                        smoothed_tvt += particle_weights[particle_index] * (
                            history_u[
                                output_history_slot,
                                particle_index,
                            ]
                            - z[output_row]
                        )
                    smoothed_paths[
                        lag_index,
                        seed_offset,
                        output_row,
                    ] = smoothed_tvt
                    diagnostics[
                        seed_offset,
                        DIAGNOSTIC_FIXED_COLUMN_COUNT + lag_index,
                    ] += 1

                    output_row += 1
                    next_output_rows[lag_index] = output_row

            # 最大 lag 已定稿的行现在才允许释放；小 lag 已输出的行仍保留到这里。
            while active_row_count > 0:
                oldest_row_id = history_row_ids[active_start_slot]
                if md[row_index] - md[oldest_row_id] < maximum_lag_ft:
                    break
                history_row_ids[active_start_slot] = -1
                active_start_slot = (
                    active_start_slot + 1
                ) % history_capacity
                active_row_count -= 1

            previous_md = md[row_index]

        # 没有足够未来 MD 距离的尾段，必须先逐 seed 精确回退旧过滤路径。
        for lag_index in range(number_of_lags):
            output_row = next_output_rows[lag_index]
            while output_row < number_of_rows:
                smoothed_paths[lag_index, seed_offset, output_row] = (
                    filtered_paths[seed_offset, output_row]
                )
                diagnostics[
                    seed_offset,
                    DIAGNOSTIC_FIXED_COLUMN_COUNT
                    + number_of_lags
                    + lag_index,
                ] += 1
                output_row += 1

        final_log_likelihoods[seed_offset] = path_log_likelihood
        diagnostics[seed_offset, DIAGNOSTIC_RESAMPLE_COUNT] = resample_count
        diagnostics[
            seed_offset,
            DIAGNOSTIC_HISTORY_ROWS_REORDERED,
        ] = history_rows_reordered_total
        diagnostics[
            seed_offset,
            DIAGNOSTIC_MAX_ACTIVE_ROWS,
        ] = maximum_active_rows_observed
        diagnostics[
            seed_offset,
            DIAGNOSTIC_REPEATED_SOURCE_RESAMPLES,
        ] = repeated_source_resample_count

    return (
        filtered_paths,
        smoothed_paths,
        final_log_likelihoods,
        diagnostics,
    )


@njit(cache=True, nogil=True)
def particle_filter_fixed_lag_all_seeds_numba(
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
    lag_distances_ft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """正式三输出接口：返回 filtered[S,H]、smoothed[L,S,H]、final_ll[S]。"""

    filtered_paths, smoothed_paths, final_ll, _ = (
        _particle_filter_fixed_lag_core_numba(
            md,
            z,
            horizontal_gr,
            typewell_gr_grid,
            typewell_min_tvt,
            typewell_step_ft,
            gr_sigma,
            initial_u,
            initial_rate,
            number_of_particles,
            number_of_seeds,
            seed_base,
            rate_momentum,
            rate_noise,
            position_noise_ft,
            resample_position_noise_ft,
            resample_rate_noise,
            resample_effective_fraction,
            initial_position_spread_ft,
            position_limit_beyond_typewell_ft,
            initial_rate_std,
            minimum_md_step_ft,
            squared_gr_residual_cap,
            likelihood_floor,
            lag_distances_ft,
        )
    )
    return filtered_paths, smoothed_paths, final_ll


@njit(cache=True, nogil=True)
def particle_filter_fixed_lag_all_seeds_with_diagnostics_numba(
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
    lag_distances_ft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """审计四输出接口；前三项与正式接口相同，第四项是逐 seed 性能计数。"""

    return _particle_filter_fixed_lag_core_numba(
        md,
        z,
        horizontal_gr,
        typewell_gr_grid,
        typewell_min_tvt,
        typewell_step_ft,
        gr_sigma,
        initial_u,
        initial_rate,
        number_of_particles,
        number_of_seeds,
        seed_base,
        rate_momentum,
        rate_noise,
        position_noise_ft,
        resample_position_noise_ft,
        resample_rate_noise,
        resample_effective_fraction,
        initial_position_spread_ft,
        position_limit_beyond_typewell_ft,
        initial_rate_std,
        minimum_md_step_ft,
        squared_gr_residual_cap,
        likelihood_floor,
        lag_distances_ft,
    )
