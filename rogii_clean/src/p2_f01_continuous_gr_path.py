"""P2-F01：把 PF 中心附近的五尺度 GR 得分面变成整段连续 offset 路径。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.p2_d01_pf_centered_gr import build_block_score_landscape


@dataclass(frozen=True)
class SecondOrderPathSolution:
    """保存二阶动态规划在控制块上的全局最优路径和诊断量。"""

    offset_path_ft: np.ndarray
    change_path_ft: np.ndarray
    selected_emission_cost: np.ndarray
    forward_backward_margin: np.ndarray
    total_path_cost: float


@dataclass(frozen=True)
class LegalContinuousPathResult:
    """保存完全不含隐藏真值的逐行路径、控制块和原始五尺度得分。"""

    row_path: pd.DataFrame
    block_path: pd.DataFrame
    offsets_ft: np.ndarray
    scale_scores: np.ndarray
    scale_pair_counts: np.ndarray
    row_to_block: np.ndarray


def build_emission_costs(
    scale_scores: np.ndarray,
    scale_pair_counts: np.ndarray,
    block_rows: np.ndarray,
    offsets_ft: np.ndarray,
    minimum_valid_scales: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把 `[尺度, 控制块, offset]` NCC 转成有限观测代价和支持度。"""

    # 五尺度 NCC，shape 为 [S, T, K]；S 是尺度数，T 是块数，K 是 offset 数。
    scores = np.asarray(scale_scores, dtype=np.float64)

    # 每个 NCC 使用的公共有效行数，shape 必须与 scores 完全相同。
    pair_counts = np.asarray(scale_pair_counts, dtype=np.float64)

    # 每个控制块的原始行数，shape 为 [T]。
    rows_per_block = np.asarray(block_rows, dtype=np.float64)

    # offset 网格，shape 为 [K]，单位 ft。
    offsets = np.asarray(offsets_ft, dtype=np.float64)

    if scores.ndim != 3:
        raise ValueError("scale_scores 必须为 [尺度, 控制块, offset] 三维数组")
    if pair_counts.shape != scores.shape:
        raise ValueError("scale_pair_counts 与 scale_scores shape 不一致")
    if rows_per_block.shape != (scores.shape[1],):
        raise ValueError("block_rows 必须与控制块数量一致")
    if offsets.shape != (scores.shape[2],):
        raise ValueError("offsets_ft 必须与 score 的 offset 维一致")
    if int(minimum_valid_scales) < 1 or int(minimum_valid_scales) > scores.shape[0]:
        raise ValueError("minimum_valid_scales 超出尺度数量")
    if not np.isfinite(rows_per_block).all() or np.any(rows_per_block <= 0.0):
        raise ValueError("block_rows 必须全部为正有限值")
    if not np.isfinite(offsets).all() or np.any(np.diff(offsets) <= 0.0):
        raise ValueError("offsets_ft 必须严格递增且有限")

    # valid 表示某尺度、某块、某 offset 是否真的算出了 NCC。
    valid = np.isfinite(scores)

    # 每个块和 offset 上有多少尺度可用，shape 从 [S,T,K] 降为 [T,K]。
    valid_scale_count = np.sum(valid, axis=0).astype(np.float64)

    # Pearson NCC 属于 [-1,1]；转成 [0,1] 损失，1 表示最差，0 表示完全一致。
    scale_loss = np.where(valid, (1.0 - np.clip(scores, -1.0, 1.0)) / 2.0, 0.0)

    # 只对有效尺度求均值；无有效尺度的位置先放 0，后面完全由回退代价接管。
    loss_sum = np.sum(scale_loss, axis=0)
    mean_loss = np.zeros_like(loss_sum)
    has_any_scale = valid_scale_count > 0.0
    mean_loss[has_any_scale] = loss_sum[has_any_scale] / valid_scale_count[has_any_scale]

    # 公共有效行比例衡量每个尺度的实际 GR 支撑，先裁到 [0,1]。
    block_denominator = rows_per_block[None, :, None]
    pair_fraction_by_scale = np.zeros_like(pair_counts)
    pair_fraction_by_scale[valid] = np.clip(
        pair_counts[valid] / np.broadcast_to(block_denominator, pair_counts.shape)[valid],
        0.0,
        1.0,
    )

    # 对有效尺度平均公共行比例，仍得到 [T,K]。
    pair_fraction_sum = np.sum(pair_fraction_by_scale, axis=0)
    mean_pair_fraction = np.zeros_like(pair_fraction_sum)
    mean_pair_fraction[has_any_scale] = (
        pair_fraction_sum[has_any_scale] / valid_scale_count[has_any_scale]
    )

    # 支持度同时考虑“每尺度有效行比例”和“五个尺度中有多少个有效”。
    valid_scale_fraction = valid_scale_count / float(scores.shape[0])
    support = np.clip(mean_pair_fraction * valid_scale_fraction, 0.0, 1.0)

    # 少于预注册尺度数时，把该状态视为无可靠观测，强制由 PF 回退代价接管。
    enough_scales = valid_scale_count >= float(minimum_valid_scales)
    support = np.where(enough_scales, support, 0.0)

    # 搜索范围最大绝对值是 40 ft；平方回退保证 GR 完全缺失时唯一最优为 0。
    maximum_absolute_offset = float(np.max(np.abs(offsets)))
    if maximum_absolute_offset <= 0.0:
        raise ValueError("offset 网格必须包含非零范围")
    fallback_cost = np.square(offsets / maximum_absolute_offset)[None, :]

    # GR 完整时主要使用 NCC，GR 缺失时连续、确定性地退回 PF 零 offset。
    emission_cost = support * mean_loss + (1.0 - support) * fallback_cost
    if not np.isfinite(emission_cost).all():
        raise ValueError("观测代价含 NaN 或 Inf")
    return emission_cost, support


def _find_exact_grid_position(grid: np.ndarray, value: float, name: str) -> int:
    """在固定 2 ft 网格中寻找一个参数的精确位置，拒绝静默取近邻。"""

    matches = np.flatnonzero(np.isclose(grid, float(value), atol=1e-9, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"{name}={value} 不在冻结网格中")
    return int(matches[0])


def solve_second_order_path(
    emission_cost: np.ndarray,
    offsets_ft: np.ndarray,
    allowed_changes_ft: np.ndarray,
    maximum_change_acceleration_ft: float,
    initial_offset_ft: float,
    initial_change_ft: float,
) -> SecondOrderPathSolution:
    """用 `(offset, change)` 状态求满足一阶和二阶硬约束的全局最小路径。"""

    # 每个控制块、每个 offset 的观测代价，shape 为 [T,K]。
    costs = np.asarray(emission_cost, dtype=np.float64)

    # offset 网格，shape 为 [K]，单位 ft。
    offsets = np.asarray(offsets_ft, dtype=np.float64)

    # 相邻 50 ft 可采用的 offset 变化量，shape 为 [V]，单位 ft。
    changes = np.asarray(allowed_changes_ft, dtype=np.float64)

    if costs.ndim != 2 or costs.shape[1] != len(offsets):
        raise ValueError("emission_cost 必须为 [控制块, offset]")
    if costs.shape[0] == 0:
        raise ValueError("至少需要一个控制块")
    if not np.isfinite(costs).all():
        raise ValueError("emission_cost 含 NaN 或 Inf")
    if not np.isfinite(offsets).all() or np.any(np.diff(offsets) <= 0.0):
        raise ValueError("offsets_ft 必须严格递增且有限")
    if not np.isfinite(changes).all() or np.any(np.diff(changes) <= 0.0):
        raise ValueError("allowed_changes_ft 必须严格递增且有限")
    if float(maximum_change_acceleration_ft) < 0.0:
        raise ValueError("maximum_change_acceleration_ft 不能为负")

    # 冻结初始 offset 和初始变化都必须存在于对应网格中。
    initial_offset_position = _find_exact_grid_position(
        offsets,
        float(initial_offset_ft),
        "initial_offset_ft",
    )
    initial_change_position = _find_exact_grid_position(
        changes,
        float(initial_change_ft),
        "initial_change_ft",
    )

    number_of_blocks, number_of_offsets = costs.shape
    number_of_changes = len(changes)

    # forward[t,k,v] 是从虚拟起点走到块 t 的状态 (offset_k, change_v) 的最小累计代价。
    forward = np.full(
        (number_of_blocks, number_of_offsets, number_of_changes),
        np.inf,
        dtype=np.float64,
    )

    # predecessor_change 只需记前一状态的 change；前一 offset 可由当前 offset-change 唯一还原。
    predecessor_change = np.full(
        (number_of_blocks, number_of_offsets, number_of_changes),
        -1,
        dtype=np.int16,
    )

    # 第一块必须能从虚拟状态 (0,0) 合法走到，避免井首凭弱 GR 突然跳层。
    for current_change_position, current_change in enumerate(changes):
        if abs(current_change - changes[initial_change_position]) > float(
            maximum_change_acceleration_ft
        ) + 1e-9:
            continue
        current_offset = offsets[initial_offset_position] + current_change
        try:
            current_offset_position = _find_exact_grid_position(
                offsets,
                current_offset,
                "首块 offset",
            )
        except ValueError:
            continue
        forward[0, current_offset_position, current_change_position] = costs[
            0,
            current_offset_position,
        ]

    # 从第二块开始，当前 offset 必须等于前一 offset 加当前 change。
    for block_position in range(1, number_of_blocks):
        for current_offset_position, current_offset in enumerate(offsets):
            for current_change_position, current_change in enumerate(changes):
                previous_offset = current_offset - current_change
                try:
                    previous_offset_position = _find_exact_grid_position(
                        offsets,
                        previous_offset,
                        "前一块 offset",
                    )
                except ValueError:
                    continue

                best_previous_cost = np.inf
                best_previous_change_position = -1
                for previous_change_position, previous_change in enumerate(changes):
                    if abs(current_change - previous_change) > float(
                        maximum_change_acceleration_ft
                    ) + 1e-9:
                        continue
                    candidate_cost = forward[
                        block_position - 1,
                        previous_offset_position,
                        previous_change_position,
                    ]
                    if candidate_cost < best_previous_cost:
                        best_previous_cost = float(candidate_cost)
                        best_previous_change_position = int(previous_change_position)

                if np.isfinite(best_previous_cost):
                    forward[
                        block_position,
                        current_offset_position,
                        current_change_position,
                    ] = best_previous_cost + costs[block_position, current_offset_position]
                    predecessor_change[
                        block_position,
                        current_offset_position,
                        current_change_position,
                    ] = best_previous_change_position

    if not np.isfinite(forward[-1]).any():
        raise ValueError("冻结连续性约束下不存在可达终点")

    # 终点不强行回到 0；直接选择使用整段证据后的全局最低累计代价状态。
    flat_end_position = int(np.argmin(forward[-1]))
    current_offset_position, current_change_position = np.unravel_index(
        flat_end_position,
        forward[-1].shape,
    )
    total_path_cost = float(forward[-1, current_offset_position, current_change_position])

    # 反向回溯得到一条满足全部硬约束的控制块路径。
    offset_positions = np.empty(number_of_blocks, dtype=np.int32)
    change_positions = np.empty(number_of_blocks, dtype=np.int32)
    offset_positions[-1] = int(current_offset_position)
    change_positions[-1] = int(current_change_position)
    for block_position in range(number_of_blocks - 1, 0, -1):
        previous_change_position = int(
            predecessor_change[
                block_position,
                current_offset_position,
                current_change_position,
            ]
        )
        if previous_change_position < 0:
            raise ValueError("回溯遇到缺失的前驱状态")
        previous_offset = offsets[current_offset_position] - changes[current_change_position]
        previous_offset_position = _find_exact_grid_position(
            offsets,
            previous_offset,
            "回溯前一块 offset",
        )
        current_offset_position = previous_offset_position
        current_change_position = previous_change_position
        offset_positions[block_position - 1] = int(current_offset_position)
        change_positions[block_position - 1] = int(current_change_position)

    # backward[t,k,v] 是已处于当前状态后，完成剩余所有块所需的最小未来代价。
    backward = np.full_like(forward, np.inf)
    backward[-1, :, :] = 0.0
    for block_position in range(number_of_blocks - 2, -1, -1):
        for current_offset_position, current_offset in enumerate(offsets):
            for current_change_position, current_change in enumerate(changes):
                best_future_cost = np.inf
                for next_change_position, next_change in enumerate(changes):
                    if abs(next_change - current_change) > float(
                        maximum_change_acceleration_ft
                    ) + 1e-9:
                        continue
                    next_offset = current_offset + next_change
                    try:
                        next_offset_position = _find_exact_grid_position(
                            offsets,
                            next_offset,
                            "后一块 offset",
                        )
                    except ValueError:
                        continue
                    candidate_future_cost = (
                        costs[block_position + 1, next_offset_position]
                        + backward[
                            block_position + 1,
                            next_offset_position,
                            next_change_position,
                        ]
                    )
                    if candidate_future_cost < best_future_cost:
                        best_future_cost = float(candidate_future_cost)
                backward[
                    block_position,
                    current_offset_position,
                    current_change_position,
                ] = best_future_cost

    # 前向加后向得到“全局路径经过当前状态”的总代价；按 offset 合并不同 change。
    through_state_cost = forward + backward
    through_offset_cost = np.min(through_state_cost, axis=2)
    forward_backward_margin = np.zeros(number_of_blocks, dtype=np.float64)
    for block_position in range(number_of_blocks):
        finite_costs = np.sort(
            through_offset_cost[block_position, np.isfinite(through_offset_cost[block_position])]
        )
        if len(finite_costs) >= 2:
            forward_backward_margin[block_position] = float(
                finite_costs[1] - finite_costs[0]
            )
        else:
            forward_backward_margin[block_position] = 0.0

    selected_emission_cost = costs[
        np.arange(number_of_blocks),
        offset_positions,
    ]
    return SecondOrderPathSolution(
        offset_path_ft=offsets[offset_positions].copy(),
        change_path_ft=changes[change_positions].copy(),
        selected_emission_cost=selected_emission_cost,
        forward_backward_margin=forward_backward_margin,
        total_path_cost=total_path_cost,
    )


def interpolate_block_values(
    row_md: np.ndarray,
    block_md_mid: np.ndarray,
    block_values: np.ndarray,
) -> np.ndarray:
    """把控制块中点上的值按 MD 线性插回逐行，首尾使用最近控制块值。"""

    rows = np.asarray(row_md, dtype=np.float64)
    block_md = np.asarray(block_md_mid, dtype=np.float64)
    values = np.asarray(block_values, dtype=np.float64)
    if rows.ndim != 1 or block_md.ndim != 1 or values.ndim != 1:
        raise ValueError("row_md、block_md_mid 和 block_values 必须为一维")
    if len(block_md) == 0 or len(block_md) != len(values):
        raise ValueError("控制块坐标和值数量不一致")
    if not np.isfinite(rows).all() or not np.isfinite(block_md).all():
        raise ValueError("MD 坐标含 NaN 或 Inf")
    if not np.isfinite(values).all():
        raise ValueError("控制块值含 NaN 或 Inf")
    if np.any(np.diff(rows) < 0.0) or np.any(np.diff(block_md) <= 0.0):
        raise ValueError("逐行 MD 必须非降，控制块中点必须严格递增")
    if len(block_md) == 1:
        return np.full(len(rows), float(values[0]), dtype=np.float64)
    return np.interp(rows, block_md, values)


def build_legal_continuous_gr_path(
    row_index: np.ndarray,
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    block_width_ft: float,
    smoothing_widths_ft: list[float],
    minimum_valid_pairs: int,
    minimum_valid_scales: int,
    allowed_changes_ft: np.ndarray,
    maximum_change_acceleration_ft: float,
    initial_offset_ft: float,
    initial_change_ft: float,
) -> LegalContinuousPathResult:
    """只用测试期合法数组生成连续路径，并保留不含 oracle 的完整 score 张量。"""

    row_keys = np.asarray(row_index, dtype=np.int64)
    row_md = np.asarray(md, dtype=np.float64)
    row_gr = np.asarray(horizontal_gr, dtype=np.float64)
    center = np.asarray(center_tvt, dtype=np.float64)
    if not (len(row_keys) == len(row_md) == len(row_gr) == len(center)):
        raise ValueError("row_index、MD、GR 和中心路径长度必须一致")
    if len(row_keys) == 0 or len(np.unique(row_keys)) != len(row_keys):
        raise ValueError("row_index 必须非空且唯一")

    # D01 scorer 只接收合法数组；返回的原始前五层分别对应五个冻结平滑尺度。
    landscape = build_block_score_landscape(
        md=row_md,
        horizontal_gr=row_gr,
        center_tvt=center,
        typewell_tvt=np.asarray(typewell_tvt, dtype=np.float64),
        typewell_gr=np.asarray(typewell_gr, dtype=np.float64),
        offsets_ft=np.asarray(offsets_ft, dtype=np.float64),
        block_width_ft=float(block_width_ft),
        smoothing_widths_ft=[float(width) for width in smoothing_widths_ft],
        minimum_valid_pairs=int(minimum_valid_pairs),
        second_peak_minimum_distance_ft=6.0,
    )
    number_of_scales = len(smoothing_widths_ft)
    scale_scores = landscape.scores[:number_of_scales].copy()
    scale_pair_counts = landscape.pair_counts[:number_of_scales].copy()
    block_rows = landscape.block_table["block_rows"].to_numpy(dtype=np.float64)

    emission_cost, support = build_emission_costs(
        scale_scores=scale_scores,
        scale_pair_counts=scale_pair_counts,
        block_rows=block_rows,
        offsets_ft=landscape.offsets_ft,
        minimum_valid_scales=int(minimum_valid_scales),
    )
    solution = solve_second_order_path(
        emission_cost=emission_cost,
        offsets_ft=landscape.offsets_ft,
        allowed_changes_ft=np.asarray(allowed_changes_ft, dtype=np.float64),
        maximum_change_acceleration_ft=float(maximum_change_acceleration_ft),
        initial_offset_ft=float(initial_offset_ft),
        initial_change_ft=float(initial_change_ft),
    )

    # 每块独立最小代价不使用连续性，是预注册的局部负对照。
    independent_offset_positions = np.argmin(emission_cost, axis=1)
    independent_offsets = landscape.offsets_ft[independent_offset_positions]

    # 取全局路径实际经过的 offset 位置，读取对应支持度。
    selected_offset_positions = np.array(
        [
            _find_exact_grid_position(landscape.offsets_ft, value, "路径 offset")
            for value in solution.offset_path_ft
        ],
        dtype=np.int32,
    )
    selected_support = support[
        np.arange(len(selected_offset_positions)),
        selected_offset_positions,
    ]
    missing_fallback = (np.max(support, axis=1) <= 0.0).astype(np.float64)
    block_md_mid = landscape.block_table["md_mid"].to_numpy(dtype=np.float64)

    # 将控制块路径连续插回每一个自然隐藏行。
    row_offset = interpolate_block_values(row_md, block_md_mid, solution.offset_path_ft)
    row_change = interpolate_block_values(row_md, block_md_mid, solution.change_path_ft)
    row_independent_offset = interpolate_block_values(
        row_md,
        block_md_mid,
        independent_offsets,
    )
    row_local_cost = interpolate_block_values(
        row_md,
        block_md_mid,
        solution.selected_emission_cost,
    )
    row_support = interpolate_block_values(row_md, block_md_mid, selected_support)
    row_margin = interpolate_block_values(
        row_md,
        block_md_mid,
        solution.forward_backward_margin,
    )
    row_missing_fallback = interpolate_block_values(
        row_md,
        block_md_mid,
        missing_fallback,
    )

    row_path = pd.DataFrame(
        {
            "row_index": row_keys,
            "f01_offset_path": row_offset,
            "f01_corrected_tvt": center + row_offset,
            "f01_offset_change": row_change,
            "f01_independent_offset": row_independent_offset,
            "f01_independent_tvt": center + row_independent_offset,
            "f01_local_gr_cost": row_local_cost,
            "f01_support_fraction": row_support,
            "f01_forward_backward_margin": row_margin,
            "f01_missing_fallback": row_missing_fallback,
        }
    )
    block_path = pd.DataFrame(
        {
            "block_position": np.arange(len(block_md_mid), dtype=np.int32),
            "block_md_mid": block_md_mid,
            "f01_offset_path": solution.offset_path_ft,
            "f01_offset_change": solution.change_path_ft,
            "f01_independent_offset": independent_offsets,
            "f01_local_gr_cost": solution.selected_emission_cost,
            "f01_support_fraction": selected_support,
            "f01_forward_backward_margin": solution.forward_backward_margin,
            "f01_missing_fallback": missing_fallback,
        }
    )
    return LegalContinuousPathResult(
        row_path=row_path,
        block_path=block_path,
        offsets_ft=landscape.offsets_ft.copy(),
        scale_scores=scale_scores,
        scale_pair_counts=scale_pair_counts,
        row_to_block=landscape.block_index.copy(),
    )
