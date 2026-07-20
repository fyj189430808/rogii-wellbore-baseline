"""P3-MDP01：只用合法 GR 与四条冻结候选生成离散动态路径。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np
import pandas as pd

from src.f01_features import fit_visible_u_trend
from src.p2_f01b_path_domain_smoothing import (
    _centered_nanmean_columns,
    _centered_window_bounds,
)
from src.p3_pf03_segmented_likelihood import _window_centers
from src.rf03_prefix_alignment import clean_typewell_arrays, fit_affine_reference


CANDIDATE_NAMES = ("P2", "low", "middle", "high")


@dataclass(frozen=True)
class DynamicPathParameters:
    """MDP01 v1 的全部冻结数值，不由 smoke 或验证分数改写。"""

    window_ft: float = 250.0
    center_step_ft: float = 125.0
    smoothing_widths_ft: tuple[float, ...] = (21.0, 51.0, 101.0)
    minimum_prefix_pairs: int = 50
    fallback_prefix_pairs: int = 10
    minimum_level_pairs: int = 30
    minimum_derivative_pairs: int = 20
    minimum_observed_rows: int = 50
    maximum_derivative_gap_ft: float = 5.0
    sigma_level_min: float = 10.0
    sigma_level_max: float = 60.0
    sigma_d1_min: float = 0.25
    sigma_d1_max: float = 10.0
    u_slope_scale: float = 0.02
    typewell_range_scale_ft: float = 25.0
    level_weights: tuple[float, ...] = (0.25, 0.15, 0.10)
    derivative_weight: float = 0.15
    correlation_weight: float = 0.20
    range_weight: float = 0.10
    u_slope_weight: float = 0.05
    mode_prior: float = 0.10
    switch_penalty: float = 0.25
    jump_weight: float = 0.20
    jump_scale_ft: float = 10.0
    second_order_weight: float = 0.15
    second_order_scale_ft: float = 5.0
    minimum_non_p2_run_blocks: int = 3


DEFAULT_PARAMETERS = DynamicPathParameters()


@dataclass(frozen=True)
class PrefixCalibration:
    slope: float
    intercept: float
    sigma_level: float
    sigma_d1: float
    visible_pair_count: int
    fallback: str


@dataclass(frozen=True)
class DynamicStateSolution:
    states: np.ndarray
    margin: np.ndarray
    objective: float


@dataclass(frozen=True)
class DynamicModePathResult:
    row_output: pd.DataFrame
    block_output: pd.DataFrame
    audit: dict[str, object]


def _mad_scale(values: np.ndarray, lower: float, upper: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(lower)
    center = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - center)))
    return float(np.clip(scale, lower, upper))


def _linear_slope(md: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(md, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return 0.0
    x = x[finite]
    y = y[finite]
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(centered, y - float(np.mean(y))) / denominator)


def _cap3(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(np.clip(value, 0.0, 3.0) / 3.0)


def fit_prefix_calibration(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: DynamicPathParameters = DEFAULT_PARAMETERS,
) -> PrefixCalibration:
    """仅在可见 TVT_input 与原始 GR 的公共行拟合 GR 仿射与尺度。"""

    required = {"MD", "GR", "TVT_input"}
    if missing := required.difference(horizontal.columns):
        raise ValueError(f"水平井缺少列：{sorted(missing)}")
    typewell_tvt, typewell_gr = clean_typewell_arrays(typewell)
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float64)
    visible_tvt = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    common = (
        np.isfinite(md)
        & np.isfinite(gr)
        & np.isfinite(visible_tvt)
        & (visible_tvt >= typewell_tvt[0])
        & (visible_tvt <= typewell_tvt[-1])
    )
    reference = np.interp(visible_tvt[common], typewell_tvt, typewell_gr)
    observed = gr[common]
    pair_count = int(len(observed))
    if (
        pair_count >= parameters.minimum_prefix_pairs
        and float(np.std(reference)) > 1e-12
    ):
        slope, intercept = fit_affine_reference(observed, reference)
        fallback = "none"
    elif pair_count >= parameters.fallback_prefix_pairs:
        slope = 1.0
        intercept = float(np.median(observed - reference))
        fallback = "offset_only"
    else:
        slope, intercept = 1.0, 0.0
        fallback = "identity"
    residual = observed - (float(slope) * reference + float(intercept))
    sigma_level = _mad_scale(
        residual,
        parameters.sigma_level_min,
        parameters.sigma_level_max,
    )
    if pair_count >= 2:
        common_md = md[common]
        steps = np.diff(common_md)
        adjacent = (steps > 0.0) & (steps <= parameters.maximum_derivative_gap_ft)
        observed_d1 = np.diff(observed)[adjacent] / steps[adjacent]
        reference_d1 = np.diff(float(slope) * reference + float(intercept))[adjacent] / steps[
            adjacent
        ]
        derivative_residual = observed_d1 - reference_d1
    else:
        derivative_residual = np.empty(0, dtype=np.float64)
    sigma_d1 = _mad_scale(
        derivative_residual,
        parameters.sigma_d1_min,
        parameters.sigma_d1_max,
    )
    return PrefixCalibration(
        slope=float(slope),
        intercept=float(intercept),
        sigma_level=sigma_level,
        sigma_d1=sigma_d1,
        visible_pair_count=pair_count,
        fallback=fallback,
    )


def circular_shift_hidden_gr(hidden_gr: np.ndarray) -> tuple[np.ndarray, int]:
    """把隐藏 GR 数值与 NaN 掩码一起固定循环平移半段。"""

    values = np.asarray(hidden_gr, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("隐藏 GR 必须是一维非空数组")
    shift_rows = int(values.size // 2)
    if values.size > 1 and shift_rows == 0:
        shift_rows = 1
    return np.roll(values, shift_rows), shift_rows


def permute_block_costs(
    local_cost: np.ndarray,
    well_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """按实验号和井号确定性打乱块成本，且多块时拒绝恒等排列。"""

    costs = np.asarray(local_cost, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[1] != len(CANDIDATE_NAMES):
        raise ValueError("local_cost 必须为 [块,4]")
    digest = hashlib.sha256(
        f"P3_MDP01_dynamic_mode_path_v1:{well_id}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    order = np.random.default_rng(seed).permutation(costs.shape[0])
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    return costs[order].copy(), order.astype(np.int64)


def _robust_level_cost(
    horizontal: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    minimum_pairs: int,
) -> float:
    valid = mask & np.isfinite(horizontal) & np.isfinite(reference)
    if int(valid.sum()) < int(minimum_pairs):
        return 0.5
    return _cap3(float(np.median(np.abs(horizontal[valid] - reference[valid]))) / sigma)


def _pearson_cost(
    horizontal: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    minimum_pairs: int,
) -> float:
    valid = mask & np.isfinite(horizontal) & np.isfinite(reference)
    if int(valid.sum()) < int(minimum_pairs):
        return 0.5
    first = horizontal[valid] - float(np.mean(horizontal[valid]))
    second = reference[valid] - float(np.mean(reference[valid]))
    denominator = float(np.sqrt(np.dot(first, first) * np.dot(second, second)))
    if denominator <= 1e-12:
        return 0.5
    rho = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return (1.0 - rho) / 2.0


def _derivative_cost(
    md: np.ndarray,
    horizontal: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    parameters: DynamicPathParameters,
) -> float:
    positions = np.flatnonzero(mask & np.isfinite(horizontal) & np.isfinite(reference))
    if len(positions) < 2:
        return 0.5
    steps = np.diff(md[positions])
    adjacent = (steps > 0.0) & (steps <= parameters.maximum_derivative_gap_ft)
    if int(adjacent.sum()) < parameters.minimum_derivative_pairs:
        return 0.5
    horizontal_d1 = np.diff(horizontal[positions])[adjacent] / steps[adjacent]
    reference_d1 = np.diff(reference[positions])[adjacent] / steps[adjacent]
    return _cap3(float(np.median(np.abs(horizontal_d1 - reference_d1))) / sigma)


def build_local_candidate_costs(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    candidate_tvt: np.ndarray,
    typewell: pd.DataFrame,
    calibration: PrefixCalibration,
    prefix_u_slope: float,
    parameters: DynamicPathParameters = DEFAULT_PARAMETERS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """构造 250/125 重叠块上的四候选固定稳健成本。"""

    row_md = np.asarray(md, dtype=np.float64)
    row_z = np.asarray(z, dtype=np.float64)
    row_gr = np.asarray(gr, dtype=np.float64)
    candidates = np.asarray(candidate_tvt, dtype=np.float64)
    if candidates.shape != (len(row_md), len(CANDIDATE_NAMES)):
        raise ValueError("candidate_tvt 必须为 [隐藏行,4]")
    if row_z.shape != row_md.shape or row_gr.shape != row_md.shape:
        raise ValueError("MD/Z/GR shape 不一致")
    if len(row_md) == 0 or not np.isfinite(row_md).all() or np.any(np.diff(row_md) < 0.0):
        raise ValueError("隐藏 MD 必须非空、有限且单调")
    if not np.isfinite(candidates).all():
        raise ValueError("四候选路径含 NaN/Inf")
    typewell_tvt, typewell_gr = clean_typewell_arrays(typewell)
    sampled_reference = np.interp(
        candidates.ravel(),
        typewell_tvt,
        typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidates.shape)
    sampled_reference = (
        calibration.slope * sampled_reference + calibration.intercept
    )
    raw_observed = np.isfinite(row_gr)
    smooth_horizontal: dict[float, np.ndarray] = {}
    smooth_reference: dict[float, np.ndarray] = {}
    for width in parameters.smoothing_widths_ft:
        left, right = _centered_window_bounds(row_md, float(width))
        smooth_horizontal[width] = _centered_nanmean_columns(
            row_gr[:, None], left, right
        )[:, 0]
        smooth_reference[width] = _centered_nanmean_columns(
            sampled_reference, left, right
        )
    centers = _window_centers(row_md, parameters.center_step_ft)
    local_cost = np.zeros((len(centers), len(CANDIDATE_NAMES)), dtype=np.float64)
    valid_ratio = np.zeros_like(local_cost)
    components: dict[str, np.ndarray] = {
        name: np.zeros_like(local_cost)
        for name in ("E21", "E51", "E101", "D1", "Corr", "Range", "USlope")
    }
    half_window = parameters.window_ft / 2.0
    for block_index, center in enumerate(centers):
        in_block = (row_md >= center - half_window) & (row_md <= center + half_window)
        block_rows = int(in_block.sum())
        observed_in_block = in_block & raw_observed
        if int(observed_in_block.sum()) == 0:
            continue
        for state in range(len(CANDIDATE_NAMES)):
            valid_raw = observed_in_block & np.isfinite(sampled_reference[:, state])
            valid_count = int(valid_raw.sum())
            q = min(1.0, valid_count / parameters.minimum_observed_rows) * (
                valid_count / block_rows
            )
            valid_ratio[block_index, state] = q
            level_costs: list[float] = []
            for label, width in zip(
                ("E21", "E51", "E101"), parameters.smoothing_widths_ft
            ):
                value = _robust_level_cost(
                    smooth_horizontal[width],
                    smooth_reference[width][:, state],
                    valid_raw,
                    calibration.sigma_level,
                    parameters.minimum_level_pairs,
                )
                components[label][block_index, state] = value
                level_costs.append(value)
            width21 = parameters.smoothing_widths_ft[0]
            derivative = _derivative_cost(
                row_md,
                smooth_horizontal[width21],
                smooth_reference[width21][:, state],
                valid_raw,
                calibration.sigma_d1,
                parameters,
            )
            correlation = _pearson_cost(
                smooth_horizontal[width21],
                smooth_reference[width21][:, state],
                valid_raw,
                parameters.minimum_level_pairs,
            )
            components["D1"][block_index, state] = derivative
            components["Corr"][block_index, state] = correlation
            block_candidate = candidates[in_block, state]
            below = np.maximum(typewell_tvt[0] - block_candidate, 0.0)
            above = np.maximum(block_candidate - typewell_tvt[-1], 0.0)
            distance = below + above
            out = distance > 0.0
            range_cost = 0.5 * float(np.mean(out))
            if np.any(out):
                range_cost += 0.5 * _cap3(
                    float(np.median(distance[out])) / parameters.typewell_range_scale_ft
                )
            candidate_u_slope = _linear_slope(
                row_md[in_block], block_candidate + row_z[in_block]
            )
            u_slope_cost = _cap3(
                abs(candidate_u_slope - float(prefix_u_slope))
                / parameters.u_slope_scale
            )
            components["Range"][block_index, state] = range_cost
            components["USlope"][block_index, state] = u_slope_cost
            observation = q * (
                float(np.dot(parameters.level_weights, level_costs))
                + parameters.derivative_weight * derivative
                + parameters.correlation_weight * correlation
            )
            local_cost[block_index, state] = (
                observation
                + parameters.range_weight * range_cost
                + parameters.u_slope_weight * u_slope_cost
            )
    components["valid_ratio"] = valid_ratio
    return centers, local_cost, components


def _transition_jump(
    candidate_center_tvt: np.ndarray,
    block_index: int,
    previous_state: int,
    current_state: int,
) -> float:
    if previous_state == current_state:
        return 0.0
    previous_mid = 0.5 * (
        candidate_center_tvt[block_index - 1, previous_state]
        + candidate_center_tvt[block_index, previous_state]
    )
    current_mid = 0.5 * (
        candidate_center_tvt[block_index - 1, current_state]
        + candidate_center_tvt[block_index, current_state]
    )
    return abs(float(current_mid - previous_mid))


def _solve_dynamic_once(
    local_cost: np.ndarray,
    candidate_center_tvt: np.ndarray,
    center_z: np.ndarray,
    parameters: DynamicPathParameters,
    forced: Mapping[int, int] | None = None,
) -> tuple[np.ndarray | None, float]:
    costs = np.asarray(local_cost, dtype=np.float64)
    paths = np.asarray(candidate_center_tvt, dtype=np.float64)
    z = np.asarray(center_z, dtype=np.float64)
    block_count = costs.shape[0]
    forced_states = dict(forced or {})
    records: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}
    for state in range(len(CANDIDATE_NAMES)):
        if 0 in forced_states and forced_states[0] != state:
            continue
        objective = float(costs[0, state])
        objective += parameters.mode_prior if state != 0 else 0.0
        objective += parameters.switch_penalty if state != 0 else 0.0
        records[(0, state, 1)] = (objective, (state,))
    for block_index in range(1, block_count):
        updated: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}
        for (previous_previous, previous, run_length), (base, history) in records.items():
            for state in range(len(CANDIDATE_NAMES)):
                if block_index in forced_states and forced_states[block_index] != state:
                    continue
                if (
                    previous != 0
                    and state != previous
                    and run_length < parameters.minimum_non_p2_run_blocks
                ):
                    continue
                new_run = min(
                    parameters.minimum_non_p2_run_blocks,
                    run_length + 1 if state == previous else 1,
                )
                objective = base + float(costs[block_index, state])
                objective += parameters.mode_prior if state != 0 else 0.0
                if state != previous:
                    objective += parameters.switch_penalty
                jump = _transition_jump(paths, block_index, previous, state)
                objective += parameters.jump_weight * _cap3(
                    jump / parameters.jump_scale_ft
                )
                if block_index >= 2:
                    u_previous_previous = (
                        paths[block_index - 2, previous_previous] + z[block_index - 2]
                    )
                    u_previous = paths[block_index - 1, previous] + z[block_index - 1]
                    u_current = paths[block_index, state] + z[block_index]
                    acceleration = abs(
                        float((u_current - u_previous) - (u_previous - u_previous_previous))
                    )
                    objective += parameters.second_order_weight * _cap3(
                        acceleration / parameters.second_order_scale_ft
                    )
                key = (previous, state, new_run)
                candidate = (objective, history + (state,))
                incumbent = updated.get(key)
                if incumbent is None or candidate < incumbent:
                    updated[key] = candidate
        records = updated
        if not records:
            return None, float("inf")
    feasible: list[tuple[float, tuple[int, ...]]] = []
    for (_, state, run_length), record in records.items():
        if state != 0 and run_length < parameters.minimum_non_p2_run_blocks:
            continue
        feasible.append(record)
    if not feasible:
        return None, float("inf")
    objective, history = min(feasible)
    return np.asarray(history, dtype=np.int64), float(objective)


def solve_dynamic_states(
    local_cost: np.ndarray,
    centers_md: np.ndarray,
    candidate_center_tvt: np.ndarray,
    center_z: np.ndarray,
    parameters: DynamicPathParameters = DEFAULT_PARAMETERS,
) -> DynamicStateSolution:
    """求固定四状态 semi-Markov 二阶最优路径；首轮暂不计算全局 margin。"""

    costs = np.asarray(local_cost, dtype=np.float64)
    centers = np.asarray(centers_md, dtype=np.float64)
    paths = np.asarray(candidate_center_tvt, dtype=np.float64)
    z = np.asarray(center_z, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[1] != len(CANDIDATE_NAMES):
        raise ValueError("local_cost 必须为 [块,4]")
    if paths.shape != costs.shape or centers.shape != (len(costs),) or z.shape != (
        len(costs),
    ):
        raise ValueError("块中心、候选中心路径或 Z shape 不一致")
    if len(costs) == 0 or not np.isfinite(costs).all() or not np.isfinite(paths).all():
        raise ValueError("DP 输入必须非空有限")
    states, objective = _solve_dynamic_once(costs, paths, z, parameters)
    if states is None:
        raise RuntimeError("冻结约束下不存在合法动态路径")
    # 精确全局 min-marginal 需要逐块约束重跑 DP。首轮路径评分不使用该量，
    # 因此显式写 NaN，避免用局部 best/second gap 冒充全局 margin。
    margin = np.full(len(states), np.nan, dtype=np.float64)
    return DynamicStateSolution(states=states, margin=margin, objective=objective)


def interpolate_state_path(
    row_md: np.ndarray,
    centers_md: np.ndarray,
    block_states: np.ndarray,
    candidate_tvt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把块状态 one-hot 沿 MD 插值，得到逐行交叉淡化路径。"""

    rows = np.asarray(row_md, dtype=np.float64)
    centers = np.asarray(centers_md, dtype=np.float64)
    states = np.asarray(block_states, dtype=np.int64)
    candidates = np.asarray(candidate_tvt, dtype=np.float64)
    if states.shape != centers.shape or candidates.shape != (
        len(rows),
        len(CANDIDATE_NAMES),
    ):
        raise ValueError("状态、中心或候选路径 shape 不一致")
    one_hot = np.eye(len(CANDIDATE_NAMES), dtype=np.float64)[states]
    weights = np.empty((len(rows), len(CANDIDATE_NAMES)), dtype=np.float64)
    for state in range(len(CANDIDATE_NAMES)):
        weights[:, state] = np.interp(rows, centers, one_hot[:, state])
    weights /= weights.sum(axis=1, keepdims=True)
    path = np.sum(weights * candidates, axis=1)
    row_states = np.argmax(weights, axis=1).astype(np.int64)
    return path, row_states, weights


def _candidate_at_centers(
    row_md: np.ndarray,
    centers: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(centers), candidates.shape[1]), dtype=np.float64)
    for state in range(candidates.shape[1]):
        output[:, state] = np.interp(centers, row_md, candidates[:, state])
    return output


def _solution_to_rows(
    row_md: np.ndarray,
    centers: np.ndarray,
    candidates: np.ndarray,
    solution: DynamicStateSolution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path, states, _ = interpolate_state_path(
        row_md, centers, solution.states, candidates
    )
    margin = np.interp(row_md, centers, solution.margin)
    return path, states, margin


def build_dynamic_mode_paths(
    *,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_index: np.ndarray,
    candidate_tvt: np.ndarray,
    last_visible_tvt: float,
    well_id: str,
    parameters: DynamicPathParameters = DEFAULT_PARAMETERS,
) -> DynamicModePathResult:
    """一次生成正常路径、safe 路径和两种 target-free 负对照。"""

    required = {"MD", "Z", "GR", "TVT_input"}
    if missing := required.difference(horizontal.columns):
        raise ValueError(f"水平井缺少合法列：{sorted(missing)}")
    keys = np.asarray(row_index, dtype=np.int64)
    if keys.ndim != 1 or len(np.unique(keys)) != len(keys):
        raise ValueError("row_index 必须一维且不重复")
    if len(keys) == 0 or keys.min() < 0 or keys.max() >= len(horizontal):
        raise ValueError("row_index 为空或越界")
    selected = horizontal.iloc[keys]
    if selected["TVT_input"].notna().any():
        raise ValueError("row_index 包含非自然隐藏行")
    row_md = selected["MD"].to_numpy(dtype=np.float64)
    row_z = selected["Z"].to_numpy(dtype=np.float64)
    row_gr = pd.to_numeric(selected["GR"], errors="coerce").to_numpy(dtype=np.float64)
    candidates = np.asarray(candidate_tvt, dtype=np.float64)
    calibration = fit_prefix_calibration(horizontal, typewell, parameters)
    prefix_u_slope, _ = fit_visible_u_trend(horizontal, window_ft=500.0)
    centers, local_cost, components = build_local_candidate_costs(
        row_md,
        row_z,
        row_gr,
        candidates,
        typewell,
        calibration,
        prefix_u_slope,
        parameters,
    )
    center_candidates = _candidate_at_centers(row_md, centers, candidates)
    center_z = np.interp(centers, row_md, row_z)
    normal_solution = solve_dynamic_states(
        local_cost, centers, center_candidates, center_z, parameters
    )
    shifted_gr, shift_rows = circular_shift_hidden_gr(row_gr)
    shifted_centers, shifted_cost, _ = build_local_candidate_costs(
        row_md,
        row_z,
        shifted_gr,
        candidates,
        typewell,
        calibration,
        prefix_u_slope,
        parameters,
    )
    if not np.array_equal(centers, shifted_centers):
        raise RuntimeError("GR control 改变了冻结块中心")
    shifted_solution = solve_dynamic_states(
        shifted_cost, centers, center_candidates, center_z, parameters
    )
    permuted_cost, permutation = permute_block_costs(local_cost, str(well_id))
    permuted_solution = solve_dynamic_states(
        permuted_cost, centers, center_candidates, center_z, parameters
    )
    dynamic_tvt, row_states, row_margin = _solution_to_rows(
        row_md, centers, candidates, normal_solution
    )
    shifted_tvt, shifted_states, _ = _solution_to_rows(
        row_md, centers, candidates, shifted_solution
    )
    permuted_tvt, permuted_states, _ = _solution_to_rows(
        row_md, centers, candidates, permuted_solution
    )
    p2_tvt = candidates[:, 0]
    safe10 = 0.9 * p2_tvt + 0.1 * dynamic_tvt
    safe25 = 0.75 * p2_tvt + 0.25 * dynamic_tvt
    shifted_safe10 = 0.9 * p2_tvt + 0.1 * shifted_tvt
    shifted_safe25 = 0.75 * p2_tvt + 0.25 * shifted_tvt
    permuted_safe10 = 0.9 * p2_tvt + 0.1 * permuted_tvt
    permuted_safe25 = 0.75 * p2_tvt + 0.25 * permuted_tvt
    row_output = pd.DataFrame(
        {
            "row_index": keys,
            "md": row_md,
            "last_visible_tvt": np.full(len(keys), float(last_visible_tvt)),
            "mdp_dynamic_tvt": dynamic_tvt,
            "mdp_dynamic_delta": dynamic_tvt - float(last_visible_tvt),
            "mdp_selected_state": row_states,
            "mdp_selected_state_name": [CANDIDATE_NAMES[index] for index in row_states],
            "mdp_margin": row_margin,
            "mdp_safe_10_tvt": safe10,
            "mdp_safe_10_delta": safe10 - float(last_visible_tvt),
            "mdp_safe_25_tvt": safe25,
            "mdp_safe_25_delta": safe25 - float(last_visible_tvt),
            "mdp_gr_shift_dynamic_tvt": shifted_tvt,
            "mdp_gr_shift_dynamic_delta": shifted_tvt - float(last_visible_tvt),
            "mdp_gr_shift_selected_state": shifted_states,
            "mdp_gr_shift_safe_10_delta": shifted_safe10 - float(last_visible_tvt),
            "mdp_gr_shift_safe_25_delta": shifted_safe25 - float(last_visible_tvt),
            "mdp_cost_permutation_dynamic_tvt": permuted_tvt,
            "mdp_cost_permutation_dynamic_delta": permuted_tvt
            - float(last_visible_tvt),
            "mdp_cost_permutation_selected_state": permuted_states,
            "mdp_cost_permutation_safe_10_delta": permuted_safe10
            - float(last_visible_tvt),
            "mdp_cost_permutation_safe_25_delta": permuted_safe25
            - float(last_visible_tvt),
        }
    )
    block_data: dict[str, object] = {
        "block_index": np.arange(len(centers), dtype=np.int64),
        "center_md": centers,
        "selected_state": normal_solution.states,
        "selected_state_name": [CANDIDATE_NAMES[index] for index in normal_solution.states],
        "margin": normal_solution.margin,
        "gr_shift_control_state": shifted_solution.states,
        "cost_permutation_control_state": permuted_solution.states,
        "cost_permutation_source_block": permutation,
    }
    for state, name in enumerate(CANDIDATE_NAMES):
        block_data[f"local_cost_{name}"] = local_cost[:, state]
        block_data[f"valid_ratio_{name}"] = components["valid_ratio"][:, state]
    block_output = pd.DataFrame(block_data)
    audit: dict[str, object] = {
        "well_id": str(well_id),
        "hidden_rows": int(len(keys)),
        "block_count": int(len(centers)),
        "hidden_observed_gr_rows": int(np.isfinite(row_gr).sum()),
        "prefix_calibration": calibration.__dict__,
        "prefix_u_slope_500": float(prefix_u_slope),
        "gr_shift_rows": int(shift_rows),
        "normal_objective": float(normal_solution.objective),
        "gr_shift_control_objective": float(shifted_solution.objective),
        "cost_permutation_control_objective": float(permuted_solution.objective),
        "hidden_target_read": False,
    }
    return DynamicModePathResult(
        row_output=row_output,
        block_output=block_output,
        audit=audit,
    )
