"""P3-PFM-D02 的纯数值核心与输入合同校验。

本模块不读写正式数据，也不生成可部署特征。runner 逐井加载数据后调用这里的
确定性函数，便于用合成数据完整验证 oracle 定义。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.p3_pfm01_ordered_pf_modes import cluster_seed_descriptors


MODE_NAMES = ("low", "middle", "high")
CANDIDATE_NAMES = ("P2", "low", "middle", "high")
LIKELIHOOD_SCALE = 8.0
_TIE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SimplexResult:
    """simplex 最小二乘结果。"""

    weights: np.ndarray
    sse: float
    active_count: int


@dataclass(frozen=True)
class DPResult:
    """固定切换惩罚动态规划结果。"""

    states: np.ndarray
    raw_sse: float
    penalized_objective: float
    switch_count: int


@dataclass(frozen=True)
class MedoidResult:
    """固定成员簇内的真实 seed medoid。"""

    index: int
    seed_id: int
    objective: float


@dataclass(frozen=True)
class OrderedRepresentatives:
    """按中心位置排为 low/middle/high 的三种固定成员代表。"""

    center_paths: np.ndarray
    rowmedian_paths: np.ndarray
    medoid_paths: np.ndarray
    medoid_seed_ids: np.ndarray
    member_seed_ids: tuple[np.ndarray, np.ndarray, np.ndarray]


def _finite_candidate_matrix(candidates: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(candidates, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("候选矩阵必须是非空二维数组 [行, 候选]")
    if truth.shape != (matrix.shape[0],):
        raise ValueError("真值必须与候选矩阵行数一致")
    if not np.isfinite(matrix).all() or not np.isfinite(truth).all():
        raise ValueError("候选路径和真值不能含 NaN/Inf")
    return matrix, truth


def solve_simplex_least_squares(candidates: np.ndarray, target: np.ndarray) -> SimplexResult:
    """枚举所有非空 active subset，精确求非负且和为一的最小二乘。

    每个 subset 内用一个成员消去等式约束，直接解等式约束最小二乘；只有解满足
    非负 KKT 可行性时才参与比较。这里没有 OLS 后裁剪。
    """

    matrix, truth = _finite_candidate_matrix(candidates, target)
    number_of_candidates = matrix.shape[1]
    best_weights: np.ndarray | None = None
    best_sse = math.inf
    best_subset: tuple[int, ...] | None = None

    for subset_size in range(1, number_of_candidates + 1):
        for subset in itertools.combinations(range(number_of_candidates), subset_size):
            active = matrix[:, subset]
            if subset_size == 1:
                active_weights = np.ones(1, dtype=np.float64)
            else:
                reference = active[:, -1]
                differences = active[:, :-1] - reference[:, None]
                free_weights, *_ = np.linalg.lstsq(differences, truth - reference, rcond=None)
                active_weights = np.concatenate(
                    [free_weights, [1.0 - float(np.sum(free_weights))]]
                )
            if np.any(active_weights < -_TIE_TOLERANCE):
                continue
            active_weights[np.abs(active_weights) <= _TIE_TOLERANCE] = 0.0
            total = float(np.sum(active_weights))
            if not np.isfinite(total) or total <= 0.0:
                continue
            active_weights /= total
            weights = np.zeros(number_of_candidates, dtype=np.float64)
            weights[list(subset)] = active_weights
            residual = matrix @ weights - truth
            sse = float(np.dot(residual, residual))
            # combinations 的自然顺序固定了完全平局时的候选优先级。
            if sse < best_sse - _TIE_TOLERANCE:
                best_weights = weights
                best_sse = sse
                best_subset = subset

    if best_weights is None or best_subset is None:
        raise RuntimeError("simplex active-set 枚举没有找到可行解")
    return SimplexResult(
        weights=best_weights,
        sse=best_sse,
        active_count=int(np.sum(best_weights > _TIE_TOLERANCE)),
    )


def solve_b25_simplex(candidates: np.ndarray, target: np.ndarray) -> SimplexResult:
    """求 q=0.75*e_P2+0.25*v 的精确强收缩 simplex 解。"""

    matrix, truth = _finite_candidate_matrix(candidates, target)
    if matrix.shape[1] != 4:
        raise ValueError("B25 必须恰好输入 P2/low/middle/high 四条候选")
    transformed = solve_simplex_least_squares(
        0.25 * matrix,
        truth - 0.75 * matrix[:, 0],
    )
    weights = 0.25 * transformed.weights
    weights[0] += 0.75
    residual = matrix @ weights - truth
    return SimplexResult(
        weights=weights,
        sse=float(np.dot(residual, residual)),
        active_count=int(np.sum(weights > _TIE_TOLERANCE)),
    )


def segment_ids_from_md(md: np.ndarray, window_ft: float) -> np.ndarray:
    """从隐藏首个 MD 起点按固定英尺窗口生成不重叠分段编号。"""

    values = np.asarray(md, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("MD 必须是一维非空有限数组")
    if not np.isfinite(window_ft) or window_ft <= 0.0:
        raise ValueError("窗口长度必须为正")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("MD 必须单调不减")
    return np.floor((values - values[0]) / float(window_ft)).astype(np.int64)


def compute_segment_sse(
    candidates: np.ndarray,
    target: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    """逐段、逐候选计算原始 SSE。"""

    matrix, truth = _finite_candidate_matrix(candidates, target)
    labels = np.asarray(segment_ids)
    if labels.shape != (matrix.shape[0],) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("segment_ids 必须是与行数一致的整数数组")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique), dtype=unique.dtype)):
        raise ValueError("segment_ids 必须从 0 开始连续编号")
    squared = np.square(matrix - truth[:, None])
    output = np.empty((len(unique), matrix.shape[1]), dtype=np.float64)
    for segment_id in unique:
        output[int(segment_id)] = np.sum(squared[labels == segment_id], axis=0)
    return output


def solve_switching_dp(segment_sse: np.ndarray, switch_penalty: float) -> DPResult:
    """四状态通用 DP；平局先保持原状态，再按较小状态编号。"""

    costs = np.asarray(segment_sse, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("分段 SSE 必须是非空二维数组")
    if not np.isfinite(costs).all() or np.any(costs < 0.0):
        raise ValueError("分段 SSE 必须是非负有限数")
    if not np.isfinite(switch_penalty) or switch_penalty < 0.0:
        raise ValueError("切换惩罚必须是非负有限数")

    number_of_segments, number_of_states = costs.shape
    previous_cost = costs[0].copy()
    parents = np.zeros((number_of_segments, number_of_states), dtype=np.int64)
    for segment_id in range(1, number_of_segments):
        current_cost = np.empty(number_of_states, dtype=np.float64)
        for state in range(number_of_states):
            transition = previous_cost + float(switch_penalty)
            transition[state] = previous_cost[state]
            minimum = float(np.min(transition))
            tied = np.flatnonzero(np.isclose(transition, minimum, rtol=0.0, atol=_TIE_TOLERANCE))
            # 若原状态在平局集合内，优先保持；否则按状态顺序选最小编号。
            parent = state if state in tied else int(tied[0])
            parents[segment_id, state] = parent
            current_cost[state] = costs[segment_id, state] + transition[parent]
        previous_cost = current_cost

    minimum = float(np.min(previous_cost))
    final_state = int(
        np.flatnonzero(np.isclose(previous_cost, minimum, rtol=0.0, atol=_TIE_TOLERANCE))[0]
    )
    states = np.empty(number_of_segments, dtype=np.int64)
    states[-1] = final_state
    for segment_id in range(number_of_segments - 1, 0, -1):
        states[segment_id - 1] = parents[segment_id, states[segment_id]]
    raw_sse = float(np.sum(costs[np.arange(number_of_segments), states]))
    switch_count = int(np.sum(states[1:] != states[:-1]))
    return DPResult(
        states=states,
        raw_sse=raw_sse,
        penalized_objective=raw_sse + float(switch_penalty) * switch_count,
        switch_count=switch_count,
    )


def _stable_member_weights(final_ll: np.ndarray, member_indices: np.ndarray) -> np.ndarray:
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    members = np.asarray(member_indices, dtype=np.int64)
    member_ll = likelihoods[members]
    if member_ll.size == 0 or not np.isfinite(member_ll).all():
        raise ValueError("模式成员 LL 必须非空且有限")
    shifted = (member_ll - float(np.max(member_ll))) / LIKELIHOOD_SCALE
    unnormalized = np.exp(shifted)
    return unnormalized / float(np.sum(unnormalized))


def choose_weighted_medoid(
    seed_paths: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    member_indices: np.ndarray,
) -> MedoidResult:
    """选择到固定成员全路径 LL 加权均方距离最小的真实 seed。"""

    paths = np.asarray(seed_paths, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    members = np.asarray(member_indices, dtype=np.int64)
    if paths.ndim != 2 or paths.shape[0] != ids.size or paths.shape[0] != len(final_ll):
        raise ValueError("seed 路径、LL 与 seed_id 数量必须一致")
    if paths.shape[1] == 0 or not np.isfinite(paths).all():
        raise ValueError("seed 路径必须非空且有限")
    if members.ndim != 1 or members.size == 0 or np.any((members < 0) | (members >= len(ids))):
        raise ValueError("模式成员索引非法")
    weights = _stable_member_weights(final_ll, members)
    member_paths = paths[members]
    weighted_center = np.sum(member_paths * weights[:, None], axis=0, dtype=np.float64)
    # 加权平方距离与到加权中心距离只差一个与候选无关的常数。
    objectives = np.mean(np.square(member_paths - weighted_center), axis=1)
    minimum = float(np.min(objectives))
    tied_local = np.flatnonzero(
        np.isclose(objectives, minimum, rtol=0.0, atol=_TIE_TOLERANCE)
    )
    tied_global = members[tied_local]
    chosen_global = int(tied_global[np.argmin(ids[tied_global])])
    chosen_local = int(np.flatnonzero(members == chosen_global)[0])
    return MedoidResult(
        index=chosen_global,
        seed_id=int(ids[chosen_global]),
        objective=float(objectives[chosen_local]),
    )


def build_ordered_representatives(
    seed_paths: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
) -> OrderedRepresentatives:
    """按冻结二维 Ward 成员，构造 center/rowmedian/medoid 三代表。"""

    paths = np.asarray(seed_paths, dtype=np.float64)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    if paths.ndim != 2 or paths.shape[0] < 3 or paths.shape[1] == 0:
        raise ValueError("至少需要三条非空 seed 路径")
    if likelihoods.shape != (paths.shape[0],) or ids.shape != (paths.shape[0],):
        raise ValueError("seed 路径、LL 与 seed_id 数量必须一致")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(paths).all() or not np.isfinite(likelihoods).all():
        raise ValueError("seed 输入含重复 id 或非有限数")

    raw_labels = cluster_seed_descriptors(paths)
    records: list[dict[str, object]] = []
    for raw_label in sorted(np.unique(raw_labels).tolist()):
        members = np.flatnonzero(raw_labels == raw_label)
        weights = _stable_member_weights(likelihoods, members)
        center = np.sum(paths[members] * weights[:, None], axis=0, dtype=np.float64)
        medoid = choose_weighted_medoid(paths, likelihoods, ids, members)
        records.append(
            {
                "center": center,
                "median": np.median(paths[members], axis=0),
                "medoid": paths[medoid.index].copy(),
                "medoid_seed_id": medoid.seed_id,
                "members": ids[members].copy(),
                "key": (float(np.mean(center)), float(center[-1]), int(np.min(ids[members]))),
            }
        )
    records.sort(key=lambda record: record["key"])
    return OrderedRepresentatives(
        center_paths=np.stack([np.asarray(record["center"]) for record in records]),
        rowmedian_paths=np.stack([np.asarray(record["median"]) for record in records]),
        medoid_paths=np.stack([np.asarray(record["medoid"]) for record in records]),
        medoid_seed_ids=np.asarray([record["medoid_seed_id"] for record in records], dtype=np.int64),
        member_seed_ids=tuple(np.asarray(record["members"], dtype=np.int64) for record in records),  # type: ignore[arg-type]
    )


def pointwise_envelope_sse(candidates: np.ndarray, target: np.ndarray) -> float:
    """逐行取最小平方误差，仅表示不可部署的 pointwise coverage。"""

    matrix, truth = _finite_candidate_matrix(candidates, target)
    return float(np.sum(np.min(np.square(matrix - truth[:, None]), axis=1)))


def pooled_micro_rmse(sse: Sequence[float], rows: Sequence[int]) -> float:
    """按总 SSE / 总行数计算 pooled micro RMSE。"""

    sse_values = np.asarray(sse, dtype=np.float64)
    row_values = np.asarray(rows, dtype=np.int64)
    if sse_values.ndim != 1 or row_values.shape != sse_values.shape or sse_values.size == 0:
        raise ValueError("SSE 与行数必须是等长非空一维数组")
    if not np.isfinite(sse_values).all() or np.any(sse_values < 0.0) or np.any(row_values <= 0):
        raise ValueError("SSE 必须非负有限，行数必须为正")
    return float(math.sqrt(float(np.sum(sse_values)) / int(np.sum(row_values))))


def validate_well_alignment(
    p2: pd.DataFrame,
    mode: pd.DataFrame,
    seed: Mapping[str, np.ndarray],
    *,
    well_id: str,
    fold: int,
    expected_rows: int,
) -> None:
    """拒绝 P2、mode、seed 的任何自然键、fold、行数或 MD 错位。"""

    required_p2 = {"well_id", "fold", "row_index", "md"}
    required_mode = {"well_id", "fold", "row_index"}
    if missing := required_p2.difference(p2.columns):
        raise ValueError(f"P2 缺列：{sorted(missing)}")
    if missing := required_mode.difference(mode.columns):
        raise ValueError(f"mode 缺列：{sorted(missing)}")
    if "row_index" not in seed or "hidden_md" not in seed:
        raise ValueError("seed 缺少 row_index 或 hidden_md")
    if len(p2) != expected_rows or len(mode) != expected_rows:
        raise ValueError("三源行数不一致")
    if not p2["well_id"].eq(well_id).all() or not mode["well_id"].eq(well_id).all():
        raise ValueError("自然键中的 well_id 不一致")
    if not p2["fold"].eq(fold).all() or not mode["fold"].eq(fold).all():
        raise ValueError("三源 fold 不一致")
    p2_key = p2["row_index"].to_numpy(dtype=np.int64)
    mode_key = mode["row_index"].to_numpy(dtype=np.int64)
    seed_key = np.asarray(seed["row_index"], dtype=np.int64)
    if seed_key.shape != (expected_rows,) or not np.array_equal(p2_key, mode_key) or not np.array_equal(p2_key, seed_key):
        raise ValueError("三源自然键 (well_id,row_index) 错位")
    if len(np.unique(p2_key)) != expected_rows:
        raise ValueError("自然键重复")
    p2_md32 = p2["md"].to_numpy(dtype=np.float64).astype(np.float32)
    seed_md32 = np.asarray(seed["hidden_md"], dtype=np.float32)
    if seed_md32.shape != (expected_rows,) or not np.array_equal(p2_md32, seed_md32):
        raise ValueError("P2 MD 与 seed hidden_md 超出 float32 舍入容差")


def validate_development_selection(selected: pd.DataFrame, shadow_wells: set[str]) -> None:
    """验证开发井选择不重不漏且绝不包含影子井。"""

    required = {"well_id", "fold", "hidden_rows"}
    if missing := required.difference(selected.columns):
        raise ValueError(f"开发井表缺列：{sorted(missing)}")
    ids = selected["well_id"].astype(str)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("开发井号为空或重复")
    overlap = set(ids).intersection(shadow_wells)
    if overlap:
        raise ValueError(f"开发选择含影子井：{sorted(overlap)[:3]}")
    if not selected["fold"].between(0, 4).all() or (selected["hidden_rows"] <= 0).any():
        raise ValueError("开发井 fold 或 hidden_rows 非法")


def resolve_run_artifact_dir(output_dir: Path, max_wells: int | None) -> Path:
    """smoke 永远写入正式目录下独立子目录，禁止覆盖正式产物。"""

    base = Path(output_dir)
    if max_wells is None:
        return base
    if max_wells not in (1, 2, 3):
        raise ValueError("--max-wells 只允许 1、2、3")
    return base / f"smoke_{max_wells}"
