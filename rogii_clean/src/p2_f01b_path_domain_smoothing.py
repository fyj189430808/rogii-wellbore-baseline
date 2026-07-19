"""P2-F01b：先采样候选 Typewell，再在同一 MD 轴平滑两条 GR 序列。

这个模块只改变 F01 v1 的得分面构造顺序。PF 中心、offset 网格、
观测代价和二阶连续路径约束都直接复用已经冻结的 F01 v1 实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.p2_d01_pf_centered_gr import _build_blocks, _clean_typewell
from src.p2_f01_continuous_gr_path import (
    LegalContinuousPathResult,
    _find_exact_grid_position,
    build_emission_costs,
    interpolate_block_values,
    solve_second_order_path,
)


@dataclass(frozen=True)
class PathDomainScoreLandscape:
    """保存同一 MD 轴五尺度得分面及其逐行、分块坐标。"""

    block_table: pd.DataFrame
    offsets_ft: np.ndarray
    score_labels: tuple[str, ...]
    scores: np.ndarray
    pair_counts: np.ndarray
    block_index: np.ndarray


def _validate_score_inputs(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    block_width_ft: float,
    smoothing_widths_md_ft: list[float],
    minimum_valid_pairs: int,
) -> None:
    """检查得分面所有数组的逐行关系、坐标顺序和冻结参数。"""

    if md.ndim != 1 or horizontal_gr.ndim != 1 or center_tvt.ndim != 1:
        raise ValueError("MD、水平井 GR 和中心 TVT 必须为一维数组")
    if not (len(md) == len(horizontal_gr) == len(center_tvt)):
        raise ValueError("MD、水平井 GR 和中心 TVT 长度必须一致")
    if len(md) == 0:
        raise ValueError("至少需要一行水平井数据")
    if not np.isfinite(md).all() or np.any(np.diff(md) < 0.0):
        raise ValueError("MD 必须全部有限且单调非降")
    if not np.isfinite(center_tvt).all():
        raise ValueError("中心 TVT 不能含 NaN 或 Inf")
    if typewell_tvt.ndim != 1 or typewell_gr.ndim != 1:
        raise ValueError("Typewell TVT 和 GR 必须为一维数组")
    if len(typewell_tvt) != len(typewell_gr):
        raise ValueError("Typewell TVT 和 GR 长度必须一致")
    if offsets_ft.ndim != 1 or len(offsets_ft) < 2:
        raise ValueError("offset 网格至少需要两个点")
    if not np.isfinite(offsets_ft).all() or np.any(np.diff(offsets_ft) <= 0.0):
        raise ValueError("offset 网格必须全部有限且严格递增")
    if float(block_width_ft) <= 0.0:
        raise ValueError("block_width_ft 必须大于 0")
    if not smoothing_widths_md_ft:
        raise ValueError("至少需要一个 MD 平滑窗口")
    if any((not np.isfinite(width)) or width <= 0.0 for width in smoothing_widths_md_ft):
        raise ValueError("MD 平滑窗口必须全部为正有限值")
    if int(minimum_valid_pairs) < 3:
        raise ValueError("minimum_valid_pairs 必须至少为 3")


def _centered_window_bounds(
    md: np.ndarray,
    smoothing_width_md_ft: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回每行居中 MD 窗口的左右位置，右位置采用开区间。"""

    half_width = float(smoothing_width_md_ft) / 2.0
    left_positions = np.searchsorted(md, md - half_width, side="left")
    right_positions = np.searchsorted(md, md + half_width, side="right")
    return left_positions, right_positions


def _centered_nanmean_columns(
    values: np.ndarray,
    left_positions: np.ndarray,
    right_positions: np.ndarray,
) -> np.ndarray:
    """用二维前缀和同时计算所有候选列在同一批 MD 窗口内的均值。"""

    value_matrix = np.asarray(values, dtype=np.float64)
    if value_matrix.ndim != 2:
        raise ValueError("values 必须为 [自然行, 序列] 二维数组")
    number_of_rows = value_matrix.shape[0]
    if left_positions.shape != (number_of_rows,) or right_positions.shape != (
        number_of_rows,
    ):
        raise ValueError("窗口边界数量必须与自然行数量一致")

    # 缺失值不参与均值。前缀数组首行补零，因此任意窗口都能用两次索引相减得到。
    finite = np.isfinite(value_matrix)
    finite_values = np.where(finite, value_matrix, 0.0)
    value_prefix = np.vstack(
        [
            np.zeros((1, value_matrix.shape[1]), dtype=np.float64),
            np.cumsum(finite_values, axis=0, dtype=np.float64),
        ]
    )
    count_prefix = np.vstack(
        [
            np.zeros((1, value_matrix.shape[1]), dtype=np.int64),
            np.cumsum(finite, axis=0, dtype=np.int64),
        ]
    )
    window_sums = value_prefix[right_positions] - value_prefix[left_positions]
    window_counts = count_prefix[right_positions] - count_prefix[left_positions]

    smoothed_values = np.full(value_matrix.shape, np.nan, dtype=np.float64)
    has_value = window_counts > 0
    smoothed_values[has_value] = window_sums[has_value] / window_counts[has_value]
    return smoothed_values


def _grouped_pearson_all_offsets(
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
    block_index: np.ndarray,
    number_of_blocks: int,
    minimum_valid_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用 ``reduceat`` 一次算完每个控制块、每个 offset 的 Pearson NCC。"""

    horizontal = np.asarray(horizontal_gr, dtype=np.float64)
    references = np.asarray(reference_gr, dtype=np.float64)
    groups = np.asarray(block_index, dtype=np.int32)
    if references.ndim != 2 or references.shape[0] != len(horizontal):
        raise ValueError("reference_gr 必须为 [自然行, offset]")
    if groups.shape != (len(horizontal),):
        raise ValueError("block_index 必须与自然行数量一致")
    if np.any(np.diff(groups) < 0.0):
        raise ValueError("block_index 必须按自然行连续非降")

    # 每个块在自然行中都是连续段，因此只需保存各段首行即可做分块归约。
    block_starts = np.flatnonzero(
        np.concatenate(([True], groups[1:] != groups[:-1]))
    )
    if len(block_starts) != int(number_of_blocks):
        raise ValueError("block_index 中的块数与 block_table 不一致")

    common = np.isfinite(horizontal)[:, None] & np.isfinite(references)
    horizontal_matrix = np.broadcast_to(horizontal[:, None], references.shape)
    x = np.where(common, horizontal_matrix, 0.0)
    y = np.where(common, references, 0.0)

    # 下列七个二维归约量足以精确还原每个块内的 Pearson 相关系数。
    pair_counts = np.add.reduceat(common.astype(np.float64), block_starts, axis=0)
    sum_x = np.add.reduceat(x, block_starts, axis=0)
    sum_y = np.add.reduceat(y, block_starts, axis=0)
    sum_x2 = np.add.reduceat(x * x, block_starts, axis=0)
    sum_y2 = np.add.reduceat(y * y, block_starts, axis=0)
    sum_xy = np.add.reduceat(x * y, block_starts, axis=0)

    enough_pairs = pair_counts >= float(minimum_valid_pairs)
    safe_counts = np.where(enough_pairs, pair_counts, 1.0)
    centered_x2 = sum_x2 - np.square(sum_x) / safe_counts
    centered_y2 = sum_y2 - np.square(sum_y) / safe_counts
    centered_xy = sum_xy - sum_x * sum_y / safe_counts
    denominator = np.sqrt(
        np.maximum(centered_x2, 0.0) * np.maximum(centered_y2, 0.0)
    )

    scores = np.full(pair_counts.shape, np.nan, dtype=np.float64)
    valid_score = enough_pairs & (denominator > 1e-12)
    scores[valid_score] = np.clip(
        centered_xy[valid_score] / denominator[valid_score],
        -1.0,
        1.0,
    )
    return scores, pair_counts


def build_path_domain_score_landscape(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    block_width_ft: float,
    smoothing_widths_md_ft: list[float],
    minimum_valid_pairs: int,
) -> PathDomainScoreLandscape:
    """先采样原始 Typewell，再沿 MD 同尺度平滑并构造分块 NCC 得分面。"""

    row_md = np.asarray(md, dtype=np.float64)
    row_gr = np.asarray(horizontal_gr, dtype=np.float64)
    center = np.asarray(center_tvt, dtype=np.float64)
    raw_typewell_tvt = np.asarray(typewell_tvt, dtype=np.float64)
    raw_typewell_gr = np.asarray(typewell_gr, dtype=np.float64)
    offsets = np.asarray(offsets_ft, dtype=np.float64)
    smoothing_widths = [float(width) for width in smoothing_widths_md_ft]
    _validate_score_inputs(
        row_md,
        row_gr,
        center,
        raw_typewell_tvt,
        raw_typewell_gr,
        offsets,
        float(block_width_ft),
        smoothing_widths,
        int(minimum_valid_pairs),
    )

    clean_typewell_tvt, clean_typewell_gr = _clean_typewell(
        raw_typewell_tvt,
        raw_typewell_gr,
    )
    block_table, block_index = _build_blocks(row_md, row_gr, float(block_width_ft))
    number_of_blocks = len(block_table)

    # candidate_tvt 和 raw_reference_gr 的 shape 都是 [自然行, offset]。
    candidate_tvt = center[:, None] + offsets[None, :]
    raw_reference_gr = np.interp(
        candidate_tvt.ravel(),
        clean_typewell_tvt,
        clean_typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    raw_reference_is_valid = np.isfinite(raw_reference_gr)

    number_of_scales = len(smoothing_widths)
    scores = np.full(
        (number_of_scales, number_of_blocks, len(offsets)),
        np.nan,
        dtype=np.float64,
    )
    pair_counts = np.zeros(scores.shape, dtype=np.float64)

    # 每次只保留一个尺度的平滑矩阵，避免五个尺度同时占用工作内存。
    for scale_position, smoothing_width in enumerate(smoothing_widths):
        left_positions, right_positions = _centered_window_bounds(
            row_md,
            smoothing_width,
        )
        smooth_horizontal_gr = _centered_nanmean_columns(
            row_gr[:, None],
            left_positions,
            right_positions,
        )[:, 0]
        smooth_reference_gr = _centered_nanmean_columns(
            raw_reference_gr,
            left_positions,
            right_positions,
        )

        # 窗口均值可能把边界外的当前行从邻行“补回来”；重新遮罩以保留真实搜索边界。
        smooth_reference_gr[~raw_reference_is_valid] = np.nan
        scale_scores, scale_pair_counts = _grouped_pearson_all_offsets(
            horizontal_gr=smooth_horizontal_gr,
            reference_gr=smooth_reference_gr,
            block_index=block_index,
            number_of_blocks=number_of_blocks,
            minimum_valid_pairs=int(minimum_valid_pairs),
        )
        scores[scale_position] = scale_scores
        pair_counts[scale_position] = scale_pair_counts

    score_labels = tuple(f"{width:g}md_ft" for width in smoothing_widths)
    return PathDomainScoreLandscape(
        block_table=block_table,
        offsets_ft=offsets.copy(),
        score_labels=score_labels,
        scores=scores,
        pair_counts=pair_counts,
        block_index=block_index.copy(),
    )


def build_legal_path_domain_gr_path(
    row_index: np.ndarray,
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    block_width_ft: float,
    smoothing_widths_md_ft: list[float],
    minimum_valid_pairs: int,
    minimum_valid_scales: int,
    allowed_changes_ft: np.ndarray,
    maximum_change_acceleration_ft: float,
    initial_offset_ft: float,
    initial_change_ft: float,
) -> LegalContinuousPathResult:
    """把同一 MD 轴得分面变成逐行连续路径，并保留全部合法诊断量。"""

    row_keys = np.asarray(row_index, dtype=np.int64)
    row_md = np.asarray(md, dtype=np.float64)
    row_gr = np.asarray(horizontal_gr, dtype=np.float64)
    center = np.asarray(center_tvt, dtype=np.float64)
    if not (len(row_keys) == len(row_md) == len(row_gr) == len(center)):
        raise ValueError("row_index、MD、GR 和中心路径长度必须一致")
    if len(row_keys) == 0 or len(np.unique(row_keys)) != len(row_keys):
        raise ValueError("row_index 必须非空且唯一")

    landscape = build_path_domain_score_landscape(
        md=row_md,
        horizontal_gr=row_gr,
        center_tvt=center,
        typewell_tvt=np.asarray(typewell_tvt, dtype=np.float64),
        typewell_gr=np.asarray(typewell_gr, dtype=np.float64),
        offsets_ft=np.asarray(offsets_ft, dtype=np.float64),
        block_width_ft=float(block_width_ft),
        smoothing_widths_md_ft=[float(width) for width in smoothing_widths_md_ft],
        minimum_valid_pairs=int(minimum_valid_pairs),
    )
    block_rows = landscape.block_table["block_rows"].to_numpy(dtype=np.float64)
    emission_cost, support = build_emission_costs(
        scale_scores=landscape.scores,
        scale_pair_counts=landscape.pair_counts,
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

    # 独立块路径是同一观测代价下、不使用连续约束的预注册负对照。
    independent_offset_positions = np.argmin(emission_cost, axis=1)
    independent_offsets = landscape.offsets_ft[independent_offset_positions]
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

    # 控制块上的离散结果按 MD 线性插回所有自然行，首尾保持最近控制块值。
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
            "f01b_offset_path": row_offset,
            "f01b_corrected_tvt": center + row_offset,
            "f01b_offset_change": row_change,
            "f01b_independent_offset": row_independent_offset,
            "f01b_independent_tvt": center + row_independent_offset,
            "f01b_local_gr_cost": row_local_cost,
            "f01b_support_fraction": row_support,
            "f01b_forward_backward_margin": row_margin,
            "f01b_missing_fallback": row_missing_fallback,
        }
    )
    block_path = pd.DataFrame(
        {
            "block_position": np.arange(len(block_md_mid), dtype=np.int32),
            "block_md_mid": block_md_mid,
            "f01b_offset_path": solution.offset_path_ft,
            "f01b_offset_change": solution.change_path_ft,
            "f01b_independent_offset": independent_offsets,
            "f01b_local_gr_cost": solution.selected_emission_cost,
            "f01b_support_fraction": selected_support,
            "f01b_forward_backward_margin": solution.forward_backward_margin,
            "f01b_missing_fallback": missing_fallback,
        }
    )
    return LegalContinuousPathResult(
        row_path=row_path,
        block_path=block_path,
        offsets_ft=landscape.offsets_ft.copy(),
        scale_scores=landscape.scores.copy(),
        scale_pair_counts=landscape.pair_counts.copy(),
        row_to_block=landscape.block_index.copy(),
    )
