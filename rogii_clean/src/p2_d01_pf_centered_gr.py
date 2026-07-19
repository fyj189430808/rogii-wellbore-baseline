"""P2-D01：围绕冻结 PF 路径计算分块、多尺度 Typewell GR 得分面。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.f03b_geometry_landscape_features import _centered_nanmean


@dataclass(frozen=True)
class BlockScoreLandscape:
    """保存 legal score 摘要及 oracle 排名所需但不落模型特征的得分矩阵。"""

    summary: pd.DataFrame
    block_table: pd.DataFrame
    offsets_ft: np.ndarray
    score_labels: tuple[str, ...]
    scores: np.ndarray
    pair_counts: np.ndarray
    block_index: np.ndarray


def _validate_legal_inputs(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    block_width_ft: float,
    smoothing_widths_ft: list[float],
    minimum_valid_pairs: int,
) -> None:
    """检查 legal scorer 的逐行对齐、物理坐标和固定搜索参数。"""

    if not (len(md) == len(horizontal_gr) == len(center_tvt)):
        raise ValueError("MD、水平井 GR 和 PF 中心路径长度必须一致")
    if len(md) == 0:
        raise ValueError("legal scorer 至少需要一行数据")
    if not np.isfinite(md).all() or not np.isfinite(center_tvt).all():
        raise ValueError("MD 或 PF 中心路径含 NaN/Inf")
    if np.any(np.diff(md) < 0.0):
        raise ValueError("MD 必须单调非降")
    if len(typewell_tvt) != len(typewell_gr):
        raise ValueError("Typewell TVT 和 GR 长度必须一致")
    if len(offsets_ft) < 2 or not np.isfinite(offsets_ft).all():
        raise ValueError("offset 网格至少需要两个有限值")
    if np.any(np.diff(offsets_ft) <= 0.0):
        raise ValueError("offset 网格必须严格递增")
    if float(block_width_ft) <= 0.0:
        raise ValueError("block_width_ft 必须大于 0")
    if not smoothing_widths_ft or any(
        float(width) <= 0.0 for width in smoothing_widths_ft
    ):
        raise ValueError("GR 平滑尺度必须全部大于 0")
    if int(minimum_valid_pairs) < 3:
        raise ValueError("minimum_valid_pairs 必须至少为 3")


def _clean_typewell(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """清除 Typewell 缺失值、排序并合并重复 TVT。"""

    finite = np.isfinite(typewell_tvt) & np.isfinite(typewell_gr)
    clean_tvt = typewell_tvt[finite].astype(np.float64, copy=False)
    clean_gr = typewell_gr[finite].astype(np.float64, copy=False)
    if len(clean_tvt) < 3:
        raise ValueError("Typewell 至少需要三个有效 TVT/GR 点")
    order = np.argsort(clean_tvt, kind="mergesort")
    clean_tvt = clean_tvt[order]
    clean_gr = clean_gr[order]
    unique_tvt, inverse = np.unique(clean_tvt, return_inverse=True)
    gr_sum = np.bincount(inverse, weights=clean_gr)
    gr_count = np.bincount(inverse)
    return unique_tvt, gr_sum / gr_count


def _build_blocks(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    block_width_ft: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """按首行 MD 起算固定物理宽度井段，并返回每行所属 block position。"""

    raw_block_id = np.floor((md - float(md[0])) / float(block_width_ft)).astype(
        np.int64
    )
    unique_block_ids, block_index = np.unique(raw_block_id, return_inverse=True)
    block_rows: list[dict[str, float | int]] = []
    for block_position, block_id in enumerate(unique_block_ids):
        positions = np.flatnonzero(block_index == block_position)
        block_gr = horizontal_gr[positions]
        block_rows.append(
            {
                "block_position": int(block_position),
                "block_id": int(block_id),
                "block_width_ft": float(block_width_ft),
                "md_start": float(md[positions[0]]),
                "md_end": float(md[positions[-1]]),
                "md_mid": float(np.median(md[positions])),
                "block_rows": int(len(positions)),
                "gr_finite_fraction": float(np.isfinite(block_gr).mean()),
            }
        )
    return pd.DataFrame(block_rows), block_index.astype(np.int32)


def _grouped_pearson(
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
    block_index: np.ndarray,
    number_of_blocks: int,
    minimum_valid_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在互不重叠的井段内计算公共有限点 Pearson NCC。"""

    common = np.isfinite(horizontal_gr) & np.isfinite(reference_gr)
    groups = block_index[common]
    x_values = horizontal_gr[common]
    y_values = reference_gr[common]
    counts = np.bincount(groups, minlength=number_of_blocks).astype(np.float64)
    sum_x = np.bincount(groups, weights=x_values, minlength=number_of_blocks)
    sum_y = np.bincount(groups, weights=y_values, minlength=number_of_blocks)
    sum_x2 = np.bincount(
        groups,
        weights=x_values * x_values,
        minlength=number_of_blocks,
    )
    sum_y2 = np.bincount(
        groups,
        weights=y_values * y_values,
        minlength=number_of_blocks,
    )
    sum_xy = np.bincount(
        groups,
        weights=x_values * y_values,
        minlength=number_of_blocks,
    )

    scores = np.full(number_of_blocks, np.nan, dtype=np.float64)
    enough = counts >= float(minimum_valid_pairs)
    centered_x2 = np.zeros(number_of_blocks, dtype=np.float64)
    centered_y2 = np.zeros(number_of_blocks, dtype=np.float64)
    centered_xy = np.zeros(number_of_blocks, dtype=np.float64)
    centered_x2[enough] = sum_x2[enough] - sum_x[enough] ** 2 / counts[enough]
    centered_y2[enough] = sum_y2[enough] - sum_y[enough] ** 2 / counts[enough]
    centered_xy[enough] = sum_xy[enough] - (
        sum_x[enough] * sum_y[enough] / counts[enough]
    )
    denominator = np.sqrt(
        np.maximum(centered_x2, 0.0) * np.maximum(centered_y2, 0.0)
    )
    valid = enough & (denominator > 1e-12)
    scores[valid] = np.clip(centered_xy[valid] / denominator[valid], -1.0, 1.0)
    return scores, counts


def _summarize_one_score_matrix(
    block_table: pd.DataFrame,
    scores: np.ndarray,
    pair_counts: np.ndarray,
    offsets_ft: np.ndarray,
    scale_label: str,
    smoothing_width_ft: float,
    second_peak_minimum_distance_ft: float,
) -> pd.DataFrame:
    """把一个尺度的 block×offset 得分压缩成合法峰值和支撑量。"""

    rows: list[dict[str, float | int | str]] = []
    zero_position = int(np.argmin(np.abs(offsets_ft)))
    for block_position in range(len(block_table)):
        block_metadata = block_table.iloc[block_position].to_dict()
        row_scores = scores[block_position]
        valid_positions = np.flatnonzero(np.isfinite(row_scores))
        summary_row: dict[str, float | int | str] = {
            **block_metadata,
            "scale_label": scale_label,
            "smoothing_width_ft": float(smoothing_width_ft),
            "best_offset_ft": np.nan,
            "best_ncc": np.nan,
            "second_offset_ft": np.nan,
            "second_ncc": np.nan,
            "peak_gap": np.nan,
            "peak_separation_ft": np.nan,
            "ncc_at_zero_offset": float(row_scores[zero_position])
            if np.isfinite(row_scores[zero_position])
            else np.nan,
            "best_pair_count": float(np.max(pair_counts[block_position])),
            "best_pair_fraction": float(np.max(pair_counts[block_position]))
            / max(float(block_metadata["block_rows"]), 1.0),
            "valid_offset_count": int(len(valid_positions)),
            "scale_best_offset_iqr_ft": np.nan,
            "scale_best_offset_range_ft": np.nan,
        }
        if len(valid_positions):
            best_position = int(
                valid_positions[np.argmax(row_scores[valid_positions])]
            )
            best_offset = float(offsets_ft[best_position])
            best_ncc = float(row_scores[best_position])
            summary_row["best_offset_ft"] = best_offset
            summary_row["best_ncc"] = best_ncc
            summary_row["best_pair_count"] = float(
                pair_counts[block_position, best_position]
            )
            summary_row["best_pair_fraction"] = float(
                pair_counts[block_position, best_position]
            ) / max(float(block_metadata["block_rows"]), 1.0)
            second_mask = np.isfinite(row_scores) & (
                np.abs(offsets_ft - best_offset)
                >= float(second_peak_minimum_distance_ft)
            )
            if second_mask.any():
                second_positions = np.flatnonzero(second_mask)
                second_position = int(
                    second_positions[np.argmax(row_scores[second_positions])]
                )
                second_offset = float(offsets_ft[second_position])
                second_ncc = float(row_scores[second_position])
                summary_row["second_offset_ft"] = second_offset
                summary_row["second_ncc"] = second_ncc
                summary_row["peak_gap"] = best_ncc - second_ncc
                summary_row["peak_separation_ft"] = abs(
                    best_offset - second_offset
                )
        rows.append(summary_row)
    return pd.DataFrame(rows)


def build_block_score_landscape(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    block_width_ft: float,
    smoothing_widths_ft: list[float],
    minimum_valid_pairs: int,
    second_peak_minimum_distance_ft: float,
) -> BlockScoreLandscape:
    """只用测试期合法数组，返回 PF 中心的分块多尺度 GR 得分面。"""

    md = np.asarray(md, dtype=np.float64)
    horizontal_gr = np.asarray(horizontal_gr, dtype=np.float64)
    center_tvt = np.asarray(center_tvt, dtype=np.float64)
    typewell_tvt = np.asarray(typewell_tvt, dtype=np.float64)
    typewell_gr = np.asarray(typewell_gr, dtype=np.float64)
    offsets_ft = np.asarray(offsets_ft, dtype=np.float64)
    normalized_scales = [float(width) for width in smoothing_widths_ft]
    _validate_legal_inputs(
        md,
        horizontal_gr,
        center_tvt,
        typewell_tvt,
        typewell_gr,
        offsets_ft,
        block_width_ft,
        normalized_scales,
        minimum_valid_pairs,
    )
    clean_typewell_tvt, clean_typewell_gr = _clean_typewell(
        typewell_tvt,
        typewell_gr,
    )
    block_table, block_index = _build_blocks(md, horizontal_gr, block_width_ft)
    number_of_blocks = len(block_table)
    scale_scores = np.full(
        (len(normalized_scales), number_of_blocks, len(offsets_ft)),
        np.nan,
        dtype=np.float64,
    )
    scale_pair_counts = np.zeros_like(scale_scores)

    for scale_position, smoothing_width_ft in enumerate(normalized_scales):
        smooth_horizontal_gr = _centered_nanmean(
            md,
            horizontal_gr,
            smoothing_width_ft,
        )
        smooth_typewell_gr = _centered_nanmean(
            clean_typewell_tvt,
            clean_typewell_gr,
            smoothing_width_ft,
        )
        for offset_position, offset_ft in enumerate(offsets_ft):
            reference_gr = np.interp(
                center_tvt + float(offset_ft),
                clean_typewell_tvt,
                smooth_typewell_gr,
                left=np.nan,
                right=np.nan,
            )
            scores, pair_counts = _grouped_pearson(
                smooth_horizontal_gr,
                reference_gr,
                block_index,
                number_of_blocks,
                int(minimum_valid_pairs),
            )
            scale_scores[scale_position, :, offset_position] = scores
            scale_pair_counts[scale_position, :, offset_position] = pair_counts

    valid_scale_count = np.sum(np.isfinite(scale_scores), axis=0)
    ensemble_sum = np.nansum(scale_scores, axis=0)
    minimum_ensemble_scales = min(3, len(normalized_scales))
    ensemble_scores = np.full_like(ensemble_sum, np.nan)
    enough_scales = valid_scale_count >= minimum_ensemble_scales
    ensemble_scores[enough_scales] = (
        ensemble_sum[enough_scales] / valid_scale_count[enough_scales]
    )
    ensemble_pair_counts = np.nanmedian(scale_pair_counts, axis=0)

    all_scores = np.concatenate([scale_scores, ensemble_scores[None, :, :]], axis=0)
    all_pair_counts = np.concatenate(
        [scale_pair_counts, ensemble_pair_counts[None, :, :]],
        axis=0,
    )
    scale_labels = tuple([f"{width:g}ft" for width in normalized_scales] + ["ensemble"])
    summary_tables: list[pd.DataFrame] = []
    for scale_position, scale_label in enumerate(scale_labels):
        smoothing_width = (
            normalized_scales[scale_position]
            if scale_position < len(normalized_scales)
            else np.nan
        )
        summary_tables.append(
            _summarize_one_score_matrix(
                block_table,
                all_scores[scale_position],
                all_pair_counts[scale_position],
                offsets_ft,
                scale_label,
                smoothing_width,
                second_peak_minimum_distance_ft,
            )
        )
    summary = pd.concat(summary_tables, ignore_index=True)

    individual_best = np.full(
        (len(normalized_scales), number_of_blocks),
        np.nan,
        dtype=np.float64,
    )
    for scale_position in range(len(normalized_scales)):
        for block_position in range(number_of_blocks):
            row_scores = scale_scores[scale_position, block_position]
            valid_positions = np.flatnonzero(np.isfinite(row_scores))
            if len(valid_positions):
                best_position = int(
                    valid_positions[np.argmax(row_scores[valid_positions])]
                )
                individual_best[scale_position, block_position] = offsets_ft[
                    best_position
                ]
    ensemble_mask = summary["scale_label"].eq("ensemble")
    for block_position in range(number_of_blocks):
        best_offsets = individual_best[:, block_position]
        best_offsets = best_offsets[np.isfinite(best_offsets)]
        if len(best_offsets):
            lower, upper = np.percentile(best_offsets, [25.0, 75.0])
            row_mask = ensemble_mask & summary["block_position"].eq(block_position)
            summary.loc[row_mask, "scale_best_offset_iqr_ft"] = upper - lower
            summary.loc[row_mask, "scale_best_offset_range_ft"] = (
                np.max(best_offsets) - np.min(best_offsets)
            )

    return BlockScoreLandscape(
        summary=summary,
        block_table=block_table,
        offsets_ft=offsets_ft.copy(),
        score_labels=scale_labels,
        scores=all_scores,
        pair_counts=all_pair_counts,
        block_index=block_index,
    )


def attach_oracle_diagnostics(
    landscape: BlockScoreLandscape,
    true_tvt: np.ndarray,
    center_tvt: np.ndarray,
) -> pd.DataFrame:
    """在 legal score 已生成后，单独附加真实 offset 的覆盖、排名和误差。"""

    true_tvt = np.asarray(true_tvt, dtype=np.float64)
    center_tvt = np.asarray(center_tvt, dtype=np.float64)
    if len(true_tvt) != len(landscape.block_index) or len(center_tvt) != len(
        landscape.block_index
    ):
        raise ValueError("oracle TVT、中心路径与 legal score 行数不一致")
    true_offset = true_tvt - center_tvt
    oracle_rows: list[dict[str, float | int | bool | str]] = []
    minimum_offset = float(np.min(landscape.offsets_ft))
    maximum_offset = float(np.max(landscape.offsets_ft))

    for block_position in range(len(landscape.block_table)):
        block_values = true_offset[landscape.block_index == block_position]
        block_values = block_values[np.isfinite(block_values)]
        if len(block_values):
            true_median = float(np.median(block_values))
            lower, upper = np.percentile(block_values, [25.0, 75.0])
            true_iqr = float(upper - lower)
            in_grid = minimum_offset <= true_median <= maximum_offset
            nearest_position = int(
                np.argmin(np.abs(landscape.offsets_ft - true_median))
            )
            nearest_offset = float(landscape.offsets_ft[nearest_position])
        else:
            true_median = np.nan
            true_iqr = np.nan
            in_grid = False
            nearest_position = -1
            nearest_offset = np.nan

        for score_position, scale_label in enumerate(landscape.score_labels):
            row_scores = landscape.scores[
                score_position,
                block_position,
            ]
            true_score = (
                float(row_scores[nearest_position])
                if nearest_position >= 0 and np.isfinite(row_scores[nearest_position])
                else np.nan
            )
            if np.isfinite(true_score):
                true_rank = float(1 + np.sum(row_scores > true_score))
                true_top5 = bool(true_rank <= 5.0)
            else:
                true_rank = np.nan
                true_top5 = False
            oracle_rows.append(
                {
                    "block_position": int(block_position),
                    "scale_label": scale_label,
                    "true_offset_median_ft": true_median,
                    "true_offset_iqr_ft": true_iqr,
                    "true_offset_in_grid": bool(in_grid),
                    "true_offset_nearest_grid_ft": nearest_offset,
                    "ncc_at_true_offset": true_score,
                    "true_offset_rank": true_rank,
                    "true_offset_top5": true_top5,
                }
            )

    oracle_table = pd.DataFrame(oracle_rows)
    result = landscape.summary.merge(
        oracle_table,
        on=["block_position", "scale_label"],
        how="left",
        validate="one_to_one",
    )
    result["best_offset_abs_error_ft"] = np.abs(
        result["best_offset_ft"] - result["true_offset_median_ft"]
    )
    return result

