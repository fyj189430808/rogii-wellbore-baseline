"""构造 F03b 固定几何路径上的逐行 Typewell offset 得分面特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd


OFFSET_GRID_FT = np.arange(-40.0, 42.0, 2.0, dtype=np.float64)
SMOOTHING_WIDTH_FT = 10.0
NCC_WINDOW_WIDTH_FT = 101.0
MINIMUM_VALID_PAIRS = 20
SECOND_PEAK_MINIMUM_DISTANCE_FT = 6.0
SOFTMAX_TEMPERATURE = 0.1
BEST_BASIN_RADIUS_FT = 4.0

F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS = [
    "f03b_best_offset",
    "f03b_best_ncc",
    "f03b_second_offset",
    "f03b_second_ncc",
    "f03b_peak_separation",
    "f03b_peak_gap",
    "f03b_soft_offset_mean",
    "f03b_soft_offset_std",
    "f03b_offset_entropy",
    "f03b_best_basin_mass",
    "f03b_valid_pair_count",
    "f03b_valid_pair_fraction",
]


def _centered_window_bounds(
    coordinates: np.ndarray,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回每个坐标点的居中窗口左右边界，右边界为开区间。"""

    half_width = float(width) / 2.0
    left = np.searchsorted(coordinates, coordinates - half_width, side="left")
    right = np.searchsorted(coordinates, coordinates + half_width, side="right")
    return left, right


def _window_sum(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """用前缀和快速计算一组不同边界窗口的和。"""

    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return prefix[right] - prefix[left]


def _centered_nanmean(
    coordinates: np.ndarray,
    values: np.ndarray,
    width: float,
) -> np.ndarray:
    """在物理坐标窗口内计算忽略缺失值的居中均值。"""

    left, right = _centered_window_bounds(coordinates, width)
    finite = np.isfinite(values)
    counts = _window_sum(finite.astype(np.float64), left, right)
    sums = _window_sum(np.where(finite, values, 0.0), left, right)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = counts > 0.0
    result[valid] = sums[valid] / counts[valid]
    return result


def _prepare_typewell(typewell_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """清洗、合并重复 TVT，并生成固定 10 ft 平滑 Typewell GR。"""

    missing_columns = {"TVT", "GR"} - set(typewell_df.columns)
    if missing_columns:
        raise ValueError(f"Typewell 缺少列：{sorted(missing_columns)}")

    tvt = typewell_df["TVT"].to_numpy(dtype=np.float64)
    gr = typewell_df["GR"].to_numpy(dtype=np.float64)
    finite = np.isfinite(tvt) & np.isfinite(gr)
    tvt = tvt[finite]
    gr = gr[finite]
    if len(tvt) < 2:
        raise ValueError("Typewell 至少需要两个有效 TVT/GR 点")

    order = np.argsort(tvt, kind="mergesort")
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, inverse = np.unique(tvt, return_inverse=True)
    gr_sum = np.bincount(inverse, weights=gr)
    gr_count = np.bincount(inverse)
    unique_gr = gr_sum / gr_count
    smooth_gr = _centered_nanmean(
        unique_tvt,
        unique_gr,
        SMOOTHING_WIDTH_FT,
    )
    return unique_tvt, smooth_gr


def _rolling_ncc(
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对所有 101 ft 窗口计算公共有效点上的 Pearson NCC。"""

    common = np.isfinite(horizontal_gr) & np.isfinite(reference_gr)
    common_float = common.astype(np.float64)
    x = np.where(common, horizontal_gr, 0.0)
    y = np.where(common, reference_gr, 0.0)

    count = _window_sum(common_float, left, right)
    sum_x = _window_sum(x, left, right)
    sum_y = _window_sum(y, left, right)
    sum_x2 = _window_sum(x * x, left, right)
    sum_y2 = _window_sum(y * y, left, right)
    sum_xy = _window_sum(x * y, left, right)

    ncc = np.full(len(horizontal_gr), np.nan, dtype=np.float64)
    enough = count >= float(MINIMUM_VALID_PAIRS)
    centered_x2 = np.zeros(len(horizontal_gr), dtype=np.float64)
    centered_y2 = np.zeros(len(horizontal_gr), dtype=np.float64)
    centered_xy = np.zeros(len(horizontal_gr), dtype=np.float64)
    centered_x2[enough] = sum_x2[enough] - sum_x[enough] ** 2 / count[enough]
    centered_y2[enough] = sum_y2[enough] - sum_y[enough] ** 2 / count[enough]
    centered_xy[enough] = sum_xy[enough] - (
        sum_x[enough] * sum_y[enough] / count[enough]
    )
    denominator = np.sqrt(
        np.maximum(centered_x2, 0.0) * np.maximum(centered_y2, 0.0)
    )
    valid = enough & (denominator > 1e-12)
    ncc[valid] = np.clip(centered_xy[valid] / denominator[valid], -1.0, 1.0)
    return ncc, count


def _summarize_score_landscape(
    scores: np.ndarray,
    pair_counts: np.ndarray,
    window_counts: np.ndarray,
) -> np.ndarray:
    """把每行 41 个 offset NCC 压缩成固定 12 个模型特征。"""

    output = np.full(
        (scores.shape[0], len(F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS)),
        np.nan,
        dtype=np.float64,
    )
    for row_position in range(scores.shape[0]):
        row_scores = scores[row_position]
        valid = np.isfinite(row_scores)
        if not valid.any():
            maximum_pairs = float(np.max(pair_counts[row_position]))
            output[row_position, 10] = maximum_pairs
            output[row_position, 11] = maximum_pairs / max(
                float(window_counts[row_position]), 1.0
            )
            continue

        valid_positions = np.flatnonzero(valid)
        best_position = int(valid_positions[np.argmax(row_scores[valid])])
        best_offset = float(OFFSET_GRID_FT[best_position])
        best_score = float(row_scores[best_position])
        output[row_position, 0] = best_offset
        output[row_position, 1] = best_score

        second_mask = valid & (
            np.abs(OFFSET_GRID_FT - best_offset)
            >= SECOND_PEAK_MINIMUM_DISTANCE_FT
        )
        if second_mask.any():
            second_positions = np.flatnonzero(second_mask)
            second_position = int(
                second_positions[np.argmax(row_scores[second_mask])]
            )
            second_offset = float(OFFSET_GRID_FT[second_position])
            second_score = float(row_scores[second_position])
            output[row_position, 2] = second_offset
            output[row_position, 3] = second_score
            output[row_position, 4] = abs(best_offset - second_offset)
            output[row_position, 5] = best_score - second_score

        valid_scores = row_scores[valid]
        valid_offsets = OFFSET_GRID_FT[valid]
        unnormalized = np.exp(
            (valid_scores - float(np.max(valid_scores))) / SOFTMAX_TEMPERATURE
        )
        weights = unnormalized / float(np.sum(unnormalized))
        soft_mean = float(np.sum(weights * valid_offsets))
        soft_variance = float(np.sum(weights * (valid_offsets - soft_mean) ** 2))
        output[row_position, 6] = soft_mean
        output[row_position, 7] = np.sqrt(max(soft_variance, 0.0))
        if len(weights) == 1:
            output[row_position, 8] = 0.0
        else:
            output[row_position, 8] = float(
                -np.sum(weights * np.log(np.maximum(weights, 1e-300)))
                / np.log(float(len(weights)))
            )
        basin = np.abs(valid_offsets - best_offset) <= BEST_BASIN_RADIUS_FT
        output[row_position, 9] = float(np.sum(weights[basin]))

        best_pair_count = float(pair_counts[row_position, best_position])
        output[row_position, 10] = best_pair_count
        output[row_position, 11] = best_pair_count / max(
            float(window_counts[row_position]), 1.0
        )
    return output


def build_geometry_landscape_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
) -> pd.DataFrame:
    """返回一口井所有隐藏行的固定几何路径 offset 得分面特征。"""

    missing_columns = {"MD", "Z", "GR", "TVT_input"} - set(horizontal_df.columns)
    if missing_columns:
        raise ValueError(f"水平井缺少列：{sorted(missing_columns)}")

    md = horizontal_df["MD"].to_numpy(dtype=np.float64)
    z = horizontal_df["Z"].to_numpy(dtype=np.float64)
    gr = horizontal_df["GR"].to_numpy(dtype=np.float64)
    tvt_input = horizontal_df["TVT_input"].to_numpy(dtype=np.float64)
    if not np.isfinite(md).all() or not np.isfinite(z).all():
        raise ValueError("水平井 MD/Z 含 NaN 或 Inf")
    if np.any(np.diff(md) < 0.0):
        raise ValueError("水平井 MD 必须单调非降")

    visible_positions = np.flatnonzero(np.isfinite(tvt_input))
    hidden_positions = np.flatnonzero(~np.isfinite(tvt_input))
    if len(visible_positions) == 0 or len(hidden_positions) == 0:
        raise ValueError("水平井必须同时包含可见前缀和隐藏后缀")
    visible_end = int(visible_positions[-1])
    if not np.array_equal(visible_positions, np.arange(visible_end + 1)):
        raise ValueError("TVT_input 可见区不是连续前缀")
    if not np.array_equal(hidden_positions, np.arange(visible_end + 1, len(md))):
        raise ValueError("TVT_input 隐藏区不是连续后缀")

    smooth_horizontal_gr = _centered_nanmean(md, gr, SMOOTHING_WIDTH_FT)
    anchor_tvt = float(tvt_input[visible_end])
    anchor_z = float(z[visible_end])
    candidate_tvt = anchor_tvt - (z - anchor_z)
    typewell_tvt, smooth_typewell_gr = _prepare_typewell(typewell_df)
    left, right = _centered_window_bounds(md, NCC_WINDOW_WIDTH_FT)

    hidden_count = len(hidden_positions)
    score_matrix = np.full(
        (hidden_count, len(OFFSET_GRID_FT)),
        np.nan,
        dtype=np.float64,
    )
    pair_count_matrix = np.zeros_like(score_matrix)
    for offset_position, offset_ft in enumerate(OFFSET_GRID_FT):
        reference_gr = np.interp(
            candidate_tvt + offset_ft,
            typewell_tvt,
            smooth_typewell_gr,
            left=np.nan,
            right=np.nan,
        )
        ncc, pair_count = _rolling_ncc(
            smooth_horizontal_gr,
            reference_gr,
            left,
            right,
        )
        score_matrix[:, offset_position] = ncc[hidden_positions]
        pair_count_matrix[:, offset_position] = pair_count[hidden_positions]

    window_counts = (right - left).astype(np.float64)[hidden_positions]
    feature_values = _summarize_score_landscape(
        score_matrix,
        pair_count_matrix,
        window_counts,
    )
    result = pd.DataFrame(
        feature_values,
        columns=F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    )
    result.insert(0, "row_index", hidden_positions.astype(np.int32))
    return result
