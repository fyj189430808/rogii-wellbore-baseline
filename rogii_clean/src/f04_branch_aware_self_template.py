"""F04：用当前井可见前缀建立按方向和连续访问分支的 GR 自模板。"""

from __future__ import annotations

import numpy as np
import pandas as pd


TVT_BIN_WIDTH_FT = 0.5
MINIMUM_SEGMENT_ROWS = 20
NCC_WINDOW_WIDTH_FT = 101.0
MINIMUM_WINDOW_PAIRS = 3
OFFSET_GRID_FT = np.array([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=np.float64)

F04_FEATURE_COLUMNS = [
    "f04_coverage_flag",
    "f04_nearest_observed_tvt_distance",
    "f04_same_direction_support_count",
    "f04_opposite_direction_support_count",
    "f04_number_of_visits",
    "f04_within_bin_gr_mad",
    "f04_self_predicted_gr",
    "f04_self_abs_error",
    "f04_self_ncc_101ft",
    "f04_best_offset",
    "f04_best_error",
    "f04_score_gap",
]

REQUIRED_COLUMNS = ["MD", "Z", "GR", "TVT_input"]


def _fill_zero_directions(directions: np.ndarray) -> np.ndarray:
    """把平坦点继承为最近的非零方向，整段平坦时保留 0。"""

    filled = np.asarray(directions, dtype=np.int8).copy()
    for index in range(1, len(filled)):
        if filled[index] == 0:
            filled[index] = filled[index - 1]
    for index in range(len(filled) - 2, -1, -1):
        if filled[index] == 0:
            filled[index] = filled[index + 1]
    return filled


def _direction_runs(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每个连续同方向 run 的左闭、右开边界。"""

    if len(directions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.flatnonzero(
        np.concatenate(([True], directions[1:] != directions[:-1]))
    )
    ends = np.concatenate((starts[1:], [len(directions)]))
    return starts.astype(np.int64), ends.astype(np.int64)


def merge_short_direction_runs(
    directions: np.ndarray,
    minimum_run_rows: int = MINIMUM_SEGMENT_ROWS,
) -> np.ndarray:
    """把少于固定行数的方向碎片并入相邻访问段。"""

    if minimum_run_rows < 1:
        raise ValueError("minimum_run_rows 必须至少为 1")
    merged = _fill_zero_directions(np.asarray(directions, dtype=np.int8))
    if len(merged) == 0:
        return merged

    while True:
        starts, ends = _direction_runs(merged)
        if len(starts) <= 1:
            break
        lengths = ends - starts
        short_runs = np.flatnonzero(lengths < minimum_run_rows)
        if len(short_runs) == 0:
            break
        run_index = int(short_runs[0])
        if run_index == 0:
            replacement = merged[int(starts[1])]
        elif run_index == len(starts) - 1:
            replacement = merged[int(starts[run_index - 1])]
        else:
            left_value = merged[int(starts[run_index - 1])]
            right_value = merged[int(starts[run_index + 1])]
            if left_value == right_value:
                replacement = left_value
            elif lengths[run_index - 1] >= lengths[run_index + 1]:
                replacement = left_value
            else:
                replacement = right_value
        merged[int(starts[run_index]) : int(ends[run_index])] = replacement
    return merged


def _directions_from_tvt(
    tvt_values: np.ndarray,
    previous_tvt: float | None = None,
) -> np.ndarray:
    """按 TVT 随 MD 的局部变化得到 -1/0/+1 方向。"""

    values = np.asarray(tvt_values, dtype=np.float64)
    if len(values) == 0:
        return np.array([], dtype=np.int8)
    differences = np.empty(len(values), dtype=np.float64)
    if previous_tvt is None:
        differences[0] = values[1] - values[0] if len(values) > 1 else 0.0
    else:
        differences[0] = values[0] - float(previous_tvt)
    if len(values) > 1:
        differences[1:] = np.diff(values)
    return _fill_zero_directions(np.sign(differences).astype(np.int8))


def _round_tvt_bin(values: np.ndarray, bin_width: float) -> np.ndarray:
    """把 TVT 固定到最近的 0.5 ft bin。"""

    return np.floor(np.asarray(values) / bin_width + 0.5) * bin_width


def _build_templates(
    visible_tvt: np.ndarray,
    visible_gr: np.ndarray,
    directions: np.ndarray,
    bin_width: float,
) -> list[dict]:
    """每个连续同方向 segment 单独保存逐 bin GR 中位数、MAD 和数量。"""

    starts, ends = _direction_runs(directions)
    templates: list[dict] = []
    for segment_id, (start, end) in enumerate(zip(starts, ends)):
        segment_tvt = visible_tvt[int(start) : int(end)]
        segment_gr = visible_gr[int(start) : int(end)]
        valid = np.isfinite(segment_tvt) & np.isfinite(segment_gr)
        if not valid.any():
            continue
        valid_tvt = segment_tvt[valid]
        valid_gr = segment_gr[valid]
        binned_tvt = _round_tvt_bin(valid_tvt, bin_width)
        unique_bins = np.unique(binned_tvt)
        medians = np.empty(len(unique_bins), dtype=np.float64)
        mads = np.empty(len(unique_bins), dtype=np.float64)
        counts = np.empty(len(unique_bins), dtype=np.float64)
        for bin_index, bin_value in enumerate(unique_bins):
            values = valid_gr[binned_tvt == bin_value]
            median = float(np.median(values))
            medians[bin_index] = median
            mads[bin_index] = float(np.median(np.abs(values - median)))
            counts[bin_index] = float(len(values))
        templates.append(
            {
                "segment_id": int(segment_id),
                "direction": int(directions[int(start)]),
                "minimum_tvt": float(np.min(valid_tvt)),
                "maximum_tvt": float(np.max(valid_tvt)),
                "bins": unique_bins,
                "median_gr": medians,
                "mad_gr": mads,
                "counts": counts,
            }
        )
    return templates


def _nearest_bin_indices(sorted_bins: np.ndarray, values: np.ndarray) -> np.ndarray:
    """在一个 segment 的已观测 bin 中找到每个候选 TVT 的最近位置。"""

    right = np.searchsorted(sorted_bins, values, side="left")
    right = np.clip(right, 0, len(sorted_bins) - 1)
    left = np.clip(right - 1, 0, len(sorted_bins) - 1)
    choose_left = np.abs(values - sorted_bins[left]) <= np.abs(
        values - sorted_bins[right]
    )
    return np.where(choose_left, left, right)


def _query_templates(
    templates: list[dict],
    query_tvt: np.ndarray,
    query_direction: np.ndarray,
    bin_width: float,
    include_support: bool,
) -> dict[str, np.ndarray]:
    """逐行选择同方向且 TVT 范围最近的 segment，并读取最近 bin。"""

    query_values = np.asarray(query_tvt, dtype=np.float64)
    directions = np.asarray(query_direction, dtype=np.int8)
    row_count = len(query_values)
    predicted_gr = np.full(row_count, np.nan, dtype=np.float64)
    predicted_mad = np.full(row_count, np.nan, dtype=np.float64)
    nearest_distance = np.full(row_count, np.nan, dtype=np.float64)
    coverage = np.zeros(row_count, dtype=np.float64)

    if not templates:
        return {
            "predicted_gr": predicted_gr,
            "mad": predicted_mad,
            "distance": nearest_distance,
            "coverage": coverage,
            "same_count": np.zeros(row_count),
            "opposite_count": np.zeros(row_count),
            "visits": np.zeros(row_count),
        }

    for direction in np.unique(directions):
        row_positions = np.flatnonzero(directions == direction)
        same_templates = [
            index
            for index, template in enumerate(templates)
            if int(template["direction"]) == int(direction)
        ]
        has_same_direction = bool(same_templates)
        candidate_templates = same_templates or list(range(len(templates)))
        values = query_values[row_positions]
        range_distances = np.empty(
            (len(values), len(candidate_templates)), dtype=np.float64
        )
        for column, template_index in enumerate(candidate_templates):
            template = templates[template_index]
            below = np.maximum(float(template["minimum_tvt"]) - values, 0.0)
            above = np.maximum(values - float(template["maximum_tvt"]), 0.0)
            range_distances[:, column] = below + above
        selected_columns = np.argmin(range_distances, axis=1)
        selected_templates = np.asarray(candidate_templates)[selected_columns]

        for template_index in np.unique(selected_templates):
            local_mask = selected_templates == template_index
            positions = row_positions[local_mask]
            local_values = query_values[positions]
            template = templates[int(template_index)]
            bin_indices = _nearest_bin_indices(template["bins"], local_values)
            selected_bins = template["bins"][bin_indices]
            distances = np.abs(local_values - selected_bins)
            predicted_gr[positions] = template["median_gr"][bin_indices]
            predicted_mad[positions] = template["mad_gr"][bin_indices]
            nearest_distance[positions] = distances
            if has_same_direction:
                coverage[positions] = (
                    distances <= (bin_width / 2.0 + 1e-9)
                ).astype(np.float64)

    same_count = np.zeros(row_count, dtype=np.float64)
    opposite_count = np.zeros(row_count, dtype=np.float64)
    visits = np.zeros(row_count, dtype=np.float64)
    if include_support:
        count_lookup: dict[tuple[int, float], float] = {}
        visit_lookup: dict[float, float] = {}
        for template in templates:
            direction = int(template["direction"])
            for bin_value, count in zip(template["bins"], template["counts"]):
                key_bin = round(float(bin_value), 6)
                key = (direction, key_bin)
                count_lookup[key] = count_lookup.get(key, 0.0) + float(count)
                visit_lookup[key_bin] = visit_lookup.get(key_bin, 0.0) + 1.0
        query_bins = _round_tvt_bin(query_values, bin_width)
        for row_index, (bin_value, direction) in enumerate(
            zip(query_bins, directions)
        ):
            key_bin = round(float(bin_value), 6)
            same_count[row_index] = count_lookup.get((int(direction), key_bin), 0.0)
            opposite_count[row_index] = count_lookup.get((-int(direction), key_bin), 0.0)
            visits[row_index] = visit_lookup.get(key_bin, 0.0)

    return {
        "predicted_gr": predicted_gr,
        "mad": predicted_mad,
        "distance": nearest_distance,
        "coverage": coverage,
        "same_count": same_count,
        "opposite_count": opposite_count,
        "visits": visits,
    }


def _window_bounds(md: np.ndarray, width_ft: float) -> tuple[np.ndarray, np.ndarray]:
    """返回以每个隐藏行为中心、固定 MD 宽度窗口的左右边界。"""

    half_width = width_ft / 2.0
    left = np.searchsorted(md, md - half_width, side="left")
    right = np.searchsorted(md, md + half_width, side="right")
    return left, right


def _rolling_mean_error(
    measured: np.ndarray,
    predicted: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """在固定 MD 窗口内计算有效 GR 对的平均绝对误差。"""

    valid = np.isfinite(measured) & np.isfinite(predicted)
    values = np.where(valid, np.abs(measured - predicted), 0.0)
    count_prefix = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    value_prefix = np.concatenate(([0.0], np.cumsum(values)))
    counts = count_prefix[right] - count_prefix[left]
    sums = value_prefix[right] - value_prefix[left]
    result = np.full(len(measured), np.nan, dtype=np.float64)
    usable = counts >= MINIMUM_WINDOW_PAIRS
    result[usable] = sums[usable] / counts[usable]
    return result


def _rolling_ncc(
    measured: np.ndarray,
    predicted: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """在固定 MD 窗口内计算实测 GR 与自模板 GR 的相关系数。"""

    valid = np.isfinite(measured) & np.isfinite(predicted)
    x = np.where(valid, measured, 0.0)
    y = np.where(valid, predicted, 0.0)
    arrays = [valid.astype(np.float64), x, y, x * x, y * y, x * y]
    window_sums = []
    for values in arrays:
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        window_sums.append(prefix[right] - prefix[left])
    count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = window_sums
    covariance = sum_xy - sum_x * sum_y / np.maximum(count, 1.0)
    variance_x = sum_xx - sum_x * sum_x / np.maximum(count, 1.0)
    variance_y = sum_yy - sum_y * sum_y / np.maximum(count, 1.0)
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    result = np.full(len(measured), np.nan, dtype=np.float64)
    usable = (count >= MINIMUM_WINDOW_PAIRS) & (denominator > 1e-12)
    result[usable] = covariance[usable] / denominator[usable]
    return np.clip(result, -1.0, 1.0)


def build_f04_self_template_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """只用当前井测试期可见列，为自然隐藏行生成 12 个 F04 特征。"""

    missing = set(REQUIRED_COLUMNS) - set(horizontal_df.columns)
    if missing:
        raise ValueError(f"水平井缺少列：{sorted(missing)}")
    visible_mask = horizontal_df["TVT_input"].notna().to_numpy()
    hidden_mask = ~visible_mask
    if not visible_mask.any() or not hidden_mask.any():
        raise ValueError(f"井 {well_id} 必须同时有可见前缀和隐藏后缀")
    visible_positions = np.flatnonzero(visible_mask)
    hidden_positions = np.flatnonzero(hidden_mask)
    visible_end = int(visible_positions[-1])
    if not np.array_equal(visible_positions, np.arange(visible_end + 1)):
        raise ValueError(f"井 {well_id} 的 TVT_input 可见区不是连续前缀")
    if not np.array_equal(hidden_positions, np.arange(visible_end + 1, len(horizontal_df))):
        raise ValueError(f"井 {well_id} 的隐藏区不是连续后缀")

    md = horizontal_df["MD"].to_numpy(dtype=np.float64)
    if np.any(np.diff(md) < 0.0):
        raise ValueError(f"井 {well_id} 的 MD 不是单调非降")
    z = horizontal_df["Z"].to_numpy(dtype=np.float64)
    gr = horizontal_df["GR"].to_numpy(dtype=np.float64)
    visible_tvt = horizontal_df.loc[visible_mask, "TVT_input"].to_numpy(
        dtype=np.float64
    )
    visible_gr = gr[visible_mask]
    visible_direction = _directions_from_tvt(visible_tvt)
    visible_direction = merge_short_direction_runs(visible_direction)
    templates = _build_templates(
        visible_tvt,
        visible_gr,
        visible_direction,
        TVT_BIN_WIDTH_FT,
    )

    last_visible_tvt = float(visible_tvt[-1])
    z_visible_end = float(z[visible_end])
    hidden_candidate_tvt = last_visible_tvt - (
        z[hidden_positions] - z_visible_end
    )
    hidden_direction = _directions_from_tvt(
        hidden_candidate_tvt,
        previous_tvt=last_visible_tvt,
    )
    main_query = _query_templates(
        templates,
        hidden_candidate_tvt,
        hidden_direction,
        TVT_BIN_WIDTH_FT,
        include_support=True,
    )

    hidden_md = md[hidden_positions]
    hidden_gr = gr[hidden_positions]
    left, right = _window_bounds(hidden_md, NCC_WINDOW_WIDTH_FT)
    self_predicted_gr = main_query["predicted_gr"]
    self_abs_error = np.abs(hidden_gr - self_predicted_gr)
    self_ncc = _rolling_ncc(hidden_gr, self_predicted_gr, left, right)

    offset_errors = np.full(
        (len(hidden_positions), len(OFFSET_GRID_FT)), np.nan, dtype=np.float64
    )
    for offset_index, offset in enumerate(OFFSET_GRID_FT):
        offset_query = _query_templates(
            templates,
            hidden_candidate_tvt + float(offset),
            hidden_direction,
            TVT_BIN_WIDTH_FT,
            include_support=False,
        )
        offset_errors[:, offset_index] = _rolling_mean_error(
            hidden_gr,
            offset_query["predicted_gr"],
            left,
            right,
        )
    sortable_errors = np.where(np.isfinite(offset_errors), offset_errors, np.inf)
    best_indices = np.argmin(sortable_errors, axis=1)
    row_indices = np.arange(len(hidden_positions))
    best_errors = sortable_errors[row_indices, best_indices]
    valid_best = np.isfinite(best_errors)
    best_offsets = OFFSET_GRID_FT[best_indices].astype(np.float64)
    best_offsets[~valid_best] = np.nan
    best_errors[~valid_best] = np.nan
    sorted_errors = np.sort(sortable_errors, axis=1)
    score_gap = sorted_errors[:, 1] - sorted_errors[:, 0]
    score_gap[~np.isfinite(score_gap)] = np.nan

    result = pd.DataFrame(
        {
            "well_id": str(well_id),
            "fold": int(fold),
            "row_index": hidden_positions.astype(np.int32),
            "f04_coverage_flag": main_query["coverage"],
            "f04_nearest_observed_tvt_distance": main_query["distance"],
            "f04_same_direction_support_count": main_query["same_count"],
            "f04_opposite_direction_support_count": main_query["opposite_count"],
            "f04_number_of_visits": main_query["visits"],
            "f04_within_bin_gr_mad": main_query["mad"],
            "f04_self_predicted_gr": self_predicted_gr,
            "f04_self_abs_error": self_abs_error,
            "f04_self_ncc_101ft": self_ncc,
            "f04_best_offset": best_offsets,
            "f04_best_error": best_errors,
            "f04_score_gap": score_gap,
        }
    )
    values = result[F04_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError(f"井 {well_id} 的 F04 特征含 Inf")
    return result
