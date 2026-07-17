"""构造严格 outer-fold 安全的邻井相对 ΔU 低频结构特征。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


F07R_FEATURE_COLUMNS = (
    "neighbor_relative_U_prior",
    "neighbor_relative_slope",
    "neighbor_curvature",
    "neighbor_MAD",
    "nearest_trajectory_distance",
    "azimuth_difference",
    "parallel_overlap_length",
    "support_well_count",
    "support_effective_weight",
    "neighbor_disagreement",
)

TARGET_INPUT_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
SOURCE_INPUT_COLUMNS = ("MD", "X", "Y", "Z", "TVT", "TVT_input")

PROFILE_STEP_FT = 25.0
PREFIX_WINDOW_FT = 1000.0
MIN_CALIBRATION_SPAN_FT = 500.0
MIN_CALIBRATION_ROWS = 20
MAX_AZIMUTH_DIFFERENCE_DEG = 45.0
MIN_PARALLEL_OVERLAP_FT = 500.0
LOCAL_DERIVATIVE_HALF_WINDOW_FT = 250.0
MAX_NEIGHBORS = 8
NO_SUPPORT_DISTANCE_FT = 1_000_000.0
NO_SUPPORT_AZIMUTH_DEG = 90.0


@dataclass(frozen=True)
class NeighborProfile:
    """保存一口允许作为标签来源的 outer-train 邻井低频路径。"""

    well_id: str
    pad_id: str
    rows: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} 缺少列：{sorted(missing)}")


def _horizontal_distance(frame: pd.DataFrame) -> np.ndarray:
    x = frame["X"].to_numpy(dtype=np.float64)
    y = frame["Y"].to_numpy(dtype=np.float64)
    step = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    return np.cumsum(step)


def _visible_anchor_position(frame: pd.DataFrame) -> int:
    visible = frame["TVT_input"].notna().to_numpy()
    hidden = ~visible
    if not visible.any() or not hidden.any():
        raise ValueError("水平井必须同时包含可见前缀和自然隐藏段")
    anchor = int(np.flatnonzero(visible)[-1])
    if not visible[: anchor + 1].all() or visible[anchor + 1 :].any():
        raise ValueError("TVT_input 必须是连续可见前缀")
    return anchor


def _downsample_indices(distance: np.ndarray, first_index: int, step_ft: float) -> np.ndarray:
    selected = [int(first_index)]
    last_distance = float(distance[first_index])
    for index in range(first_index + 1, len(distance)):
        current_distance = float(distance[index])
        if current_distance - last_distance >= float(step_ft):
            selected.append(index)
            last_distance = current_distance
    if selected[-1] != len(distance) - 1:
        selected.append(len(distance) - 1)
    return np.asarray(selected, dtype=np.int64)


def build_source_profile(horizontal_df: pd.DataFrame) -> pd.DataFrame:
    """从 outer-train 真值构造 25 ft 低频 XY/U 路径。"""

    _require_columns(horizontal_df, SOURCE_INPUT_COLUMNS, "邻井")
    anchor = _visible_anchor_position(horizontal_df)
    horizontal_s = _horizontal_distance(horizontal_df)
    start_s = float(horizontal_s[anchor]) - PREFIX_WINDOW_FT
    first_index = int(np.searchsorted(horizontal_s, start_s, side="left"))
    selected = _downsample_indices(horizontal_s, first_index, PROFILE_STEP_FT)
    x = horizontal_df["X"].to_numpy(dtype=np.float64)[selected]
    y = horizontal_df["Y"].to_numpy(dtype=np.float64)[selected]
    z = horizontal_df["Z"].to_numpy(dtype=np.float64)[selected]
    tvt = horizontal_df["TVT"].to_numpy(dtype=np.float64)[selected]
    if not np.isfinite(np.column_stack([x, y, z, tvt])).all():
        raise ValueError("邻井 XY/Z/TVT 含非有限值")
    return pd.DataFrame(
        {
            "X": x,
            "Y": y,
            "U": tvt + z,
            "source_s": horizontal_s[selected],
        }
    )


def allowed_source_ids(
    registry_df: pd.DataFrame,
    outer_fold: int,
    target_well_id: str,
) -> list[str]:
    """返回 outer-train 且不同井、不同 pad 的稳定井号列表。"""

    _require_columns(registry_df, ("well_id", "pad_id", "fold"), "fold 注册表")
    well_ids = registry_df["well_id"].astype(str)
    target_rows = registry_df.loc[well_ids == str(target_well_id)]
    if len(target_rows) != 1:
        raise ValueError("目标井必须在 fold 注册表中恰好出现一次")
    target_pad = str(target_rows.iloc[0]["pad_id"])
    allowed = registry_df.loc[
        (registry_df["fold"].astype(int) != int(outer_fold))
        & (registry_df["pad_id"].astype(str) != target_pad)
        & (well_ids != str(target_well_id)),
        "well_id",
    ]
    return sorted(allowed.astype(str).tolist())


def _target_coordinates(horizontal_df: pd.DataFrame) -> dict[str, np.ndarray | int | float]:
    _require_columns(horizontal_df, TARGET_INPUT_COLUMNS, "目标井")
    anchor = _visible_anchor_position(horizontal_df)
    x = horizontal_df["X"].to_numpy(dtype=np.float64)
    y = horizontal_df["Y"].to_numpy(dtype=np.float64)
    z = horizontal_df["Z"].to_numpy(dtype=np.float64)
    tvt_input = horizontal_df["TVT_input"].to_numpy(dtype=np.float64)
    if not np.isfinite(np.column_stack([x, y, z])).all():
        raise ValueError("目标井 XY/Z 含非有限值")
    horizontal_s = _horizontal_distance(horizontal_df)
    h = horizontal_s - float(horizontal_s[anchor])
    direction = np.asarray([x[-1] - x[anchor], y[-1] - y[anchor]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("最后可见点到井尾没有有效水平位移")
    direction /= norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    relative_xy = np.column_stack([x - x[anchor], y - y[anchor]])
    q = relative_xy @ direction
    r = relative_xy @ normal
    return {
        "anchor": anchor,
        "x": x,
        "y": y,
        "z": z,
        "tvt_input": tvt_input,
        "horizontal_s": horizontal_s,
        "h": h,
        "direction": direction,
        "normal": normal,
        "q": q,
        "r": r,
    }


def compute_pair_geometry(
    target_horizontal_df: pd.DataFrame,
    source_profile_df: pd.DataFrame,
) -> dict[str, float]:
    """计算一对轨迹的方向、距离、平行重叠和无量纲排序分数。"""

    _require_columns(source_profile_df, ("X", "Y", "U", "source_s"), "邻井 profile")
    target = _target_coordinates(target_horizontal_df)
    anchor = int(target["anchor"])
    horizontal_s = np.asarray(target["horizontal_s"], dtype=np.float64)
    start_s = float(horizontal_s[anchor]) - PREFIX_WINDOW_FT
    target_segment = horizontal_s >= start_s
    target_xy = np.column_stack(
        [
            np.asarray(target["x"], dtype=np.float64)[target_segment],
            np.asarray(target["y"], dtype=np.float64)[target_segment],
        ]
    )
    source_xy = source_profile_df[["X", "Y"]].to_numpy(dtype=np.float64)
    source_tree = cKDTree(source_xy)
    trajectory_min_distance = float(source_tree.query(target_xy, k=1)[0].min())

    direction = np.asarray(target["direction"], dtype=np.float64)
    normal = np.asarray(target["normal"], dtype=np.float64)
    anchor_xy = np.asarray(
        [
            np.asarray(target["x"], dtype=np.float64)[anchor],
            np.asarray(target["y"], dtype=np.float64)[anchor],
        ]
    )
    source_relative = source_xy - anchor_xy
    source_q = source_relative @ direction
    source_r = source_relative @ normal
    target_q = np.asarray(target["q"], dtype=np.float64)[target_segment]
    target_r = np.asarray(target["r"], dtype=np.float64)[target_segment]

    source_direction = source_xy[-1] - source_xy[0]
    source_norm = float(np.linalg.norm(source_direction))
    if source_norm <= 1.0e-12:
        azimuth_difference = 90.0
    else:
        source_direction /= source_norm
        cosine = float(np.clip(abs(np.dot(direction, source_direction)), 0.0, 1.0))
        azimuth_difference = float(np.degrees(np.arccos(cosine)))

    target_min_q = float(np.min(target_q))
    target_max_q = float(np.max(target_q))
    source_min_q = float(np.min(source_q))
    source_max_q = float(np.max(source_q))
    overlap = max(0.0, min(target_max_q, source_max_q) - max(target_min_q, source_min_q))
    along_track = abs(
        0.5 * (source_min_q + source_max_q)
        - 0.5 * (target_min_q + target_max_q)
    )
    cross_track = abs(float(np.median(source_r)) - float(np.median(target_r)))

    target_endpoints = np.asarray([target_xy[0], target_xy[-1]])
    source_endpoints = np.asarray([source_xy[0], source_xy[-1]])
    direct = np.asarray(
        [
            np.linalg.norm(target_endpoints[0] - source_endpoints[0]),
            np.linalg.norm(target_endpoints[1] - source_endpoints[1]),
        ]
    )
    reversed_pair = np.asarray(
        [
            np.linalg.norm(target_endpoints[0] - source_endpoints[1]),
            np.linalg.norm(target_endpoints[1] - source_endpoints[0]),
        ]
    )
    endpoint_distances = direct if direct.sum() <= reversed_pair.sum() else reversed_pair
    heel_distance = float(endpoint_distances[0])
    toe_distance = float(endpoint_distances[1])

    target_length = max(float(horizontal_s[-1] - start_s), 1.0)
    overlap_fraction = float(np.clip(overlap / target_length, 0.0, 1.0))
    geometry_score = (
        trajectory_min_distance / target_length
        + float(np.sin(np.radians(azimuth_difference)))
        + (1.0 - overlap_fraction)
        + along_track / target_length
        + cross_track / target_length
        + 0.5 * (heel_distance + toe_distance) / target_length
    )
    return {
        "trajectory_min_distance": trajectory_min_distance,
        "azimuth_difference": azimuth_difference,
        "parallel_overlap_length": float(overlap),
        "along_track_distance": float(along_track),
        "cross_track_distance": float(cross_track),
        "heel_distance": heel_distance,
        "toe_distance": toe_distance,
        "geometry_score": float(geometry_score),
    }


def _collapse_projection(q: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(q, kind="mergesort")
    ordered_q = q[order]
    ordered_values = values[order]
    unique_q, starts, counts = np.unique(ordered_q, return_index=True, return_counts=True)
    collapsed = np.empty(len(unique_q), dtype=np.float64)
    for index, (start, count) in enumerate(zip(starts, counts, strict=True)):
        collapsed[index] = float(np.median(ordered_values[start : start + count]))
    return unique_q, collapsed


def _interpolate_inside(q_grid: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    result = np.full(len(query), np.nan, dtype=np.float64)
    supported = (query >= q_grid[0]) & (query <= q_grid[-1])
    result[supported] = np.interp(query[supported], q_grid, values)
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    clean_values = values[valid]
    clean_weights = weights[valid]
    order = np.argsort(clean_values, kind="mergesort")
    clean_values = clean_values[order]
    clean_weights = clean_weights[order]
    if len(clean_values) == 1:
        return float(clean_values[0])
    total_weight = float(np.sum(clean_weights))
    centered_cdf = (np.cumsum(clean_weights) - 0.5 * clean_weights) / total_weight
    return float(np.interp(0.5, centered_cdf, clean_values))


def _candidate_for_neighbor(
    target_horizontal_df: pd.DataFrame,
    target: dict[str, np.ndarray | int | float],
    neighbor: NeighborProfile,
) -> dict[str, object] | None:
    geometry = compute_pair_geometry(target_horizontal_df, neighbor.rows)
    if geometry["azimuth_difference"] > MAX_AZIMUTH_DIFFERENCE_DEG:
        return None
    if geometry["parallel_overlap_length"] < MIN_PARALLEL_OVERLAP_FT:
        return None

    anchor = int(target["anchor"])
    target_x = np.asarray(target["x"], dtype=np.float64)
    target_y = np.asarray(target["y"], dtype=np.float64)
    target_q = np.asarray(target["q"], dtype=np.float64)
    target_h = np.asarray(target["h"], dtype=np.float64)
    direction = np.asarray(target["direction"], dtype=np.float64)
    anchor_xy = np.asarray([target_x[anchor], target_y[anchor]])
    source_xy = neighbor.rows[["X", "Y"]].to_numpy(dtype=np.float64)
    source_q_raw = (source_xy - anchor_xy) @ direction
    source_u_raw = neighbor.rows["U"].to_numpy(dtype=np.float64)
    source_q, source_u = _collapse_projection(source_q_raw, source_u_raw)
    if len(source_q) < 2 or not (source_q[0] <= 0.0 <= source_q[-1]):
        return None

    visible = target_horizontal_df["TVT_input"].notna().to_numpy()
    calibration = visible & (target_h >= -PREFIX_WINDOW_FT)
    source_visible_u = _interpolate_inside(source_q, source_u, target_q)
    calibration &= np.isfinite(source_visible_u)
    calibration_indices = np.flatnonzero(calibration)
    if len(calibration_indices) < MIN_CALIBRATION_ROWS:
        return None
    calibration_span = float(np.ptp(target_h[calibration_indices]))
    if calibration_span < MIN_CALIBRATION_SPAN_FT:
        return None

    target_visible_u = (
        np.asarray(target["tvt_input"], dtype=np.float64)
        + np.asarray(target["z"], dtype=np.float64)
    )
    residual = target_visible_u[calibration_indices] - source_visible_u[calibration_indices]
    design = np.column_stack(
        [
            np.ones(len(calibration_indices), dtype=np.float64),
            target_h[calibration_indices],
        ]
    )
    coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
    intercept = float(coefficients[0])
    slope_correction = float(coefficients[1])
    fitted = design @ coefficients
    calibration_rmse = float(np.sqrt(np.mean((residual - fitted) ** 2)))

    hidden_indices = np.flatnonzero(~visible)
    hidden_q = target_q[hidden_indices]
    hidden_h = target_h[hidden_indices]
    source_hidden_u = _interpolate_inside(source_q, source_u, hidden_q)
    calibrated_u = source_hidden_u + intercept + slope_correction * hidden_h
    target_anchor_u = float(target_visible_u[anchor])
    relative_u = calibrated_u - target_anchor_u

    half_window = LOCAL_DERIVATIVE_HALF_WINDOW_FT
    source_minus = _interpolate_inside(source_q, source_u, hidden_q - half_window)
    source_plus = _interpolate_inside(source_q, source_u, hidden_q + half_window)
    local_slope = (
        source_plus
        - source_minus
        + 2.0 * half_window * slope_correction
    ) / (2.0 * half_window)
    local_curvature = (
        source_plus - 2.0 * source_hidden_u + source_minus
    ) / (half_window**2)
    row_distance = cKDTree(source_xy).query(
        np.column_stack([target_x[hidden_indices], target_y[hidden_indices]]),
        k=1,
    )[0]
    return {
        "geometry": geometry,
        "calibration_rmse": calibration_rmse,
        "relative_u": relative_u,
        "local_slope": local_slope,
        "local_curvature": local_curvature,
        "row_distance": np.asarray(row_distance, dtype=np.float64),
    }


def _validate_neighbor_pool(
    neighbors: list[NeighborProfile],
    *,
    target_well_id: str,
    target_pad_id: str,
) -> None:
    """阻止目标井自身、同 pad 井或重复井进入标签来源池。"""

    seen_well_ids: set[str] = set()
    for neighbor in neighbors:
        source_well_id = str(neighbor.well_id)
        source_pad_id = str(neighbor.pad_id)
        if source_well_id == str(target_well_id):
            raise ValueError("邻井池包含目标井自身")
        if source_pad_id == str(target_pad_id):
            raise ValueError("邻井池包含与目标井同一 pad 的井")
        if source_well_id in seen_well_ids:
            raise ValueError("邻井池包含重复 well_id")
        seen_well_ids.add(source_well_id)


def build_neighbor_prior_features(
    target_horizontal_df: pd.DataFrame,
    neighbors: list[NeighborProfile],
    *,
    target_well_id: str,
    target_pad_id: str,
) -> pd.DataFrame:
    """为一口目标井的自然隐藏行构造固定 10 列 F07R 特征。"""

    _validate_neighbor_pool(
        neighbors,
        target_well_id=str(target_well_id),
        target_pad_id=str(target_pad_id),
    )
    target = _target_coordinates(target_horizontal_df)
    anchor = int(target["anchor"])
    hidden_indices = np.flatnonzero(
        target_horizontal_df["TVT_input"].isna().to_numpy()
    )
    candidates: list[dict[str, object]] = []
    for neighbor in neighbors:
        candidate = _candidate_for_neighbor(target_horizontal_df, target, neighbor)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: float(item["geometry"]["geometry_score"]))
    candidates = candidates[:MAX_NEIGHBORS]

    hidden_row_count = len(hidden_indices)
    output = {"row_index": hidden_indices.astype(np.int64)}
    output["neighbor_relative_U_prior"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["neighbor_relative_slope"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["neighbor_curvature"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["neighbor_MAD"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["nearest_trajectory_distance"] = np.full(
        hidden_row_count,
        NO_SUPPORT_DISTANCE_FT,
        dtype=np.float64,
    )
    output["azimuth_difference"] = np.full(
        hidden_row_count,
        NO_SUPPORT_AZIMUTH_DEG,
        dtype=np.float64,
    )
    output["parallel_overlap_length"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["support_well_count"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["support_effective_weight"] = np.zeros(hidden_row_count, dtype=np.float64)
    output["neighbor_disagreement"] = np.zeros(hidden_row_count, dtype=np.float64)
    if not candidates:
        return pd.DataFrame(output)

    rmse_values = np.asarray(
        [float(candidate["calibration_rmse"]) for candidate in candidates],
        dtype=np.float64,
    )
    positive_rmse = rmse_values[rmse_values > 1.0e-12]
    rmse_scale = float(np.median(positive_rmse)) if len(positive_rmse) else 1.0
    base_weights = np.asarray(
        [
            np.exp(-float(candidate["geometry"]["geometry_score"]))
            / (1.0 + float(candidate["calibration_rmse"]) / rmse_scale)
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    relative = np.vstack(
        [np.asarray(candidate["relative_u"], dtype=np.float64) for candidate in candidates]
    )
    slopes = np.vstack(
        [np.asarray(candidate["local_slope"], dtype=np.float64) for candidate in candidates]
    )
    curvatures = np.vstack(
        [np.asarray(candidate["local_curvature"], dtype=np.float64) for candidate in candidates]
    )
    row_distances = np.vstack(
        [np.asarray(candidate["row_distance"], dtype=np.float64) for candidate in candidates]
    )
    azimuths = np.asarray(
        [float(candidate["geometry"]["azimuth_difference"]) for candidate in candidates]
    )
    overlaps = np.asarray(
        [float(candidate["geometry"]["parallel_overlap_length"]) for candidate in candidates]
    )

    for row in range(len(hidden_indices)):
        supported = np.isfinite(relative[:, row]) & (base_weights > 0.0)
        if not supported.any():
            continue
        values = relative[supported, row]
        weights = base_weights[supported]
        total_weight = float(np.sum(weights))
        mean = float(np.sum(values * weights) / total_weight)
        median = _weighted_median(values, weights)
        output["neighbor_relative_U_prior"][row] = mean
        slope_value = _weighted_mean(
            slopes[supported, row], weights
        )
        if np.isfinite(slope_value):
            output["neighbor_relative_slope"][row] = slope_value
        curvature_value = _weighted_mean(
            curvatures[supported, row], weights
        )
        if np.isfinite(curvature_value):
            output["neighbor_curvature"][row] = curvature_value
        output["neighbor_MAD"][row] = _weighted_median(
            np.abs(values - median), weights
        )
        output["nearest_trajectory_distance"][row] = float(
            np.min(row_distances[supported, row])
        )
        output["azimuth_difference"][row] = _weighted_mean(azimuths[supported], weights)
        output["parallel_overlap_length"][row] = _weighted_mean(overlaps[supported], weights)
        output["support_well_count"][row] = float(np.sum(supported))
        output["support_effective_weight"][row] = float(
            total_weight**2 / np.sum(weights**2)
        )
        if len(values) == 1:
            output["neighbor_disagreement"][row] = 0.0
        else:
            output["neighbor_disagreement"][row] = float(
                np.sqrt(np.sum(weights * (values - mean) ** 2) / total_weight)
            )
    result = pd.DataFrame(output)
    feature_values = result[list(F07R_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("F07R 特征含非有限值")
    if int(anchor) < 0:
        raise AssertionError("不可达的锚点")
    return result
