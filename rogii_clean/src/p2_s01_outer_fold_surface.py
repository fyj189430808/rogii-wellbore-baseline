"""P2-S01：用严格 outer-fold 地层面生成目标井的确定性 TVT 路径。"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# 合法目标函数只会读取这五列；TVT 和六个 surface 即使存在也不会被访问。
LEGAL_TARGET_COLUMNS: Final[tuple[str, ...]] = ("MD", "X", "Y", "Z", "TVT_input")

# source 表一口井只能贡献一行，避免长井因为采样点多而支配局部平面。
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "well_id",
    "source_x",
    "source_y",
    "source_surface",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], frame_name: str) -> None:
    """检查输入列，尽早暴露数据版本或拼接错误。"""

    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} 缺少列：{missing_columns}")


def build_source_representative(
    well_id: str,
    horizontal_df: pd.DataFrame,
    surface_name: str,
) -> dict[str, float | str]:
    """把一口 outer-train 井压缩成一个等权的 XY/surface 中位代表点。"""

    required_columns = ("X", "Y", str(surface_name))
    _require_columns(horizontal_df, required_columns, f"source 井 {well_id}")

    # 三列必须在同一行同时有效，避免三次中位数来自互不重叠的数据区间。
    numeric_rows = horizontal_df.loc[:, list(required_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid_rows = numeric_rows.notna().all(axis=1)
    if not bool(valid_rows.any()):
        raise ValueError(f"source 井 {well_id} 没有有效的 X/Y/{surface_name} 行")

    valid_values = numeric_rows.loc[valid_rows]
    return {
        "well_id": str(well_id),
        "source_x": float(valid_values["X"].median()),
        "source_y": float(valid_values["Y"].median()),
        "source_surface": float(valid_values[str(surface_name)].median()),
    }


def allowed_surface_source_ids(
    registry_df: pd.DataFrame,
    outer_fold: int,
    target_well_id: str | None = None,
) -> list[str]:
    """返回合法 source 井：排除完整 outer-valid fold，并按需排除目标井自身。"""

    _require_columns(registry_df, ("well_id", "fold"), "fold 注册表")
    registry = registry_df.loc[:, ["well_id", "fold"]].copy()
    registry["well_id"] = registry["well_id"].astype(str)
    registry["fold"] = pd.to_numeric(registry["fold"], errors="raise").astype(int)
    if bool(registry["well_id"].duplicated().any()):
        raise ValueError("fold 注册表含重复 well_id")

    allowed_mask = registry["fold"].ne(int(outer_fold))
    if target_well_id is not None:
        allowed_mask &= registry["well_id"].ne(str(target_well_id))

    return sorted(registry.loc[allowed_mask, "well_id"].tolist())


def horizontal_distance(horizontal_df: pd.DataFrame) -> np.ndarray:
    """计算沿井轨迹累计的 XY 水平距离，输出 shape=[行数]，单位沿用原始 XY。"""

    _require_columns(horizontal_df, ("X", "Y"), "目标水平井")
    xy = horizontal_df.loc[:, ["X", "Y"]].apply(pd.to_numeric, errors="coerce")
    xy_values = xy.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(xy_values).all()):
        raise ValueError("目标井 X/Y 含缺失或非有限值")

    if len(xy_values) == 0:
        return np.empty(0, dtype=np.float64)

    step_distance = np.linalg.norm(np.diff(xy_values, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(step_distance, dtype=np.float64)]


def select_control_indices(
    cumulative_horizontal_distance: np.ndarray,
    visible_mask: np.ndarray,
    step_ft: float,
) -> np.ndarray:
    """每隔固定水平距离选控制点，并强制保留首行、最后可见行和末行。"""

    distance = np.asarray(cumulative_horizontal_distance, dtype=np.float64)
    visible = np.asarray(visible_mask, dtype=bool)
    if distance.ndim != 1 or visible.ndim != 1 or len(distance) != len(visible):
        raise ValueError("距离和 visible_mask 必须是一维且长度相同")
    if len(distance) == 0:
        raise ValueError("目标井不能为空")
    if not bool(np.isfinite(distance).all()) or bool(np.any(np.diff(distance) < 0.0)):
        raise ValueError("累计水平距离必须有限且单调不减")
    if not np.isfinite(step_ft) or float(step_ft) <= 0.0:
        raise ValueError("控制点间隔必须为正数")

    visible_positions = np.flatnonzero(visible)
    if visible_positions.size == 0:
        raise ValueError("目标井没有可见 TVT_input 前缀")

    selected_indices: set[int] = {0, len(distance) - 1, int(visible_positions[-1])}
    next_distance = float(distance[0]) + float(step_ft)
    final_distance = float(distance[-1])
    while next_distance < final_distance:
        candidate_index = int(np.searchsorted(distance, next_distance, side="left"))
        if candidate_index < len(distance):
            selected_indices.add(candidate_index)
        next_distance += float(step_ft)

    return np.asarray(sorted(selected_indices), dtype=np.int64)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """计算加权中位数，供局部平面秩不足时稳定回退。"""

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    position = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    position = min(position, len(ordered_values) - 1)
    return float(ordered_values[position])


def _validate_source_table(source_table: pd.DataFrame) -> pd.DataFrame:
    """规范 source 表顺序和类型，使读取顺序不会改变近邻结果。"""

    _require_columns(source_table, SOURCE_COLUMNS, "surface source 表")
    sources = source_table.loc[:, list(SOURCE_COLUMNS)].copy()
    sources["well_id"] = sources["well_id"].astype(str)
    for column in ("source_x", "source_y", "source_surface"):
        sources[column] = pd.to_numeric(sources[column], errors="coerce")
    if bool(sources["well_id"].duplicated().any()):
        raise ValueError("surface source 表中一口井出现了多行")
    if not bool(np.isfinite(sources[["source_x", "source_y", "source_surface"]]).all().all()):
        raise ValueError("surface source 表含缺失或非有限值")
    return sources.sort_values("well_id", kind="stable").reset_index(drop=True)


def local_plane_predictions(
    query_xy: np.ndarray,
    source_table: pd.DataFrame,
    nearest_source_wells: int,
    distance_weight_epsilon: float,
    excluded_well_id: str | None = None,
) -> pd.DataFrame:
    """在每个查询 XY 处用最近若干口不同 source 井拟合加权局部平面。"""

    queries = np.asarray(query_xy, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query_xy 必须为 shape=[查询点数, 2]")
    if not bool(np.isfinite(queries).all()):
        raise ValueError("query_xy 含缺失或非有限值")
    if int(nearest_source_wells) < 3:
        raise ValueError("局部平面至少需要 3 口 source 井")
    if not np.isfinite(distance_weight_epsilon) or float(distance_weight_epsilon) <= 0.0:
        raise ValueError("距离权重 epsilon 必须为正数")

    sources = _validate_source_table(source_table)
    if excluded_well_id is not None:
        sources = sources.loc[sources["well_id"].ne(str(excluded_well_id))].reset_index(drop=True)
    if len(sources) < int(nearest_source_wells):
        raise ValueError("排除非法 source 后，剩余井数不足以拟合固定局部平面")

    source_xy = sources[["source_x", "source_y"]].to_numpy(dtype=np.float64)
    source_surface = sources["source_surface"].to_numpy(dtype=np.float64)
    source_tree = cKDTree(source_xy)
    neighbor_distance, neighbor_index = source_tree.query(
        queries,
        k=int(nearest_source_wells),
        workers=1,
    )
    if int(nearest_source_wells) == 1:
        neighbor_distance = neighbor_distance[:, None]
        neighbor_index = neighbor_index[:, None]

    output_rows: list[dict[str, float]] = []
    for query_number in range(len(queries)):
        distances = np.asarray(neighbor_distance[query_number], dtype=np.float64)
        indices = np.asarray(neighbor_index[query_number], dtype=np.int64)
        neighbor_xy = source_xy[indices]
        neighbor_surface = source_surface[indices]

        # 尺度只用于数值稳定；最小 1 个原始坐标单位避免重合点导致除零。
        positive_distances = distances[distances > 0.0]
        if positive_distances.size:
            coordinate_scale = max(float(np.median(positive_distances)), 1.0)
        else:
            coordinate_scale = 1.0

        centered_xy = (neighbor_xy - queries[query_number]) / coordinate_scale
        design_matrix = np.column_stack(
            [
                np.ones(len(indices), dtype=np.float64),
                centered_xy[:, 0],
                centered_xy[:, 1],
            ]
        )
        normalized_distance = distances / coordinate_scale
        weights = 1.0 / (normalized_distance + float(distance_weight_epsilon))
        square_root_weights = np.sqrt(weights)
        weighted_design = design_matrix * square_root_weights[:, None]
        weighted_surface = neighbor_surface * square_root_weights
        coefficients, _, matrix_rank, singular_values = np.linalg.lstsq(
            weighted_design,
            weighted_surface,
            rcond=None,
        )

        used_fallback = float(int(matrix_rank) < 3)
        if used_fallback:
            predicted_surface = _weighted_median(neighbor_surface, weights)
            fitted_neighbor_surface = np.full(len(indices), predicted_surface, dtype=np.float64)
        else:
            predicted_surface = float(coefficients[0])
            fitted_neighbor_surface = design_matrix @ coefficients

        weighted_squared_error = weights * np.square(neighbor_surface - fitted_neighbor_surface)
        local_plane_rmse = float(np.sqrt(np.sum(weighted_squared_error) / np.sum(weights)))
        if len(singular_values) and float(np.min(singular_values)) > 0.0:
            condition_number = float(np.max(singular_values) / np.min(singular_values))
        else:
            condition_number = float("inf")

        output_rows.append(
            {
                "predicted_surface": float(predicted_surface),
                "nearest_source_distance": float(distances[0]),
                "kth_source_distance": float(distances[-1]),
                "local_plane_rmse": local_plane_rmse,
                "local_plane_rank": float(matrix_rank),
                "local_plane_condition": condition_number,
                "used_fallback": used_fallback,
            }
        )

    return pd.DataFrame(output_rows)


def _interpolate_control_values(
    all_distance: np.ndarray,
    control_distance: np.ndarray,
    control_values: np.ndarray,
) -> np.ndarray:
    """把控制点数值线性插回逐行；相同水平距离只保留最后一个控制点。"""

    # np.interp 要求横坐标严格递增；竖直段会产生重复水平距离。
    reversed_unique_distance, reversed_index = np.unique(
        control_distance[::-1],
        return_index=True,
    )
    kept_reversed_positions = reversed_index
    kept_original_positions = len(control_distance) - 1 - kept_reversed_positions
    order = np.argsort(control_distance[kept_original_positions], kind="stable")
    unique_positions = kept_original_positions[order]
    unique_distance = control_distance[unique_positions]
    unique_values = control_values[unique_positions]

    if len(unique_distance) == 1:
        return np.full(len(all_distance), float(unique_values[0]), dtype=np.float64)
    return np.interp(all_distance, unique_distance, unique_values)


def build_legal_surface_path(
    target_horizontal_df: pd.DataFrame,
    source_table: pd.DataFrame,
    nearest_source_wells: int,
    control_step_horizontal_ft: float,
    distance_weight_epsilon: float,
    excluded_source_well_id: str | None = None,
) -> pd.DataFrame:
    """仅凭合法目标列和 outer-train surface，生成自然隐藏段的 TVT 候选路径。"""

    _require_columns(target_horizontal_df, LEGAL_TARGET_COLUMNS, "目标水平井")

    # 立即复制白名单列，保证后续代码不可能意外读取目标 TVT 或 surface。
    target = target_horizontal_df.loc[:, list(LEGAL_TARGET_COLUMNS)].copy()
    for column in LEGAL_TARGET_COLUMNS:
        target[column] = pd.to_numeric(target[column], errors="coerce")

    visible_mask = target["TVT_input"].notna().to_numpy(dtype=bool)
    hidden_mask = ~visible_mask
    visible_positions = np.flatnonzero(visible_mask)
    hidden_positions = np.flatnonzero(hidden_mask)
    if visible_positions.size == 0 or hidden_positions.size == 0:
        raise ValueError("目标井必须同时包含可见前缀和自然隐藏后缀")
    if int(visible_positions[-1]) + 1 != int(hidden_positions[0]) or bool(visible_mask[hidden_positions[0]:].any()):
        raise ValueError("TVT_input 必须是连续可见前缀，之后全部为隐藏段")
    if not bool(np.isfinite(target[["MD", "X", "Y", "Z"]]).all().all()):
        raise ValueError("目标井 MD/X/Y/Z 含缺失或非有限值")

    distance = horizontal_distance(target)
    control_indices = select_control_indices(
        cumulative_horizontal_distance=distance,
        visible_mask=visible_mask,
        step_ft=float(control_step_horizontal_ft),
    )
    control_xy = target.loc[control_indices, ["X", "Y"]].to_numpy(dtype=np.float64)
    control_prediction = local_plane_predictions(
        query_xy=control_xy,
        source_table=source_table,
        nearest_source_wells=int(nearest_source_wells),
        distance_weight_epsilon=float(distance_weight_epsilon),
        excluded_well_id=excluded_source_well_id,
    )

    control_distance = distance[control_indices]
    interpolated_columns: dict[str, np.ndarray] = {}
    for column in control_prediction.columns:
        interpolated_columns[column] = _interpolate_control_values(
            all_distance=distance,
            control_distance=control_distance,
            control_values=control_prediction[column].to_numpy(dtype=np.float64),
        )

    predicted_surface = interpolated_columns["predicted_surface"]
    visible_tvt = target.loc[visible_mask, "TVT_input"].to_numpy(dtype=np.float64)
    visible_z = target.loc[visible_mask, "Z"].to_numpy(dtype=np.float64)
    visible_surface = predicted_surface[visible_mask]
    visible_residual = visible_tvt + visible_z - visible_surface
    prefix_offset = float(np.median(visible_residual))
    centered_visible_residual = visible_residual - prefix_offset
    prefix_calibration_rmse = float(np.sqrt(np.mean(np.square(centered_visible_residual))))

    z_all = target["Z"].to_numpy(dtype=np.float64)
    surface_tvt_all = predicted_surface + prefix_offset - z_all
    last_visible_tvt = float(target.loc[visible_positions[-1], "TVT_input"])

    output = pd.DataFrame(
        {
            "row_index": hidden_positions.astype(np.int64),
            "surface_hat": predicted_surface[hidden_mask],
            "surface_tvt": surface_tvt_all[hidden_mask],
            "surface_tvt_delta": surface_tvt_all[hidden_mask] - last_visible_tvt,
            "nearest_source_distance": interpolated_columns["nearest_source_distance"][hidden_mask],
            "kth_source_distance": interpolated_columns["kth_source_distance"][hidden_mask],
            "local_plane_rmse": interpolated_columns["local_plane_rmse"][hidden_mask],
            "local_plane_rank": interpolated_columns["local_plane_rank"][hidden_mask],
            "local_plane_condition": interpolated_columns["local_plane_condition"][hidden_mask],
            "used_fallback": interpolated_columns["used_fallback"][hidden_mask],
            "prefix_offset": np.full(hidden_positions.size, prefix_offset, dtype=np.float64),
            "prefix_calibration_rmse": np.full(
                hidden_positions.size,
                prefix_calibration_rmse,
                dtype=np.float64,
            ),
        }
    )
    return output


def permute_source_surfaces(source_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """固定打乱 source_surface 与 XY 的对应关系，生成空间负对照。"""

    sources = _validate_source_table(source_table)
    random_generator = np.random.default_rng(int(seed))
    permutation = random_generator.permutation(len(sources))
    if len(sources) > 1 and bool(np.array_equal(permutation, np.arange(len(sources)))):
        permutation = np.roll(permutation, 1)
    permuted = sources.copy()
    permuted["source_surface"] = sources["source_surface"].to_numpy(dtype=np.float64)[permutation]
    return permuted
