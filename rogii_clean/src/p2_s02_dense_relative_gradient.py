"""P2-S02：从 outer-train dense EGFDU 点估计局部梯度并积分相对结构路径。"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.p2_s01_outer_fold_surface import (
    LEGAL_TARGET_COLUMNS,
    horizontal_distance,
    select_control_indices,
)


DENSE_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_well_id",
    "source_row_index",
    "source_distance_ft",
    "source_x",
    "source_y",
    "source_surface",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], frame_name: str) -> None:
    """检查必要列，避免把错误数据版本带入空间监督。"""

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} 缺少列：{missing}")


def build_dense_source_profile(
    well_id: str,
    horizontal_df: pd.DataFrame,
    surface_name: str,
    control_step_horizontal_ft: float,
) -> pd.DataFrame:
    """沿一口 source 井的累计 XY 距离等距保留 EGFDU 控制点。"""

    required = ("X", "Y", str(surface_name))
    _require_columns(horizontal_df, required, f"source 井 {well_id}")
    numeric = horizontal_df.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    valid_mask = numeric.notna().all(axis=1).to_numpy(dtype=bool)
    if not bool(valid_mask.any()):
        raise ValueError(f"source 井 {well_id} 没有有效 X/Y/{surface_name}")
    if not np.isfinite(control_step_horizontal_ft) or float(control_step_horizontal_ft) <= 0.0:
        raise ValueError("source 控制点间隔必须为正数")

    original_rows = np.flatnonzero(valid_mask)
    valid = numeric.loc[valid_mask].reset_index(drop=True)
    distance = horizontal_distance(valid)
    selected: set[int] = {0, len(valid) - 1}
    next_distance = float(control_step_horizontal_ft)
    while next_distance < float(distance[-1]):
        selected.add(int(np.searchsorted(distance, next_distance, side="left")))
        next_distance += float(control_step_horizontal_ft)
    selected_positions = np.asarray(sorted(selected), dtype=np.int64)

    return pd.DataFrame(
        {
            "source_well_id": np.full(len(selected_positions), str(well_id), dtype=object),
            "source_row_index": original_rows[selected_positions].astype(np.int64),
            "source_distance_ft": distance[selected_positions],
            "source_x": valid.loc[selected_positions, "X"].to_numpy(dtype=np.float64),
            "source_y": valid.loc[selected_positions, "Y"].to_numpy(dtype=np.float64),
            "source_surface": valid.loc[
                selected_positions,
                str(surface_name),
            ].to_numpy(dtype=np.float64),
        }
    )


def _validate_dense_source_points(source_points: pd.DataFrame) -> pd.DataFrame:
    """规范 dense source 点的类型和排序。"""

    _require_columns(source_points, DENSE_SOURCE_COLUMNS, "dense surface source 表")
    points = source_points.loc[:, list(DENSE_SOURCE_COLUMNS)].copy()
    points["source_well_id"] = points["source_well_id"].astype(str)
    numeric_columns = [column for column in DENSE_SOURCE_COLUMNS if column != "source_well_id"]
    for column in numeric_columns:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    if not bool(np.isfinite(points[numeric_columns]).all().all()):
        raise ValueError("dense surface source 表含缺失或非有限值")
    if bool(points.duplicated(["source_well_id", "source_row_index"]).any()):
        raise ValueError("dense surface source 表含重复井内行键")
    return points.sort_values(
        ["source_well_id", "source_distance_ft", "source_row_index"],
        kind="stable",
    ).reset_index(drop=True)


def compute_outer_train_gradient_cap(source_points: pd.DataFrame, quantile: float) -> float:
    """只用 outer-train 井自身相邻控制点计算方向斜率分位数上限。"""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("梯度上限分位数必须位于 0 和 1 之间")
    points = _validate_dense_source_points(source_points)
    directional_slopes: list[np.ndarray] = []
    for _, profile in points.groupby("source_well_id", sort=False):
        distance = profile["source_distance_ft"].to_numpy(dtype=np.float64)
        surface = profile["source_surface"].to_numpy(dtype=np.float64)
        distance_change = np.diff(distance)
        surface_change = np.diff(surface)
        valid = distance_change > 0.0
        if bool(valid.any()):
            directional_slopes.append(np.abs(surface_change[valid] / distance_change[valid]))
    if not directional_slopes:
        raise ValueError("outer-train dense source 无法计算任何井内方向斜率")
    all_slopes = np.concatenate(directional_slopes)
    cap = float(np.quantile(all_slopes, float(quantile)))
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("outer-train P99 梯度上限不是正有限值")
    return cap


def reverse_source_profile_gradients(source_points: pd.DataFrame) -> pd.DataFrame:
    """保持每井 surface 中位值不变，把井内相对变化方向反转。"""

    points = _validate_dense_source_points(source_points)
    reversed_points = points.copy()
    per_well_median = reversed_points.groupby("source_well_id")["source_surface"].transform(
        "median"
    )
    reversed_points["source_surface"] = (
        2.0 * per_well_median - reversed_points["source_surface"]
    )
    return reversed_points


def _maximum_angular_gap_degrees(query_xy: np.ndarray, neighbor_xy: np.ndarray) -> float:
    """计算邻点方位最大空缺角；大于 180 度表示查询点不被邻点包围。"""

    vectors = neighbor_xy - query_xy
    nonzero = np.linalg.norm(vectors, axis=1) > 0.0
    vectors = vectors[nonzero]
    if len(vectors) < 3:
        return 360.0
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    angles = np.sort(angles)
    wrapped = np.r_[angles, angles[0] + 2.0 * np.pi]
    return float(np.degrees(np.max(np.diff(wrapped))))


class DenseSurfaceGradientIndex:
    """保存一个 outer fold 的 dense source 点，并查询不同井的局部梯度。"""

    def __init__(self, source_points: pd.DataFrame) -> None:
        self.points = _validate_dense_source_points(source_points)
        self.source_xy = self.points[["source_x", "source_y"]].to_numpy(dtype=np.float64)
        self.source_surface = self.points["source_surface"].to_numpy(dtype=np.float64)
        self.source_well_ids = self.points["source_well_id"].to_numpy(dtype=str)
        # 唯一井集合只构造一次；逐控制点重复把约十万行字符串转成 set 会占掉绝大多数时间。
        self.unique_source_well_ids = frozenset(self.source_well_ids.tolist())
        self.tree = cKDTree(self.source_xy)

    def _nearest_distinct_source_points(
        self,
        query_xy: np.ndarray,
        number_of_wells: int,
        excluded_well_id: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按距离扫描 dense 点，每口 source 井只保留最近的一个点。"""

        available_well_count = len(self.unique_source_well_ids)
        if excluded_well_id is not None and str(excluded_well_id) in self.unique_source_well_ids:
            available_well_count -= 1
        if available_well_count < int(number_of_wells):
            raise ValueError("排除目标井后，不同 source 井数量不足")

        candidate_count = min(len(self.points), max(256, int(number_of_wells) * 8))
        while True:
            distances, indices = self.tree.query(query_xy, k=candidate_count, workers=1)
            distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
            indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
            selected_indices: list[int] = []
            selected_distances: list[float] = []
            seen_wells: set[str] = set()
            for distance, point_index in zip(distances, indices):
                source_well_id = str(self.source_well_ids[point_index])
                if excluded_well_id is not None and source_well_id == str(excluded_well_id):
                    continue
                if source_well_id in seen_wells:
                    continue
                seen_wells.add(source_well_id)
                selected_indices.append(int(point_index))
                selected_distances.append(float(distance))
                if len(selected_indices) == int(number_of_wells):
                    return (
                        np.asarray(selected_indices, dtype=np.int64),
                        np.asarray(selected_distances, dtype=np.float64),
                    )
            if candidate_count == len(self.points):
                raise ValueError("扫描全部 dense 点后仍找不到足够的不同 source 井")
            candidate_count = min(len(self.points), candidate_count * 2)

    def predict_gradients(
        self,
        query_xy: np.ndarray,
        nearest_distinct_source_wells: int,
        distance_weight_epsilon: float,
        gradient_magnitude_cap: float,
        excluded_well_id: str | None = None,
    ) -> pd.DataFrame:
        """在每个查询点拟合局部平面，只返回受 outer-train 上限约束的梯度。"""

        queries = np.asarray(query_xy, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 2:
            raise ValueError("query_xy 必须为 shape=[查询点数, 2]")
        if not bool(np.isfinite(queries).all()):
            raise ValueError("query_xy 含缺失或非有限值")
        if int(nearest_distinct_source_wells) < 3:
            raise ValueError("局部梯度至少需要 3 口不同 source 井")
        if float(distance_weight_epsilon) <= 0.0:
            raise ValueError("距离权重 epsilon 必须为正数")
        if not np.isfinite(gradient_magnitude_cap) or float(gradient_magnitude_cap) <= 0.0:
            raise ValueError("梯度模长上限必须为正有限值")

        output_rows: list[dict[str, Any]] = []
        for query in queries:
            selected_indices, distances = self._nearest_distinct_source_points(
                query_xy=query,
                number_of_wells=int(nearest_distinct_source_wells),
                excluded_well_id=excluded_well_id,
            )
            neighbor_xy = self.source_xy[selected_indices]
            neighbor_surface = self.source_surface[selected_indices]
            neighbor_well_ids = tuple(self.source_well_ids[selected_indices].tolist())

            positive_distances = distances[distances > 0.0]
            coordinate_scale = (
                max(float(np.median(positive_distances)), 1.0)
                if positive_distances.size
                else 1.0
            )
            centered_xy = (neighbor_xy - query) / coordinate_scale
            design_matrix = np.column_stack(
                [
                    np.ones(len(selected_indices), dtype=np.float64),
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

            fitted_surface = design_matrix @ coefficients
            local_plane_rmse = float(
                np.sqrt(
                    np.sum(weights * np.square(neighbor_surface - fitted_surface))
                    / np.sum(weights)
                )
            )
            if len(singular_values) and float(np.min(singular_values)) > 0.0:
                condition_number = float(np.max(singular_values) / np.min(singular_values))
            else:
                condition_number = float("inf")

            if int(matrix_rank) < 3:
                raw_gradient_x = 0.0
                raw_gradient_y = 0.0
            else:
                raw_gradient_x = float(coefficients[1] / coordinate_scale)
                raw_gradient_y = float(coefficients[2] / coordinate_scale)
            raw_magnitude = float(np.hypot(raw_gradient_x, raw_gradient_y))
            if raw_magnitude > float(gradient_magnitude_cap):
                shrinkage = float(gradient_magnitude_cap) / raw_magnitude
                gradient_x = raw_gradient_x * shrinkage
                gradient_y = raw_gradient_y * shrinkage
                gradient_was_clipped = 1.0
            else:
                gradient_x = raw_gradient_x
                gradient_y = raw_gradient_y
                gradient_was_clipped = 0.0
            gradient_magnitude = float(np.hypot(gradient_x, gradient_y))
            maximum_gap = _maximum_angular_gap_degrees(query, neighbor_xy)

            output_rows.append(
                {
                    "gradient_x": gradient_x,
                    "gradient_y": gradient_y,
                    "raw_gradient_magnitude": raw_magnitude,
                    "gradient_magnitude": gradient_magnitude,
                    "gradient_was_clipped": gradient_was_clipped,
                    "nearest_source_distance": float(distances[0]),
                    "kth_source_distance": float(distances[-1]),
                    "local_plane_rmse": local_plane_rmse,
                    "local_plane_rank": float(matrix_rank),
                    "local_plane_condition": condition_number,
                    "maximum_angular_gap_degrees": maximum_gap,
                    "outside_support": float(maximum_gap > 180.0),
                    "neighbor_well_ids": neighbor_well_ids,
                }
            )
        return pd.DataFrame(output_rows)


def _interpolate_control_values(
    all_distance: np.ndarray,
    control_distance: np.ndarray,
    control_values: np.ndarray,
) -> np.ndarray:
    """把数值控制点插回逐行，并处理竖直段的重复水平距离。"""

    _, reversed_index = np.unique(control_distance[::-1], return_index=True)
    original_positions = len(control_distance) - 1 - reversed_index
    order = np.argsort(control_distance[original_positions], kind="stable")
    positions = original_positions[order]
    unique_distance = control_distance[positions]
    unique_values = control_values[positions]
    if len(unique_distance) == 1:
        return np.full(len(all_distance), float(unique_values[0]), dtype=np.float64)
    return np.interp(all_distance, unique_distance, unique_values)


def build_legal_relative_gradient_path(
    target_horizontal_df: pd.DataFrame,
    dense_surface_index: Any,
    nearest_distinct_source_wells: int,
    target_control_step_horizontal_ft: float,
    distance_weight_epsilon: float,
    gradient_magnitude_cap: float,
    excluded_source_well_id: str | None = None,
) -> pd.DataFrame:
    """只读目标合法列，从最后可见点开始积分 outer-train surface 梯度。"""

    _require_columns(target_horizontal_df, LEGAL_TARGET_COLUMNS, "目标水平井")
    target = target_horizontal_df.loc[:, list(LEGAL_TARGET_COLUMNS)].copy()
    for column in LEGAL_TARGET_COLUMNS:
        target[column] = pd.to_numeric(target[column], errors="coerce")
    if not bool(np.isfinite(target[["MD", "X", "Y", "Z"]]).all().all()):
        raise ValueError("目标井 MD/X/Y/Z 含缺失或非有限值")

    visible_mask = target["TVT_input"].notna().to_numpy(dtype=bool)
    hidden_mask = ~visible_mask
    visible_positions = np.flatnonzero(visible_mask)
    hidden_positions = np.flatnonzero(hidden_mask)
    if visible_positions.size == 0 or hidden_positions.size == 0:
        raise ValueError("目标井必须有可见前缀和隐藏后缀")
    if int(visible_positions[-1]) + 1 != int(hidden_positions[0]) or bool(visible_mask[hidden_positions[0]:].any()):
        raise ValueError("TVT_input 必须为连续可见前缀")

    distance = horizontal_distance(target)
    control_indices = select_control_indices(
        cumulative_horizontal_distance=distance,
        visible_mask=visible_mask,
        step_ft=float(target_control_step_horizontal_ft),
    )
    anchor_row_index = int(visible_positions[-1])
    anchor_control_positions = np.flatnonzero(control_indices == anchor_row_index)
    if anchor_control_positions.size != 1:
        raise ValueError("最后可见锚点没有被控制点表唯一保留")
    anchor_control_position = int(anchor_control_positions[0])

    control_xy = target.loc[control_indices, ["X", "Y"]].to_numpy(dtype=np.float64)
    control_prediction = dense_surface_index.predict_gradients(
        query_xy=control_xy,
        nearest_distinct_source_wells=int(nearest_distinct_source_wells),
        distance_weight_epsilon=float(distance_weight_epsilon),
        gradient_magnitude_cap=float(gradient_magnitude_cap),
        excluded_well_id=excluded_source_well_id,
    )

    gradient_xy = control_prediction[["gradient_x", "gradient_y"]].to_numpy(dtype=np.float64)
    relative_surface = np.zeros(len(control_indices), dtype=np.float64)
    for control_position in range(anchor_control_position + 1, len(control_indices)):
        xy_change = control_xy[control_position] - control_xy[control_position - 1]
        average_gradient = 0.5 * (
            gradient_xy[control_position] + gradient_xy[control_position - 1]
        )
        relative_surface[control_position] = (
            relative_surface[control_position - 1]
            + float(np.dot(average_gradient, xy_change))
        )
    for control_position in range(anchor_control_position - 1, -1, -1):
        xy_change = control_xy[control_position + 1] - control_xy[control_position]
        average_gradient = 0.5 * (
            gradient_xy[control_position + 1] + gradient_xy[control_position]
        )
        relative_surface[control_position] = (
            relative_surface[control_position + 1]
            - float(np.dot(average_gradient, xy_change))
        )

    neighbor_change = np.zeros(len(control_indices), dtype=np.float64)
    neighbor_sets = [set(value) for value in control_prediction["neighbor_well_ids"]]
    for control_position in range(1, len(control_indices)):
        overlap = len(neighbor_sets[control_position] & neighbor_sets[control_position - 1])
        denominator = max(len(neighbor_sets[control_position]), 1)
        neighbor_change[control_position] = 1.0 - overlap / denominator

    control_distance = distance[control_indices]
    relative_surface_all = _interpolate_control_values(
        all_distance=distance,
        control_distance=control_distance,
        control_values=relative_surface,
    )
    interpolated: dict[str, np.ndarray] = {}
    numeric_columns = [
        column for column in control_prediction.columns if column != "neighbor_well_ids"
    ]
    for column in numeric_columns:
        interpolated[column] = _interpolate_control_values(
            all_distance=distance,
            control_distance=control_distance,
            control_values=control_prediction[column].to_numpy(dtype=np.float64),
        )
    interpolated_neighbor_change = _interpolate_control_values(
        all_distance=distance,
        control_distance=control_distance,
        control_values=neighbor_change,
    )

    last_visible_tvt = float(target.loc[anchor_row_index, "TVT_input"])
    anchor_z = float(target.loc[anchor_row_index, "Z"])
    z_all = target["Z"].to_numpy(dtype=np.float64)
    predicted_tvt_all = last_visible_tvt + relative_surface_all - (z_all - anchor_z)
    visible_truth = target.loc[visible_mask, "TVT_input"].to_numpy(dtype=np.float64)
    prefix_backtest_rmse = float(
        np.sqrt(np.mean(np.square(predicted_tvt_all[visible_mask] - visible_truth)))
    )

    output: dict[str, np.ndarray] = {
        "row_index": hidden_positions.astype(np.int64),
        "relative_surface_delta": relative_surface_all[hidden_mask],
        "surface_tvt": predicted_tvt_all[hidden_mask],
        "surface_tvt_delta": predicted_tvt_all[hidden_mask] - last_visible_tvt,
    }
    for column, values in interpolated.items():
        output[column] = values[hidden_mask]
    output["neighbor_change_fraction"] = interpolated_neighbor_change[hidden_mask]
    output["prefix_backtest_rmse"] = np.full(
        hidden_positions.size,
        prefix_backtest_rmse,
        dtype=np.float64,
    )
    output["gradient_magnitude_cap"] = np.full(
        hidden_positions.size,
        float(gradient_magnitude_cap),
        dtype=np.float64,
    )
    return pd.DataFrame(output)
