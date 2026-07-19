"""P2-S03：用最近 outer-train dense EGFDU 点构造无平面外推的 TVT 路径。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.p2_s01_outer_fold_surface import (
    LEGAL_TARGET_COLUMNS,
    horizontal_distance,
    select_control_indices,
)
from src.p2_s02_dense_relative_gradient import _validate_dense_source_points


class DenseNearestSurfaceIndex:
    """保存一个 outer fold 的 dense source 点，并返回查询 XY 的最近合法点。"""

    def __init__(self, source_points: pd.DataFrame) -> None:
        self.points = _validate_dense_source_points(source_points)
        self.source_xy = self.points[["source_x", "source_y"]].to_numpy(dtype=np.float64)
        self.source_surface = self.points["source_surface"].to_numpy(dtype=np.float64)
        self.source_well_ids = self.points["source_well_id"].to_numpy(dtype=str)
        self.unique_source_well_ids = frozenset(self.source_well_ids.tolist())
        self.tree = cKDTree(self.source_xy)

    def predict_nearest_surface(
        self,
        query_xy: np.ndarray,
        excluded_well_id: str | None = None,
    ) -> pd.DataFrame:
        """批量查询最近点；正式训练特征阶段可额外排除目标井自身。"""

        queries = np.asarray(query_xy, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 2:
            raise ValueError("query_xy 必须为 shape=[查询点数, 2]")
        if not bool(np.isfinite(queries).all()):
            raise ValueError("query_xy 含缺失或非有限值")
        if excluded_well_id is not None and len(self.unique_source_well_ids) <= 1:
            raise ValueError("排除目标井后没有可用 source 井")

        if excluded_well_id is None or str(excluded_well_id) not in self.unique_source_well_ids:
            distances, indices = self.tree.query(queries, k=1, workers=1)
            distances = np.asarray(distances, dtype=np.float64)
            indices = np.asarray(indices, dtype=np.int64)
        else:
            selected_distances: list[float] = []
            selected_indices: list[int] = []
            for query in queries:
                candidate_count = min(len(self.points), 32)
                while True:
                    candidate_distances, candidate_indices = self.tree.query(
                        query,
                        k=candidate_count,
                        workers=1,
                    )
                    candidate_distances = np.atleast_1d(candidate_distances)
                    candidate_indices = np.atleast_1d(candidate_indices)
                    chosen_index: int | None = None
                    chosen_distance = float("nan")
                    for distance, point_index in zip(candidate_distances, candidate_indices):
                        if str(self.source_well_ids[int(point_index)]) == str(excluded_well_id):
                            continue
                        chosen_index = int(point_index)
                        chosen_distance = float(distance)
                        break
                    if chosen_index is not None:
                        selected_indices.append(chosen_index)
                        selected_distances.append(chosen_distance)
                        break
                    if candidate_count == len(self.points):
                        raise ValueError("扫描全部 dense 点后仍只有被排除的目标井")
                    candidate_count = min(len(self.points), candidate_count * 2)
            indices = np.asarray(selected_indices, dtype=np.int64)
            distances = np.asarray(selected_distances, dtype=np.float64)

        return pd.DataFrame(
            {
                "nearest_surface": self.source_surface[indices],
                "nearest_source_distance": distances,
                "nearest_source_well_id": self.source_well_ids[indices],
            }
        )


def _interpolate_control_values(
    all_distance: np.ndarray,
    control_distance: np.ndarray,
    control_values: np.ndarray,
) -> np.ndarray:
    """把控制点插回逐行，并删除竖直段的重复水平距离。"""

    _, reversed_index = np.unique(control_distance[::-1], return_index=True)
    original_positions = len(control_distance) - 1 - reversed_index
    order = np.argsort(control_distance[original_positions], kind="stable")
    positions = original_positions[order]
    unique_distance = control_distance[positions]
    unique_values = control_values[positions]
    if len(unique_distance) == 1:
        return np.full(len(all_distance), float(unique_values[0]), dtype=np.float64)
    return np.interp(all_distance, unique_distance, unique_values)


def build_legal_nearest_surface_path(
    target_horizontal_df: pd.DataFrame,
    nearest_surface_index: Any,
    target_control_step_horizontal_ft: float,
    excluded_source_well_id: str | None = None,
) -> pd.DataFrame:
    """只读合法目标列，用最近 dense surface 和可见前缀校准生成隐藏路径。"""

    missing = [column for column in LEGAL_TARGET_COLUMNS if column not in target_horizontal_df.columns]
    if missing:
        raise ValueError(f"目标水平井缺少列：{missing}")
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
    control_xy = target.loc[control_indices, ["X", "Y"]].to_numpy(dtype=np.float64)
    control_prediction = nearest_surface_index.predict_nearest_surface(
        control_xy,
        excluded_well_id=excluded_source_well_id,
    )
    control_distance = distance[control_indices]
    nearest_surface_all = _interpolate_control_values(
        distance,
        control_distance,
        control_prediction["nearest_surface"].to_numpy(dtype=np.float64),
    )
    nearest_distance_all = _interpolate_control_values(
        distance,
        control_distance,
        control_prediction["nearest_source_distance"].to_numpy(dtype=np.float64),
    )

    source_ids = control_prediction["nearest_source_well_id"].astype(str).tolist()
    source_changed = np.zeros(len(source_ids), dtype=np.float64)
    for position in range(1, len(source_ids)):
        source_changed[position] = float(source_ids[position] != source_ids[position - 1])
    source_change_all = _interpolate_control_values(
        distance,
        control_distance,
        source_changed,
    )

    visible_tvt = target.loc[visible_mask, "TVT_input"].to_numpy(dtype=np.float64)
    visible_z = target.loc[visible_mask, "Z"].to_numpy(dtype=np.float64)
    visible_residual = visible_tvt + visible_z - nearest_surface_all[visible_mask]
    prefix_offset = float(np.median(visible_residual))
    prefix_calibration_rmse = float(
        np.sqrt(np.mean(np.square(visible_residual - prefix_offset)))
    )
    z_all = target["Z"].to_numpy(dtype=np.float64)
    predicted_tvt_all = nearest_surface_all + prefix_offset - z_all
    last_visible_tvt = float(target.loc[visible_positions[-1], "TVT_input"])

    return pd.DataFrame(
        {
            "row_index": hidden_positions.astype(np.int64),
            "nearest_surface": nearest_surface_all[hidden_mask],
            "surface_tvt": predicted_tvt_all[hidden_mask],
            "surface_tvt_delta": predicted_tvt_all[hidden_mask] - last_visible_tvt,
            "nearest_source_distance": nearest_distance_all[hidden_mask],
            "source_change_fraction": source_change_all[hidden_mask],
            "prefix_offset": np.full(hidden_positions.size, prefix_offset, dtype=np.float64),
            "prefix_calibration_rmse": np.full(
                hidden_positions.size,
                prefix_calibration_rmse,
                dtype=np.float64,
            ),
            "unique_source_wells_on_controls": np.full(
                hidden_positions.size,
                len(set(source_ids)),
                dtype=np.float64,
            ),
        }
    )
