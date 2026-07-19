"""P3-N01m：邻井平均残差的加权中位数、固定收缩和负对照工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.p3_n01a_neighbor_residual_audit import (
    TrajectoryGeometry,
    build_fixed_shuffle,
)


ETA_PRIOR_WEIGHT = 1.0
MAXIMUM_NEIGHBORS = 8
SHUFFLE_SEED = 20260719


def _finite_xy(values: np.ndarray, name: str) -> np.ndarray:
    xy = np.asarray(values, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError(f"{name} 必须是至少两行的 [行数, 2] XY 数组")
    if not np.isfinite(xy).all():
        raise ValueError(f"{name} 含非有限值")
    return xy


def trajectory_direction(xy: np.ndarray) -> np.ndarray:
    """用与冻结 N01a 完全相同的首尾差计算单位方向。"""

    values = _finite_xy(xy, "xy")
    direction = values[-1] - values[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("xy 首尾重合，无法定义方向")
    return direction / norm


def compute_cached_trajectory_geometry(
    target_xy: np.ndarray,
    target_direction: np.ndarray,
    source_xy: np.ndarray,
    source_direction: np.ndarray,
    source_spatial_index: cKDTree,
    minimum_distance_ft: float | None = None,
) -> TrajectoryGeometry:
    """复用 source KDTree/方向，同时保持冻结几何公式逐字段不变。"""

    target = _finite_xy(target_xy, "target_xy")
    source = _finite_xy(source_xy, "source_xy")
    target_unit = np.asarray(target_direction, dtype=np.float64)
    source_unit = np.asarray(source_direction, dtype=np.float64)
    if target_unit.shape != (2,) or source_unit.shape != (2,):
        raise ValueError("target/source direction 必须是长度为 2 的向量")
    if not np.isfinite(target_unit).all() or not np.isfinite(source_unit).all():
        raise ValueError("target/source direction 含非有限值")
    if minimum_distance_ft is None:
        minimum_distance = float(source_spatial_index.query(target, k=1)[0].min())
    else:
        minimum_distance = float(minimum_distance_ft)
        if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
            raise ValueError("minimum_distance_ft 必须是非负有限值")
    signed_cosine = float(np.clip(np.dot(target_unit, source_unit), -1.0, 1.0))
    undirected_cosine = abs(signed_cosine)
    azimuth_difference = float(np.degrees(np.arccos(undirected_cosine)))
    target_projection = target @ target_unit
    source_projection = source @ target_unit
    overlap = max(
        0.0,
        min(float(target_projection.max()), float(source_projection.max()))
        - max(float(target_projection.min()), float(source_projection.min())),
    )
    return TrajectoryGeometry(
        minimum_distance_ft=minimum_distance,
        azimuth_difference_deg=azimuth_difference,
        parallel_overlap_ft=float(overlap),
        reverse_source_controls=bool(signed_cosine < 0.0),
    )


@dataclass(frozen=True)
class NeighborMeanEstimate:
    """一口目标井的邻井加权中位数及收缩后常数偏移。"""

    weighted_median_residual: float
    predicted_offset_ft: float
    total_weight: float
    eta: float
    neighbor_count: int


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """返回排序后累计权重第一次达到总权重一半的残差值。"""

    residuals = np.asarray(values, dtype=np.float64)
    residual_weights = np.asarray(weights, dtype=np.float64)
    if residuals.ndim != 1 or residual_weights.shape != residuals.shape:
        raise ValueError("values 和 weights 必须是一维等长数组")
    if len(residuals) == 0:
        raise ValueError("加权中位数至少需要一个值")
    if not np.isfinite(residuals).all() or not np.isfinite(residual_weights).all():
        raise ValueError("残差或权重含 NaN/Inf")
    if (residual_weights <= 0.0).any():
        raise ValueError("加权中位数的权重必须全部大于零")
    order = np.argsort(residuals, kind="mergesort")
    ordered_values = residuals[order]
    ordered_weights = residual_weights[order]
    threshold = 0.5 * float(np.sum(ordered_weights))
    position = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[position])


def aggregate_neighbor_mean(
    residuals: np.ndarray,
    weights: np.ndarray,
) -> NeighborMeanEstimate:
    """先取加权中位数，再用 eta=W/(W+1) 向 0 收缩；无邻居固定回零。"""

    values = np.asarray(residuals, dtype=np.float64)
    neighbor_weights = np.asarray(weights, dtype=np.float64)
    if values.shape != neighbor_weights.shape or values.ndim != 1:
        raise ValueError("residuals 和 weights 必须是一维等长数组")
    if len(values) == 0:
        return NeighborMeanEstimate(0.0, 0.0, 0.0, 0.0, 0)
    median = weighted_median(values, neighbor_weights)
    total_weight = float(np.sum(neighbor_weights))
    eta = total_weight / (total_weight + ETA_PRIOR_WEIGHT)
    return NeighborMeanEstimate(
        weighted_median_residual=median,
        predicted_offset_ft=float(eta * median),
        total_weight=total_weight,
        eta=float(eta),
        neighbor_count=int(len(values)),
    )


def build_within_fold_derangement(
    sources: pd.DataFrame,
    seed: int = SHUFFLE_SEED,
) -> pd.DataFrame:
    """每个 inner fold 独立使用同一固定 seed，生成无固定点 donor 映射。"""

    required = {"well_id", "fold"}
    if missing := required.difference(sources.columns):
        raise ValueError(f"source 表缺列：{sorted(missing)}")
    keys = sources[["well_id", "fold"]].copy()
    keys["well_id"] = keys["well_id"].astype(str)
    if keys["well_id"].duplicated().any():
        raise ValueError("source 井号重复")
    rows: list[dict[str, int | str]] = []
    for fold, fold_rows in keys.groupby("fold", sort=True):
        source_ids = fold_rows["well_id"].astype(str).tolist()
        mapping = build_fixed_shuffle(source_ids, seed=int(seed))
        rows.extend(
            {
                "source_well_id": source_id,
                "donor_well_id": mapping[source_id],
                "fold": int(fold),
            }
            for source_id in sorted(source_ids)
        )
    return pd.DataFrame(rows).sort_values("source_well_id").reset_index(drop=True)


def select_top_neighbors(
    candidates: pd.DataFrame,
    target_well_id: str,
    maximum_neighbors: int = MAXIMUM_NEIGHBORS,
) -> pd.DataFrame:
    """排除自身后，按权重降序、距离升序、井号升序固定取前八口。"""

    required = {"source_well_id", "weight", "distance_ft"}
    if missing := required.difference(candidates.columns):
        raise ValueError(f"邻井候选缺列：{sorted(missing)}")
    selected = candidates.copy()
    selected["source_well_id"] = selected["source_well_id"].astype(str)
    selected = selected.loc[selected["source_well_id"].ne(str(target_well_id))].copy()
    numeric = selected[["weight", "distance_ft"]].to_numpy(dtype=np.float64)
    if len(selected) and not np.isfinite(numeric).all():
        raise ValueError("邻井候选权重或距离含 NaN/Inf")
    selected = selected.loc[selected["weight"].gt(0.0)].copy()
    selected = selected.sort_values(
        ["weight", "distance_ft", "source_well_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return selected.head(int(maximum_neighbors)).reset_index(drop=True)


def summarize_source_signal(pairs: pd.DataFrame) -> dict[str, object]:
    """按预注册距离、夹角和 Typewell 切片汇总真实/错配残差差异。"""

    required = {
        "distance_ft",
        "azimuth_difference_deg",
        "same_typewell_tail",
        "real_absdiff",
        "shuffled_absdiff",
    }
    if missing := required.difference(pairs.columns):
        raise ValueError(f"source pair signal 缺列：{sorted(missing)}")
    numeric_columns = [
        "distance_ft",
        "azimuth_difference_deg",
        "real_absdiff",
        "shuffled_absdiff",
    ]
    if len(pairs) and not np.isfinite(
        pairs[numeric_columns].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("source pair signal 含 NaN/Inf")

    def summarize(mask: pd.Series | np.ndarray) -> dict[str, int | float | None]:
        subset = pairs.loc[np.asarray(mask, dtype=bool)]
        if subset.empty:
            return {
                "pairs": 0,
                "real_mean_absdiff": None,
                "real_median_absdiff": None,
                "shuffled_mean_absdiff": None,
                "shuffled_median_absdiff": None,
            }
        real = subset["real_absdiff"].to_numpy(dtype=np.float64)
        shuffled = subset["shuffled_absdiff"].to_numpy(dtype=np.float64)
        return {
            "pairs": int(len(subset)),
            "real_mean_absdiff": float(np.mean(real)),
            "real_median_absdiff": float(np.median(real)),
            "shuffled_mean_absdiff": float(np.mean(shuffled)),
            "shuffled_median_absdiff": float(np.median(shuffled)),
        }

    distance = pairs["distance_ft"].to_numpy(dtype=np.float64)
    azimuth = pairs["azimuth_difference_deg"].to_numpy(dtype=np.float64)
    same_typewell = pairs["same_typewell_tail"].astype(bool).to_numpy()
    return {
        "overall": summarize(np.ones(len(pairs), dtype=bool)),
        "distance_slices": {
            "le_500ft": summarize(distance <= 500.0),
            "500_to_1000ft": summarize((distance > 500.0) & (distance <= 1000.0)),
            "1000_to_2500ft": summarize((distance > 1000.0) & (distance <= 2500.0)),
        },
        "azimuth_slices": {
            "le_15deg": summarize(azimuth <= 15.0),
            "15_to_45deg": summarize((azimuth > 15.0) & (azimuth <= 45.0)),
        },
        "typewell_slices": {
            "same_template": summarize(same_typewell),
            "different_template": summarize(~same_typewell),
        },
    }
