"""P3-N01a 邻井低维残差曲线迁移的纯计算函数。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.p3_d00_residual_structure import fit_control_point_residual, normalized_progress


CONTROL_COUNT = 4
MAX_DISTANCE_FT = 2500.0
MAX_AZIMUTH_DIFFERENCE_DEG = 45.0
MIN_PARALLEL_OVERLAP_FT = 500.0
MAX_NEIGHBORS = 8
DISTANCE_DECAY_FT = 1000.0
OVERLAP_FULL_WEIGHT_FT = 1000.0
SAME_TYPEWELL_MULTIPLIER = 1.5
ETA_PRIOR_WEIGHT = 1.0


@dataclass(frozen=True)
class TrajectoryGeometry:
    """两条隐藏段 XY 轨迹之间的固定几何关系。"""

    minimum_distance_ft: float
    azimuth_difference_deg: float
    parallel_overlap_ft: float
    reverse_source_controls: bool


@dataclass(frozen=True)
class AggregatedControls:
    """邻井控制点的加权结果和收缩强度。"""

    weighted_controls: np.ndarray
    shrunk_controls: np.ndarray
    total_weight: float
    eta: float


def _finite_xy(values: np.ndarray, name: str) -> np.ndarray:
    xy = np.asarray(values, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError(f"{name} 必须是至少两行的 [行数, 2] XY 数组")
    if not np.isfinite(xy).all():
        raise ValueError(f"{name} 含非有限值")
    return xy


def _trajectory_direction(xy: np.ndarray, name: str) -> np.ndarray:
    direction = xy[-1] - xy[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} 首尾 XY 重合，无法定义方向")
    return direction / norm


def compute_trajectory_geometry(
    target_xy: np.ndarray,
    source_xy: np.ndarray,
) -> TrajectoryGeometry:
    """用两口井完整隐藏段 XY 点列计算距离、夹角、平行重叠和方向。"""

    target = _finite_xy(target_xy, "target_xy")
    source = _finite_xy(source_xy, "source_xy")
    target_direction = _trajectory_direction(target, "target_xy")
    source_direction = _trajectory_direction(source, "source_xy")

    minimum_distance = float(cKDTree(source).query(target, k=1)[0].min())
    signed_cosine = float(np.clip(np.dot(target_direction, source_direction), -1.0, 1.0))
    undirected_cosine = abs(signed_cosine)
    azimuth_difference = float(np.degrees(np.arccos(undirected_cosine)))

    target_projection = target @ target_direction
    source_projection = source @ target_direction
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


def geometry_is_eligible(geometry: TrajectoryGeometry) -> bool:
    """执行 N01a 冻结的三项邻井几何门槛。"""

    return bool(
        geometry.minimum_distance_ft <= MAX_DISTANCE_FT
        and geometry.azimuth_difference_deg <= MAX_AZIMUTH_DIFFERENCE_DEG
        and geometry.parallel_overlap_ft >= MIN_PARALLEL_OVERLAP_FT
    )


def neighbor_weight(
    distance_ft: float,
    azimuth_difference_deg: float,
    overlap_ft: float,
    same_typewell_tail: bool,
) -> float:
    """按距离、平行度、重叠长度和同 Typewell 尾段指纹计算固定权重。"""

    distance = max(float(distance_ft), 0.0)
    angle = float(np.clip(azimuth_difference_deg, 0.0, 90.0))
    overlap = max(float(overlap_ft), 0.0)
    distance_weight = float(np.exp(-distance / DISTANCE_DECAY_FT))
    direction_weight = float(np.cos(np.deg2rad(angle)) ** 2)
    overlap_weight = float(min(overlap / OVERLAP_FULL_WEIGHT_FT, 1.0))
    typewell_weight = SAME_TYPEWELL_MULTIPLIER if same_typewell_tail else 1.0
    return distance_weight * direction_weight * overlap_weight * typewell_weight


def aggregate_control_profiles(
    controls: list[np.ndarray],
    weights: np.ndarray,
    reverse_flags: np.ndarray,
) -> AggregatedControls:
    """把最多八口邻井的四控制点加权，并用 eta=sumw/(sumw+1) 收缩。"""

    if not controls:
        zeros = np.zeros(CONTROL_COUNT, dtype=np.float64)
        return AggregatedControls(zeros, zeros.copy(), 0.0, 0.0)
    matrix = np.vstack([np.asarray(value, dtype=np.float64) for value in controls])
    if matrix.shape != (len(controls), CONTROL_COUNT):
        raise ValueError("每条残差 profile 必须恰好包含四个控制点")
    profile_weights = np.asarray(weights, dtype=np.float64)
    reverse = np.asarray(reverse_flags, dtype=bool)
    if profile_weights.shape != (len(controls),) or reverse.shape != (len(controls),):
        raise ValueError("权重和反向标记必须与 profile 数量一致")
    if not np.isfinite(matrix).all() or not np.isfinite(profile_weights).all():
        raise ValueError("控制点或权重含非有限值")
    if (profile_weights <= 0.0).any():
        raise ValueError("所有入选邻井权重必须大于零")

    oriented = matrix.copy()
    oriented[reverse] = oriented[reverse, ::-1]
    total_weight = float(profile_weights.sum())
    weighted_controls = np.average(oriented, axis=0, weights=profile_weights)
    eta = total_weight / (total_weight + ETA_PRIOR_WEIGHT)
    return AggregatedControls(
        weighted_controls=np.asarray(weighted_controls, dtype=np.float64),
        shrunk_controls=np.asarray(weighted_controls * eta, dtype=np.float64),
        total_weight=total_weight,
        eta=float(eta),
    )


def interpolate_control4(controls: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """把四个等距控制点线性插值回目标井逐行隐藏段。"""

    values = np.asarray(controls, dtype=np.float64)
    query = np.asarray(progress, dtype=np.float64)
    if values.shape != (CONTROL_COUNT,):
        raise ValueError("controls 必须恰好有四个值")
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("progress 必须是一维有限数组")
    if ((query < 0.0) | (query > 1.0)).any():
        raise ValueError("progress 必须位于 [0, 1]")
    knots = np.linspace(0.0, 1.0, CONTROL_COUNT, dtype=np.float64)
    return np.interp(query, knots, values)


def fit_source_control4(
    hidden_md: np.ndarray,
    target_tvt: np.ndarray,
    direct_path_tvt: np.ndarray,
) -> np.ndarray:
    """按 D00 的 control4_unanchored 定义拟合一口 source 井的残差控制点。"""

    md = np.asarray(hidden_md, dtype=np.float64)
    target = np.asarray(target_tvt, dtype=np.float64)
    direct = np.asarray(direct_path_tvt, dtype=np.float64)
    if md.ndim != 1 or target.shape != md.shape or direct.shape != md.shape:
        raise ValueError("hidden_md、target_tvt 和 direct_path_tvt 必须是同形一维数组")
    if not np.isfinite(np.column_stack([md, target, direct])).all():
        raise ValueError("source 残差拟合输入含非有限值")
    fit = fit_control_point_residual(
        normalized_progress(md),
        target - direct,
        number_of_points=CONTROL_COUNT,
        anchored=False,
    )
    return np.asarray(
        [fit.coefficients[f"control_point_{index}"] for index in range(CONTROL_COUNT)],
        dtype=np.float64,
    )


def build_fixed_shuffle(source_ids: list[str], seed: int) -> dict[str, str]:
    """生成与输入顺序无关、无固定点的固定随机 profile 映射。"""

    ordered = sorted(str(value) for value in source_ids)
    if len(ordered) < 2:
        raise ValueError("打乱负对照至少需要两口 source 井")
    generator = np.random.default_rng(int(seed))
    shuffled = list(np.asarray(ordered, dtype=object)[generator.permutation(len(ordered))])
    for _attempt in range(len(ordered)):
        if all(source_id != profile_id for source_id, profile_id in zip(ordered, shuffled)):
            break
        shuffled = shuffled[1:] + shuffled[:1]
    if any(source_id == profile_id for source_id, profile_id in zip(ordered, shuffled)):
        shuffled = ordered[1:] + ordered[:1]
    return dict(zip(ordered, (str(value) for value in shuffled), strict=True))
