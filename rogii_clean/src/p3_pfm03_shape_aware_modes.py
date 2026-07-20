"""P3-PFM03：用完整路径形状把 128 条 PF seed 压缩为低、中、高三条路径。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage


# 正式实验固定使用 128 条 seed；改变它会改变候选池，不能静默调整。
FORMAL_NUMBER_OF_SEEDS = 128
# 64 点能够保留前段分叉和中段弯曲，同时让不同长度的井具有相同表示长度。
FORMAL_SAMPLE_POINTS = 64
# K=3 在 oracle 诊断之前已经冻结，不能根据正式 CV 或隐藏真值选择。
FORMAL_NUMBER_OF_MODES = 3
# 模式中心沿用 PF 的固定温度 8，使代表路径与现有 scale8 路径口径一致。
FORMAL_LIKELIHOOD_SCALE = 8.0
MODE_NAMES = ("low", "middle", "high")
FORMAL_FEATURE_COLUMNS = (
    "shape_mode_low_delta",
    "shape_mode_middle_delta",
    "shape_mode_high_delta",
)


@dataclass(frozen=True)
class ShapeClusterResult:
    """保存 64 点形状残差和每条 seed 的 Ward 簇编号。"""

    sampled_paths: np.ndarray
    sampled_reference: np.ndarray
    shape_residuals: np.ndarray
    labels: np.ndarray


def _validate_paths_and_reference(
    seed_delta: np.ndarray,
    hidden_md: np.ndarray,
    reference_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把输入转为 float64，并核对 [seed, 行]、[行]、[行] 三个形状。"""

    paths = np.asarray(seed_delta, dtype=np.float64)
    md = np.asarray(hidden_md, dtype=np.float64)
    reference = np.asarray(reference_delta, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[0] < FORMAL_NUMBER_OF_MODES:
        raise ValueError("seed_delta 必须是至少三条路径组成的二维数组")
    if paths.shape[1] <= 0:
        raise ValueError("隐藏路径不能是空路径")
    if md.shape != (paths.shape[1],) or reference.shape != (paths.shape[1],):
        raise ValueError("hidden_md 和 reference_delta 必须与路径逐行对齐")
    if not np.isfinite(paths).all():
        raise ValueError("seed_delta 含 NaN/Inf")
    if not np.isfinite(md).all() or not np.isfinite(reference).all():
        raise ValueError("hidden_md 或 reference_delta 含 NaN/Inf")
    if np.any(np.diff(md) < 0.0):
        raise ValueError("hidden_md 必须单调不减")
    return paths, md, reference


def _sample_rows_on_normalized_md(
    row_values: np.ndarray,
    hidden_md: np.ndarray,
    sample_points: int = FORMAL_SAMPLE_POINTS,
) -> np.ndarray:
    """把一条或多条逐行路径插值到固定的 0～1 井深进度网格。"""

    values = np.asarray(row_values, dtype=np.float64)
    md = np.asarray(hidden_md, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != md.size:
        raise ValueError("待采样路径与 hidden_md 长度不一致")
    if sample_points < 3:
        raise ValueError("固定采样点至少为 3")

    target_progress = np.linspace(0.0, 1.0, sample_points, dtype=np.float64)
    md_span = float(md[-1] - md[0])
    if md_span <= 0.0:
        return np.repeat(values[:, :1], sample_points, axis=1)
    source_progress = (md - md[0]) / md_span
    sampled = np.empty((values.shape[0], sample_points), dtype=np.float64)
    for path_number in range(values.shape[0]):
        sampled[path_number] = np.interp(
            target_progress,
            source_progress,
            values[path_number],
        )
    return sampled


def cluster_shape_residuals(
    seed_delta: np.ndarray,
    hidden_md: np.ndarray,
    reference_delta: np.ndarray,
) -> ShapeClusterResult:
    """在 64 维相对参考路径上做无标准化 Ward K=3 聚类。"""

    paths, md, reference = _validate_paths_and_reference(
        seed_delta,
        hidden_md,
        reference_delta,
    )
    sampled_paths = _sample_rows_on_normalized_md(paths, md)
    sampled_reference = _sample_rows_on_normalized_md(reference, md)[0]
    # 每条 seed 都减同一条合法 scale8 参考路径；这里故意不做逐维 z-score。
    shape_residuals = sampled_paths - sampled_reference[None, :]
    hierarchy = linkage(shape_residuals, method="ward", optimal_ordering=False)
    labels = cut_tree(
        hierarchy,
        n_clusters=[FORMAL_NUMBER_OF_MODES],
    ).reshape(-1)
    labels = labels.astype(np.int64, copy=False)
    if np.unique(labels).size != FORMAL_NUMBER_OF_MODES:
        raise RuntimeError("Ward 聚类没有得到固定的三个非空模式")
    return ShapeClusterResult(
        sampled_paths=sampled_paths,
        sampled_reference=sampled_reference,
        shape_residuals=shape_residuals,
        labels=labels,
    )


def _within_mode_likelihood_weights(member_ll: np.ndarray) -> np.ndarray:
    """对一个簇内部的最终 log-likelihood 做稳定的温度 8 softmax。"""

    likelihoods = np.asarray(member_ll, dtype=np.float64)
    if likelihoods.ndim != 1 or likelihoods.size == 0:
        raise ValueError("模式内部 final_ll 不能为空")
    if not np.isfinite(likelihoods).all():
        raise ValueError("final_ll 含 NaN/Inf")
    shifted = (likelihoods - float(np.max(likelihoods))) / FORMAL_LIKELIHOOD_SCALE
    unnormalized = np.exp(shifted)
    denominator = float(np.sum(unnormalized))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise RuntimeError("模式内部似然权重无法归一化")
    return unnormalized / denominator


def build_shape_mode_paths(
    seed_delta: np.ndarray,
    hidden_md: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    reference_delta: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """输出逐行三条 LL 加权中心 delta；整个函数不接触隐藏 TVT。"""

    paths, md, reference = _validate_paths_and_reference(
        seed_delta,
        hidden_md,
        reference_delta,
    )
    if paths.shape[0] != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("正式 PFM03 必须恰好输入 128 条 seed 路径")
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids)
    if likelihoods.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("final_ll 必须与 128 条 seed 一一对应")
    if ids.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("seed_ids 必须与 128 条 seed 一一对应")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("seed_ids 必须是整数")
    ids = ids.astype(np.int64, copy=False)
    if not np.array_equal(ids, np.arange(FORMAL_NUMBER_OF_SEEDS, dtype=np.int64)):
        raise ValueError("正式 seed_ids 必须严格为 0～127")

    clustered = cluster_shape_residuals(paths, md, reference)
    unordered_records: list[dict[str, Any]] = []
    for raw_label in sorted(np.unique(clustered.labels).tolist()):
        member_mask = clustered.labels == raw_label
        member_indices = np.flatnonzero(member_mask)
        member_weights = _within_mode_likelihood_weights(likelihoods[member_mask])
        center_path = np.sum(
            paths[member_mask] * member_weights[:, None],
            axis=0,
            dtype=np.float64,
        )
        relative_center = center_path - reference
        unordered_records.append(
            {
                "raw_label": int(raw_label),
                "center_path": center_path,
                "mean_relative_delta": float(np.mean(relative_center)),
                "endpoint_relative_delta": float(relative_center[-1]),
                "minimum_seed_id": int(np.min(ids[member_indices])),
                "member_seed_ids": ids[member_indices].astype(int).tolist(),
                "seed_count": int(member_indices.size),
            }
        )

    # low/middle/high 始终表示相对 scale8 参考路径的位置，不按似然质量编号。
    ordered_records = sorted(
        unordered_records,
        key=lambda record: (
            float(record["mean_relative_delta"]),
            float(record["endpoint_relative_delta"]),
            int(record["minimum_seed_id"]),
        ),
    )
    feature_frame = pd.DataFrame(
        {
            feature_name: np.asarray(record["center_path"], dtype=np.float64)
            for feature_name, record in zip(
                FORMAL_FEATURE_COLUMNS,
                ordered_records,
                strict=True,
            )
        },
        columns=list(FORMAL_FEATURE_COLUMNS),
    )
    diagnostics: dict[str, Any] = {
        "number_of_seeds": FORMAL_NUMBER_OF_SEEDS,
        "sample_points": FORMAL_SAMPLE_POINTS,
        "ward_k": FORMAL_NUMBER_OF_MODES,
        "per_dimension_standardization": False,
        "reference_path": "pf128_scale_8_delta",
        "representative": "final_ll_scale8_weighted_center",
        "hidden_tvt_read": False,
        "ordered_modes": [],
    }
    for mode_name, record in zip(MODE_NAMES, ordered_records, strict=True):
        diagnostics["ordered_modes"].append(
            {
                "name": mode_name,
                "raw_label": int(record["raw_label"]),
                "mean_relative_delta": float(record["mean_relative_delta"]),
                "endpoint_relative_delta": float(record["endpoint_relative_delta"]),
                "minimum_seed_id": int(record["minimum_seed_id"]),
                "seed_count": int(record["seed_count"]),
                "member_seed_ids": list(record["member_seed_ids"]),
            }
        )
    return feature_frame, diagnostics

