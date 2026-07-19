"""P3-PF03a：用五条冻结聚合路径做 500 ft 局部 GR 混合代理。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.p3_d01_pf_observation_audit import seed_weights
from src.p3_pf02_target_ess_paths import solve_temperature_for_target_ess


CANDIDATE_PATH_COLUMNS = (
    "pf128_mean_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
)
OUTPUT_PATH_COLUMN = "pf128_local500_delta"


def _mean_path_fallback(reason: str, observed_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    """没有足够局部证据时，回退到 128 个种子的冻结均值路径。"""
    weights = np.zeros(len(CANDIDATE_PATH_COLUMNS), dtype=np.float64)
    weights[0] = 1.0
    return weights, {
        "fallback_used": True,
        "fallback_reason": reason,
        "observed_rows": int(observed_rows),
        "temperature": None,
        "achieved_ess": 1.0,
    }


def score_one_segment(
    observed_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    gr_sigma: float,
    target_ess: float,
    minimum_observed_rows: int,
    squared_residual_cap: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """只用一段内原始有效 GR，为五条候选路径求目标 ESS 权重。"""
    observed = np.asarray(observed_gr, dtype=np.float64)
    candidates = np.asarray(candidate_tvt, dtype=np.float64)
    typewell_depth = np.asarray(typewell_tvt, dtype=np.float64)
    typewell_curve = np.asarray(typewell_gr, dtype=np.float64)
    if observed.ndim != 1 or candidates.shape != (len(observed), len(CANDIDATE_PATH_COLUMNS)):
        raise ValueError("candidate_tvt 必须为 [有效 GR 行数, 5]")
    if not np.isfinite(observed).all() or not np.isfinite(candidates).all():
        raise ValueError("局部 GR 与候选 TVT 必须全部有限")
    if len(observed) < int(minimum_observed_rows):
        return _mean_path_fallback("observed_gr_rows_below_minimum", len(observed))
    if (
        typewell_depth.ndim != 1
        or typewell_curve.shape != typewell_depth.shape
        or len(typewell_depth) < 2
        or not np.isfinite(typewell_depth).all()
        or not np.isfinite(typewell_curve).all()
        or np.any(np.diff(typewell_depth) <= 0.0)
    ):
        raise ValueError("Typewell TVT/GR 必须有限、等长且 TVT 严格递增")
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        raise ValueError("gr_sigma 必须为有限正数")
    if not np.isfinite(squared_residual_cap) or squared_residual_cap <= 0.0:
        raise ValueError("squared_residual_cap 必须为有限正数")

    local_log_scores = np.empty(len(CANDIDATE_PATH_COLUMNS), dtype=np.float64)
    for candidate_index in range(len(CANDIDATE_PATH_COLUMNS)):
        expected_gr = np.interp(
            candidates[:, candidate_index],
            typewell_depth,
            typewell_curve,
        )
        standardized_residual = (observed - expected_gr) / float(gr_sigma)
        squared_residual = np.square(standardized_residual)
        squared_residual = np.minimum(squared_residual, float(squared_residual_cap))
        local_log_scores[candidate_index] = -0.5 * float(squared_residual.sum())

    # 五条聚合路径在这一段给出完全相同的 GR 证据时，温度不可辨识。
    if np.ptp(local_log_scores) <= 1e-12:
        return _mean_path_fallback("candidate_scores_indistinguishable", len(observed))

    solution = solve_temperature_for_target_ess(
        seed_log_likelihoods=local_log_scores,
        target_ess=float(target_ess),
    )
    weights = seed_weights(local_log_scores, float(solution["temperature"]))
    return weights, {
        "fallback_used": False,
        "fallback_reason": "",
        "observed_rows": int(len(observed)),
        "temperature": float(solution["temperature"]),
        "achieved_ess": float(solution["achieved_ess"]),
        "score_range": float(np.ptp(local_log_scores)),
    }


def interpolate_segment_weights(
    row_md: np.ndarray,
    segment_centers_md: np.ndarray,
    segment_weights: np.ndarray,
) -> np.ndarray:
    """在相邻 500 ft 段中心之间逐候选线性插值，再逐行归一化。"""
    md = np.asarray(row_md, dtype=np.float64)
    centers = np.asarray(segment_centers_md, dtype=np.float64)
    weights = np.asarray(segment_weights, dtype=np.float64)
    if md.ndim != 1 or centers.ndim != 1:
        raise ValueError("row_md 和 segment_centers_md 必须是一维")
    if weights.shape != (len(centers), len(CANDIDATE_PATH_COLUMNS)):
        raise ValueError("segment_weights 必须为 [段数, 5]")
    if len(centers) == 0 or np.any(np.diff(centers) <= 0.0):
        raise ValueError("段中心必须非空且严格递增")

    row_weights = np.empty((len(md), len(CANDIDATE_PATH_COLUMNS)), dtype=np.float64)
    for candidate_index in range(len(CANDIDATE_PATH_COLUMNS)):
        row_weights[:, candidate_index] = np.interp(
            md,
            centers,
            weights[:, candidate_index],
        )
    weight_sums = row_weights.sum(axis=1)
    if np.any(weight_sums <= 0.0) or not np.isfinite(row_weights).all():
        raise ValueError("插值后的路径权重非法")
    row_weights /= weight_sums[:, None]
    return row_weights


def build_local500_proxy_path(
    frozen_paths: pd.DataFrame,
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    gr_sigma: float,
    segment_length_ft: float,
    minimum_observed_rows: int,
    target_ess: float,
    squared_residual_cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从五条冻结绝对路径生成一条沿 MD 动态加权的 local500 delta。"""
    required_path_columns = [
        "well_id",
        "row_index",
        "last_visible_tvt",
        *CANDIDATE_PATH_COLUMNS,
    ]
    missing_path_columns = [
        column for column in required_path_columns if column not in frozen_paths.columns
    ]
    if missing_path_columns or frozen_paths.empty:
        raise ValueError(f"冻结路径为空或缺列: {missing_path_columns}")
    if frozen_paths["well_id"].astype(str).nunique() != 1:
        raise ValueError("一次只能处理一口井")
    for column in ("MD", "GR", "TVT_input"):
        if column not in horizontal_well.columns:
            raise ValueError(f"水平井缺少合法输入列 {column}")
    if "TVT" not in typewell.columns or "GR" not in typewell.columns:
        raise ValueError("Typewell 缺少 TVT 或 GR")
    if not np.isfinite(segment_length_ft) or segment_length_ft <= 0.0:
        raise ValueError("segment_length_ft 必须为有限正数")

    frozen = frozen_paths.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    row_index = frozen["row_index"].to_numpy(dtype=np.int64)
    if (
        len(np.unique(row_index)) != len(row_index)
        or row_index.min() < 0
        or row_index.max() >= len(horizontal_well)
    ):
        raise ValueError("冻结路径 row_index 越界或重复")
    selected_horizontal = horizontal_well.iloc[row_index]
    if selected_horizontal["TVT_input"].notna().any():
        raise ValueError("冻结路径包含非自然隐藏行")

    hidden_md = selected_horizontal["MD"].to_numpy(dtype=np.float64)
    hidden_gr = selected_horizontal["GR"].to_numpy(dtype=np.float64)
    if not np.isfinite(hidden_md).all() or np.any(np.diff(hidden_md) < 0.0):
        raise ValueError("隐藏 MD 必须有限且不下降")
    last_visible_tvt = frozen["last_visible_tvt"].to_numpy(dtype=np.float64)
    candidate_delta = frozen[list(CANDIDATE_PATH_COLUMNS)].to_numpy(dtype=np.float64)
    candidate_absolute_tvt = last_visible_tvt[:, None] + candidate_delta
    if not np.isfinite(candidate_absolute_tvt).all():
        raise ValueError("五条候选路径含 NaN 或无穷值")

    sorted_typewell = typewell.sort_values("TVT", kind="mergesort")
    typewell_tvt = sorted_typewell["TVT"].to_numpy(dtype=np.float64)
    typewell_gr_series = sorted_typewell["GR"].astype(np.float64)
    typewell_gr_mean = float(typewell_gr_series.mean(skipna=True))
    if not np.isfinite(typewell_gr_mean):
        raise ValueError("Typewell GR 全部缺失")
    typewell_gr = typewell_gr_series.fillna(typewell_gr_mean).to_numpy(dtype=np.float64)

    first_hidden_md = float(hidden_md[0])
    segment_ids = np.floor((hidden_md - first_hidden_md) / float(segment_length_ft)).astype(int)
    unique_segment_ids = np.unique(segment_ids)
    segment_centers: list[float] = []
    segment_weight_rows: list[np.ndarray] = []
    segment_reports: list[dict[str, Any]] = []
    for segment_id in unique_segment_ids:
        in_segment = segment_ids == segment_id
        observed_in_segment = in_segment & np.isfinite(hidden_gr)
        weights, segment_report = score_one_segment(
            observed_gr=hidden_gr[observed_in_segment],
            candidate_tvt=candidate_absolute_tvt[observed_in_segment],
            typewell_tvt=typewell_tvt,
            typewell_gr=typewell_gr,
            gr_sigma=float(gr_sigma),
            target_ess=float(target_ess),
            minimum_observed_rows=int(minimum_observed_rows),
            squared_residual_cap=float(squared_residual_cap),
        )
        center_md = first_hidden_md + (float(segment_id) + 0.5) * float(segment_length_ft)
        segment_centers.append(center_md)
        segment_weight_rows.append(weights)
        segment_reports.append({"segment_id": int(segment_id), "center_md": center_md, **segment_report})

    row_weights = interpolate_segment_weights(
        row_md=hidden_md,
        segment_centers_md=np.asarray(segment_centers, dtype=np.float64),
        segment_weights=np.vstack(segment_weight_rows),
    )
    local_delta = np.sum(row_weights * candidate_delta, axis=1).astype(np.float32)
    output = frozen[["row_index", "last_visible_tvt"]].copy()
    output[OUTPUT_PATH_COLUMN] = local_delta
    temperatures = [
        float(item["temperature"])
        for item in segment_reports
        if not bool(item["fallback_used"])
    ]
    report: dict[str, Any] = {
        "segments": int(len(segment_reports)),
        "scored_segments": int(sum(not bool(item["fallback_used"]) for item in segment_reports)),
        "fallback_segments": int(sum(bool(item["fallback_used"]) for item in segment_reports)),
        "observed_hidden_gr_rows": int(np.isfinite(hidden_gr).sum()),
        "target_ess": float(target_ess),
        "temperature_min": float(min(temperatures)) if temperatures else None,
        "temperature_median": float(np.median(temperatures)) if temperatures else None,
        "temperature_max": float(max(temperatures)) if temperatures else None,
        "segment_reports": segment_reports,
    }
    return output, report
