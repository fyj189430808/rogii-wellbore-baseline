"""P3-GRM01：用整段原始 GR 在严格基础路径附近搜索一个常数偏移。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from src.rf03_prefix_alignment import clean_typewell_arrays


# 这 61 个候选是实验卡冻结的唯一网格，不能根据 outer0 结果调整。
CANDIDATE_OFFSETS_FT = np.arange(-30, 31, 1, dtype=np.float64)
BLOCK_WIDTH_FT = 300.0
MINIMUM_POINTS_PER_BLOCK = 30
MINIMUM_TOTAL_POINTS = 60
MINIMUM_VALID_BLOCKS = 2
ABSOLUTE_LOSS_CAP = 3.0
ZERO_PENALTY_WEIGHT = 0.05


@dataclass(frozen=True)
class GlobalOffsetResult:
    """保存一口井不含真值的合法候选得分和最终选择。"""

    scores: pd.DataFrame
    selected_offset_ft: float
    evidence_sufficient: bool
    common_points: int
    valid_blocks: int
    affine_slope: float
    affine_intercept: float
    prefix_pairs: int
    robust_scale: float
    prefix_fallback: bool


def rank_candidate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """按总损失、绝对偏移、带符号偏移稳定排序，并写入从 1 开始的名次。"""

    required = {"offset_ft", "total_score"}
    if not required.issubset(scores.columns):
        raise ValueError(f"候选得分缺列：{sorted(required - set(scores.columns))}")
    ranked = scores.copy()
    finite_scores = np.isfinite(ranked["total_score"].to_numpy(dtype=np.float64))
    ranked["rank"] = np.nan
    ranked.loc[finite_scores, "rank"] = ranked.loc[
        finite_scores, "total_score"
    ].rank(method="average", ascending=True)
    ranked["_absolute_offset"] = np.abs(ranked["offset_ft"].to_numpy(dtype=np.float64))
    ranked = ranked.sort_values(
        ["total_score", "_absolute_offset", "offset_ft"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked.drop(columns="_absolute_offset")


def circular_shift_finite_gr(gr_values: np.ndarray) -> np.ndarray:
    """只循环移动有限 GR 的值序列一半长度，原始 NaN 位置保持不动。"""

    original = np.asarray(gr_values, dtype=np.float64)
    shifted = original.copy()
    finite_positions = np.flatnonzero(np.isfinite(original))
    if len(finite_positions) <= 1:
        return shifted
    finite_values = original[finite_positions]
    shift = len(finite_values) // 2
    shifted[finite_positions] = np.roll(finite_values, -shift)
    return shifted


def nearest_grid_offset(mean_residual_ft: float) -> float:
    """按距离、偏移绝对值、带符号偏移依次打破最近网格的平分。"""

    mean_residual = float(mean_residual_ft)
    if not np.isfinite(mean_residual):
        raise ValueError("真实平均残差必须有限")
    ordered = sorted(
        CANDIDATE_OFFSETS_FT.tolist(),
        key=lambda candidate: (
            abs(float(candidate) - mean_residual),
            abs(float(candidate)),
            float(candidate),
        ),
    )
    return float(ordered[0])


def _fit_prefix_huber(
    horizontal_df: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[float, float, int, float, bool]:
    """仅用 TVT_input 可见前缀拟合 horizontal_GR=a*typewell_GR+b。"""

    visible_tvt = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce").to_numpy(dtype=np.float64)
    horizontal_gr = pd.to_numeric(horizontal_df["GR"], errors="coerce").to_numpy(dtype=np.float64)
    mask = (
        np.isfinite(visible_tvt)
        & np.isfinite(horizontal_gr)
        & (visible_tvt >= float(typewell_tvt[0]))
        & (visible_tvt <= float(typewell_tvt[-1]))
    )
    pair_count = int(mask.sum())
    if pair_count < 30:
        return 1.0, 0.0, pair_count, 1.0, True
    reference = np.interp(visible_tvt[mask], typewell_tvt, typewell_gr)
    if (
        float(np.std(reference)) <= 1e-12
        or float(np.std(horizontal_gr[mask])) <= 1e-12
    ):
        return 1.0, 0.0, pair_count, 1.0, True
    try:
        model = HuberRegressor(
            epsilon=1.35,
            alpha=0.0,
            max_iter=200,
            fit_intercept=True,
        )
        model.fit(reference.reshape(-1, 1), horizontal_gr[mask])
        affine_slope = float(model.coef_[0])
        affine_intercept = float(model.intercept_)
        if not np.isfinite(affine_slope) or not np.isfinite(affine_intercept):
            raise ValueError("Huber 返回非有限参数")
        residual = horizontal_gr[mask] - (
            affine_slope * reference + affine_intercept
        )
        residual_median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_median)))
        robust_scale = max(1.0, 1.4826 * mad)
        if not np.isfinite(robust_scale):
            raise ValueError("Huber 残差尺度非有限")
        return (
            affine_slope,
            affine_intercept,
            pair_count,
            robust_scale,
            False,
        )
    except Exception:
        # Huber 数值异常时不猜测参数；固定回退会让合法选择变成 0 ft。
        return 1.0, 0.0, pair_count, 1.0, True


def score_global_offsets(
    base_tvt: np.ndarray,
    hidden_md: np.ndarray,
    hidden_gr: np.ndarray,
    typewell_df: pd.DataFrame,
    affine_slope: float,
    affine_intercept: float,
    robust_scale: float = 1.0,
    calibration_valid: bool = True,
) -> GlobalOffsetResult:
    """在完全相同的原始有限 GR 行和 200 ft 有效块上评价 61 个候选。"""

    base = np.asarray(base_tvt, dtype=np.float64)
    md = np.asarray(hidden_md, dtype=np.float64)
    observed_gr = np.asarray(hidden_gr, dtype=np.float64)
    if not (base.shape == md.shape == observed_gr.shape) or base.ndim != 1:
        raise ValueError("base_tvt、hidden_md、hidden_gr 必须是一维等长数组")
    typewell_tvt, typewell_gr = clean_typewell_arrays(typewell_df)

    # 所有候选共用同一行集合：原始 GR 有限，且 ±30 ft 都在 Typewell 支持内。
    common_mask = (
        np.isfinite(base)
        & np.isfinite(md)
        & np.isfinite(observed_gr)
        & (base + float(CANDIDATE_OFFSETS_FT[0]) >= float(typewell_tvt[0]))
        & (base + float(CANDIDATE_OFFSETS_FT[-1]) <= float(typewell_tvt[-1]))
    )
    common_base = base[common_mask]
    common_md = md[common_mask]
    common_gr = observed_gr[common_mask]
    common_points = int(len(common_base))

    block_ids = np.empty(0, dtype=np.int64)
    valid_block_ids = np.empty(0, dtype=np.int64)
    if common_points:
        hidden_start_md = float(np.nanmin(md))
        block_ids = np.floor((common_md - hidden_start_md) / BLOCK_WIDTH_FT).astype(np.int64)
        unique_blocks, counts = np.unique(block_ids, return_counts=True)
        valid_block_ids = unique_blocks[counts >= MINIMUM_POINTS_PER_BLOCK]
    valid_blocks = int(len(valid_block_ids))
    scale = float(robust_scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("robust_scale 必须是至少 1.0 的有限数")
    evidence_sufficient = bool(
        calibration_valid
        and
        common_points >= MINIMUM_TOTAL_POINTS and valid_blocks >= MINIMUM_VALID_BLOCKS
    )

    rows: list[dict[str, float | int | bool]] = []
    for offset_ft in CANDIDATE_OFFSETS_FT:
        penalty = ZERO_PENALTY_WEIGHT * (float(offset_ft) / 30.0) ** 2
        data_loss = float("nan")
        if evidence_sufficient:
            reference_gr = np.interp(
                common_base + float(offset_ft), typewell_tvt, typewell_gr
            )
            calibrated = float(affine_slope) * reference_gr + float(affine_intercept)
            standardized_error = np.abs(common_gr - calibrated) / scale
            point_loss = np.minimum(standardized_error, ABSOLUTE_LOSS_CAP)
            block_losses = [
                float(np.mean(point_loss[block_ids == block_id]))
                for block_id in valid_block_ids
            ]
            data_loss = float(np.mean(block_losses))
        rows.append(
            {
                "offset_ft": float(offset_ft),
                "data_loss": data_loss,
                "penalty": float(penalty),
                "total_score": data_loss + penalty if np.isfinite(data_loss) else float("nan"),
                "common_points": common_points,
                "valid_blocks": valid_blocks,
                "evidence_sufficient": evidence_sufficient,
            }
        )
    scores = pd.DataFrame(rows)
    if evidence_sufficient:
        selected_offset = float(rank_candidate_scores(scores)["offset_ft"].iloc[0])
    else:
        selected_offset = 0.0
    return GlobalOffsetResult(
        scores=scores,
        selected_offset_ft=selected_offset,
        evidence_sufficient=evidence_sufficient,
        common_points=common_points,
        valid_blocks=valid_blocks,
        affine_slope=float(affine_slope),
        affine_intercept=float(affine_intercept),
        prefix_pairs=0,
        robust_scale=scale,
        prefix_fallback=not bool(calibration_valid),
    )


def score_legal_well(
    base_tvt: np.ndarray,
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    hidden_gr_override: np.ndarray | None = None,
) -> GlobalOffsetResult:
    """合法阶段入口；函数签名和读取列都没有 target_tvt。"""

    typewell_tvt, typewell_gr = clean_typewell_arrays(typewell_df)
    (
        affine_slope,
        affine_intercept,
        prefix_pairs,
        robust_scale,
        prefix_fallback,
    ) = _fit_prefix_huber(
        horizontal_df, typewell_tvt, typewell_gr
    )
    hidden_mask = horizontal_df["TVT_input"].isna().to_numpy()
    hidden_md = pd.to_numeric(horizontal_df.loc[hidden_mask, "MD"], errors="coerce").to_numpy(dtype=np.float64)
    original_hidden_gr = pd.to_numeric(horizontal_df.loc[hidden_mask, "GR"], errors="coerce").to_numpy(dtype=np.float64)
    hidden_gr = original_hidden_gr if hidden_gr_override is None else np.asarray(hidden_gr_override, dtype=np.float64)
    result = score_global_offsets(
        base_tvt,
        hidden_md,
        hidden_gr,
        typewell_df,
        affine_slope,
        affine_intercept,
        robust_scale=robust_scale,
        calibration_valid=not prefix_fallback,
    )
    return GlobalOffsetResult(
        scores=result.scores,
        selected_offset_ft=result.selected_offset_ft,
        evidence_sufficient=result.evidence_sufficient,
        common_points=result.common_points,
        valid_blocks=result.valid_blocks,
        affine_slope=affine_slope,
        affine_intercept=affine_intercept,
        prefix_pairs=prefix_pairs,
        robust_scale=robust_scale,
        prefix_fallback=prefix_fallback,
    )
