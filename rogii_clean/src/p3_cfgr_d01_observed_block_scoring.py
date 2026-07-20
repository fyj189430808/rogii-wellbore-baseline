"""P3-CFGR-D01：仅以原始观测 GR 对冻结候选路径做分块评分。"""

from __future__ import annotations

import numpy as np


def assign_hidden_blocks(
    hidden_md: np.ndarray, *, min_hidden_md: float, block_size_ft: float = 250.0
) -> np.ndarray:
    """返回非重叠 250 ft 块号；MD 无效行保留为 -1。"""
    md = np.asarray(hidden_md, dtype=np.float64)
    result = np.full(md.shape, -1, dtype=np.int64)
    finite = np.isfinite(md)
    result[finite] = np.floor((md[finite] - min_hidden_md) / block_size_ft).astype(np.int64)
    return result


def deterministic_hidden_gr_roll(observed_gr: np.ndarray, fraction: float) -> np.ndarray:
    """确定性循环移位完整观测序列；NaN 与数值同行移动，绝不插补。"""
    values = np.asarray(observed_gr, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("observed_gr 必须是一维")
    shift = int(round(values.size * float(fraction))) % max(values.size, 1)
    return np.roll(values, shift)


def block_candidate_costs(
    observed_gr: np.ndarray,
    expected_gr: np.ndarray,
    block_ids: np.ndarray,
    *,
    minimum_observed_points: int = 20,
    common_sigma: float = 1.0,
    out_of_range: np.ndarray | None = None,
    out_of_range_penalty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按块等权中位绝对标准化残差，返回候选成本、有效块和每块原始点数。"""
    observed = np.asarray(observed_gr, dtype=np.float64)
    expected = np.asarray(expected_gr, dtype=np.float64)
    blocks = np.asarray(block_ids, dtype=np.int64)
    if expected.ndim != 2 or expected.shape[0] != observed.size or blocks.size != observed.size:
        raise ValueError("GR、候选和块号行数必须一致")
    valid_ids = np.unique(blocks[blocks >= 0])
    counts = np.array([int(np.isfinite(observed[blocks == item]).sum()) for item in valid_ids])
    usable = valid_ids[counts >= minimum_observed_points]
    if usable.size == 0:
        return np.full(expected.shape[1], np.nan), usable, counts
    sigma = float(common_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("common_sigma 必须为正的有限共同前缀尺度")
    penalties = np.zeros_like(expected, dtype=np.float64)
    if out_of_range is not None:
        outside = np.asarray(out_of_range, dtype=bool)
        if outside.shape != expected.shape:
            raise ValueError("out_of_range 必须与 expected_gr 同形状")
        penalties[outside] = float(out_of_range_penalty)
    block_cost_rows: list[np.ndarray] = []
    for item in usable:
        selector = (blocks == item) & np.isfinite(observed)
        valid_expected = np.isfinite(expected[selector])
        # 候选需在同一块具备完整的最小原始观测配对数；少量有限点不能取巧。
        candidate_counts = valid_expected.sum(axis=0)
        residual = np.abs(expected[selector] - observed[selector, None]) / sigma
        row_cost = np.full(expected.shape[1], np.nan)
        enough = candidate_counts >= minimum_observed_points
        row_cost[enough] = np.nanmedian(residual[:, enough], axis=0) + np.nanmean(penalties[selector][:, enough], axis=0)
        block_cost_rows.append(row_cost)
    stacked = np.vstack(block_cost_rows)
    finite_counts = np.isfinite(stacked).sum(axis=0)
    summed = np.nansum(stacked, axis=0)
    result = np.full(expected.shape[1], np.nan)
    result[finite_counts > 0] = summed[finite_counts > 0] / finite_counts[finite_counts > 0]
    return result, usable, counts


def candidate_ranks(values: np.ndarray) -> np.ndarray:
    """稳定的从小到大候选名次（1 为最佳），避免 scipy 依赖。"""
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks
