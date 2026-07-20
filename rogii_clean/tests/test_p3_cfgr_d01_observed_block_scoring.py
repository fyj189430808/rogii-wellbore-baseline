"""P3-CFGR-D01 的纯数值合同测试。"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.p3_cfgr_d01_observed_block_scoring import (
    assign_hidden_blocks,
    block_candidate_costs,
    deterministic_hidden_gr_roll,
)
from scripts.run_p3_cfgr_d01_observed_block_candidate_scoring import _rank_metrics
from scripts.run_p3_cfgr_d01_observed_block_candidate_scoring import legal_cache_dir
from scripts.run_p3_cfgr_d01_observed_block_candidate_scoring import build_control_contrasts


def test_block_costs_use_only_observed_gr_and_require_twenty_points() -> None:
    """NaN GR 不得填补，且不足 20 个原始点的块不得进入得分。"""
    md = np.r_[np.arange(20.0), np.arange(20.0, 30.0) + 250.0]
    observed_gr = np.r_[np.full(20, 100.0), np.full(10, np.nan)]
    expected_gr = np.column_stack([np.full(30, 100.0), np.full(30, 110.0)])
    blocks = assign_hidden_blocks(md, min_hidden_md=0.0, block_size_ft=250.0)

    costs, valid_blocks, observed_counts = block_candidate_costs(
        observed_gr, expected_gr, blocks, minimum_observed_points=20
    )

    np.testing.assert_allclose(costs, [0.0, 10.0])
    assert valid_blocks.tolist() == [0]
    assert observed_counts.tolist() == [20, 0]


def test_hidden_gr_roll_preserves_nan_mask_and_is_deterministic() -> None:
    """循环错移须把完整 GR 和 NaN 掩码一起移动，不能产生插补。"""
    gr = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
    first = deterministic_hidden_gr_roll(gr, 0.40)
    second = deterministic_hidden_gr_roll(gr, 0.40)

    np.testing.assert_array_equal(np.isnan(first), np.isnan(np.roll(gr, 2)))
    np.testing.assert_allclose(first, second, equal_nan=True)


def test_block_cost_uses_shared_calibration_scale_not_candidate_mad() -> None:
    """所有候选必须共享前缀校准尺度，不能用自身残差波动稀释成本。"""
    observed = np.full(20, 100.0)
    expected = np.column_stack([np.full(20, 102.0), np.r_[np.zeros(10), np.full(10, 200.0)]])
    costs, _, _ = block_candidate_costs(observed, expected, np.zeros(20, dtype=int), common_sigma=2.0)
    np.testing.assert_allclose(costs, [1.0, 50.0])


def test_block_cost_excludes_candidate_with_fewer_than_twenty_finite_expected_points() -> None:
    """候选自身期望 GR 不足20点时，即使观测充分也不得形成低成本。"""
    observed = np.full(20, 100.0)
    expected = np.column_stack([np.full(20, 101.0), np.r_[np.full(19, 100.0), np.nan]])
    outside = np.column_stack([np.zeros(20, bool), np.ones(20, bool)])
    costs, valid, _ = block_candidate_costs(observed, expected, np.zeros(20, dtype=int), common_sigma=1.0, out_of_range=outside)
    assert valid.tolist() == [0]
    assert costs[0] == 1.0
    assert np.isnan(costs[1])


def test_rank_metrics_top1_compares_actual_best_candidate_not_rank_at_index_zero() -> None:
    """候选0名次相同但非最优时，不得误报 top1 命中。"""
    metrics = _rank_metrics(np.array([2.0, 1.0, 3.0]), np.array([2.0, 3.0, 1.0]))
    assert metrics["top1_hit"] is False
    assert metrics["top3_contains_true_best"] is True


def test_rank_metrics_returns_nan_for_no_usable_score() -> None:
    """空/全 NaN 评分不能生成虚假的排名、命中或相关。"""
    metrics = _rank_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["top1_hit"])
    assert np.isnan(metrics["score_rank"]).all()
    assert np.isnan(metrics["truth_rank"]).all()


def test_rank_metrics_returns_nan_correlation_for_all_tied_scores() -> None:
    """所有候选同分没有排序信息，Spearman 必须是 NaN 而非警告/伪信号。"""
    metrics = _rank_metrics(np.array([1.0, 1.0, 1.0]), np.array([3.0, 2.0, 1.0]))
    assert np.isnan(metrics["spearman"])


def test_smoke_and_all_legal_caches_are_physically_isolated() -> None:
    """smoke 产物绝不能成为 all/evaluate 可复用的 legal cache。"""
    assert legal_cache_dir(Path("artifact"), "smoke") != legal_cache_dir(Path("artifact"), "all")


def test_control_contrast_aggregates_five_shifts_within_each_well() -> None:
    """shift 对照先在同一井平均，不能把五条 shift 行直接与 normal 混合。"""
    frame = __import__("pandas").DataFrame([
        {"well_id": "a", "control": "normal", "spearman": 10.0},
        {"well_id": "a", "control": "shift_0.20", "spearman": 2.0},
        {"well_id": "a", "control": "shift_0.35", "spearman": 4.0},
        {"well_id": "b", "control": "normal", "spearman": 20.0},
        {"well_id": "b", "control": "shift_0.20", "spearman": 6.0},
        {"well_id": "b", "control": "shift_0.35", "spearman": 8.0},
    ])
    contrast = build_control_contrasts(frame, ["spearman"])["normal_minus_shift_mean"]["spearman"]
    assert contrast["mean_delta"] == 10.0
    assert contrast["valid_wells"] == 2
