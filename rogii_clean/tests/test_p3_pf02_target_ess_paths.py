"""P3-PF02：精确目标 ESS 聚合路径的最小数值测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.p3_pf02_target_ess_paths import (
    aggregate_target_ess_paths,
    require_exact_log_likelihood_match,
    solve_temperature_for_target_ess,
)


def test_bisection_reaches_each_requested_effective_path_count() -> None:
    log_likelihoods = np.array([-6.0, -3.0, -1.0, 0.0, 0.5, 1.0])

    results = [
        solve_temperature_for_target_ess(log_likelihoods, target_ess=target)
        for target in (2.0, 3.0, 5.0)
    ]

    for target, result in zip((2.0, 3.0, 5.0), results, strict=True):
        assert result["achieved_ess"] == pytest.approx(target, abs=1e-6)
        assert result["temperature"] > 0.0
    assert results[0]["temperature"] < results[1]["temperature"] < results[2]["temperature"]


def test_aggregate_returns_one_delta_path_for_each_target_without_changing_seeds() -> None:
    seed_predictions = np.array(
        [[100.0, 101.0], [102.0, 103.0], [104.0, 105.0], [106.0, 107.0]],
        dtype=np.float64,
    )
    original = seed_predictions.copy()
    log_likelihoods = np.array([-4.0, -2.0, -1.0, 0.0], dtype=np.float64)

    paths, report = aggregate_target_ess_paths(
        seed_predictions=seed_predictions,
        seed_log_likelihoods=log_likelihoods,
        last_visible_tvt=99.0,
        target_effective_sample_sizes=(2, 3),
    )

    assert set(paths) == {"pf128_ess2_delta", "pf128_ess3_delta"}
    assert paths["pf128_ess2_delta"].shape == (2,)
    assert report["ess2_achieved_ess"] == pytest.approx(2.0, abs=1e-6)
    assert report["ess3_achieved_ess"] == pytest.approx(3.0, abs=1e-6)
    np.testing.assert_array_equal(seed_predictions, original)


def test_log_likelihood_reconciliation_requires_bitwise_identity() -> None:
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    require_exact_log_likelihood_match(expected.copy(), expected)

    changed = expected.copy()
    changed[1] = np.nextafter(changed[1], np.inf)
    with pytest.raises(RuntimeError, match="逐位一致"):
        require_exact_log_likelihood_match(changed, expected)


def test_delta_follows_frozen_absolute_float32_then_subtract_float32_convention() -> None:
    seed_predictions = np.array(
        [[100_000_001.0], [100_000_002.0], [100_000_003.0], [100_000_004.0]],
        dtype=np.float64,
    )
    log_likelihoods = np.array([-4.0, -2.0, -1.0, 0.0], dtype=np.float64)

    paths, _ = aggregate_target_ess_paths(
        seed_predictions=seed_predictions,
        seed_log_likelihoods=log_likelihoods,
        last_visible_tvt=100_000_000.0,
        target_effective_sample_sizes=(2,),
    )

    # 旧 P2-P01 先把绝对路径压成 float32，再减 float32 的末个可见 TVT。
    np.testing.assert_array_equal(paths["pf128_ess2_delta"], np.array([0.0], dtype=np.float32))
