from __future__ import annotations

import numpy as np

from src.p3_up06_projection_strength_oracle import (
    alpha_rmse_curve,
    best_grid_alpha,
    projection_candidate,
    quadratic_sse_terms,
    sse_curve_from_terms,
)


def test_projection_candidate_uses_requested_strength() -> None:
    base = np.array([10.0, 20.0, 30.0])
    projected = np.array([12.0, 18.0, 34.0])

    candidate = projection_candidate(base, projected, alpha=0.25)

    np.testing.assert_allclose(candidate, np.array([10.5, 19.5, 31.0]))


def test_best_grid_alpha_finds_known_half_strength() -> None:
    base = np.array([0.0, 0.0, 0.0])
    projected = np.array([2.0, 4.0, 6.0])
    target = np.array([1.0, 2.0, 3.0])
    alpha_grid = np.arange(0.0, 1.25 + 0.001, 0.05)

    best_alpha, best_rmse = best_grid_alpha(target, base, projected, alpha_grid)

    assert best_alpha == 0.5
    assert best_rmse == 0.0


def test_alpha_curve_is_row_weighted_micro_rmse() -> None:
    target = np.array([0.0, 0.0, 10.0])
    base = np.array([0.0, 0.0, 0.0])
    projected = np.array([0.0, 0.0, 20.0])
    alpha_grid = np.array([0.0, 0.5, 1.0])

    curve = alpha_rmse_curve(target, base, projected, alpha_grid)

    np.testing.assert_allclose(
        curve,
        np.array([np.sqrt(100.0 / 3.0), 0.0, np.sqrt(100.0 / 3.0)]),
    )


def test_fast_quadratic_sse_curve_matches_direct_candidates() -> None:
    target = np.array([1.0, -2.0, 4.0, 8.0])
    base = np.array([0.5, -1.0, 5.0, 7.0])
    projected = np.array([2.0, -3.0, 2.0, 10.0])
    alpha_grid = np.array([0.0, 0.35, 1.0, 1.25])

    terms = quadratic_sse_terms(target, base, projected)
    fast_curve = sse_curve_from_terms(terms, alpha_grid)
    direct_curve = np.array(
        [
            np.sum((target - projection_candidate(base, projected, alpha)) ** 2)
            for alpha in alpha_grid
        ]
    )

    np.testing.assert_allclose(fast_curve, direct_curve, rtol=0.0, atol=1.0e-12)
