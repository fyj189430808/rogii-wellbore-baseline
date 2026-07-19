import numpy as np

from src.p3_n01b_half_strength_neighbor_residual import (
    apply_neighbor_multiplier,
    pooled_multiplier_grid,
    select_grid_multiplier,
)


def test_half_strength_is_midpoint_between_baseline_and_full_correction() -> None:
    baseline = np.asarray([10.0, 20.0, 30.0])
    full = np.asarray([14.0, 18.0, 36.0])
    result = apply_neighbor_multiplier(baseline, full, multiplier=0.5)
    assert np.allclose(result, np.asarray([12.0, 19.0, 33.0]))


def test_multiplier_grid_uses_pooled_row_level_rmse() -> None:
    target_chunks = [np.asarray([2.0, 2.0]), np.asarray([8.0])]
    baseline_chunks = [np.asarray([0.0, 0.0]), np.asarray([10.0])]
    full_chunks = [np.asarray([4.0, 4.0]), np.asarray([6.0])]

    rows = pooled_multiplier_grid(
        target_chunks,
        baseline_chunks,
        full_chunks,
        multipliers=[0.0, 0.5, 1.0],
    )

    assert [row["multiplier"] for row in rows] == [0.0, 0.5, 1.0]
    assert np.isclose(rows[1]["rmse"], 0.0)
    assert np.isclose(rows[0]["rmse"], rows[2]["rmse"])


def test_grid_selection_has_stable_smallest_multiplier_tie_break() -> None:
    rows = [
        {"multiplier": 0.75, "rmse": 3.0},
        {"multiplier": 0.5, "rmse": 3.0},
        {"multiplier": 0.25, "rmse": 4.0},
    ]
    assert select_grid_multiplier(rows) == 0.5

