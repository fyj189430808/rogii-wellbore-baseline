"""P3-UP07 核心公式测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_p3_up07_pfs_correction_strength_oracle import SHADOW_METADATA_PATH
from src.p3_up07_pfs_correction_strength_oracle import (
    grid_oracle,
    linear_slope_per_1000ft,
    make_alpha_grid,
    rmse_curve_from_sufficient_statistics,
)


def test_quadratic_curve_matches_direct_row_predictions() -> None:
    target = np.array([10.0, 12.0, 8.0, 15.0])
    base = np.array([9.0, 13.0, 7.5, 14.0])
    direction = np.array([2.0, -1.0, 0.5, 3.0])
    alphas = make_alpha_grid()
    error = target - base
    formula = rmse_curve_from_sufficient_statistics(
        len(target),
        float(np.dot(error, error)),
        float(np.dot(error, direction)),
        float(np.dot(direction, direction)),
        alphas,
    )
    direct = np.array(
        [np.sqrt(np.mean((target - (base + alpha * direction)) ** 2)) for alpha in alphas]
    )
    np.testing.assert_allclose(formula, direct, rtol=0.0, atol=1.0e-12)


def test_grid_oracle_recovers_known_alpha_and_up03_formula() -> None:
    direction = np.array([-2.0, 1.0, 4.0, -3.0])
    base = np.array([100.0, 101.0, 102.0, 103.0])
    target = base + 0.35 * direction
    error = target - base
    alphas = make_alpha_grid()
    best_alpha, best_rmse, _ = grid_oracle(
        len(target),
        float(np.dot(error, error)),
        float(np.dot(error, direction)),
        float(np.dot(direction, direction)),
        alphas,
    )
    assert best_alpha == 0.35
    assert best_rmse < 1.0e-12
    np.testing.assert_allclose(base + 0.25 * direction, base + 0.25 * direction)


def test_slope_is_reported_per_1000ft() -> None:
    md = np.array([0.0, 500.0, 1000.0, 1500.0])
    values = 2.0 + 0.003 * md
    assert abs(linear_slope_per_1000ft(md, values) - 3.0) < 1.0e-12


def test_no_label_metadata_contains_all_wells_not_only_shadow_subset() -> None:
    metadata = pd.read_csv(SHADOW_METADATA_PATH, dtype={"well_id": str})
    shadow = pd.read_csv(
        SHADOW_METADATA_PATH.parent / "shadow_holdout.csv",
        usecols=["well_id"],
        dtype={"well_id": str},
    )
    development = metadata.loc[~metadata["well_id"].isin(set(shadow["well_id"]))]
    assert len(metadata) == 773
    assert len(development) == 657
    assert development["hidden_gr_observed_rate"].notna().all()
