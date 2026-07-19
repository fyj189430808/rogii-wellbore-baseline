from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_r01a_linear_residual import (
    apply_linear_residual,
    fit_linear_residual,
    fit_predict_ridge_coefficients,
    normalized_progress,
    paired_well_bootstrap_delta,
)


def test_fit_linear_residual_recovers_unanchored_intercept_and_slope() -> None:
    md = np.array([100.0, 110.0, 120.0, 130.0])
    base = np.array([10.0, 10.0, 10.0, 10.0])
    target = base + 2.5 - 4.0 * normalized_progress(md)

    coefficients = fit_linear_residual(md, base, target)

    np.testing.assert_allclose(coefficients, [2.5, -4.0], atol=1e-12)


def test_apply_linear_residual_uses_normalized_md_progress() -> None:
    md = np.array([20.0, 30.0, 50.0])
    base = np.array([100.0, 101.0, 102.0])

    corrected = apply_linear_residual(md, base, intercept=3.0, slope=6.0)

    np.testing.assert_allclose(corrected, [103.0, 106.0, 111.0])


def test_multioutput_ridge_is_fit_only_on_training_rows() -> None:
    train_features = pd.DataFrame({"f": [-2.0, -1.0, 1.0, 2.0]})
    train_targets = np.column_stack(
        [2.0 * train_features["f"], -3.0 * train_features["f"]]
    )
    validation_features = pd.DataFrame({"f": [3.0]})

    prediction, _ = fit_predict_ridge_coefficients(
        train_features,
        train_targets,
        validation_features,
        alpha=0.0,
    )

    np.testing.assert_allclose(prediction, [[6.0, -9.0]], atol=1e-10)


def test_bootstrap_returns_negative_delta_for_uniform_improvement() -> None:
    per_well = pd.DataFrame(
        {
            "well_id": ["a", "b", "c"],
            "rows": [10, 20, 30],
            "base_sse": [100.0, 200.0, 300.0],
            "candidate_sse": [25.0, 50.0, 75.0],
        }
    )

    result = paired_well_bootstrap_delta(per_well, n_resamples=100, seed=7)

    assert result["ci95_high"] < 0.0
    assert result["probability_better"] == 1.0
