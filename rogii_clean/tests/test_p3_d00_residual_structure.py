from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_d00_residual_structure import (  # noqa: E402
    audit_well_residuals,
    fit_control_point_residual,
    fit_polynomial_residual,
    normalized_progress,
)


@pytest.mark.parametrize(
    ("degree", "residual"),
    [
        (0, lambda t: np.full_like(t, 2.5)),
        (1, lambda t: -1.0 + 3.0 * t),
        (2, lambda t: 1.0 - 2.0 * t + 4.0 * t**2),
        (3, lambda t: -2.0 + 1.5 * t - 3.0 * t**2 + 2.0 * t**3),
    ],
)
def test_unanchored_polynomial_exactly_recovers_each_supported_degree(
    degree: int,
    residual: object,
) -> None:
    t = np.linspace(0.0, 1.0, 17)
    expected = residual(t)  # type: ignore[operator]

    fit = fit_polynomial_residual(t, expected, degree=degree, anchored=False)

    np.testing.assert_allclose(fit.fitted, expected, atol=1e-12)
    assert fit.sse == pytest.approx(0.0, abs=1e-20)
    assert set(fit.coefficients) == {"intercept", *[f"t^{power}" for power in range(1, degree + 1)]}


def test_anchored_polynomial_forces_zero_start_and_recovers_anchored_curve() -> None:
    t = np.linspace(0.0, 1.0, 13)
    residual = 2.0 * t - 3.0 * t**2 + 5.0 * t**3

    fit = fit_polynomial_residual(t, residual, degree=3, anchored=True)

    np.testing.assert_allclose(fit.fitted, residual, atol=1e-12)
    assert fit.coefficients["intercept"] == 0.0
    assert fit.fitted[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("number_of_points", [4, 8])
def test_control_point_basis_is_partition_of_unity_and_recovers_piecewise_curve(
    number_of_points: int,
) -> None:
    t = np.linspace(0.0, 1.0, 41)
    control_values = np.linspace(-3.0, 5.0, number_of_points)
    residual = np.interp(t, np.linspace(0.0, 1.0, number_of_points), control_values)

    fit = fit_control_point_residual(
        t,
        residual,
        number_of_points=number_of_points,
        anchored=False,
    )

    np.testing.assert_allclose(fit.basis.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(fit.fitted, residual, atol=1e-12)
    assert fit.sse == pytest.approx(0.0, abs=1e-20)
    assert list(fit.coefficients) == [f"control_point_{index}" for index in range(number_of_points)]


def test_anchored_control_points_set_first_control_value_to_zero() -> None:
    t = np.linspace(0.0, 1.0, 31)
    residual = np.interp(t, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], [0.0, 2.0, -1.0, 4.0])

    fit = fit_control_point_residual(t, residual, number_of_points=4, anchored=True)

    np.testing.assert_allclose(fit.fitted, residual, atol=1e-12)
    assert fit.coefficients["control_point_0"] == 0.0
    assert fit.fitted[0] == pytest.approx(0.0, abs=1e-12)


def test_single_row_well_has_finite_progress_fits_and_audit() -> None:
    well = pd.DataFrame(
        {
            "well_id": ["only"],
            "md": [123.0],
            "target_tvt": [12.0],
            "pred_tvt": [10.0],
        }
    )
    progress = normalized_progress(well["md"].to_numpy())

    assert progress.tolist() == [0.0]
    fits = [
        fit_polynomial_residual(progress, np.array([2.0]), degree=degree, anchored=anchored)
        for degree in range(4)
        for anchored in (False, True)
    ] + [
        fit_control_point_residual(progress, np.array([2.0]), number_of_points=points, anchored=anchored)
        for points in (4, 8)
        for anchored in (False, True)
    ]
    assert all(np.isfinite(fit.fitted).all() and np.isfinite(fit.sse) for fit in fits)

    audit = audit_well_residuals(well)
    assert audit["well_id"] == "only"
    assert audit["rows"] == 1
    assert audit["baseline_sse"] == pytest.approx(4.0)
    assert all(np.isfinite(result["sse"]) for result in audit["fits"].values())


def test_unanchored_sse_never_increases_when_polynomial_or_control_basis_gains_dimensions() -> None:
    t = np.linspace(0.0, 1.0, 29)
    residual = np.sin(9.0 * t) + 0.2 * t

    polynomial_sses = [
        fit_polynomial_residual(t, residual, degree=degree, anchored=False).sse
        for degree in range(4)
    ]
    control_sses = [
        fit_control_point_residual(t, residual, number_of_points=points, anchored=False).sse
        for points in (4, 8)
    ]

    assert np.all(np.diff(polynomial_sses) <= 1e-12)
    assert control_sses[1] <= control_sses[0] + 1e-12
