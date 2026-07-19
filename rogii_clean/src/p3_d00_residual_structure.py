"""Low-dimensional, oracle-only residual structure fits for P3-D00."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_SUPPORTED_POLYNOMIAL_DEGREES = frozenset(range(4))
_SUPPORTED_CONTROL_POINT_COUNTS = frozenset((4, 8))


@dataclass(frozen=True)
class ResidualFit:
    """A fitted residual basis with row-wise fitted values kept in memory only."""

    family: str
    anchored: bool
    dimensions: int
    coefficients: dict[str, float]
    basis: np.ndarray
    fitted: np.ndarray
    sse: float


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_fit_inputs(t: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    progress = _as_finite_vector(t, "t")
    residual_values = _as_finite_vector(residual, "residual")
    if progress.shape != residual_values.shape:
        raise ValueError("t and residual must have identical shapes")
    if (progress < 0.0).any() or (progress > 1.0).any():
        raise ValueError("t must be normalized to the closed interval [0, 1]")
    return progress, residual_values


def _least_squares_coefficients(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if design.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design,
        residual,
        rcond=None,
    )
    return np.asarray(coefficients, dtype=np.float64)


def _finish_fit(
    family: str,
    anchored: bool,
    full_basis: np.ndarray,
    coefficient_names: list[str],
    free_columns: np.ndarray,
    residual: np.ndarray,
) -> ResidualFit:
    full_coefficients = np.zeros(full_basis.shape[1], dtype=np.float64)
    full_coefficients[free_columns] = _least_squares_coefficients(
        full_basis[:, free_columns],
        residual,
    )
    fitted = full_basis @ full_coefficients
    error = residual - fitted
    sse = float(error @ error)
    return ResidualFit(
        family=family,
        anchored=bool(anchored),
        dimensions=int(len(free_columns)),
        coefficients={
            name: float(value)
            for name, value in zip(coefficient_names, full_coefficients, strict=True)
        },
        basis=full_basis,
        fitted=fitted,
        sse=sse,
    )


def normalized_progress(md: np.ndarray) -> np.ndarray:
    """Map a well's MD values to [0, 1], with degenerate wells fixed at zero."""

    md_values = _as_finite_vector(md, "md")
    minimum = float(np.min(md_values))
    maximum = float(np.max(md_values))
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros_like(md_values, dtype=np.float64)
    return (md_values - minimum) / span


def fit_polynomial_residual(
    t: np.ndarray,
    residual: np.ndarray,
    degree: int,
    anchored: bool,
) -> ResidualFit:
    """Fit a degree 0--3 polynomial; anchoring imposes e(0) = 0."""

    if int(degree) != degree or degree not in _SUPPORTED_POLYNOMIAL_DEGREES:
        raise ValueError("degree must be one of 0, 1, 2, or 3")
    progress, residual_values = _validate_fit_inputs(t, residual)
    degree = int(degree)
    full_basis = np.column_stack([progress**power for power in range(degree + 1)])
    names = ["intercept", *[f"t^{power}" for power in range(1, degree + 1)]]
    free_columns = np.arange(1 if anchored else 0, degree + 1, dtype=np.int64)
    return _finish_fit(
        family=f"polynomial_degree_{degree}",
        anchored=anchored,
        full_basis=full_basis,
        coefficient_names=names,
        free_columns=free_columns,
        residual=residual_values,
    )


def _control_point_basis(t: np.ndarray, number_of_points: int) -> np.ndarray:
    knots = np.linspace(0.0, 1.0, number_of_points, dtype=np.float64)
    interval = np.searchsorted(knots, t, side="right") - 1
    interval = np.clip(interval, 0, number_of_points - 2)
    left_knot = knots[interval]
    fraction = (t - left_knot) / (knots[interval + 1] - left_knot)
    basis = np.zeros((len(t), number_of_points), dtype=np.float64)
    rows = np.arange(len(t))
    basis[rows, interval] = 1.0 - fraction
    basis[rows, interval + 1] = fraction
    return basis


def fit_control_point_residual(
    t: np.ndarray,
    residual: np.ndarray,
    number_of_points: int,
    anchored: bool,
) -> ResidualFit:
    """Fit 4 or 8 equally spaced piecewise-linear control-point values."""

    if (
        int(number_of_points) != number_of_points
        or number_of_points not in _SUPPORTED_CONTROL_POINT_COUNTS
    ):
        raise ValueError("number_of_points must be 4 or 8")
    progress, residual_values = _validate_fit_inputs(t, residual)
    number_of_points = int(number_of_points)
    full_basis = _control_point_basis(progress, number_of_points)
    names = [f"control_point_{index}" for index in range(number_of_points)]
    free_columns = np.arange(
        1 if anchored else 0,
        number_of_points,
        dtype=np.int64,
    )
    return _finish_fit(
        family=f"control_points_{number_of_points}",
        anchored=anchored,
        full_basis=full_basis,
        coefficient_names=names,
        free_columns=free_columns,
        residual=residual_values,
    )


def _fit_summary(fit: ResidualFit) -> dict[str, object]:
    rows = len(fit.fitted)
    return {
        "family": fit.family,
        "anchored": fit.anchored,
        "dimensions": fit.dimensions,
        "sse": fit.sse,
        "rmse": float(np.sqrt(fit.sse / rows)),
        "coefficients": fit.coefficients,
    }


def audit_well_residuals(well_frame: pd.DataFrame) -> dict[str, object]:
    """Summarize all fixed D00 bases for one development well.

    The returned dictionary intentionally contains only well-level diagnostics;
    it does not expose row-level residuals or fitted oracle values.
    """

    required_columns = {"well_id", "md", "target_tvt", "pred_tvt"}
    missing_columns = sorted(required_columns - set(well_frame.columns))
    if missing_columns:
        raise ValueError(f"well_frame missing required columns: {missing_columns}")
    if len(well_frame) == 0:
        raise ValueError("well_frame must not be empty")
    well_ids = well_frame["well_id"].dropna().astype(str).unique()
    if len(well_ids) != 1:
        raise ValueError("well_frame must contain exactly one well_id")

    progress = normalized_progress(well_frame["md"].to_numpy(dtype=np.float64))
    target = _as_finite_vector(well_frame["target_tvt"].to_numpy(dtype=np.float64), "target_tvt")
    prediction = _as_finite_vector(well_frame["pred_tvt"].to_numpy(dtype=np.float64), "pred_tvt")
    residual = target - prediction
    baseline_sse = float(residual @ residual)

    fits: dict[str, dict[str, object]] = {}
    for anchored in (False, True):
        anchor_name = "anchored" if anchored else "unanchored"
        for degree in range(4):
            key = f"polynomial_degree_{degree}_{anchor_name}"
            fits[key] = _fit_summary(
                fit_polynomial_residual(progress, residual, degree, anchored)
            )
        for number_of_points in (4, 8):
            key = f"control_points_{number_of_points}_{anchor_name}"
            fits[key] = _fit_summary(
                fit_control_point_residual(
                    progress,
                    residual,
                    number_of_points,
                    anchored,
                )
            )

    return {
        "well_id": str(well_ids[0]),
        "rows": int(len(well_frame)),
        "baseline_sse": baseline_sse,
        "baseline_rmse": float(np.sqrt(baseline_sse / len(well_frame))),
        "fits": fits,
    }
