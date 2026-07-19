import numpy as np

from src.p3_n01a_neighbor_residual_audit import (
    aggregate_control_profiles,
    build_fixed_shuffle,
    compute_trajectory_geometry,
    fit_source_control4,
    interpolate_control4,
    neighbor_weight,
)


def _line(start_x: float, end_x: float, y: float, rows: int = 11) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(start_x, end_x, rows),
            np.full(rows, y, dtype=np.float64),
        ]
    )


def test_geometry_uses_full_parallel_trajectory_and_detects_reverse_direction() -> None:
    target_xy = _line(0.0, 1000.0, 0.0)
    source_xy = _line(1000.0, 0.0, 100.0)

    geometry = compute_trajectory_geometry(target_xy, source_xy)

    assert np.isclose(geometry.minimum_distance_ft, 100.0)
    assert np.isclose(geometry.azimuth_difference_deg, 0.0)
    assert np.isclose(geometry.parallel_overlap_ft, 1000.0)
    assert geometry.reverse_source_controls is True


def test_neighbor_weight_matches_frozen_formula() -> None:
    weight = neighbor_weight(
        distance_ft=1000.0,
        azimuth_difference_deg=30.0,
        overlap_ft=500.0,
        same_typewell_tail=True,
    )
    expected = np.exp(-1.0) * (np.cos(np.deg2rad(30.0)) ** 2) * 0.5 * 1.5
    assert np.isclose(weight, expected)


def test_aggregate_reverses_controls_and_applies_eta_shrinkage() -> None:
    controls = [np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([2.0, 2.0, 2.0, 2.0])]
    weights = np.asarray([2.0, 1.0])
    reverse = np.asarray([True, False])

    result = aggregate_control_profiles(controls, weights, reverse)

    expected_mean = (2.0 * np.asarray([4.0, 3.0, 2.0, 1.0]) + np.asarray([2.0] * 4)) / 3.0
    assert np.allclose(result.weighted_controls, expected_mean)
    assert np.isclose(result.eta, 3.0 / 4.0)
    assert np.allclose(result.shrunk_controls, expected_mean * 0.75)


def test_control4_interpolation_uses_four_equal_progress_knots() -> None:
    progress = np.asarray([0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0])
    values = interpolate_control4(np.asarray([0.0, 3.0, 6.0, 9.0]), progress)
    assert np.allclose(values, np.asarray([0.0, 1.5, 3.0, 4.5, 6.0, 9.0]))


def test_fixed_shuffle_is_deterministic_and_has_no_fixed_points() -> None:
    source_ids = [f"well_{index}" for index in range(9)]
    first = build_fixed_shuffle(source_ids, seed=20260719)
    second = build_fixed_shuffle(list(reversed(source_ids)), seed=20260719)

    assert first == second
    assert set(first) == set(source_ids)
    assert set(first.values()) == set(source_ids)
    assert all(source_id != profile_id for source_id, profile_id in first.items())


def test_source_control4_fits_true_minus_direct_path() -> None:
    md = np.linspace(100.0, 200.0, 101)
    progress = (md - md.min()) / (md.max() - md.min())
    expected_controls = np.asarray([2.0, -1.0, 4.0, 3.0])
    direct_path = np.full(len(md), 1000.0)
    target = direct_path + interpolate_control4(expected_controls, progress)

    fitted = fit_source_control4(md, target, direct_path)

    assert np.allclose(fitted, expected_controls)
