from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_s02_dense_relative_gradient import (
    DenseSurfaceGradientIndex,
    build_dense_source_profile,
    build_legal_relative_gradient_path,
    compute_outer_train_gradient_cap,
    reverse_source_profile_gradients,
)


def test_dense_source_profile_keeps_fixed_horizontal_control_points() -> None:
    horizontal = pd.DataFrame(
        {
            "X": [0.0, 20.0, 49.0, 51.0, 99.0, 101.0],
            "Y": [0.0] * 6,
            "EGFDU": [100.0, 102.0, 104.9, 105.1, 109.9, 110.1],
        }
    )

    profile = build_dense_source_profile(
        well_id="source_a",
        horizontal_df=horizontal,
        surface_name="EGFDU",
        control_step_horizontal_ft=50.0,
    )

    assert profile["source_row_index"].tolist() == [0, 3, 5]
    assert profile["source_well_id"].eq("source_a").all()
    np.testing.assert_allclose(profile["source_distance_ft"], [0.0, 51.0, 101.0])


def test_dense_index_uses_at_most_one_point_from_each_source_well() -> None:
    rows = []
    for point_number in range(20):
        rows.append(
            {
                "source_well_id": "dense_near_well",
                "source_row_index": point_number,
                "source_distance_ft": float(point_number),
                "source_x": float(point_number) * 0.01,
                "source_y": 0.0,
                "source_surface": 100.0,
            }
        )
    for well_number in range(1, 11):
        rows.append(
            {
                "source_well_id": f"well_{well_number}",
                "source_row_index": 0,
                "source_distance_ft": 0.0,
                "source_x": float(well_number),
                "source_y": float((well_number % 2) * 2 - 1),
                "source_surface": 100.0 + float(well_number),
            }
        )
    index = DenseSurfaceGradientIndex(pd.DataFrame(rows))

    result = index.predict_gradients(
        query_xy=np.array([[0.0, 0.0]]),
        nearest_distinct_source_wells=10,
        distance_weight_epsilon=0.001,
        gradient_magnitude_cap=100.0,
    )

    neighbor_ids = result.loc[0, "neighbor_well_ids"]
    assert len(neighbor_ids) == 10
    assert len(set(neighbor_ids)) == 10


def test_dense_local_plane_recovers_known_xy_gradient() -> None:
    rows = []
    points = [
        (-2.0, -2.0), (-2.0, 0.0), (-2.0, 2.0), (0.0, -2.0), (0.0, 2.0),
        (2.0, -2.0), (2.0, 0.0), (2.0, 2.0), (-1.0, 1.0), (1.0, -1.0),
    ]
    for well_number, (source_x, source_y) in enumerate(points):
        rows.append(
            {
                "source_well_id": f"well_{well_number}",
                "source_row_index": 0,
                "source_distance_ft": 0.0,
                "source_x": source_x,
                "source_y": source_y,
                "source_surface": 100.0 + 2.0 * source_x - 3.0 * source_y,
            }
        )
    index = DenseSurfaceGradientIndex(pd.DataFrame(rows))

    result = index.predict_gradients(
        query_xy=np.array([[0.5, -0.5]]),
        nearest_distinct_source_wells=10,
        distance_weight_epsilon=0.001,
        gradient_magnitude_cap=10.0,
    )

    assert result.loc[0, "gradient_x"] == pytest.approx(2.0, abs=1e-10)
    assert result.loc[0, "gradient_y"] == pytest.approx(-3.0, abs=1e-10)
    assert result.loc[0, "gradient_was_clipped"] == 0.0


def test_outer_train_gradient_cap_uses_within_well_directional_slopes() -> None:
    points = pd.DataFrame(
        {
            "source_well_id": ["a", "a", "a", "b", "b", "b"],
            "source_row_index": [0, 1, 2, 0, 1, 2],
            "source_distance_ft": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "source_x": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "source_y": [0.0] * 6,
            "source_surface": [0.0, 1.0, 2.0, 0.0, 2.0, 4.0],
        }
    )

    cap = compute_outer_train_gradient_cap(points, quantile=0.99)

    assert 0.1 < cap <= 0.2


def test_gradient_magnitude_is_clipped_without_changing_direction() -> None:
    rows = []
    for well_number, (source_x, source_y) in enumerate(
        [(-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, -1.0), (0.0, 1.0),
         (1.0, -1.0), (1.0, 0.0), (1.0, 1.0), (-0.5, 0.5), (0.5, -0.5)]
    ):
        rows.append(
            {
                "source_well_id": f"w{well_number}",
                "source_row_index": 0,
                "source_distance_ft": 0.0,
                "source_x": source_x,
                "source_y": source_y,
                "source_surface": 10.0 * source_x,
            }
        )
    index = DenseSurfaceGradientIndex(pd.DataFrame(rows))

    result = index.predict_gradients(
        query_xy=np.array([[0.0, 0.0]]),
        nearest_distinct_source_wells=10,
        distance_weight_epsilon=0.001,
        gradient_magnitude_cap=0.25,
    )

    assert result.loc[0, "raw_gradient_magnitude"] == pytest.approx(10.0)
    assert result.loc[0, "gradient_magnitude"] == pytest.approx(0.25)
    assert result.loc[0, "gradient_x"] == pytest.approx(0.25)
    assert result.loc[0, "gradient_y"] == pytest.approx(0.0, abs=1e-12)
    assert result.loc[0, "gradient_was_clipped"] == 1.0


class _ConstantGradientIndex:
    def predict_gradients(self, query_xy: np.ndarray, **_: object) -> pd.DataFrame:
        row_count = len(query_xy)
        return pd.DataFrame(
            {
                "gradient_x": np.full(row_count, 0.2),
                "gradient_y": np.full(row_count, 0.1),
                "raw_gradient_magnitude": np.full(row_count, np.hypot(0.2, 0.1)),
                "gradient_magnitude": np.full(row_count, np.hypot(0.2, 0.1)),
                "gradient_was_clipped": np.zeros(row_count),
                "nearest_source_distance": np.ones(row_count),
                "kth_source_distance": np.full(row_count, 10.0),
                "local_plane_rmse": np.zeros(row_count),
                "local_plane_rank": np.full(row_count, 3.0),
                "local_plane_condition": np.ones(row_count),
                "maximum_angular_gap_degrees": np.full(row_count, 90.0),
                "outside_support": np.zeros(row_count),
                "neighbor_well_ids": [tuple(f"w{i}" for i in range(10))] * row_count,
            }
        )


def test_relative_gradient_path_integrates_from_last_visible_anchor() -> None:
    row_count = 11
    x = np.linspace(0.0, 100.0, row_count)
    y = np.zeros(row_count)
    z = np.linspace(-5000.0, -4990.0, row_count)
    surface = 1000.0 + 0.2 * x + 0.1 * y
    marker = 11600.0
    true_tvt = surface + marker - z
    tvt_input = true_tvt.copy()
    tvt_input[6:] = np.nan
    target = pd.DataFrame(
        {
            "MD": np.arange(row_count, dtype=np.float64),
            "X": x,
            "Y": y,
            "Z": z,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
            "EGFDU": surface,
        }
    )

    legal_path = build_legal_relative_gradient_path(
        target_horizontal_df=target,
        dense_surface_index=_ConstantGradientIndex(),
        nearest_distinct_source_wells=10,
        target_control_step_horizontal_ft=25.0,
        distance_weight_epsilon=0.001,
        gradient_magnitude_cap=1.0,
    )

    np.testing.assert_allclose(legal_path["surface_tvt"], true_tvt[6:], atol=1e-10)

    changed_forbidden = target.copy()
    changed_forbidden["TVT"] = -99999.0
    changed_forbidden["EGFDU"] = 99999.0
    changed_path = build_legal_relative_gradient_path(
        target_horizontal_df=changed_forbidden,
        dense_surface_index=_ConstantGradientIndex(),
        nearest_distinct_source_wells=10,
        target_control_step_horizontal_ft=25.0,
        distance_weight_epsilon=0.001,
        gradient_magnitude_cap=1.0,
    )
    pd.testing.assert_frame_equal(legal_path, changed_path)


def test_reversed_gradient_control_preserves_well_median_and_flips_change() -> None:
    points = pd.DataFrame(
        {
            "source_well_id": ["a", "a", "a", "b", "b", "b"],
            "source_row_index": [0, 1, 2, 0, 1, 2],
            "source_distance_ft": [0.0, 50.0, 100.0, 0.0, 50.0, 100.0],
            "source_x": [0.0, 50.0, 100.0, 0.0, 50.0, 100.0],
            "source_y": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "source_surface": [10.0, 20.0, 30.0, 100.0, 120.0, 140.0],
        }
    )

    reversed_points = reverse_source_profile_gradients(points)

    for well_id in ["a", "b"]:
        original = points.loc[points["source_well_id"].eq(well_id), "source_surface"].to_numpy()
        reversed_values = reversed_points.loc[
            reversed_points["source_well_id"].eq(well_id), "source_surface"
        ].to_numpy()
        assert np.median(reversed_values) == pytest.approx(np.median(original))
        np.testing.assert_allclose(np.diff(reversed_values), -np.diff(original))
