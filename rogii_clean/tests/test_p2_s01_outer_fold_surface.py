from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_s01_outer_fold_surface import (
    LEGAL_TARGET_COLUMNS,
    allowed_surface_source_ids,
    build_legal_surface_path,
    build_source_representative,
    local_plane_predictions,
    permute_source_surfaces,
    select_control_indices,
)


def test_build_source_representative_uses_one_median_point_per_well() -> None:
    horizontal = pd.DataFrame(
        {
            "X": [0.0, 10.0, 20.0],
            "Y": [1.0, 3.0, 9.0],
            "EGFDU": [100.0, 104.0, 120.0],
        }
    )

    representative = build_source_representative("well_a", horizontal, "EGFDU")

    assert representative == {
        "well_id": "well_a",
        "source_x": 10.0,
        "source_y": 3.0,
        "source_surface": 104.0,
    }


def test_select_control_indices_keeps_endpoints_and_visible_anchor() -> None:
    horizontal_distance = np.array([0.0, 20.0, 49.0, 51.0, 99.0, 101.0])
    visible_mask = np.array([True, True, True, False, False, False])

    indices = select_control_indices(horizontal_distance, visible_mask, step_ft=50.0)

    assert indices.tolist() == [0, 2, 3, 5]


def test_local_plane_recovers_a_known_surface_plane() -> None:
    source_rows = []
    for well_number, (source_x, source_y) in enumerate(
        [(-2.0, -2.0), (-2.0, 0.0), (-2.0, 2.0), (0.0, -2.0), (0.0, 2.0),
         (2.0, -2.0), (2.0, 0.0), (2.0, 2.0), (-1.0, 1.0), (1.0, -1.0)]
    ):
        source_rows.append(
            {
                "well_id": f"source_{well_number}",
                "source_x": source_x,
                "source_y": source_y,
                "source_surface": 100.0 + 2.0 * source_x - 3.0 * source_y,
            }
        )
    source_table = pd.DataFrame(source_rows)
    query_xy = np.array([[0.5, -0.5], [1.5, 0.5]], dtype=np.float64)

    prediction = local_plane_predictions(
        query_xy=query_xy,
        source_table=source_table,
        nearest_source_wells=10,
        distance_weight_epsilon=0.001,
    )

    expected = 100.0 + 2.0 * query_xy[:, 0] - 3.0 * query_xy[:, 1]
    np.testing.assert_allclose(prediction["predicted_surface"], expected, atol=1e-10)
    np.testing.assert_allclose(prediction["local_plane_rmse"], 0.0, atol=1e-10)
    assert prediction["local_plane_rank"].eq(3).all()


def test_local_plane_can_exclude_the_target_source_well() -> None:
    source_table = pd.DataFrame(
        {
            "well_id": ["target", "a", "b", "c", "d"],
            "source_x": [0.0, -1.0, 1.0, 0.0, 0.0],
            "source_y": [0.0, 0.0, 0.0, -1.0, 1.0],
            "source_surface": [999.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    prediction = local_plane_predictions(
        query_xy=np.array([[0.0, 0.0]]),
        source_table=source_table,
        nearest_source_wells=4,
        distance_weight_epsilon=0.001,
        excluded_well_id="target",
    )

    assert prediction.loc[0, "predicted_surface"] == pytest.approx(10.0)
    assert prediction.loc[0, "nearest_source_distance"] == pytest.approx(1.0)


def test_legal_surface_path_uses_only_visible_target_columns() -> None:
    source_rows = []
    for well_number, source_x in enumerate(np.linspace(-50.0, 150.0, 10)):
        source_y = float((well_number % 2) * 50.0 - 25.0)
        source_rows.append(
            {
                "well_id": f"source_{well_number}",
                "source_x": float(source_x),
                "source_y": source_y,
                "source_surface": 1000.0 + 0.2 * source_x + 0.1 * source_y,
            }
        )
    source_table = pd.DataFrame(source_rows)

    row_count = 11
    target_x = np.linspace(0.0, 100.0, row_count)
    target_y = np.zeros(row_count)
    target_z = np.linspace(-5000.0, -4990.0, row_count)
    true_surface = 1000.0 + 0.2 * target_x + 0.1 * target_y
    marker_offset = 11600.0
    true_tvt = true_surface + marker_offset - target_z
    tvt_input = true_tvt.copy()
    tvt_input[6:] = np.nan
    target = pd.DataFrame(
        {
            "MD": np.arange(row_count, dtype=np.float64),
            "X": target_x,
            "Y": target_y,
            "Z": target_z,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
            "EGFDU": true_surface,
        }
    )

    legal_path = build_legal_surface_path(
        target_horizontal_df=target,
        source_table=source_table,
        nearest_source_wells=10,
        control_step_horizontal_ft=25.0,
        distance_weight_epsilon=0.001,
    )

    np.testing.assert_allclose(legal_path["surface_tvt"], true_tvt[6:], atol=1e-8)
    assert set(LEGAL_TARGET_COLUMNS) == {"MD", "X", "Y", "Z", "TVT_input"}
    assert legal_path["row_index"].tolist() == list(range(6, row_count))

    changed_forbidden_columns = target.copy()
    changed_forbidden_columns["TVT"] = -123456.0
    changed_forbidden_columns["EGFDU"] = 987654.0
    changed_path = build_legal_surface_path(
        target_horizontal_df=changed_forbidden_columns,
        source_table=source_table,
        nearest_source_wells=10,
        control_step_horizontal_ft=25.0,
        distance_weight_epsilon=0.001,
    )
    pd.testing.assert_frame_equal(legal_path, changed_path)


def test_allowed_surface_sources_exclude_outer_fold_and_target_well() -> None:
    registry = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "fold": [0, 1, 1, 2],
        }
    )

    source_ids = allowed_surface_source_ids(
        registry_df=registry,
        outer_fold=0,
        target_well_id="b",
    )

    assert source_ids == ["c", "d"]


def test_permuted_surface_negative_control_is_deterministic_and_keeps_xy() -> None:
    source_table = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "source_x": [0.0, 1.0, 2.0, 3.0],
            "source_y": [4.0, 5.0, 6.0, 7.0],
            "source_surface": [10.0, 20.0, 30.0, 40.0],
        }
    )

    first = permute_source_surfaces(source_table, seed=20260719)
    second = permute_source_surfaces(source_table, seed=20260719)

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(first[["source_x", "source_y"]], source_table[["source_x", "source_y"]])
    assert sorted(first["source_surface"].tolist()) == sorted(source_table["source_surface"].tolist())
    assert not first["source_surface"].equals(source_table["source_surface"])


def test_local_plane_rank_deficiency_uses_fallback() -> None:
    source_table = pd.DataFrame(
        {
            "well_id": [f"w{i}" for i in range(10)],
            "source_x": np.arange(10, dtype=np.float64),
            "source_y": np.zeros(10, dtype=np.float64),
            "source_surface": np.arange(10, dtype=np.float64),
        }
    )

    prediction = local_plane_predictions(
        query_xy=np.array([[4.5, 0.0]]),
        source_table=source_table,
        nearest_source_wells=10,
        distance_weight_epsilon=0.001,
    )

    assert prediction.loc[0, "used_fallback"] == 1.0
    assert np.isfinite(prediction.loc[0, "predicted_surface"])
