from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_s03_nearest_dense_surface import (
    DenseNearestSurfaceIndex,
    build_legal_nearest_surface_path,
)


def _source_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_well_id": ["a", "a", "b", "c"],
            "source_row_index": [0, 1, 0, 0],
            "source_distance_ft": [0.0, 50.0, 0.0, 0.0],
            "source_x": [0.0, 1.0, 10.0, -10.0],
            "source_y": [0.0, 0.0, 0.0, 0.0],
            "source_surface": [100.0, 101.0, 110.0, 90.0],
        }
    )


def test_nearest_dense_index_returns_exact_nearest_point() -> None:
    index = DenseNearestSurfaceIndex(_source_points())

    result = index.predict_nearest_surface(np.array([[0.9, 0.0], [9.0, 0.0]]))

    assert result["nearest_source_well_id"].tolist() == ["a", "b"]
    np.testing.assert_allclose(result["nearest_surface"], [101.0, 110.0])
    np.testing.assert_allclose(result["nearest_source_distance"], [0.1, 1.0])


def test_nearest_dense_index_can_exclude_target_well() -> None:
    index = DenseNearestSurfaceIndex(_source_points())

    result = index.predict_nearest_surface(
        np.array([[0.0, 0.0]]),
        excluded_well_id="a",
    )

    assert result.loc[0, "nearest_source_well_id"] in {"b", "c"}
    assert result.loc[0, "nearest_source_distance"] == 10.0


class _ExactSurfaceIndex:
    def predict_nearest_surface(self, query_xy: np.ndarray, **_: object) -> pd.DataFrame:
        surface = 1000.0 + 0.2 * query_xy[:, 0] + 0.1 * query_xy[:, 1]
        return pd.DataFrame(
            {
                "nearest_surface": surface,
                "nearest_source_distance": np.ones(len(query_xy)),
                "nearest_source_well_id": ["source"] * len(query_xy),
            }
        )


def test_legal_nearest_surface_path_calibrates_visible_prefix_and_ignores_truth_columns() -> None:
    row_count = 11
    x = np.linspace(0.0, 100.0, row_count)
    y = np.zeros(row_count)
    z = np.linspace(-5000.0, -4990.0, row_count)
    surface = 1000.0 + 0.2 * x
    true_tvt = surface + 11600.0 - z
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

    path = build_legal_nearest_surface_path(
        target_horizontal_df=target,
        nearest_surface_index=_ExactSurfaceIndex(),
        target_control_step_horizontal_ft=25.0,
    )

    np.testing.assert_allclose(path["surface_tvt"], true_tvt[6:], atol=1e-10)
    changed = target.copy()
    changed["TVT"] = -999.0
    changed["EGFDU"] = 999.0
    changed_path = build_legal_nearest_surface_path(
        target_horizontal_df=changed,
        nearest_surface_index=_ExactSurfaceIndex(),
        target_control_step_horizontal_ft=25.0,
    )
    pd.testing.assert_frame_equal(path, changed_path)
