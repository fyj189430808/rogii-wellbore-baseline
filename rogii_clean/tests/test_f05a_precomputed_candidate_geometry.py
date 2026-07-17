from __future__ import annotations

import numpy as np
import pandas as pd

from src.f05a_precomputed_candidate_geometry import (
    CANDIDATE_PATH_COLUMNS,
    F05A_FEATURE_COLUMNS,
    build_candidate_geometry,
)


def _synthetic_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for well_id, row_indices, shift in [
        ("well_a", [10, 11], 0.0),
        ("well_b", [20], 100.0),
    ]:
        for within_well_index, row_index in enumerate(row_indices):
            row: dict[str, object] = {
                "well": well_id,
                "id": f"{well_id}_{row_index}",
                "pf_ancc_std": 1.5 + within_well_index,
                "pf_vs_z": -2.0 - within_well_index,
                "sc8_sc": 1.0,
                "sc15_sc": 3.0,
                "sc25_sc": 5.0,
                "target": 999999.0,
            }
            for candidate_index, column_name in enumerate(CANDIDATE_PATH_COLUMNS):
                row[column_name] = shift + within_well_index * 10.0 + candidate_index
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_candidate_geometry_returns_only_keys_and_fixed_features() -> None:
    result = build_candidate_geometry(_synthetic_rows())

    assert list(result.columns) == ["well_id", "row_index", *F05A_FEATURE_COLUMNS]
    assert result[["well_id", "row_index"]].to_dict("records") == [
        {"well_id": "well_a", "row_index": 10},
        {"well_id": "well_a", "row_index": 11},
        {"well_id": "well_b", "row_index": 20},
    ]
    assert "target" not in result.columns


def test_row_and_endpoint_geometry_match_simple_candidates() -> None:
    result = build_candidate_geometry(_synthetic_rows())
    first = result.iloc[0]
    second = result.iloc[1]

    assert np.isclose(first["candidate_median"], 7.0)
    assert np.isclose(first["candidate_iqr"], 7.0)
    assert np.isclose(first["candidate_mad"], 4.0)
    assert np.isclose(first["candidate_span"], 14.0)
    assert np.isclose(first["candidate_largest_gap"], 1.0)
    assert np.isclose(first["candidate_typical_separation"], 7.0)
    assert np.isclose(first["candidate_tail_separation"], 11.2)
    assert np.isclose(first["pf_vs_candidate_median_abs"], 7.0)
    assert np.isclose(first["pf_seed_std"], 1.5)
    assert np.isclose(first["abs_pf_vs_z"], 2.0)
    assert np.isclose(first["scale_score_std"], np.std([1.0, 3.0, 5.0]))

    # well_a 的末行候选为 10..24，因此井级终点中位数应为 17。
    assert np.isclose(first["endpoint_candidate_median"], 17.0)
    assert np.isclose(second["endpoint_candidate_median"], 17.0)
    assert np.isclose(first["candidate_span_growth"], 0.0)
    assert np.isclose(second["candidate_span_growth"], 0.0)


def test_target_values_cannot_change_features() -> None:
    source = _synthetic_rows()
    changed = source.copy()
    changed["target"] = [-1.0, 2.0, 3.0]

    original_features = build_candidate_geometry(source)
    changed_features = build_candidate_geometry(changed)

    pd.testing.assert_frame_equal(original_features, changed_features)
