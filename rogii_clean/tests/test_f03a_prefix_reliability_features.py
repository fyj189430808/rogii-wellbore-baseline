import numpy as np
import pandas as pd

from src.f03a_prefix_reliability_features import (
    F03A_FEATURE_COLUMNS,
    build_well_prefix_reliability_features,
)


def test_builds_one_well_row_from_two_d0_scopes() -> None:
    registry = pd.DataFrame(
        {
            "well_id": ["well_a"],
            "fold": [2],
            "visible_rows": [200],
        }
    )
    offset_scores = pd.DataFrame(
        {
            "well_id": ["well_a", "well_a"],
            "fold": [2, 2],
            "scope": ["all_visible", "tail_1000ft"],
            "offset_ft": [0.0, 0.0],
            "n_points": [160, 80],
            "raw_ncc": [0.70, 0.85],
            "affine_median_ae": [8.0, 6.0],
        }
    )
    margins = pd.DataFrame(
        {
            "well_id": ["well_a", "well_a"],
            "fold": [2, 2],
            "scope": ["all_visible", "tail_1000ft"],
            "n_points": [160, 80],
            "ncc_margin_vs_best_wrong": [0.10, 0.25],
            "mae_margin_vs_best_wrong": [1.5, 3.0],
        }
    )

    actual = build_well_prefix_reliability_features(
        offset_scores=offset_scores,
        margins=margins,
        registry=registry,
    )

    assert actual.columns.tolist() == ["well_id", "fold", *F03A_FEATURE_COLUMNS]
    assert len(actual) == 1
    row = actual.iloc[0]
    assert row["f03a_all_valid_pair_count"] == 160
    assert np.isclose(row["f03a_all_valid_pair_fraction"], 0.8)
    assert np.isclose(row["f03a_tail_valid_pair_fraction"], 0.4)
    assert np.isclose(row["f03a_tail_minus_all_raw_ncc"], 0.15)
    assert np.isclose(row["f03a_tail_minus_all_affine_mae"], -2.0)
    assert np.isclose(row["f03a_tail_minus_all_ncc_margin"], 0.15)
    assert np.isclose(row["f03a_tail_minus_all_mae_margin"], 1.5)
