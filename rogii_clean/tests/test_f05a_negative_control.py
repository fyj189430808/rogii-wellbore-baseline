import numpy as np
import pandas as pd

from src.f05a_negative_control import circular_shift_candidates_within_well


def test_circular_shift_preserves_each_well_distribution_but_breaks_row_alignment():
    source = pd.DataFrame(
        {
            "well_id": ["a", "a", "a", "a", "b", "b", "b"],
            "row_index": [0, 1, 2, 3, 0, 1, 2],
            "candidate_1": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
            "candidate_2": [5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0],
        }
    )

    shifted = circular_shift_candidates_within_well(
        source,
        candidate_columns=["candidate_1", "candidate_2"],
    )

    assert shifted[["well_id", "row_index"]].equals(
        source[["well_id", "row_index"]]
    )
    np.testing.assert_array_equal(
        shifted.loc[shifted["well_id"] == "a", "candidate_1"],
        np.array([3.0, 4.0, 1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        shifted.loc[shifted["well_id"] == "b", "candidate_1"],
        np.array([30.0, 10.0, 20.0]),
    )
    for well_id in ["a", "b"]:
        original_values = source.loc[source["well_id"] == well_id, "candidate_2"]
        shifted_values = shifted.loc[shifted["well_id"] == well_id, "candidate_2"]
        assert sorted(original_values.tolist()) == sorted(shifted_values.tolist())
