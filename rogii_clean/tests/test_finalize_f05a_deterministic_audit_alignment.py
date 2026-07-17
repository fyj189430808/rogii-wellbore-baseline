from __future__ import annotations

import pandas as pd

from scripts.finalize_f05a_deterministic_audit import (
    align_and_validate_audit_inputs,
)


KEY_COLUMNS = ["well_id", "row_index"]


def test_aligns_all_inputs_to_official_prediction_order_before_validation() -> None:
    official = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "row_index": [9, 2, 7],
            "fold": [1, 0, 0],
            "target_tvt": [109.0, 102.0, 107.0],
            "pred_tvt": [108.0, 101.0, 106.0],
            "carry_tvt": [110.0, 103.0, 108.0],
        }
    )
    official["row_index"] = official["row_index"].astype("int32")
    candidates = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b", "well_a"],
            "row_index": [7, 9, 2],
            "pf_ancc_std": [70.0, 90.0, 20.0],
            "beam_std_d": [7.0, 9.0, 2.0],
            "sc25_sc": [0.7, 0.9, 0.2],
        }
    )
    candidates["row_index"] = candidates["row_index"].astype("int64")
    base = pd.DataFrame(
        {
            "well_id": ["well_a", "well_a", "well_b"],
            "row_index": [2, 7, 9],
            "gr_missing": [0.2, 0.7, 0.9],
            "hidden_fraction": [0.02, 0.07, 0.09],
        }
    )
    base["row_index"] = base["row_index"].astype("int32")
    b00 = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b", "well_a"],
            "row_index": [7, 9, 2],
            "target_tvt": [107.0, 109.0, 102.0],
            "pred_tvt": [105.0, 107.0, 100.0],
        }
    )
    b00["row_index"] = b00["row_index"].astype("int32")

    aligned_candidates, aligned_base, aligned_b00, well_table, alignment = (
        align_and_validate_audit_inputs(candidates, base, official, b00)
    )

    expected_keys = official[KEY_COLUMNS].reset_index(drop=True)
    for aligned in (aligned_candidates, aligned_base, aligned_b00):
        pd.testing.assert_frame_equal(
            aligned[KEY_COLUMNS].reset_index(drop=True),
            expected_keys,
        )

    assert aligned_candidates["pf_ancc_std"].tolist() == [90.0, 20.0, 70.0]
    assert aligned_base["gr_missing"].tolist() == [0.9, 0.2, 0.7]
    assert aligned_b00["pred_tvt"].tolist() == [107.0, 100.0, 105.0]
    assert well_table["well_id"].tolist() == ["well_b", "well_a"]
    assert alignment["candidate_prediction_keys_exact"] is True
    assert alignment["candidate_base_keys_exact"] is True
    assert alignment["candidate_b00_keys_exact"] is True
