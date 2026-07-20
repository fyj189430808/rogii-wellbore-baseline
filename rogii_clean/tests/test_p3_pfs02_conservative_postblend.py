import numpy as np
import pandas as pd


def test_build_postblend_candidates_uses_fixed_formula_and_order():
    from src.p3_pfs02_conservative_postblend import build_postblend_candidates

    legal_rows = pd.DataFrame(
        {
            "p3b00_pred_tvt": [100.0, 110.0],
            "last_visible_tvt": [90.0, 90.0],
            "pfs_lag250_delta": [20.0, 40.0],
            "pfs_lag500_delta": [10.0, 30.0],
            "pfs_lag1000_delta": [0.0, 20.0],
        }
    )

    result, candidate_names = build_postblend_candidates(
        legal_rows,
        lag_distances=(250, 500, 1000),
        blend_fractions=(0.10, 0.25, 0.50),
    )

    assert candidate_names == [
        "lag250_alpha10",
        "lag250_alpha25",
        "lag250_alpha50",
        "lag500_alpha10",
        "lag500_alpha25",
        "lag500_alpha50",
        "lag1000_alpha10",
        "lag1000_alpha25",
        "lag1000_alpha50",
    ]
    np.testing.assert_allclose(result["lag250_alpha25_pred_tvt"], [102.5, 115.0])
    np.testing.assert_allclose(result["lag1000_alpha50_pred_tvt"], [95.0, 110.0])


def test_score_screen_selects_unique_best_pooled_improvement_and_applies_gate():
    from src.p3_pfs02_conservative_postblend import score_candidates

    legal = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "row_index": [0, 1, 0, 1],
            "p3b00_pred_tvt": [2.0, 2.0, 2.0, 2.0],
            "good_pred_tvt": [1.0, 1.0, 1.0, 1.0],
            "bad_pred_tvt": [3.0, 3.0, 3.0, 3.0],
        }
    )
    targets = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "row_index": [0, 1, 0, 1],
            "target_tvt": [0.0, 0.0, 0.0, 0.0],
        }
    )

    metrics, _, _ = score_candidates(
        legal,
        targets,
        candidate_names=["good", "bad"],
        minimum_pooled_improvement_ft=0.20,
        maximum_any_fold_degradation_ft=0.10,
    )

    assert metrics["selected_candidate"] == "good"
    assert metrics["gate"]["passed"] is True
    assert metrics["candidates"]["good"]["pooled_improvement_ft"] == 1.0
    assert metrics["candidates"]["bad"]["gate_passed"] is False


def test_score_screen_stops_when_best_pooled_candidate_fails_single_fold_limit():
    from src.p3_pfs02_conservative_postblend import score_candidates

    legal = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [0, 1],
            "row_index": [0, 0],
            "p3b00_pred_tvt": [1.0, 1.0],
            "candidate_pred_tvt": [0.0, 3.0],
        }
    )
    targets = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [0, 1],
            "row_index": [0, 0],
            "target_tvt": [0.0, 0.0],
        }
    )

    metrics, _, _ = score_candidates(
        legal,
        targets,
        candidate_names=["candidate"],
        minimum_pooled_improvement_ft=-10.0,
        maximum_any_fold_degradation_ft=0.10,
    )

    assert metrics["selected_candidate"] == "candidate"
    assert metrics["gate"]["passed"] is False
    assert metrics["candidates"]["candidate"]["maximum_fold_degradation_ft"] == 2.0
