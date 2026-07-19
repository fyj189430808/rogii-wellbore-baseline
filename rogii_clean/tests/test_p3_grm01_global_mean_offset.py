from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_grm01_global_mean_offset import (  # noqa: E402
    BLOCK_WIDTH_FT,
    CANDIDATE_OFFSETS_FT,
    circular_shift_finite_gr,
    nearest_grid_offset,
    rank_candidate_scores,
    score_global_offsets,
    score_legal_well,
)
from scripts.diagnose_p3_grm01_global_mean_offset import (  # noqa: E402
    build_top_k_flags,
    summarize_path_distribution,
    validate_legal_outer_contract,
)


def _linear_typewell() -> pd.DataFrame:
    tvt = np.arange(0.0, 401.0, 1.0)
    return pd.DataFrame({"TVT": tvt, "GR": 0.2 * tvt + 10.0})


def test_candidate_grid_and_ties_prefer_zero_then_smaller_absolute_offset() -> None:
    np.testing.assert_array_equal(CANDIDATE_OFFSETS_FT, np.arange(-30, 31, 1))
    scores = pd.DataFrame(
        {"offset_ft": [-2.0, 2.0, 0.0, -1.0], "total_score": [1.0] * 4}
    )

    ranked = rank_candidate_scores(scores)

    assert ranked["offset_ft"].tolist() == [0.0, -1.0, -2.0, 2.0]
    assert ranked["rank"].tolist() == [2.5, 2.5, 2.5, 2.5]


def test_all_candidates_use_identical_typewell_support_and_quadratic_penalty() -> None:
    typewell = _linear_typewell()
    md = np.arange(0.0, 400.0, 1.0)
    base_tvt = np.linspace(100.0, 200.0, len(md))
    hidden_gr = np.interp(base_tvt, typewell["TVT"], typewell["GR"])

    result = score_global_offsets(
        base_tvt=base_tvt,
        hidden_md=md,
        hidden_gr=hidden_gr,
        typewell_df=typewell,
        affine_slope=1.0,
        affine_intercept=0.0,
    )

    assert result.evidence_sufficient
    assert result.scores["common_points"].nunique() == 1
    zero_penalty = result.scores.loc[result.scores["offset_ft"].eq(0), "penalty"].iloc[0]
    edge_penalty = result.scores.loc[result.scores["offset_ft"].eq(30), "penalty"].iloc[0]
    assert zero_penalty == 0.0
    assert np.isclose(edge_penalty, 0.05)


def test_three_hundred_ft_blocks_receive_equal_weight() -> None:
    assert BLOCK_WIDTH_FT == 300.0
    typewell = pd.DataFrame(
        {"TVT": np.arange(0.0, 501.0), "GR": np.arange(0.0, 501.0)}
    )
    first_md = np.arange(0.0, 300.0, 1.0)
    second_md = np.arange(300.0, 340.0, 1.0)
    md = np.concatenate([first_md, second_md])
    base_tvt = np.full(len(md), 200.0)
    hidden_gr = np.concatenate([np.full(len(first_md), 200.0), np.full(len(second_md), 202.0)])

    result = score_global_offsets(
        base_tvt,
        md,
        hidden_gr,
        typewell,
        affine_slope=1.0,
        affine_intercept=0.0,
    )
    zero = result.scores.loc[result.scores["offset_ft"].eq(0)].iloc[0]

    # 第一块损失为 0，第二块损失为 2；块等权后是 1，而不是逐行加权的 1/3。
    assert np.isclose(zero["data_loss"], 1.0)
    assert zero["valid_blocks"] == 2


def test_hidden_loss_is_normalized_by_prefix_mad_scale_then_capped() -> None:
    typewell = pd.DataFrame(
        {"TVT": np.arange(0.0, 501.0), "GR": np.arange(0.0, 501.0)}
    )
    md = np.arange(0.0, 600.0, 1.0)
    base_tvt = np.full(len(md), 200.0)
    hidden_gr = np.full(len(md), 204.0)

    result = score_global_offsets(
        base_tvt,
        md,
        hidden_gr,
        typewell,
        affine_slope=1.0,
        affine_intercept=0.0,
        robust_scale=2.0,
    )
    zero = result.scores.loc[result.scores["offset_ft"].eq(0)].iloc[0]

    assert np.isclose(zero["data_loss"], 2.0)


def test_insufficient_evidence_falls_back_to_zero() -> None:
    typewell = _linear_typewell()
    md = np.arange(50.0)
    base_tvt = np.full(50, 200.0)
    hidden_gr = np.full(50, 50.0)

    result = score_global_offsets(
        base_tvt,
        md,
        hidden_gr,
        typewell,
        affine_slope=1.0,
        affine_intercept=0.0,
    )

    assert not result.evidence_sufficient
    assert result.selected_offset_ft == 0.0
    ranked = rank_candidate_scores(result.scores)
    assert ranked["rank"].isna().all()


def test_circular_shift_moves_only_finite_gr_and_preserves_nan_positions() -> None:
    gr = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])

    shifted = circular_shift_finite_gr(gr)

    np.testing.assert_array_equal(np.isnan(shifted), np.isnan(gr))
    np.testing.assert_array_equal(shifted[np.isfinite(shifted)], [3.0, 4.0, 1.0, 2.0])


def test_known_synthetic_constant_offset_is_recovered() -> None:
    typewell = _linear_typewell()
    md = np.arange(0.0, 400.0, 1.0)
    base_tvt = np.linspace(100.0, 250.0, len(md))
    true_offset = 7.0
    hidden_gr = np.interp(base_tvt + true_offset, typewell["TVT"], typewell["GR"])

    result = score_global_offsets(
        base_tvt,
        md,
        hidden_gr,
        typewell,
        affine_slope=1.0,
        affine_intercept=0.0,
    )

    assert result.selected_offset_ft == true_offset


def test_legal_scoring_interface_cannot_receive_target_tvt() -> None:
    signature = inspect.signature(score_legal_well)

    assert "target_tvt" not in signature.parameters


def test_prefix_huber_requires_thirty_pairs_and_falls_back_deterministically() -> None:
    visible_rows = 29
    hidden_rows = 100
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(visible_rows + hidden_rows, dtype=np.float64),
            "GR": np.linspace(10.0, 30.0, visible_rows + hidden_rows),
            "TVT_input": np.concatenate(
                [np.linspace(100.0, 128.0, visible_rows), np.full(hidden_rows, np.nan)]
            ),
        }
    )
    typewell = _linear_typewell()

    result = score_legal_well(
        np.full(hidden_rows, 200.0),
        horizontal,
        typewell,
    )

    assert result.prefix_pairs == 29
    assert result.prefix_fallback
    assert result.robust_scale == 1.0
    assert not result.evidence_sufficient
    assert result.selected_offset_ft == 0.0


def test_prefix_scale_is_one_point_four_eight_two_six_times_huber_residual_mad() -> None:
    visible_rows = 60
    hidden_rows = 600
    visible_tvt = np.linspace(100.0, 159.0, visible_rows)
    reference_gr = 0.2 * visible_tvt + 10.0
    visible_gr = 1.4 * reference_gr + 5.0 + np.tile([-2.0, 2.0], 30)
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(visible_rows + hidden_rows, dtype=np.float64),
            "GR": np.concatenate([visible_gr, np.full(hidden_rows, 50.0)]),
            "TVT_input": np.concatenate(
                [visible_tvt, np.full(hidden_rows, np.nan)]
            ),
        }
    )

    result = score_legal_well(
        np.full(hidden_rows, 200.0), horizontal, _linear_typewell()
    )
    fitted = result.affine_slope * reference_gr + result.affine_intercept
    residual = visible_gr - fitted
    expected_scale = max(
        1.0,
        1.4826 * np.median(np.abs(residual - np.median(residual))),
    )

    assert not result.prefix_fallback
    assert np.isclose(result.robust_scale, expected_scale)


def test_strict_outer_contract_rejects_duplicate_or_wrong_row_coverage() -> None:
    registry = pd.DataFrame(
        {"well_id": ["a", "b"], "fold": [0, 0], "hidden_rows": [2, 1]}
    )
    legal = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "fold": [0, 0, 0],
            "row_index": [10, 11, 20],
            "md": [1.0, 2.0, 3.0],
            "pred_tvt": [100.0, 101.0, 102.0],
        }
    )
    validate_legal_outer_contract(legal, ["a", "b"], registry, expected_rows=3)

    duplicated = pd.concat([legal, legal.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="重复"):
        validate_legal_outer_contract(
            duplicated, ["a", "b"], registry, expected_rows=4
        )


def test_nearest_grid_tie_breaks_by_distance_then_absolute_value_then_sign() -> None:
    assert nearest_grid_offset(-0.5) == 0.0
    assert nearest_grid_offset(0.5) == 0.0
    assert nearest_grid_offset(-1.5) == -1.0
    assert nearest_grid_offset(30.8) == 30.0


def test_top_k_requires_both_grid_coverage_and_sufficient_evidence() -> None:
    outside = build_top_k_flags(rank=1.0, grid_covered=False, evidence_sufficient=True)
    no_evidence = build_top_k_flags(rank=1.0, grid_covered=True, evidence_sufficient=False)
    qualified = build_top_k_flags(rank=3.5, grid_covered=True, evidence_sufficient=True)

    assert outside == {"qualified": False, "top1": False, "top3": False, "top5": False}
    assert no_evidence == {"qualified": False, "top1": False, "top3": False, "top5": False}
    assert qualified == {"qualified": True, "top1": False, "top3": False, "top5": True}


def test_path_distribution_reports_micro_macro_median_p90_and_worst() -> None:
    target = np.zeros(3)
    prediction = np.array([3.0, 4.0, 10.0])
    well_ids = np.array(["a", "a", "b"])

    summary = summarize_path_distribution(target, prediction, well_ids)

    assert np.isclose(summary["micro_rmse"], np.sqrt(125.0 / 3.0))
    assert np.isclose(summary["macro_well_rmse"], (np.sqrt(12.5) + 10.0) / 2.0)
    assert np.isclose(summary["median_well_rmse"], (np.sqrt(12.5) + 10.0) / 2.0)
    assert np.isclose(summary["p90_well_rmse"], np.quantile([np.sqrt(12.5), 10.0], 0.9))
    assert summary["worst_well_rmse"] == 10.0
