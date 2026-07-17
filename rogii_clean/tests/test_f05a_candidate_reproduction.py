from __future__ import annotations

import numpy as np
import pandas as pd

from src.f05a_candidate_reproduction import (
    _beam_delta_summaries,
    build_candidate_features,
)
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造一口很小但同时覆盖 PF、Beam 和 NCC 的合成井。"""

    typewell_tvt = np.arange(11000.0, 11080.5, 0.5, dtype=np.float64)
    typewell_gr = (
        70.0
        + 20.0 * np.sin((typewell_tvt - 11000.0) / 4.0)
        + 8.0 * np.cos((typewell_tvt - 11000.0) / 1.7)
    )
    typewell_df = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})

    visible_rows = 80
    hidden_rows = 20
    total_rows = visible_rows + hidden_rows
    md = np.arange(total_rows, dtype=np.float64)
    z = 1000.0 + 0.02 * md + 0.0002 * md**2
    true_tvt = 11025.0 + 0.025 * md + 0.20 * np.sin(md / 15.0)
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)
    horizontal_gr = horizontal_gr + 1.5 * np.sin(md / 3.0)
    horizontal_gr[[84, 91]] = np.nan

    tvt_input = true_tvt.copy()
    tvt_input[visible_rows:] = np.nan
    horizontal_df = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": horizontal_gr,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
        }
    )
    return horizontal_df, typewell_df


def test_candidate_features_have_fixed_shape_order_and_finite_values() -> None:
    horizontal_df, typewell_df = _synthetic_well()

    result = build_candidate_features(horizontal_df, typewell_df, seed=42)

    assert len(DIRECT_CANDIDATE_COLUMNS) == 24
    assert list(result.columns) == ["row_index", *DIRECT_CANDIDATE_COLUMNS]
    assert result["row_index"].tolist() == list(range(80, 100))
    assert result.shape == (20, 25)
    assert np.isfinite(result[DIRECT_CANDIDATE_COLUMNS].to_numpy()).all()


def test_same_seed_reproduces_identical_candidate_features() -> None:
    horizontal_df, typewell_df = _synthetic_well()

    first = build_candidate_features(horizontal_df, typewell_df, seed=17)
    second = build_candidate_features(horizontal_df, typewell_df, seed=17)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_hidden_tvt_mutation_or_removal_cannot_change_features() -> None:
    horizontal_df, typewell_df = _synthetic_well()
    hidden_mask = horizontal_df["TVT_input"].isna()

    mutated = horizontal_df.copy()
    mutated.loc[hidden_mask, "TVT"] = np.linspace(-1_000_000.0, 1_000_000.0, hidden_mask.sum())
    removed = horizontal_df.drop(columns="TVT")

    original_result = build_candidate_features(horizontal_df, typewell_df, seed=29)
    mutated_result = build_candidate_features(mutated, typewell_df, seed=29)
    removed_result = build_candidate_features(removed, typewell_df, seed=29)

    pd.testing.assert_frame_equal(original_result, mutated_result, check_exact=True)
    pd.testing.assert_frame_equal(original_result, removed_result, check_exact=True)


def test_beam_summaries_subtract_anchor_before_reduction() -> None:
    beam_paths = {
        f"beam_{index}": np.array([11734.45 + index * 0.5], dtype=np.float32)
        for index in range(7)
    }
    last_known_tvt = 11747.37

    delta_matrix, mean_delta, std_delta, median_delta = _beam_delta_summaries(
        beam_paths,
        last_known_tvt,
    )

    expected_deltas = np.stack(
        [path - np.float32(last_known_tvt) for path in beam_paths.values()],
        axis=1,
    )
    np.testing.assert_array_equal(delta_matrix, expected_deltas)
    np.testing.assert_array_equal(mean_delta, expected_deltas.mean(axis=1))
    np.testing.assert_array_equal(std_delta, expected_deltas.std(axis=1))
    np.testing.assert_array_equal(median_delta, np.median(expected_deltas, axis=1))
