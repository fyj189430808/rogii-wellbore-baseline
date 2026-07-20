from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_mdp01_dynamic_mode_path import (  # noqa: E402
    CANDIDATE_NAMES,
    DEFAULT_PARAMETERS,
    build_dynamic_mode_paths,
    circular_shift_hidden_gr,
    fit_prefix_calibration,
    interpolate_state_path,
    permute_block_costs,
    solve_dynamic_states,
)


def _prefix_frames(hidden_gr: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible_md = np.arange(0.0, 100.0)
    visible_tvt = 1000.0 + 0.1 * visible_md
    typewell_tvt = np.arange(990.0, 1061.0, 0.5)
    typewell_gr = 20.0 + 0.5 * typewell_tvt
    visible_gr = 2.0 * np.interp(visible_tvt, typewell_tvt, typewell_gr) + 5.0
    hidden_md = 100.0 + np.arange(len(hidden_gr), dtype=np.float64)
    horizontal = pd.DataFrame(
        {
            "MD": np.concatenate([visible_md, hidden_md]),
            "Z": np.zeros(100 + len(hidden_gr)),
            "GR": np.concatenate([visible_gr, hidden_gr]),
            "TVT_input": np.concatenate(
                [visible_tvt, np.full(len(hidden_gr), np.nan)]
            ),
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell


def test_prefix_calibration_uses_only_visible_original_gr() -> None:
    horizontal, typewell = _prefix_frames(np.full(20, np.nan))
    calibration = fit_prefix_calibration(horizontal, typewell)

    assert calibration.visible_pair_count == 100
    np.testing.assert_allclose(calibration.slope, 2.0, atol=1e-10)
    np.testing.assert_allclose(calibration.intercept, 5.0, atol=1e-8)

    changed = horizontal.copy()
    changed.loc[changed["TVT_input"].isna(), "GR"] = 1e9
    assert fit_prefix_calibration(changed, typewell) == calibration


def test_circular_shift_moves_values_and_nan_mask_together() -> None:
    gr = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
    shifted, shift_rows = circular_shift_hidden_gr(gr)

    assert shift_rows == 2
    np.testing.assert_allclose(shifted, np.roll(gr, 2), equal_nan=True)


def test_cost_permutation_is_deterministic_and_non_identity() -> None:
    costs = np.arange(20, dtype=np.float64).reshape(5, 4)
    first, first_order = permute_block_costs(costs, "well-a")
    second, second_order = permute_block_costs(costs, "well-a")

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_order, second_order)
    assert not np.array_equal(first_order, np.arange(5))
    np.testing.assert_array_equal(first, costs[first_order])


def test_non_p2_state_requires_three_consecutive_blocks() -> None:
    centers = np.arange(5, dtype=np.float64) * 125.0
    center_z = np.zeros(5)
    paths = np.zeros((5, 4), dtype=np.float64)
    two_block = np.full((5, 4), 100.0)
    two_block[:, 0] = 0.0
    two_block[1:3, 1] = -10.0

    result = solve_dynamic_states(two_block, centers, paths, center_z)
    np.testing.assert_array_equal(result.states, np.zeros(5, dtype=np.int64))

    three_block = two_block.copy()
    three_block[3, 1] = -10.0
    result = solve_dynamic_states(three_block, centers, paths, center_z)
    np.testing.assert_array_equal(result.states, np.array([0, 1, 1, 1, 0]))


def test_jump_and_second_order_terms_change_the_optimal_path() -> None:
    centers = np.arange(4, dtype=np.float64) * 125.0
    center_z = np.zeros(4)
    paths = np.zeros((4, 4), dtype=np.float64)
    paths[:, 1] = np.array([0.0, 30.0, 0.0, 30.0])
    local = np.zeros((4, 4), dtype=np.float64)
    local[:, 0] = 0.5
    local[:, 1] = 0.0
    local[:, 2:] = 100.0

    no_geometry = replace(
        DEFAULT_PARAMETERS,
        mode_prior=0.0,
        switch_penalty=0.0,
        jump_weight=0.0,
        second_order_weight=0.0,
    )
    geometric = replace(
        no_geometry,
        jump_weight=1.0,
        second_order_weight=1.0,
    )
    assert np.all(
        solve_dynamic_states(local, centers, paths, center_z, no_geometry).states
        == 1
    )
    assert np.all(
        solve_dynamic_states(local, centers, paths, center_z, geometric).states
        == 0
    )


def test_first_mvp_marks_global_margin_as_deferred() -> None:
    centers = np.arange(4, dtype=np.float64) * 125.0
    paths = np.zeros((4, 4), dtype=np.float64)
    local = np.zeros((4, 4), dtype=np.float64)

    result = solve_dynamic_states(local, centers, paths, np.zeros(4))

    np.testing.assert_array_equal(result.states, np.zeros(4, dtype=np.int64))
    assert np.isnan(result.margin).all()
    assert np.isfinite(result.objective)


def test_interpolation_reproduces_candidate_and_crossfades_one_hot_states() -> None:
    row_md = np.array([0.0, 62.5, 125.0])
    centers = np.array([0.0, 125.0])
    states = np.array([0, 1], dtype=np.int64)
    candidates = np.column_stack(
        [
            np.zeros(3),
            np.full(3, 10.0),
            np.full(3, 20.0),
            np.full(3, 30.0),
        ]
    )

    path, row_states, weights = interpolate_state_path(
        row_md, centers, states, candidates
    )

    np.testing.assert_allclose(path, [0.0, 5.0, 10.0])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_array_equal(row_states, [0, 0, 1])


def test_all_missing_hidden_gr_returns_p2_and_fixed_safe_paths() -> None:
    hidden_rows = 501
    horizontal, typewell = _prefix_frames(np.full(hidden_rows, np.nan))
    hidden_md = horizontal.loc[horizontal["TVT_input"].isna(), "MD"].to_numpy()
    p2 = 1010.0 + 0.01 * (hidden_md - hidden_md[0])
    candidates = np.column_stack([p2, p2 + 10.0, p2 + 20.0, p2 + 30.0])

    result = build_dynamic_mode_paths(
        horizontal=horizontal,
        typewell=typewell,
        row_index=np.flatnonzero(horizontal["TVT_input"].isna().to_numpy()),
        candidate_tvt=candidates,
        last_visible_tvt=1009.9,
        well_id="well-a",
    )

    np.testing.assert_allclose(result.row_output["mdp_dynamic_tvt"], p2)
    np.testing.assert_array_equal(
        result.row_output["mdp_selected_state"],
        np.full(hidden_rows, CANDIDATE_NAMES.index("P2")),
    )
    np.testing.assert_allclose(result.row_output["mdp_safe_10_tvt"], p2)
    np.testing.assert_allclose(result.row_output["mdp_safe_25_tvt"], p2)
    expected_control_columns = {
        "mdp_gr_shift_dynamic_tvt",
        "mdp_gr_shift_dynamic_delta",
        "mdp_gr_shift_selected_state",
        "mdp_gr_shift_safe_10_delta",
        "mdp_gr_shift_safe_25_delta",
        "mdp_cost_permutation_dynamic_tvt",
        "mdp_cost_permutation_dynamic_delta",
        "mdp_cost_permutation_selected_state",
        "mdp_cost_permutation_safe_10_delta",
        "mdp_cost_permutation_safe_25_delta",
    }
    assert expected_control_columns.issubset(result.row_output.columns)
    assert result.audit["hidden_target_read"] is False
