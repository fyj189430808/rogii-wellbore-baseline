from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# 从仓库父目录运行时，强制导入本项目的 src，而不是环境中的同名包。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_pfmd02_representation_oracle import (
    build_ordered_representatives,
    choose_weighted_medoid,
    compute_segment_sse,
    pointwise_envelope_sse,
    pooled_micro_rmse,
    resolve_run_artifact_dir,
    segment_ids_from_md,
    solve_b25_simplex,
    solve_simplex_least_squares,
    solve_switching_dp,
    validate_development_selection,
    validate_well_alignment,
)


def test_simplex_solver_finds_known_interior_solution() -> None:
    candidates = np.array([[0.0, 1.0], [1.0, 0.0]])
    target = np.array([0.25, 0.75])

    result = solve_simplex_least_squares(candidates, target)

    np.testing.assert_allclose(result.weights, [0.75, 0.25], atol=1e-12)
    assert result.sse == pytest.approx(0.0, abs=1e-14)
    assert result.active_count == 2


def test_b25_keeps_p2_weight_at_or_above_three_quarters() -> None:
    candidates = np.column_stack(
        [np.zeros(3), np.full(3, 10.0), np.full(3, 20.0), np.full(3, 30.0)]
    )
    target = np.full(3, 30.0)

    result = solve_b25_simplex(candidates, target)

    assert result.weights[0] >= 0.75 - 1e-12
    assert result.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(candidates @ result.weights, np.full(3, 7.5))


def test_md_segmentation_uses_first_hidden_md_and_nonoverlapping_boundaries() -> None:
    md = np.array([100.0, 200.0, 349.999, 350.0, 599.9, 600.0])

    segment_ids = segment_ids_from_md(md, 250.0)

    np.testing.assert_array_equal(segment_ids, [0, 0, 0, 1, 1, 2])


def test_dp_applies_switch_penalty_and_backtracks_selected_states() -> None:
    segment_sse = np.array([[0.0, 4.0], [4.0, 0.0], [4.0, 0.0]])

    result = solve_switching_dp(segment_sse, switch_penalty=3.0)

    np.testing.assert_array_equal(result.states, [0, 1, 1])
    assert result.raw_sse == pytest.approx(0.0)
    assert result.penalized_objective == pytest.approx(3.0)
    assert result.switch_count == 1


def test_dp_ties_prefer_staying_then_lower_state_index() -> None:
    result = solve_switching_dp(np.zeros((3, 4)), switch_penalty=0.0)

    np.testing.assert_array_equal(result.states, [0, 0, 0])


def test_medoid_is_always_one_of_the_fixed_member_seeds() -> None:
    paths = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [100.0, 100.0]])
    seed_ids = np.array([10, 11, 12, 13])

    result = choose_weighted_medoid(
        paths,
        final_ll=np.array([-2.0, 0.0, -1.0, 1000.0]),
        seed_ids=seed_ids,
        member_indices=np.array([0, 1, 2]),
    )

    assert result.seed_id in {10, 11, 12}
    assert result.index in {0, 1, 2}


def test_pointwise_envelope_sse_selects_best_candidate_on_each_row() -> None:
    candidates = np.array([[0.0, 10.0], [10.0, 0.0], [4.0, 4.0]])
    target = np.array([0.0, 0.0, 5.0])

    assert pointwise_envelope_sse(candidates, target) == pytest.approx(1.0)


def test_micro_rmse_pools_sse_and_rows_instead_of_averaging_wells() -> None:
    assert pooled_micro_rmse([4.0, 9.0], [1, 3]) == pytest.approx(np.sqrt(13.0 / 4.0))


def test_natural_key_misalignment_is_rejected() -> None:
    p2 = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "fold": [2, 2],
            "row_index": [5, 6],
            "md": [100.0, 100.5],
        }
    )
    mode = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "fold": [2, 2],
            "row_index": [5, 7],
        }
    )
    seed = {"row_index": np.array([5, 6]), "hidden_md": np.array([100.0, 100.5], dtype=np.float32)}

    with pytest.raises(ValueError, match="自然键"):
        validate_well_alignment(p2, mode, seed, well_id="w", fold=2, expected_rows=2)


def test_float32_equivalent_md_is_accepted_but_real_md_mismatch_is_rejected() -> None:
    p2 = pd.DataFrame(
        {"well_id": ["w"], "fold": [1], "row_index": [8], "md": [123.456789]}
    )
    mode = pd.DataFrame({"well_id": ["w"], "fold": [1], "row_index": [8]})
    seed = {"row_index": np.array([8]), "hidden_md": np.array([np.float32(123.456789)])}
    validate_well_alignment(p2, mode, seed, well_id="w", fold=1, expected_rows=1)

    seed["hidden_md"] = np.array([124.0], dtype=np.float32)
    with pytest.raises(ValueError, match="MD"):
        validate_well_alignment(p2, mode, seed, well_id="w", fold=1, expected_rows=1)


def test_shadow_wells_are_rejected_from_development_selection() -> None:
    selected = pd.DataFrame({"well_id": ["dev", "shadow"], "fold": [0, 1], "hidden_rows": [2, 3]})

    with pytest.raises(ValueError, match="影子"):
        validate_development_selection(selected, {"shadow"})


def test_smoke_and_formal_directories_are_physically_separate() -> None:
    base = Path("artifacts") / "formal"
    formal = resolve_run_artifact_dir(base, None)
    smoke = resolve_run_artifact_dir(base, 3)

    assert formal == base
    assert smoke == base / "smoke_3"
    assert smoke != formal
    with pytest.raises(ValueError, match="1、2、3"):
        resolve_run_artifact_dir(base, 4)


def test_extreme_cross_mode_likelihoods_keep_centers_finite() -> None:
    paths = np.vstack(
        [
            np.tile([-20.0, -19.0, -18.0], (3, 1)),
            np.tile([0.0, 1.0, 2.0], (3, 1)),
            np.tile([20.0, 21.0, 22.0], (3, 1)),
        ]
    )
    paths += np.arange(9)[:, None] * 0.01
    final_ll = np.array([-1e300, -1e300 - 1e6, -1e300 - 2e6, 1e300, 1e300 - 1e6, 1e300 - 2e6, -5e299, -5e299 - 1e6, -5e299 - 2e6])

    representatives = build_ordered_representatives(paths, final_ll, np.arange(9))

    assert np.isfinite(representatives.center_paths).all()
    assert np.isfinite(representatives.rowmedian_paths).all()
    assert set(representatives.medoid_seed_ids).issubset(set(range(9)))


def test_segment_sse_returns_one_row_per_segment_and_candidate() -> None:
    candidates = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    target = np.array([0.0, 1.0, 4.0])

    result = compute_segment_sse(candidates, target, np.array([0, 0, 1]))

    np.testing.assert_allclose(result, [[0.0, 2.0], [4.0, 1.0]])
