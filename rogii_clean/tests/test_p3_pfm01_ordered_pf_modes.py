"""P3-PFM01 的固定语义、物理隔离和汇总公式测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_p3_pfm01_ordered_pf_modes as pfm01_runner
from src.p3_pfm01_ordered_pf_modes import (
    build_seed_descriptors,
    compute_direction_metrics,
    compute_p2_position,
    compute_three_vs_five_oracle,
    fixed_mode_reweight,
    order_mode_records,
    scale8_weights,
    summarize_ordered_modes,
)
from scripts.run_p3_pfm01_ordered_pf_modes import (
    assert_legal_columns_safe,
    build_deranged_indices,
    guarded_target_read,
    resolve_run_artifact_dir,
    validate_outer_contract,
    validate_row_alignment,
)


def test_seed_descriptors_are_only_full_mean_and_endpoint_without_zscore() -> None:
    paths = np.asarray([[0.0, 2.0, 4.0], [10.0, 10.0, 13.0]])
    observed = build_seed_descriptors(paths)
    expected = np.asarray([[2.0, 4.0], [11.0, 13.0]])
    np.testing.assert_allclose(observed, expected)


def test_scale8_weights_use_frozen_temperature() -> None:
    weights = scale8_weights(np.asarray([0.0, 8.0 * np.log(3.0)]))
    np.testing.assert_allclose(weights, [0.25, 0.75])


def test_mode_center_is_likelihood_weighted_not_plain_mean() -> None:
    paths = np.asarray(
        [
            [0.0, 0.0],
            [4.0, 4.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [20.0, 20.0],
            [21.0, 21.0],
        ]
    )
    likelihoods = np.asarray([0.0, 8.0 * np.log(3.0), 0.0, 0.0, 0.0, 0.0])
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    modes = summarize_ordered_modes(paths, likelihoods, np.arange(6), labels)
    np.testing.assert_allclose(modes["low"]["center_path"], [3.0, 3.0])


def test_mode_names_follow_mean_then_endpoint_then_minimum_seed_id() -> None:
    records = [
        {"mean_delta": 1.0, "endpoint_delta": 2.0, "minimum_seed_id": 4},
        {"mean_delta": 1.0, "endpoint_delta": 0.0, "minimum_seed_id": 9},
        {"mean_delta": 1.0, "endpoint_delta": 0.0, "minimum_seed_id": 3},
    ]
    ordered = order_mode_records(records)
    assert [record["minimum_seed_id"] for record in ordered] == [3, 9, 4]


def test_rotate64_reweights_exactly_the_same_memberships() -> None:
    paths = np.arange(24, dtype=np.float64).reshape(6, 4)
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    modes = summarize_ordered_modes(paths, np.arange(6, dtype=float), np.arange(6), labels)
    memberships = {
        name: np.asarray(mode["member_seed_ids"], dtype=np.int64)
        for name, mode in modes.items()
    }
    rotated = fixed_mode_reweight(
        paths,
        np.roll(np.arange(6, dtype=float), 3),
        np.arange(6),
        memberships,
    )
    for name in ("low", "middle", "high"):
        np.testing.assert_array_equal(rotated[name]["member_seed_ids"], memberships[name])


def test_degenerate_mode_envelope_returns_half() -> None:
    raw, clipped, degenerate = compute_p2_position(
        p2_delta=np.asarray([2.0, 2.0]),
        low_path=np.asarray([1.0, 1.0]),
        high_path=np.asarray([1.0 + 1e-8, 1.0 + 1e-8]),
    )
    assert (raw, clipped, degenerate) == (0.5, 0.5, True)


def test_p2_position_preserves_raw_outside_value_and_clips_feature() -> None:
    raw, clipped, degenerate = compute_p2_position(
        p2_delta=np.asarray([3.0, 3.0]),
        low_path=np.asarray([0.0, 0.0]),
        high_path=np.asarray([2.0, 2.0]),
    )
    assert raw == pytest.approx(1.5)
    assert clipped == pytest.approx(1.0)
    assert not degenerate


def test_outer_contract_rejects_shadow_and_wrong_formal_count() -> None:
    registry = pd.DataFrame(
        {"well_id": ["a", "b"], "fold": [0, 0], "hidden_rows": [2, 1]}
    )
    rows = pd.DataFrame(
        {
            "well_id": ["a", "a", "b"],
            "fold": [0, 0, 0],
            "row_index": [4, 5, 9],
            "md": [1.0, 2.0, 3.0],
            "pred_tvt": [7.0, 8.0, 9.0],
        }
    )
    with pytest.raises(ValueError, match="影子"):
        validate_outer_contract(rows, registry, {"b"}, formal=False)
    with pytest.raises(ValueError, match="131"):
        validate_outer_contract(rows, registry, set(), formal=True)


def test_strict_outer_loader_rejects_sha_before_opening_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = pd.DataFrame({"well_id": ["a"], "fold": [0], "hidden_rows": [1]})
    dataset_was_opened = False

    def forbidden_dataset(*args: object, **kwargs: object) -> object:
        nonlocal dataset_was_opened
        dataset_was_opened = True
        raise AssertionError("SHA 失败后不应打开 parquet")

    monkeypatch.setattr(pfm01_runner, "file_sha256", lambda path: "bad-sha")
    monkeypatch.setattr(pfm01_runner.arrow_dataset, "dataset", forbidden_dataset)
    with pytest.raises(ValueError, match="SHA"):
        pfm01_runner.load_strict_outer_legal(selected)
    assert not dataset_was_opened


def test_row_alignment_requires_identical_ordered_keys() -> None:
    expected = np.asarray([3, 4, 5], dtype=np.int64)
    validate_row_alignment("ok", expected, np.asarray([3, 4, 5]))
    with pytest.raises(ValueError, match="row_index"):
        validate_row_alignment("bad", expected, np.asarray([3, 5, 4]))


@pytest.mark.parametrize(
    "column",
    ["target_tvt", "true_value", "mean_residual", "abs_error", "rmse", "oracle_rank", "best_path"],
)
def test_legal_columns_reject_all_forbidden_target_terms(column: str) -> None:
    with pytest.raises(ValueError, match="禁止"):
        assert_legal_columns_safe(["well_id", column])


def test_incomplete_legal_cache_blocks_target_reader(tmp_path: Path) -> None:
    selected = pd.DataFrame(
        {"well_id": ["a"], "fold": [0], "hidden_rows": [2]}
    )
    target_was_read = False

    def reader() -> pd.DataFrame:
        nonlocal target_was_read
        target_was_read = True
        return pd.DataFrame()

    with pytest.raises(RuntimeError, match="合法缓存"):
        guarded_target_read(selected, tmp_path, reader)
    assert not target_was_read


def test_cross_well_derangement_has_no_fixed_points_and_is_repeatable() -> None:
    first = build_deranged_indices(131, seed=20260719)
    second = build_deranged_indices(131, seed=20260719)
    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(131))
    assert sorted(first.tolist()) == list(range(131))


def test_oracle_uses_pooled_sse_not_mean_of_well_rmse() -> None:
    wells = pd.DataFrame(
        {
            "hidden_rows": [1, 3],
            "low_sse": [1.0, 48.0],
            "middle_sse": [4.0, 27.0],
            "high_sse": [9.0, 12.0],
            "mean_sse": [16.0, 3.0],
            "scale3_sse": [25.0, 6.0],
            "scale5_sse": [36.0, 9.0],
            "scale8_sse": [49.0, 12.0],
            "scale12_sse": [64.0, 15.0],
        }
    )
    metrics = compute_three_vs_five_oracle(wells)
    # 三模式逐井选择 low(1) 和 high(12)，所以 pooled RMSE=sqrt(13/4)。
    assert metrics["three_mode_oracle_pooled_rmse"] == pytest.approx(np.sqrt(13.0 / 4.0))
    # 五温度逐井都选择 mean，所以 pooled RMSE=sqrt(19/4)。
    assert metrics["five_temperature_oracle_pooled_rmse"] == pytest.approx(np.sqrt(19.0 / 4.0))


def test_direction_auc_keeps_preregistered_sign_and_never_flips() -> None:
    residual = np.asarray([-4.0, -3.0, 3.0, 4.0])
    anti_direction = np.asarray([2.0, 1.0, -1.0, -2.0])
    metrics = compute_direction_metrics(anti_direction, residual, minimum_abs_residual=2.0)
    assert metrics["roc_auc"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.0)


def test_smoke_artifacts_are_physically_separate_from_formal_directory(tmp_path: Path) -> None:
    assert resolve_run_artifact_dir(tmp_path, None) == tmp_path
    assert resolve_run_artifact_dir(tmp_path, 3) == tmp_path / "smoke_3"
