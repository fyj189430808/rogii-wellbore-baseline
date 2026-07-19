from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_r01b_mean_residual_ridge import (  # noqa: E402
    SOURCE_PREDICTION_SHA256,
    OUTER_FEATURE_COLUMNS,
    _assert_one_row_per_expected_well,
    _assert_prediction_keys,
    apply_mean_residual,
    fit_mean_residual_target,
    read_outer_feature_predictions,
    read_outer_scoring_predictions,
    shuffle_mean_residual_targets,
    verify_prediction_sources,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "fold": [1, 1, 2, 2],
            "row_index": [0, 1, 2, 3],
            "pred_tvt": [10.0, 12.0, 20.0, 25.0],
            "target_tvt": [11.0, 15.0, 16.0, 23.0],
        }
    )


def test_fit_mean_residual_target_is_row_residual_mean_per_well() -> None:
    result = fit_mean_residual_target(_predictions())

    expected = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [1, 2],
            "mean_residual": [2.0, -3.0],
        }
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_apply_mean_residual_adds_one_constant_to_every_row_of_a_well() -> None:
    predictions = _predictions()
    corrections = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [1, 2],
            "pred_mean_residual": [1.25, -0.5],
        }
    )

    result = apply_mean_residual(predictions, corrections)

    np.testing.assert_allclose(
        result["corrected_tvt"] - result["pred_tvt"],
        [1.25, 1.25, -0.5, -0.5],
    )


def test_shuffle_mean_residual_targets_preserves_wells_but_changes_assignment() -> None:
    targets = fit_mean_residual_target(_predictions())

    shuffled = shuffle_mean_residual_targets(targets, seed=20260719)

    assert shuffled[["well_id", "fold"]].equals(targets[["well_id", "fold"]])
    assert sorted(shuffled["mean_residual"]) == sorted(targets["mean_residual"])
    assert not shuffled["mean_residual"].equals(targets["mean_residual"])


def test_prediction_source_hashes_are_frozen() -> None:
    assert SOURCE_PREDICTION_SHA256 == {
        "outer0": "cb0d1b778a2193507204fd6b104efbdb4fd44f20a4a96d3f9eff139759df9bad",
        "inner1": "3e87cf380084fab3eb8b70487ca58f1b82158f68848ed68be842fda5a24d5a8c",
        "inner2": "0937be78c54a0357034ff39fc90b887e832bdf3003af95f7cb04141130dfecab",
        "inner3": "3032458872d966ff166782a621e7afeb5799be12623edf006be42433123d6942",
        "inner4": "1aec1c40e3e0e510bd6deda9d7ea0e77cf179ee1146e4850e08d6d108e7447ff",
    }


def test_verify_prediction_sources_rejects_sha256_mismatch() -> None:
    with tempfile.TemporaryDirectory(dir=CLEAN_ROOT) as temporary_dir:
        source_dir = Path(temporary_dir)
        (source_dir / "prediction.parquet").write_bytes(b"not-the-frozen-prediction")

        with pytest.raises(RuntimeError, match="outer0.*SHA256"):
            verify_prediction_sources(
                source_dir,
                source_paths={"outer0": Path("prediction.parquet")},
                expected_hashes={"outer0": "0" * 64},
            )


def test_assert_prediction_keys_rejects_fold_different_from_registry() -> None:
    predictions = _predictions().iloc[[0, 2]].copy()
    predictions.loc[predictions["well_id"].eq("b"), "fold"] = 1
    table = predictions[["well_id", "row_index"]].copy()
    registry = pd.DataFrame({"well_id": ["a", "b"], "fold": [1, 2]})

    with pytest.raises(RuntimeError, match="registry fold"):
        _assert_prediction_keys(
            "synthetic", predictions, table, registry, {"a", "b"}
        )


def test_assert_one_row_per_expected_well_rejects_silent_well_loss() -> None:
    dropped = pd.DataFrame({"well_id": ["a"], "fold": [1]})

    with pytest.raises(RuntimeError, match="train features.*2.*1"):
        _assert_one_row_per_expected_well(
            "train features", dropped, {"a", "b"}
        )


def test_outer_target_is_loaded_only_by_second_scoring_read(monkeypatch) -> None:
    feature_rows = pd.DataFrame(
        {
            "well_id": ["a"],
            "fold": [0],
            "row_index": [7],
            "md": [100.0],
            "pred_tvt": [9.0],
        }
    )
    scoring_rows = feature_rows.assign(target_tvt=[10.0])
    calls: list[object] = []

    def fake_read_parquet(path: Path, columns=None):
        calls.append(columns)
        return feature_rows.copy() if columns is not None else scoring_rows.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    feature_stage = read_outer_feature_predictions(Path("outer.parquet"))
    assert "target_tvt" not in feature_stage
    scoring_stage = read_outer_scoring_predictions(
        Path("outer.parquet"), feature_stage
    )

    assert calls == [OUTER_FEATURE_COLUMNS, None]
    assert "target_tvt" in scoring_stage
