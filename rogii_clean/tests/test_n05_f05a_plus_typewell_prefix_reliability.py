from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f03a_prefix_reliability_features import F03A_FEATURE_COLUMNS
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CONFIG_PATH = (
    CLEAN_ROOT / "configs" / "n05_f05a_plus_typewell_prefix_reliability_v1.json"
)
RUNNER_PATH = (
    CLEAN_ROOT / "scripts" / "run_n05_f05a_plus_typewell_prefix_reliability_cv.py"
)


def _load_runner():
    assert RUNNER_PATH.is_file(), "N05 runner 尚未实现"
    spec = importlib.util.spec_from_file_location("run_n05_combo", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f05a_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["well_a", "well_a", "well_b"],
            "fold": [1, 1, 3],
            "row_index": [10, 11, 20],
            "target_delta": [1.0, 2.0, 3.0],
        }
    )


def _well_features() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a"],
            "fold": [3, 1],
            "hidden_tvt_that_must_not_join": [9999.0, 9999.0],
        }
    )
    for column_index, column in enumerate(F03A_FEATURE_COLUMNS):
        rows[column] = [
            float(200 + column_index),
            float(100 + column_index),
        ]
    return rows


def test_config_freezes_f05a_and_only_adds_eighteen_f03a_features() -> None:
    assert CONFIG_PATH.is_file(), "N05 配置尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *F03A_FEATURE_COLUMNS,
    ]

    assert config["experiment_id"] == "N05_f05a_plus_typewell_prefix_reliability_v1"
    assert config["baseline_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_count"] == 54
    assert config["feature_columns"] == expected_features
    assert config["candidate_feature_columns"] == DIRECT_CANDIDATE_COLUMNS
    assert config["new_feature_columns"] == F03A_FEATURE_COLUMNS
    assert config["offset_scores_sha256"] == (
        "13790c39a28a74f62ec149bef5efb011d46c742431b50cb36a6fea52a2c9bc34"
    )
    assert config["per_well_margins_sha256"] == (
        "bd407c5cb92c074429c0716c81707a7c4b127bf7adfa8f5972f3a89a6b5add47"
    )
    assert config["f03a_feature_code_sha256"] == (
        "c31509be89d3caf91d3a3e798ac34941e9bb7fe2701f1685b0e6485a0103a9e6"
    )
    assert config["cache_rebuild_allowed"] is False
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"

    runner = _load_runner()
    assert runner._validate_config(config) == expected_features
    assert runner.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert runner.parse_args(["--mode", "fold4"]).mode == "fold4"


def test_well_join_repeats_features_and_preserves_hidden_row_order() -> None:
    runner = _load_runner()
    result = runner.add_prefix_reliability_features(_f05a_rows(), _well_features())

    assert result[["well_id", "row_index"]].to_dict("records") == [
        {"well_id": "well_a", "row_index": 10},
        {"well_id": "well_a", "row_index": 11},
        {"well_id": "well_b", "row_index": 20},
    ]
    assert result[F03A_FEATURE_COLUMNS[0]].tolist() == [100.0, 100.0, 200.0]
    assert "hidden_tvt_that_must_not_join" not in result.columns


@pytest.mark.parametrize("failure_kind", ["duplicate", "missing", "extra", "fold"])
def test_well_join_rejects_bad_lineage(failure_kind: str) -> None:
    runner = _load_runner()
    well_features = _well_features()
    if failure_kind == "duplicate":
        well_features = pd.concat(
            [well_features, well_features.iloc[[0]]], ignore_index=True
        )
    elif failure_kind == "missing":
        well_features = well_features.iloc[:-1].copy()
    elif failure_kind == "extra":
        extra = well_features.iloc[[0]].copy()
        extra["well_id"] = "well_extra"
        well_features = pd.concat([well_features, extra], ignore_index=True)
    else:
        well_features.loc[well_features["well_id"] == "well_a", "fold"] = 4

    with pytest.raises(ValueError, match="井级特征|fold"):
        runner.add_prefix_reliability_features(_f05a_rows(), well_features)


def test_experiment_fingerprint_tracks_both_d0_inputs_and_feature_code() -> None:
    runner = _load_runner()
    common = {
        "config": {"experiment_id": "n05"},
        "model_params": {"n_estimators": 1734, "random_state": 29},
        "registry_hash": "registry-sha",
        "runner_hash": "runner-sha",
        "candidate_cache_sha256": "candidate-cache",
        "candidate_metadata_sha256": "candidate-meta",
        "offset_scores_sha256": "offset-scores",
        "per_well_margins_sha256": "margins",
        "feature_code_sha256": "feature-code",
    }

    first = runner.build_experiment_fingerprint(**common)
    for key in (
        "offset_scores_sha256",
        "per_well_margins_sha256",
        "feature_code_sha256",
    ):
        changed = dict(common)
        changed[key] = f"{changed[key]}-changed"
        assert first != runner.build_experiment_fingerprint(**changed)
