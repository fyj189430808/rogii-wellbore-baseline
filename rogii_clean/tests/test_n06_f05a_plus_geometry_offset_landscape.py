from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f03b_geometry_landscape_features import (
    F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
)
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CONFIG_PATH = (
    CLEAN_ROOT / "configs" / "n06_f05a_plus_geometry_offset_landscape_v1.json"
)
RUNNER_PATH = (
    CLEAN_ROOT
    / "scripts"
    / "run_n06_f05a_plus_geometry_offset_landscape_cv.py"
)


def _load_runner():
    assert RUNNER_PATH.is_file(), "N06 runner 尚未实现"
    spec = importlib.util.spec_from_file_location("run_n06_combo", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f05a_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["well_a", "well_a", "well_b"],
            "row_index": [10, 11, 20],
            "target_delta": [1.0, 2.0, 3.0],
        }
    )


def _landscape_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "row_index": [20, 11, 10],
            "target_tvt_that_must_not_join": [9999.0, 9999.0, 9999.0],
        }
    )
    for column_index, column in enumerate(
        F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS
    ):
        rows[column] = [
            float(300 + column_index),
            float(200 + column_index),
            float(100 + column_index),
        ]
    return rows


def test_config_freezes_f05a_and_only_adds_twelve_landscape_columns() -> None:
    assert CONFIG_PATH.is_file(), "N06 配置尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    ]

    assert config["experiment_id"] == "N06_f05a_plus_geometry_offset_landscape_v1"
    assert config["baseline_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_count"] == 48
    assert config["feature_columns"] == expected_features
    assert config["candidate_feature_columns"] == DIRECT_CANDIDATE_COLUMNS
    assert config["new_feature_columns"] == F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS
    assert config["landscape_feature_cache_sha256"] == (
        "3cda07536fd3e227f237345f3beafeb44df7c19f063941097726b447a38bbcc7"
    )
    assert config["landscape_feature_cache_metadata_sha256"] == (
        "0b35d232b42c5a36132ef8b781f400ca1528b8fde6466b9067433be8f84ea529"
    )
    assert config["candidate_cache_rebuild_allowed"] is False
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"

    runner = _load_runner()
    assert runner._validate_config(config) == expected_features
    assert runner.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert runner.parse_args(["--mode", "fold4"]).mode == "fold4"


def test_strict_key_join_restores_f05a_order_and_selects_only_twelve_columns() -> None:
    runner = _load_runner()
    result = runner.add_landscape_features(_f05a_rows(), _landscape_rows())

    assert result[["well_id", "row_index"]].to_dict("records") == [
        {"well_id": "well_a", "row_index": 10},
        {"well_id": "well_a", "row_index": 11},
        {"well_id": "well_b", "row_index": 20},
    ]
    first = F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS[0]
    assert result[first].tolist() == [100.0, 200.0, 300.0]
    assert "target_tvt_that_must_not_join" not in result.columns


@pytest.mark.parametrize("failure_kind", ["duplicate", "missing", "extra"])
def test_strict_key_join_rejects_non_bijective_cache(failure_kind: str) -> None:
    runner = _load_runner()
    landscape_rows = _landscape_rows()
    if failure_kind == "duplicate":
        landscape_rows = pd.concat(
            [landscape_rows, landscape_rows.iloc[[0]]], ignore_index=True
        )
    elif failure_kind == "missing":
        landscape_rows = landscape_rows.iloc[:-1].copy()
    else:
        extra = landscape_rows.iloc[[0]].copy()
        extra["well_id"] = "well_extra"
        extra["row_index"] = 999
        landscape_rows = pd.concat([landscape_rows, extra], ignore_index=True)

    with pytest.raises(ValueError, match="重复键|一一对应"):
        runner.add_landscape_features(_f05a_rows(), landscape_rows)


def test_landscape_validation_allows_registered_nan_but_rejects_inf() -> None:
    runner = _load_runner()
    rows = _landscape_rows()
    rows.loc[0, "f03b_best_ncc"] = np.nan
    runner._validate_landscape_values(rows)

    rows.loc[0, "f03b_best_ncc"] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        runner._validate_landscape_values(rows)


def test_experiment_fingerprint_tracks_landscape_cache_and_metadata() -> None:
    runner = _load_runner()
    common = {
        "config": {"experiment_id": "n06"},
        "model_params": {"n_estimators": 1734, "random_state": 29},
        "registry_hash": "registry-sha",
        "runner_hash": "runner-sha",
        "candidate_cache_sha256": "candidate-cache",
        "candidate_metadata_sha256": "candidate-meta",
        "landscape_cache_sha256": "landscape-cache",
        "landscape_metadata_sha256": "landscape-meta",
    }
    first = runner.build_experiment_fingerprint(**common)
    for key in ("landscape_cache_sha256", "landscape_metadata_sha256"):
        changed = dict(common)
        changed[key] = f"{changed[key]}-changed"
        assert first != runner.build_experiment_fingerprint(**changed)
