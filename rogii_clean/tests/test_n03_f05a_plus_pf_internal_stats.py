from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.f05b_internal_stats import F05B_FEATURE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CONFIG_PATH = CLEAN_ROOT / "configs" / "n03_f05a_plus_pf_internal_stats_v1.json"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_n03_f05a_plus_pf_internal_stats_cv.py"
F05B_COLUMNS = list(F05B_FEATURE_COLUMNS)


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_n03_combo_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_runner():
    assert RUNNER_PATH.is_file(), "N03 runner 尚未实现"
    spec = importlib.util.spec_from_file_location("run_n03_combo", RUNNER_PATH)
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


def _internal_stats_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "row_index": [20, 11, 10],
            "hidden_tvt_that_must_not_join": [9999.0, 9999.0, 9999.0],
        }
    )
    for column_index, column in enumerate(F05B_COLUMNS):
        rows[column] = [
            float(300 + column_index),
            float(200 + column_index),
            float(100 + column_index),
        ]
    return rows


def test_config_freezes_f05a_and_only_adds_ten_f05b_features() -> None:
    assert CONFIG_PATH.is_file(), "N03 配置尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *F05B_COLUMNS,
    ]
    assert config["experiment_id"] == "N03_f05a_plus_pf_internal_stats_v1"
    assert config["baseline_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_count"] == 46
    assert config["feature_columns"] == expected_features
    assert config["candidate_feature_columns"] == DIRECT_CANDIDATE_COLUMNS
    assert config["new_feature_columns"] == F05B_COLUMNS
    assert config["candidate_cache_sha256"] == (
        "66b32f8ed790ea53184d43bc02d6899031348deec418d7933366bbef64105076"
    )
    assert config["candidate_cache_metadata_sha256"] == (
        "1b152c05ce48b89ef683a545dcdeae16d1d72d58a57f6c723ecb74d41126c5d4"
    )
    assert config["internal_stats_cache_sha256"] == (
        "d44cd4c28629f678f4ba4123c9b88e75078a7d51ac3995b1ca37d15924944687"
    )
    assert config["internal_stats_cache_metadata_sha256"] == (
        "90ed5e7ed8edac47c9f5be1144ac4bed379bcd66cc2f1077dd1fba1bde618791"
    )
    assert config["cache_seed"] == 42
    assert config["cache_rebuild_allowed"] is False
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"

    runner = _load_runner()
    assert runner._validate_config(config) == expected_features
    assert runner.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert runner.parse_args(["--mode", "fold4"]).mode == "fold4"


def test_strict_key_join_restores_f05a_order_and_selects_only_ten_columns() -> None:
    runner = _load_runner()

    result = runner.add_internal_stats_features(
        _f05a_rows(),
        _internal_stats_rows(),
    )

    assert result[["well_id", "row_index"]].to_dict("records") == [
        {"well_id": "well_a", "row_index": 10},
        {"well_id": "well_a", "row_index": 11},
        {"well_id": "well_b", "row_index": 20},
    ]
    assert result[F05B_COLUMNS[0]].tolist() == [100.0, 200.0, 300.0]
    assert "hidden_tvt_that_must_not_join" not in result.columns


@pytest.mark.parametrize("failure_kind", ["duplicate", "missing", "extra"])
def test_strict_key_join_rejects_non_bijective_cache(failure_kind: str) -> None:
    runner = _load_runner()
    internal_stats_rows = _internal_stats_rows()
    if failure_kind == "duplicate":
        internal_stats_rows = pd.concat(
            [internal_stats_rows, internal_stats_rows.iloc[[0]]],
            ignore_index=True,
        )
    elif failure_kind == "missing":
        internal_stats_rows = internal_stats_rows.iloc[:-1].copy()
    else:
        extra = internal_stats_rows.iloc[[0]].copy()
        extra["well_id"] = "well_extra"
        extra["row_index"] = 999
        internal_stats_rows = pd.concat(
            [internal_stats_rows, extra],
            ignore_index=True,
        )

    with pytest.raises(ValueError, match="重复键|一一对应"):
        runner.add_internal_stats_features(_f05a_rows(), internal_stats_rows)


def test_missing_cache_bundle_never_rebuilds(local_tmp_path: Path) -> None:
    runner = _load_runner()
    paths = [
        local_tmp_path / "base.parquet",
        local_tmp_path / "f05a.parquet",
        local_tmp_path / "f05a_meta.json",
        local_tmp_path / "f05b.parquet",
        local_tmp_path / "f05b_meta.json",
    ]

    with pytest.raises(FileNotFoundError, match="不会重建"):
        runner.require_cache_bundle(*paths)

    assert not any(path.exists() for path in paths)


def test_experiment_fingerprint_tracks_both_cache_and_metadata_sha() -> None:
    runner = _load_runner()
    common = {
        "config": {"experiment_id": "n03"},
        "model_params": {"n_estimators": 1734, "random_state": 29},
        "registry_hash": "registry-sha",
        "runner_hash": "runner-sha",
        "candidate_cache_sha256": "candidate-cache",
        "candidate_metadata_sha256": "candidate-meta",
        "internal_stats_cache_sha256": "stats-cache",
        "internal_stats_metadata_sha256": "stats-meta",
    }

    first = runner.build_experiment_fingerprint(**common)
    for key in (
        "candidate_cache_sha256",
        "candidate_metadata_sha256",
        "internal_stats_cache_sha256",
        "internal_stats_metadata_sha256",
    ):
        changed = dict(common)
        changed[key] = f"{changed[key]}-changed"
        assert first != runner.build_experiment_fingerprint(**changed)
