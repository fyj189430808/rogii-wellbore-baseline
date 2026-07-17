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

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CONFIG_PATH = CLEAN_ROOT / "configs" / "n04_f05a_plus_local_gr_quality_v1.json"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_n04_f05a_plus_local_gr_quality_cv.py"
LOCAL_GR_COLUMNS = [
    "gr_local_observed_count_50",
    "gr_local_mad_50",
    "gr_local_variance_50",
    "gr_local_observed_count_100",
    "gr_local_mad_100",
    "gr_local_variance_100",
    "gr_local_observed_count_200",
    "gr_local_mad_200",
    "gr_local_variance_200",
]


def _load_runner():
    assert RUNNER_PATH.is_file(), "N04 runner 尚未实现"
    spec = importlib.util.spec_from_file_location("run_n04_combo", RUNNER_PATH)
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


def _local_gr_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "row_index": [20, 11, 10],
            "target_tvt_that_must_not_join": [9999.0, 9999.0, 9999.0],
        }
    )
    for column_index, column in enumerate(LOCAL_GR_COLUMNS):
        rows[column] = [
            float(300 + column_index),
            float(200 + column_index),
            float(100 + column_index),
        ]
    return rows


def test_config_freezes_f05a_and_only_adds_nine_local_gr_columns() -> None:
    assert CONFIG_PATH.is_file(), "N04 配置尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *LOCAL_GR_COLUMNS,
    ]

    assert config["experiment_id"] == "N04_f05a_plus_local_gr_quality_v1"
    assert config["baseline_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_count"] == 45
    assert config["feature_columns"] == expected_features
    assert config["candidate_feature_columns"] == DIRECT_CANDIDATE_COLUMNS
    assert config["new_feature_columns"] == LOCAL_GR_COLUMNS
    assert config["local_gr_feature_cache_sha256"] == (
        "ddc14dea316c2a5b40fb43f5cc8b523e909af9efca8b5bfdd7d5cb5943928707"
    )
    assert config["local_gr_feature_cache_metadata_sha256"] == (
        "89e91d7d00fe88397702a820620838be749b399b3dcfb1d331d6dc9683503d01"
    )
    assert config["candidate_cache_rebuild_allowed"] is False
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"

    runner = _load_runner()
    assert runner.LOCAL_GR_COLUMNS == LOCAL_GR_COLUMNS
    assert runner._validate_config(config) == expected_features
    assert runner.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert runner.parse_args(["--mode", "fold4"]).mode == "fold4"


def test_strict_key_join_restores_f05a_order_and_selects_only_nine_columns() -> None:
    runner = _load_runner()
    result = runner.add_local_gr_features(_f05a_rows(), _local_gr_rows())

    assert result[["well_id", "row_index"]].to_dict("records") == [
        {"well_id": "well_a", "row_index": 10},
        {"well_id": "well_a", "row_index": 11},
        {"well_id": "well_b", "row_index": 20},
    ]
    assert result[LOCAL_GR_COLUMNS[0]].tolist() == [100.0, 200.0, 300.0]
    assert "target_tvt_that_must_not_join" not in result.columns


@pytest.mark.parametrize("failure_kind", ["duplicate", "missing", "extra"])
def test_strict_key_join_rejects_non_bijective_cache(failure_kind: str) -> None:
    runner = _load_runner()
    local_gr_rows = _local_gr_rows()
    if failure_kind == "duplicate":
        local_gr_rows = pd.concat(
            [local_gr_rows, local_gr_rows.iloc[[0]]], ignore_index=True
        )
    elif failure_kind == "missing":
        local_gr_rows = local_gr_rows.iloc[:-1].copy()
    else:
        extra = local_gr_rows.iloc[[0]].copy()
        extra["well_id"] = "well_extra"
        extra["row_index"] = 999
        local_gr_rows = pd.concat([local_gr_rows, extra], ignore_index=True)

    with pytest.raises(ValueError, match="重复键|一一对应"):
        runner.add_local_gr_features(_f05a_rows(), local_gr_rows)


def test_local_gr_validation_allows_only_registered_nan_columns() -> None:
    runner = _load_runner()
    rows = _local_gr_rows()
    rows.loc[0, "gr_local_mad_50"] = np.nan
    runner._validate_local_gr_values(rows)

    rows.loc[0, "gr_local_observed_count_50"] = np.nan
    with pytest.raises(ValueError, match="count.*NaN"):
        runner._validate_local_gr_values(rows)

    rows.loc[0, "gr_local_observed_count_50"] = 1.0
    rows.loc[0, "gr_local_variance_100"] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        runner._validate_local_gr_values(rows)
