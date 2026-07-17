from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CLEAN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CLEAN_ROOT / "configs" / "n02_f05a_plus_gr_gap_geometry_v1.json"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_n02_f05a_plus_gr_gap_geometry_cv.py"

GR_GAP_COLUMNS = [
    "gr_gap_length_ft",
    "gr_distance_to_left_observed_ft",
    "gr_distance_to_right_observed_ft",
    "gr_relative_position_inside_gap",
    "gr_valid_fraction_50",
    "gr_valid_fraction_100",
    "gr_valid_fraction_200",
    "well_hidden_gr_valid_fraction",
    "well_hidden_longest_gr_gap_ft",
]


def _load_runner():
    assert RUNNER_PATH.is_file(), "N02 runner 尚未实现"
    spec = importlib.util.spec_from_file_location(
        "run_n02_f05a_plus_gr_gap_geometry_cv",
        RUNNER_PATH,
    )
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


def _gr_gap_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "row_index": [20, 11, 10],
            "target_tvt": [9999.0, 9999.0, 9999.0],
        }
    )
    for column_index, column in enumerate(GR_GAP_COLUMNS):
        rows[column] = [
            float(300 + column_index),
            float(200 + column_index),
            float(100 + column_index),
        ]
    return rows


def test_config_freezes_f05a_plus_nine_gr_gap_columns_and_source_sha() -> None:
    assert CONFIG_PATH.is_file(), "N02 config 尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["experiment_id"] == "N02_f05a_plus_gr_gap_geometry_v1"
    assert config["baseline_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_columns"] == [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *GR_GAP_COLUMNS,
    ]
    assert config["new_feature_columns"] == GR_GAP_COLUMNS
    assert config["gr_gap_feature_cache"] == (
        "artifacts/RF02a_gr_missing_geometry_v1/feature_cache.parquet"
    )
    assert config["gr_gap_feature_cache_sha256"] == (
        "d0cc87dee182b6992464cb09623c7a428c8f3acba2b6158b786e3b2076cecc98"
    )
    assert config["candidate_cache_rebuild_allowed"] is False

    runner = _load_runner()
    assert runner.GR_GAP_COLUMNS == GR_GAP_COLUMNS
    assert runner._validate_config(config) == config["feature_columns"]
    assert runner.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert runner.parse_args(["--mode", "fold4"]).mode == "fold4"


def test_strict_key_join_restores_f05a_order_and_ignores_unselected_columns() -> None:
    runner = _load_runner()

    result = runner.add_gr_gap_features(_f05a_rows(), _gr_gap_rows())

    assert result[["well_id", "row_index"]].to_dict("records") == [
        {"well_id": "well_a", "row_index": 10},
        {"well_id": "well_a", "row_index": 11},
        {"well_id": "well_b", "row_index": 20},
    ]
    assert result[GR_GAP_COLUMNS[0]].tolist() == [100.0, 200.0, 300.0]
    assert "target_tvt" not in result.columns


@pytest.mark.parametrize("failure_kind", ["duplicate", "missing", "extra"])
def test_strict_key_join_rejects_non_bijective_cache(failure_kind: str) -> None:
    runner = _load_runner()
    gr_gap_rows = _gr_gap_rows()
    if failure_kind == "duplicate":
        gr_gap_rows = pd.concat([gr_gap_rows, gr_gap_rows.iloc[[0]]], ignore_index=True)
    elif failure_kind == "missing":
        gr_gap_rows = gr_gap_rows.iloc[:-1].copy()
    else:
        extra = gr_gap_rows.iloc[[0]].copy()
        extra["well_id"] = "well_extra"
        extra["row_index"] = 999
        gr_gap_rows = pd.concat([gr_gap_rows, extra], ignore_index=True)

    with pytest.raises(ValueError, match="键|重复|一一对应"):
        runner.add_gr_gap_features(_f05a_rows(), gr_gap_rows)

