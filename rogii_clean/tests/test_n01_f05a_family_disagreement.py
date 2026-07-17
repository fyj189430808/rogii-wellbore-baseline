from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS
from src.n01_f05a_family_disagreement import (
    FAMILY_DISAGREEMENT_COLUMNS,
    REPRESENTATIVE_PATH_COLUMNS,
    build_family_disagreement_features,
)


CLEAN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CLEAN_ROOT / "configs" / "n01_f05a_family_disagreement_v1.json"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_n01_f05a_family_disagreement_cv.py"


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["well_a", "well_a"],
            "row_index": [10, 11],
            "pf_ancc_delta": [10.0, -4.0],
            "pf_z_delta": [4.0, 8.0],
            "beam_mean_d": [7.0, 2.0],
            "sc_ens_d": [1.0, 6.0],
            "target_delta": [9999.0, -9999.0],
        }
    )


def _load_runner():
    assert RUNNER_PATH.is_file(), "N01 runner 尚未实现"
    spec = importlib.util.spec_from_file_location(
        "run_n01_f05a_family_disagreement_cv",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_disagreement_formulas_and_column_order() -> None:
    result = build_family_disagreement_features(_candidate_rows())

    assert REPRESENTATIVE_PATH_COLUMNS == [
        "pf_ancc_delta",
        "pf_z_delta",
        "beam_mean_d",
        "sc_ens_d",
    ]
    assert list(result.columns) == [
        "well_id",
        "row_index",
        *FAMILY_DISAGREEMENT_COLUMNS,
    ]

    first = result.iloc[0]
    assert first["pf_ancc_minus_beam"] == pytest.approx(3.0)
    assert first["pf_ancc_minus_sc"] == pytest.approx(9.0)
    assert first["beam_minus_sc"] == pytest.approx(6.0)
    assert first["abs_pf_ancc_minus_beam"] == pytest.approx(3.0)
    assert first["abs_pf_ancc_minus_sc"] == pytest.approx(9.0)
    assert first["abs_beam_minus_sc"] == pytest.approx(6.0)

    four_paths = np.asarray([10.0, 4.0, 7.0, 1.0], dtype=np.float32)
    median = np.median(four_paths)
    expected_mad = np.median(np.abs(four_paths - median))
    assert first["family_path_std"] == pytest.approx(np.std(four_paths))
    assert first["family_path_range"] == pytest.approx(9.0)
    assert first["family_path_mad"] == pytest.approx(expected_mad)


def test_unrelated_columns_cannot_change_family_disagreement_features() -> None:
    source = _candidate_rows()
    changed = source.copy()
    changed["target_delta"] = [123456.0, 654321.0]
    changed["hidden_tvt"] = [1.0, 2.0]

    original = build_family_disagreement_features(source)
    mutated = build_family_disagreement_features(changed)

    pd.testing.assert_frame_equal(original, mutated)


def test_nonfinite_representative_path_is_rejected() -> None:
    source = _candidate_rows()
    source.loc[0, "sc_ens_d"] = np.nan

    with pytest.raises(ValueError, match="NaN|Inf"):
        build_family_disagreement_features(source)


def test_config_and_runner_freeze_f05a_plus_nine_new_columns() -> None:
    assert CONFIG_PATH.is_file(), "N01 config 尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["experiment_id"] == "N01_f05a_family_disagreement_v1"
    assert config["baseline_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_columns"] == [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *FAMILY_DISAGREEMENT_COLUMNS,
    ]
    assert config["new_feature_columns"] == FAMILY_DISAGREEMENT_COLUMNS
    assert config["candidate_cache_rebuild_allowed"] is False

    runner = _load_runner()
    assert runner._validate_config(config) == config["feature_columns"]
    assert runner.parse_args(["--mode", "smoke"]).mode == "smoke"
    assert runner.parse_args(["--mode", "fold4"]).mode == "fold4"

