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
RUNNER_PATH = CLEAN_ROOT / "scripts" / "diagnose_p2_s01_outer_fold_surface.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_s01_outer_fold_surface_path_v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p2_s01_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 P2-S01 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_freezes_the_preregistered_surface_design() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["surface_name"] == "EGFDU"
    assert config["nearest_source_wells"] == 10
    assert config["control_step_horizontal_ft"] == 50.0
    assert config["fold_registry_sha256"] == (
        "a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c"
    )
    assert config["success_conditions"] == {
        "maximum_own_surface_oracle_micro_rmse_ft": 0.6,
        "minimum_improvement_vs_carry_ft": 0.5,
        "minimum_folds_better_than_carry": 4,
        "minimum_improvement_vs_permuted_surface_ft": 1.0,
    }


def test_fold_source_table_excludes_the_complete_outer_fold() -> None:
    runner = _load_runner()
    registry = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "fold": [0, 0, 1, 2],
        }
    )
    representatives = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "source_x": [0.0, 1.0, 2.0, 3.0],
            "source_y": [0.0, 1.0, 2.0, 3.0],
            "source_surface": [10.0, 11.0, 12.0, 13.0],
        }
    )

    source_table = runner.build_fold_source_table(registry, representatives, outer_fold=0)

    assert source_table["well_id"].tolist() == ["c", "d"]


def test_own_surface_oracle_checks_formula_without_entering_legal_path() -> None:
    runner = _load_runner()
    own_surface = np.array([100.0, 101.0, 102.0, 103.0])
    z = np.array([-500.0, -499.0, -498.0, -497.0])
    marker = 1000.0
    tvt = own_surface + marker - z
    tvt_input = tvt.copy()
    tvt_input[2:] = np.nan
    oracle_frame = pd.DataFrame(
        {
            "Z": z,
            "TVT_input": tvt_input,
            "EGFDU": own_surface,
        }
    )

    prediction = runner.build_own_surface_oracle_path(oracle_frame, "EGFDU")

    np.testing.assert_allclose(prediction, tvt[2:])


def test_summary_applies_all_preregistered_gates() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    per_well = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "fold": [0, 1, 2, 3, 4],
            "hidden_rows": [100, 100, 100, 100, 100],
            "sse_surface": [100.0] * 5,
            "sse_negative": [900.0] * 5,
            "sse_own_surface": [4.0] * 5,
            "sse_carry": [400.0] * 5,
            "sse_p2b00": [144.0] * 5,
            "surface_rmse": [1.0] * 5,
            "negative_rmse": [3.0] * 5,
            "own_surface_rmse": [0.2] * 5,
            "carry_rmse": [2.0] * 5,
            "p2b00_rmse": [1.2] * 5,
            "median_nearest_source_distance": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    summary = runner.build_summary(per_well, config, wall_seconds=12.5)

    assert summary["micro_rmse"]["surface"] == pytest.approx(1.0)
    assert summary["micro_rmse"]["carry"] == pytest.approx(2.0)
    assert summary["checks"]["own_surface_oracle_pass"] is True
    assert summary["checks"]["carry_improvement_pass"] is True
    assert summary["checks"]["fold_consistency_pass"] is True
    assert summary["checks"]["negative_control_pass"] is True
    assert summary["surface_path_supported"] is True


def test_summary_rejects_a_path_that_only_beats_one_fold() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    per_well = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "fold": [0, 1, 2, 3, 4],
            "hidden_rows": [100] * 5,
            "sse_surface": [100.0, 900.0, 900.0, 900.0, 900.0],
            "sse_negative": [1600.0] * 5,
            "sse_own_surface": [4.0] * 5,
            "sse_carry": [400.0] * 5,
            "sse_p2b00": [144.0] * 5,
            "surface_rmse": [1.0, 3.0, 3.0, 3.0, 3.0],
            "negative_rmse": [4.0] * 5,
            "own_surface_rmse": [0.2] * 5,
            "carry_rmse": [2.0] * 5,
            "p2b00_rmse": [1.2] * 5,
            "median_nearest_source_distance": [10.0] * 5,
        }
    )

    summary = runner.build_summary(per_well, config, wall_seconds=12.5)

    assert summary["folds_better_than_carry"] == 1
    assert summary["checks"]["fold_consistency_pass"] is False
    assert summary["surface_path_supported"] is False
