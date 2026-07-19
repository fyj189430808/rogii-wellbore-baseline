from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))
RUNNER_PATH = CLEAN_ROOT / "scripts" / "diagnose_p2_s02_dense_relative_gradient.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_s02_dense_relative_gradient_v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p2_s02_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 P2-S02 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_freezes_dense_relative_gradient_design() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["surface_name"] == "EGFDU"
    assert config["source_control_step_horizontal_ft"] == 50.0
    assert config["target_control_step_horizontal_ft"] == 50.0
    assert config["nearest_distinct_source_wells"] == 10
    assert config["gradient_cap_quantile"] == 0.99


def test_fold_dense_source_points_exclude_all_outer_valid_wells() -> None:
    runner = _load_runner()
    registry = pd.DataFrame(
        {"well_id": ["a", "b", "c"], "fold": [0, 1, 2], "hidden_rows": [1, 1, 1]}
    )
    dense_points = pd.DataFrame(
        {
            "source_well_id": ["a", "a", "b", "c"],
            "source_row_index": [0, 1, 0, 0],
            "source_distance_ft": [0.0, 50.0, 0.0, 0.0],
            "source_x": [0.0, 1.0, 2.0, 3.0],
            "source_y": [0.0, 0.0, 0.0, 0.0],
            "source_surface": [10.0, 11.0, 12.0, 13.0],
        }
    )

    source_points = runner.build_fold_source_points(registry, dense_points, outer_fold=0)

    assert set(source_points["source_well_id"]) == {"b", "c"}


def test_summary_applies_dense_gradient_gates() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    per_well = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "fold": [0, 1, 2, 3, 4],
            "hidden_rows": [100] * 5,
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
            "median_nearest_source_distance": [10.0] * 5,
        }
    )

    summary = runner.build_summary(per_well, config, wall_seconds=5.0)

    assert summary["micro_rmse"]["surface"] == pytest.approx(1.0)
    assert summary["checks"] == {
        "own_surface_oracle_pass": True,
        "carry_improvement_pass": True,
        "fold_consistency_pass": True,
        "reversed_gradient_control_pass": True,
    }
    assert summary["surface_path_supported"] is True
