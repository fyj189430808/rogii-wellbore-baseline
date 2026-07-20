"""P3-PFM03 正式 41+3 单模 LightGBM runner 合同测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pfm03_shape_aware_modes_cv as runner


def test_real_config_contains_frozen_development_counts_required_by_registry_loader() -> None:
    """正式 registry loader 依赖这三项，真实配置必须完整冻结而不能靠默认值。"""

    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert config["development_fold_well_counts"] == {
        "0": 131,
        "1": 132,
        "2": 131,
        "3": 132,
        "4": 131,
    }
    assert config["development_fold_hidden_row_counts"] == {
        "0": 651_881,
        "1": 630_395,
        "2": 645_557,
        "3": 649_717,
        "4": 634_322,
    }
    assert config["baseline_development_micro_rmse"] == 10.272146267501086


def test_formal_feature_list_keeps_original_41_and_only_adds_shape_paths() -> None:
    features = runner.build_formal_feature_names()

    assert features[:41] == runner.P3B00_FEATURES
    assert features[41:] == [
        "shape_mode_low_delta",
        "shape_mode_middle_delta",
        "shape_mode_high_delta",
    ]
    assert len(features) == len(set(features)) == 44


def test_formal_gate_starts_with_folds_1_and_2_and_never_uses_fold0() -> None:
    assert runner.parse_fold_spec("1,2") == [1, 2]
    assert runner.parse_fold_spec("1,2,3,4") == [1, 2, 3, 4]
    for forbidden in ("0,1", "all", "0"):
        with pytest.raises(ValueError, match="1,2"):
            runner.parse_fold_spec(forbidden)

    passed = {"success_checks": {"folds12_pass": True}}
    failed = {"success_checks": {"folds12_pass": False}}
    assert runner.remaining_folds_after_screen([1, 2], passed) == []
    assert runner.remaining_folds_after_screen([1, 2, 3, 4], passed) == [3, 4]
    assert runner.remaining_folds_after_screen([1, 2, 3, 4], failed) == []


def test_folds12_gate_uses_average_fold_improvement_and_worst_fold_limit() -> None:
    assert runner.folds12_gate({1: 0.20, 2: 0.10})["folds12_pass"] is True
    assert runner.folds12_gate({1: 0.50, 2: -0.16})["folds12_pass"] is False
    assert runner.folds12_gate({1: 0.14, 2: 0.14})["folds12_pass"] is False
