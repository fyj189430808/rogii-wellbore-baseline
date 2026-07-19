"""P2-P01 runner 的冻结配置、合法缓存和晋级门槛测试。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p2_p01_multiseed_pf_mean import (  # noqa: E402
    validate_last_visible_matches_carry,
    evaluate_path_success_checks,
    select_mode_registry,
    validate_config,
    validate_legal_cache,
)


def test_last_visible_comparison_uses_frozen_float32_semantics() -> None:
    """旧基线的 float64 carry 只要转成 float32 后相同，就应视为同一路径起点。"""

    carry_float64 = pd.Series([11747.37, 11747.37])
    cached_float32 = pd.Series([11747.37, 11747.37], dtype="float32")

    validate_last_visible_matches_carry(carry_float64, cached_float32)

    wrong_cached = pd.Series([11747.37, 11748.37], dtype="float32")
    with pytest.raises(ValueError, match="last_visible_tvt"):
        validate_last_visible_matches_carry(carry_float64, wrong_cached)


def _load_config() -> dict:
    path = CLEAN_ROOT / "configs" / "p2_p01_multiseed_pf_mean_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_config_rejects_parameter_or_formal_feature_changes() -> None:
    """结果出现后不能换 seed 数、粒子数或把 scale 路径偷偷加入模型。"""

    config = _load_config()
    validate_config(config)

    changed_seeds = copy.deepcopy(config)
    changed_seeds["particle_filter"]["number_of_seeds"] = 64
    with pytest.raises(ValueError, match="number_of_seeds"):
        validate_config(changed_seeds)

    changed_feature = copy.deepcopy(config)
    changed_feature["formal_feature_name"] = "pf128_scale_5_delta"
    with pytest.raises(ValueError, match="formal_feature_name"):
        validate_config(changed_feature)


def test_mode_registry_is_fixed_and_smoke_is_first_fold0_well() -> None:
    """smoke、fold0、all 的井集合由注册表固定，不能看结果后挑井。"""

    registry = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "fold": [0, 1, 0, 4],
            "hidden_rows": [10, 20, 30, 40],
        }
    )
    assert select_mode_registry(registry, "smoke")["well_id"].tolist() == ["a"]
    assert select_mode_registry(registry, "fold0")["well_id"].tolist() == ["a", "c"]
    assert select_mode_registry(registry, "all")["well_id"].tolist() == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_legal_cache_rejects_target_surface_and_wrong_keys() -> None:
    """合法特征缓存不能含真值、surface、重复键或错误井号。"""

    legal = pd.DataFrame(
        {
            "well_id": ["a", "a"],
            "row_index": [10, 11],
            "last_visible_tvt": [100.0, 100.0],
            "pf128_mean_tvt": [101.0, 102.0],
            "pf128_mean_delta": [1.0, 2.0],
            "pf128_seed0_delta": [0.9, 1.9],
            "pf128_scale_3_delta": [1.0, 2.0],
            "pf128_scale_5_delta": [1.0, 2.0],
            "pf128_scale_8_delta": [1.0, 2.0],
            "pf128_scale_12_delta": [1.0, 2.0],
            "pf128_seed_std": [0.5, 0.6],
            "_cache_fingerprint": ["hash", "hash"],
        }
    )
    validate_legal_cache(legal, expected_well_id="a", expected_rows=2)

    for forbidden_column in ["target", "TVT", "ANCC", "oracle_rank"]:
        leaked = legal.copy()
        leaked[forbidden_column] = [101.0, 102.0]
        with pytest.raises(ValueError, match="禁止列|未登记列|列集合"):
            validate_legal_cache(leaked, expected_well_id="a", expected_rows=2)

    duplicated = pd.concat([legal, legal.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="重复"):
        validate_legal_cache(duplicated, expected_well_id="a", expected_rows=3)


def test_only_program_and_legality_controls_gate_full_feature_generation() -> None:
    """路径 RMSE 只作诊断；全量生成只由确定性和合法性决定。"""

    config = _load_config()
    passing = {
        "maximum_repeat_difference_ft": 0.0,
        "maximum_concurrent_difference_ft": 0.0,
        "maximum_hidden_tvt_mutation_difference_ft": 0.0,
        "fold0_improvement_vs_single_pf_ft": -10.0,
        "well_win_rate_vs_single_pf": 0.0,
    }
    checks = evaluate_path_success_checks(passing, config)
    assert checks["feature_generation_supported"] is True

    failing = dict(passing)
    failing["maximum_repeat_difference_ft"] = 1e-6
    failed_checks = evaluate_path_success_checks(failing, config)
    assert failed_checks["feature_generation_supported"] is False
    assert failed_checks["repeat_determinism_pass"] is False
