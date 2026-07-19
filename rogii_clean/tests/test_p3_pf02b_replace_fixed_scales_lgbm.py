"""PF02b 等维替换实验的最小合同测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pf02_target_ess_lgbm_cv import P3B00_FEATURES
from scripts.run_p3_pf02b_replace_fixed_scales_lgbm_v1 import (
    DEFAULT_CONFIG,
    REPLACEMENT_MAPPING,
    build_formal_feature_names,
    parse_fold_spec,
    validate_frozen_contract,
)


def test_feature_contract_replaces_four_paths_in_place_and_stays_at_41() -> None:
    features = build_formal_feature_names()

    assert len(features) == 41
    assert len(set(features)) == 41
    assert features[-5] == "pf128_mean_delta"
    assert features[-4:] == list(REPLACEMENT_MAPPING.values())
    assert not set(REPLACEMENT_MAPPING).intersection(features)

    unchanged = [name for name in P3B00_FEATURES if name not in REPLACEMENT_MAPPING]
    assert len(unchanged) == 37
    for name in unchanged:
        assert features[P3B00_FEATURES.index(name)] == name


def test_fold_spec_keeps_only_registered_screen_and_full_modes() -> None:
    assert parse_fold_spec("0,1") == [0, 1]
    assert parse_fold_spec("all") == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="0,1"):
        parse_fold_spec("0")


def test_checked_in_config_matches_frozen_replacement_contract() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    features, params = validate_frozen_contract(config)

    assert len(features) == 41
    assert config["replacement_mapping"] == REPLACEMENT_MAPPING
    assert params["n_estimators"] == 1734
    assert params["random_state"] == 29
