"""PFM04 冻结 41+3 特征合同与井级广播测试。"""

from __future__ import annotations

import sys
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pfm04_mode_direction_summary_cv as runner


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """使用项目盘临时目录，避开当前 Windows 用户临时目录的权限问题。"""

    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pfm04_runner_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def test_feature_contract_is_exactly_original_41_plus_three_summaries() -> None:
    """禁止把模式路径、质量或其他审计字段悄悄加入模型。"""

    features = runner.build_formal_feature_names()
    assert len(features) == 44
    assert features[:41] == runner.P3B00_FEATURES
    assert features[41:] == [
        "direction_score",
        "p2_position_raw",
        "high_minus_low_separation",
    ]
    assert not any("mode_low_delta" in name for name in features)
    assert runner.parse_fold_spec("1,2") == [1, 2]


def test_well_level_summaries_are_broadcast_by_natural_well_id_only(
    workspace_tmp_path: Path,
) -> None:
    """同井各行拿到相同摘要，不使用行序碰巧对齐。"""

    feature_table = pd.DataFrame(
        {
            "well_id": ["00001234", "00001234", "well-b"],
            "row_index": [7, 8, 3],
            "fold": [1, 1, 2],
        }
    )
    registry = pd.DataFrame(
        {
            "well_id": ["00001234", "well-b"],
            "fold": [1, 2],
            "hidden_rows": [2, 1],
        }
    )
    summary_path = workspace_tmp_path / "per_well.parquet"
    pd.DataFrame(
        {
            "well_id": ["well-b", "00001234"],
            "fold": [2, 1],
            "direction_score": [0.8, -0.2],
            "p2_position_raw": [0.7, 0.3],
            "high_minus_low_separation": [12.0, 8.0],
            "source_seed_sha256": ["seed-b", "seed-a"],
            "source_p2_sha256": ["p2", "p2"],
            "_cache_fingerprint": ["fingerprint", "fingerprint"],
        }
    ).to_parquet(summary_path, index=False)

    merged, fingerprint = runner.broadcast_mode_summaries(
        feature_table,
        registry,
        summary_path,
        {"shadow-well"},
    )

    assert fingerprint == "fingerprint"
    np.testing.assert_allclose(merged["direction_score"], [-0.2, -0.2, 0.8])
    np.testing.assert_allclose(merged["p2_position_raw"], [0.3, 0.3, 0.7])
    np.testing.assert_allclose(merged["high_minus_low_separation"], [8.0, 8.0, 12.0])


def test_real_config_keeps_frozen_lightgbm_model_and_hash() -> None:
    """真实入口必须核对基线模型文件 SHA、1734 棵树和 seed29。"""

    config = runner.read_json(runner.DEFAULT_CONFIG)
    features, params = runner.validate_frozen_contract(config)
    assert len(features) == 44
    assert params["n_estimators"] == 1734
    assert params["random_state"] == 29
    assert runner.file_sha256(runner.resolve_clean_path(config["model_config"])) == config[
        "model_config_sha256"
    ]


def test_real_config_contains_complete_frozen_development_fold_contract() -> None:
    """正式配置必须提供公共 registry 加载器要求的逐折井数、行数和基线分数。"""

    config = runner.read_json(runner.DEFAULT_CONFIG)
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
    registry, shadow_ids = runner.load_development_registry(
        runner.resolve_clean_path(config["fold_registry"]),
        runner.resolve_clean_path(config["shadow_registry"]),
        config,
    )
    assert len(registry) == 657
    assert len(shadow_ids) == 116
