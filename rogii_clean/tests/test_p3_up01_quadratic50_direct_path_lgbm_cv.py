"""UP01 二次投影 50% 单路径进入冻结 LightGBM 的合同测试。"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_up01_quadratic50_direct_path_lgbm_cv as runner  # noqa: E402


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="up01_lgbm_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_config() -> dict[str, object]:
    return json.loads(
        (CLEAN_ROOT / "configs" / "p3_up01_quadratic50_direct_path_lgbm_v1.json")
        .read_text(encoding="utf-8")
    )


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "fold": np.array([3, 3, 4, 4], dtype=np.int64),
            "row_index": np.array([10, 11, 20, 21], dtype=np.int32),
            "md": [1.0, 2.0, 1.0, 2.0],
            "p2_pred_tvt": [100.0, 101.0, 200.0, 201.0],
            "degree2_blend25_pred_tvt": [100.1, 101.1, 200.1, 201.1],
            "degree2_blend50_pred_tvt": [100.2, 101.2, 200.2, 201.2],
            "degree2_blend75_pred_tvt": [100.3, 101.3, 200.3, 201.3],
            "degree3_blend25_pred_tvt": [100.4, 101.4, 200.4, 201.4],
            "degree3_blend50_pred_tvt": [100.5, 101.5, 200.5, 201.5],
            "degree3_blend75_pred_tvt": [100.6, 101.6, 200.6, 201.6],
        }
    )


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "pad_id": ["pa", "pb"],
            "fold": [3, 4],
            "hidden_rows": [2, 2],
        }
    )


def test_feature_contract_is_exact_41_plus_selected_path() -> None:
    features = runner.build_formal_feature_names()

    assert runner.NEW_UP01_FEATURES == ["up01_degree2_blend50_delta"]
    assert features == [*runner.P3B00_FEATURES, "up01_degree2_blend50_delta"]
    assert len(features) == len(set(features)) == 42


def test_config_freezes_selection_confirmation_and_model() -> None:
    config = _load_config()

    assert config["experiment_id"] == "P3_UP01_quadratic50_direct_path_lgbm_v1"
    assert config["selected_candidate"] == "degree2_blend50"
    assert config["selected_candidate_column"] == "degree2_blend50_pred_tvt"
    assert config["selection_folds"] == [1, 2]
    assert config["independent_confirmation_folds"] == [3, 4]
    assert config["post_confirmation_fold"] == [0]
    assert config["selection_folds_full5_only"] == [1, 2]
    assert config["formal_feature_count"] == 42
    assert config["model_training"] is True
    assert config["shadow_target_access"] is False
    assert config["confirmation_gate"] == {
        "minimum_arithmetic_mean_fold_improvement_ft": 0.15,
        "maximum_any_fold_degradation_ft": 0.15,
    }

    features, model_params = runner.validate_frozen_contract(
        config,
        enforce_config_file_hash=False,
    )
    assert len(features) == 42
    assert model_params["n_estimators"] == 1734
    assert model_params["random_state"] == 29


def test_frozen_contract_rejects_selected_candidate_or_model_flag_change() -> None:
    config = _load_config()
    config["selected_candidate"] = "degree2_blend75"
    with pytest.raises(ValueError, match="冻结|候选|合同"):
        runner.validate_frozen_contract(config, enforce_config_file_hash=False)

    config = _load_config()
    config["model_training"] = False
    with pytest.raises(ValueError, match="冻结|模型|合同"):
        runner.validate_frozen_contract(config, enforce_config_file_hash=False)


def test_selection_provenance_locks_best_mean_candidate_on_folds12() -> None:
    metrics = {
        "folds": [1, 2],
        "model_training": False,
        "candidate_generation_hidden_target_read": False,
        "shadow_target_read": False,
        "candidates": {
            "degree2_blend25": {"arithmetic_mean_fold_improvement_ft": 0.25},
            "degree2_blend50": {"arithmetic_mean_fold_improvement_ft": 0.37},
            "degree2_blend75": {"arithmetic_mean_fold_improvement_ft": 0.35},
        },
    }

    provenance = runner.validate_selection_provenance(
        metrics,
        selected_candidate="degree2_blend50",
        selection_folds=[1, 2],
    )

    assert provenance["selected_candidate"] == "degree2_blend50"
    assert provenance["selection_used_target"] is True
    assert provenance["independent_confirmation_folds"] == [3, 4]

    metrics["candidates"]["degree2_blend25"][
        "arithmetic_mean_fold_improvement_ft"
    ] = 0.40
    with pytest.raises(ValueError, match="最好|选择"):
        runner.validate_selection_provenance(
            metrics,
            selected_candidate="degree2_blend50",
            selection_folds=[1, 2],
        )


def test_legal_candidate_cache_is_complete_safe_and_naturally_aligned(
    workspace_tmp_path: Path,
) -> None:
    candidate_path = workspace_tmp_path / "legal_candidates.parquet"
    frame = _candidate_frame()
    frame.to_parquet(candidate_path, index=False)
    manifest = {
        "experiment_id": "P3_UP01_robust_u_projection_v1",
        "candidate_generation_complete": True,
        "hidden_target_read": False,
        "shadow_overlap_wells": 0,
        "folds": [0, 1, 2, 3, 4],
        "wells": 2,
        "rows": 4,
        "legal_candidates_sha256": runner.file_sha256(candidate_path),
    }
    safe_keys = frame[["well_id", "fold", "row_index"]].copy()

    selected, audit = runner.validate_legal_candidate_cache(
        candidate_path=candidate_path,
        manifest=manifest,
        registry=_registry(),
        shadow_ids={"shadow"},
        safe_source_keys=safe_keys,
        expected_sha256=runner.file_sha256(candidate_path),
        expected_wells=2,
        expected_rows=4,
    )

    assert tuple(selected.columns) == (
        "well_id",
        "row_index",
        "degree2_blend50_pred_tvt",
    )
    assert audit["legal_cache_complete"] is True
    assert audit["hidden_target_read"] is False
    assert audit["shadow_overlap_wells"] == 0
    assert audit["natural_key_match"] is True


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("hidden_target", "隐藏|target"),
        ("shadow", "影子|shadow"),
        ("row_count", "行数"),
        ("duplicate", "重复"),
        ("fold", "fold"),
        ("nonfinite", "NaN|Inf|有限"),
        ("key_mismatch", "自然键|行键"),
        ("sha", "SHA|哈希"),
    ],
)
def test_legal_candidate_cache_rejects_tainted_contract(
    workspace_tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    candidate_path = workspace_tmp_path / "legal_candidates.parquet"
    frame = _candidate_frame()
    if damage == "duplicate":
        frame.loc[1, "row_index"] = frame.loc[0, "row_index"]
    elif damage == "fold":
        frame.loc[0, "fold"] = 0
    elif damage == "nonfinite":
        frame.loc[0, "degree2_blend50_pred_tvt"] = np.inf
    frame.to_parquet(candidate_path, index=False)
    manifest = {
        "experiment_id": "P3_UP01_robust_u_projection_v1",
        "candidate_generation_complete": True,
        "hidden_target_read": damage == "hidden_target",
        "shadow_overlap_wells": 1 if damage == "shadow" else 0,
        "folds": [0, 1, 2, 3, 4],
        "wells": 2,
        "rows": 3 if damage == "row_count" else 4,
        "legal_candidates_sha256": runner.file_sha256(candidate_path),
    }
    safe_keys = _candidate_frame()[["well_id", "fold", "row_index"]].copy()
    if damage == "key_mismatch":
        safe_keys.loc[0, "row_index"] = 999
    expected_sha = "0" * 64 if damage == "sha" else runner.file_sha256(candidate_path)

    with pytest.raises(ValueError, match=message):
        runner.validate_legal_candidate_cache(
            candidate_path=candidate_path,
            manifest=manifest,
            registry=_registry(),
            shadow_ids={"shadow"},
            safe_source_keys=safe_keys,
            expected_sha256=expected_sha,
            expected_wells=2,
            expected_rows=4,
        )


def test_selected_absolute_path_becomes_only_one_delta_and_keeps_order() -> None:
    feature_table = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "row_index": [11, 10, 21, 20],
            "last_visible_tvt": [99.0, 99.0, 199.0, 199.0],
            "old": [1.0, 2.0, 3.0, 4.0],
        }
    )
    selected = _candidate_frame()[
        ["well_id", "row_index", "degree2_blend50_pred_tvt"]
    ]

    merged = runner.merge_selected_path(feature_table, selected)

    assert merged["row_index"].tolist() == [11, 10, 21, 20]
    assert merged["up01_degree2_blend50_delta"].tolist() == pytest.approx(
        [2.2, 1.2, 2.2, 1.2]
    )
    assert not any("blend25" in name or "blend75" in name for name in merged.columns)


def test_confirmation_gate_uses_mean_improvement_and_max_degradation() -> None:
    passed = pd.DataFrame(
        {"fold": [3, 4], "improvement_ft": [0.31, -0.01]}
    )
    failed_mean = pd.DataFrame(
        {"fold": [3, 4], "improvement_ft": [0.30, -0.01]}
    )
    failed_fold = pd.DataFrame(
        {"fold": [3, 4], "improvement_ft": [0.50, -0.151]}
    )

    assert runner.evaluate_confirmation_gate(passed)["confirmation_pass"] is True
    assert runner.evaluate_confirmation_gate(failed_mean)["confirmation_pass"] is False
    assert runner.evaluate_confirmation_gate(failed_fold)["confirmation_pass"] is False


def test_only_confirmation_folds_run_before_gate_and_full5_is_labeled() -> None:
    passed = {"confirmation_pass": True}
    failed = {"confirmation_pass": False}

    assert runner.parse_stage("confirm") == "confirm"
    assert runner.parse_stage("all") == "all"
    with pytest.raises(ValueError, match="confirm|all"):
        runner.parse_stage("3")
    assert runner.continuation_folds("confirm", passed) == []
    assert runner.continuation_folds("all", failed) == []
    assert runner.continuation_folds("all", passed) == [0, 1, 2]
    labels = runner.fold_evidence_labels()
    assert labels == {
        "0": "post_confirmation_additional_fold",
        "1": "used_for_candidate_selection_full5_only",
        "2": "used_for_candidate_selection_full5_only",
        "3": "independent_confirmation",
        "4": "independent_confirmation",
    }


def test_formal_feature_list_contains_no_other_up01_candidate_or_summary() -> None:
    features = runner.build_formal_feature_names()

    assert [name for name in features if name.startswith("up01_")] == [
        "up01_degree2_blend50_delta"
    ]
    forbidden = ("blend25", "blend75", "degree3", "mass", "score", "oracle")
    assert not any(token in name.lower() for name in features for token in forbidden)


def test_large_unchanged_baseline_parquets_are_not_rehashed_in_up01_startup() -> None:
    """UP01 只重验新候选；旧基线大文件沿用冻结哈希并通过实际行键再核对。"""

    assert runner.LARGE_REGISTERED_SOURCE_KEYS == {
        "base_feature_cache",
        "candidate_feature_cache",
        "source_p3b00_predictions",
    }
    assert "source_up01_legal_candidates" not in runner.LARGE_REGISTERED_SOURCE_KEYS
