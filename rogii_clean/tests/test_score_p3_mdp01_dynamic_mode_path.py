"""P3-MDP01 独立评分器的完整性门和指标测试。"""

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
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_mdp01_dynamic_mode_path as runner
from scripts import score_p3_mdp01_dynamic_mode_path as scorer


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mdp01_score_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _legal_frame(well_id: str, fold: int, fingerprint: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [well_id, well_id],
            "fold": [fold, fold],
            "row_index": [3, 4],
            "md": [10.0, 11.0],
            "last_visible_tvt": [100.0, 100.0],
            "mdp_dynamic_delta": [1.0, 2.0],
            "mdp_selected_state": [1, 1],
            "mdp_margin": [np.nan, np.nan],
            "mdp_safe_10_delta": [0.1, 0.2],
            "mdp_safe_25_delta": [0.25, 0.5],
            "mdp_gr_shift_dynamic_delta": [2.0, 3.0],
            "mdp_gr_shift_selected_state": [2, 2],
            "mdp_gr_shift_safe_10_delta": [0.2, 0.3],
            "mdp_gr_shift_safe_25_delta": [0.5, 0.75],
            "mdp_cost_permutation_dynamic_delta": [3.0, 4.0],
            "mdp_cost_permutation_selected_state": [3, 3],
            "mdp_cost_permutation_safe_10_delta": [0.3, 0.4],
            "mdp_cost_permutation_safe_25_delta": [0.75, 1.0],
            "_cache_fingerprint": [fingerprint, fingerprint],
        }
    )[runner.LEGAL_CACHE_COLUMNS]


def test_scorer_refuses_incomplete_legal_generation_before_target_loader(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = workspace_tmp_path
    called = False

    def forbidden_target_loader(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal called
        called = True
        raise AssertionError("完整性检查之前不应读取 target_tvt")

    monkeypatch.setattr(scorer, "load_target_rows", forbidden_target_loader)
    registry = pd.DataFrame(
        {"well_id": ["00001234"], "fold": [1], "hidden_rows": [2]}
    )

    with pytest.raises(ValueError, match="legal generation"):
        scorer.score_experiment(
            registry=registry,
            artifact_dir=tmp_path,
            p2_predictions_path=tmp_path / "predictions.parquet",
            run_mode="folds12",
        )
    assert called is False


def test_score_one_well_uses_fixed_paths_without_oracle_selection() -> None:
    fingerprint = "fp"
    legal = _legal_frame("00001234", 1, fingerprint)
    target = pd.DataFrame(
        {
            "well_id": ["00001234", "00001234"],
            "fold": [1, 1],
            "row_index": [3, 4],
            "target_tvt": [101.0, 102.0],
            "pred_tvt": [100.0, 100.0],
        }
    )

    record = scorer.score_one_well(
        target,
        legal,
        well_id="00001234",
        fold=1,
        hidden_rows=2,
        fingerprint=fingerprint,
    )

    assert record["p2_sse"] == 5.0
    assert record["mdp_dynamic_sse"] == 0.0
    assert record["mdp_gr_shift_dynamic_sse"] == 2.0
    assert record["mdp_cost_permutation_dynamic_sse"] == 8.0


def test_folds12_scorer_accepts_completed_all_run_as_legal_superset(
    workspace_tmp_path: Path,
) -> None:
    fingerprint = "a" * 64
    artifact_dir = workspace_tmp_path / "artifact"
    cache_path = artifact_dir / "legal_cache" / "00001234.parquet"
    runtime_path = artifact_dir / "legal_runtime" / "00001234.json"
    cache_path.parent.mkdir(parents=True)
    runtime_path.parent.mkdir(parents=True)
    legal = _legal_frame("00001234", 1, fingerprint)
    legal.to_parquet(cache_path, index=False)
    runtime_path.write_text(
        json.dumps(
            {
                "experiment_fingerprint": fingerprint,
                "well_id": "00001234",
                "fold": 1,
                "hidden_rows": 2,
                "hidden_tvt_read": False,
                "cache_sha256": runner.file_sha256(cache_path),
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "runtime_all.json").write_text(
        json.dumps(
            {
                "experiment_id": scorer.EXPERIMENT_ID,
                "run_mode": "all",
                "completed": True,
                "experiment_fingerprint": fingerprint,
                "selected_wells": 657,
                "selected_rows": 3_211_872,
                "completed_wells": 657,
                "completed_rows": 3_211_872,
                "shadow_overlap": 0,
                "hidden_tvt_read": False,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    registry = pd.DataFrame(
        {"well_id": ["00001234"], "fold": [1], "hidden_rows": [2]}
    )

    run_dir, observed_fingerprint, paths = scorer.validate_legal_generation_complete(
        registry, artifact_dir, "folds12"
    )

    assert run_dir == artifact_dir
    assert observed_fingerprint == fingerprint
    assert paths == {"00001234": cache_path}


def test_metric_summary_reports_pooled_and_fold_improvement() -> None:
    per_well = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [1, 2],
            "rows": [2, 2],
            "p2_sse": [8.0, 8.0],
            "p2_rmse": [2.0, 2.0],
            "mdp_dynamic_sse": [2.0, 2.0],
            "mdp_dynamic_rmse": [1.0, 1.0],
        }
    )

    metrics, per_fold = scorer.summarize_method(
        per_well,
        method="mdp_dynamic",
        baseline_method="p2",
    )

    assert metrics["rmse"] == 1.0
    assert metrics["baseline_rmse"] == 2.0
    assert metrics["improvement_ft"] == 1.0
    assert per_fold["improvement_ft"].tolist() == [1.0, 1.0]
