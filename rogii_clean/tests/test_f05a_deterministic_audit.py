from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CLEAN_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = CLEAN_ROOT / "scripts" / "audit_f05a_deterministic_results.py"


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_f05a_deterministic_audit_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_runner():
    assert RUNNER_PATH.is_file(), "deterministic F05a audit runner 尚未实现"
    spec = importlib.util.spec_from_file_location(
        "audit_f05a_deterministic_results",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_artifacts(root: Path) -> tuple[Path, Path, Path]:
    artifact_dir = root / "experiment"
    cache_dir = root / "cache"
    b00_dir = root / "b00"
    artifact_dir.mkdir()
    cache_dir.mkdir()
    b00_dir.mkdir()

    well_ids = np.repeat(["well_a", "well_b", "well_c", "well_d"], 2)
    row_indices = np.tile([1, 2], 4)
    folds = np.repeat([0, 1, 2, 3], 2)
    keys = pd.DataFrame({"well_id": well_ids, "row_index": row_indices})

    candidates = keys.copy()
    for feature_index, feature in enumerate(DIRECT_CANDIDATE_COLUMNS):
        candidates[feature] = (
            np.arange(len(candidates), dtype=np.float64) + feature_index + 1.0
        )
    candidate_path = cache_dir / "candidate_feature_cache.parquet"
    candidates.to_parquet(candidate_path, index=False)
    cache_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "completed": True,
                "rows": len(candidates),
                "wells": 4,
                "seed": 42,
                "feature_columns": DIRECT_CANDIDATE_COLUMNS,
                "candidate_cache_sha256": cache_sha256,
                "base_lineage": {"fingerprint": "synthetic"},
            }
        ),
        encoding="utf-8",
    )

    base = keys.copy()
    base["row_index"] = base["row_index"].astype(np.int32)
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        base[feature] = np.arange(len(base), dtype=np.float64) + feature_index
    base["gr_missing"] = np.tile([0.0, 1.0], 4)
    base.to_parquet(b00_dir / "feature_cache.parquet", index=False)

    target = np.arange(len(keys), dtype=np.float64) + 100.0
    predictions = keys.assign(
        fold=folds,
        target_tvt=target,
        pred_tvt=target + 1.0,
        carry_tvt=target + 3.0,
    )
    b00_predictions = keys.assign(
        fold=folds,
        target_tvt=target,
        pred_tvt=target + 2.0,
        carry_tvt=target + 3.0,
    )
    predictions["row_index"] = predictions["row_index"].astype(np.int32)
    b00_predictions["row_index"] = b00_predictions["row_index"].astype(np.int32)
    predictions.to_parquet(artifact_dir / "predictions.parquet", index=False)
    b00_predictions.to_parquet(b00_dir / "predictions.parquet", index=False)

    all_features = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
    for fold_id in range(5):
        fold_dir = artifact_dir / f"fold_{fold_id}"
        fold_dir.mkdir()
        pd.DataFrame(
            {
                "feature": all_features,
                "gain": np.arange(len(all_features), dtype=np.float64) + fold_id,
                "split": np.arange(len(all_features), dtype=np.float64) + 2 * fold_id,
            }
        ).to_csv(fold_dir / "feature_importance.csv", index=False)
    return artifact_dir, cache_dir, b00_dir


def test_run_audit_writes_read_only_quality_slice_bootstrap_and_importance(
    local_tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifact_dir, cache_dir, b00_dir = _write_synthetic_artifacts(local_tmp_path)
    prediction_path = artifact_dir / "predictions.parquet"
    prediction_sha256_before = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    summary = runner.run_audit(
        artifact_dir=artifact_dir,
        candidate_cache_dir=cache_dir,
        b00_dir=b00_dir,
        bootstrap_repeats=20,
    )

    assert summary["rows"] == 8
    assert summary["wells"] == 4
    assert summary["micro_rmse"] == pytest.approx(1.0)
    assert summary["delta_vs_b00"] == pytest.approx(-1.0)
    assert len(pd.read_csv(artifact_dir / "feature_quality.csv")) == len(
        DIRECT_CANDIDATE_COLUMNS
    )
    assert len(pd.read_csv(artifact_dir / "slice_metrics.csv")) == 20
    assert len(pd.read_parquet(artifact_dir / "bootstrap_replicates.parquet")) == 20
    importance = pd.read_csv(artifact_dir / "feature_importance_audit.csv")
    assert set(importance["feature"]) == set(FEATURE_COLUMNS + DIRECT_CANDIDATE_COLUMNS)
    assert importance.loc[
        importance["feature"] == FEATURE_COLUMNS[0], "gain_mean"
    ].item() == pytest.approx(2.0)
    per_fold = pd.read_csv(artifact_dir / "per_fold.csv")
    assert per_fold["fold"].tolist() == [0, 1, 2, 3]
    assert per_fold.notna().all().all()
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == prediction_sha256_before


def test_run_audit_requires_completed_five_fold_predictions(
    local_tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifact_dir = local_tmp_path / "experiment"
    artifact_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="predictions.parquet"):
        runner.run_audit(
            artifact_dir=artifact_dir,
            candidate_cache_dir=local_tmp_path / "cache",
            b00_dir=local_tmp_path / "b00",
            bootstrap_repeats=20,
        )


def test_run_audit_aligns_shuffled_candidate_cache_by_unique_keys(
    local_tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifact_dir, cache_dir, b00_dir = _write_synthetic_artifacts(local_tmp_path)
    candidate_path = cache_dir / "candidate_feature_cache.parquet"
    shuffled = pd.read_parquet(candidate_path).sample(frac=1.0, random_state=17)
    shuffled.to_parquet(candidate_path, index=False)
    metadata_path = cache_dir / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["candidate_cache_sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = runner.run_audit(
        artifact_dir=artifact_dir,
        candidate_cache_dir=cache_dir,
        b00_dir=b00_dir,
        bootstrap_repeats=20,
    )

    assert summary["alignment"]["candidate_prediction_keys_exact"] is True
    assert summary["micro_rmse"] == pytest.approx(1.0)


def test_saved_model_accepts_lightgbm_positional_names_but_rejects_wrong_count(
    local_tmp_path: Path,
) -> None:
    runner = _load_runner()
    expected = ["feature_a", "feature_b", "feature_c"]

    runner.validate_saved_model_feature_names(
        ["Column_0", "Column_1", "Column_2"],
        expected,
    )

    with pytest.raises(ValueError, match="特征列"):
        runner.validate_saved_model_feature_names(
            ["Column_0", "Column_1"],
            expected,
        )


def test_cli_wrapper_can_import_project_when_executed_as_a_file() -> None:
    command = (
        "import runpy; "
        f"runpy.run_path({str(RUNNER_PATH)!r}, run_name='audit_import_only')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=CLEAN_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_lightweight_entry_owns_main_instead_of_forwarding_full_audit() -> None:
    runner = _load_runner()

    assert runner.main.__module__ == "audit_f05a_deterministic_results"
