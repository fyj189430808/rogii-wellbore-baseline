from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


CLEAN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CLEAN_ROOT / "configs" / "f05a_deterministic_candidates_v1.json"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_f05a_deterministic_candidates_cv.py"


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_f05a_deterministic_runner_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_runner():
    assert RUNNER_PATH.is_file(), "deterministic candidate CV runner 尚未实现"
    spec = importlib.util.spec_from_file_location(
        "run_f05a_deterministic_candidates_cv",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_freezes_b00_plus_direct_candidate_columns() -> None:
    assert CONFIG_PATH.is_file(), "deterministic candidate CV 配置尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["experiment_id"] == "F05a_deterministic_candidates_v1"
    assert config["feature_columns"] == [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
    assert config["candidate_feature_columns"] == DIRECT_CANDIDATE_COLUMNS
    assert config["candidate_cache_rebuild_allowed"] is False


def test_missing_candidate_cache_fails_without_creating_files(
    local_tmp_path: Path,
) -> None:
    runner = _load_runner()
    cache_path = local_tmp_path / "candidate_feature_cache.parquet"
    metadata_path = local_tmp_path / "meta.json"

    with pytest.raises(FileNotFoundError, match="不会重建"):
        runner.require_candidate_cache_files(cache_path, metadata_path)

    assert not cache_path.exists()
    assert not metadata_path.exists()


def test_candidate_keys_must_match_b00_in_exact_order() -> None:
    runner = _load_runner()
    base_rows = pd.DataFrame(
        {"well_id": ["well_a", "well_b"], "row_index": [10, 20]}
    )
    candidate_rows = pd.DataFrame(
        {"well_id": ["well_b", "well_a"], "row_index": [20, 10]}
    )

    with pytest.raises(ValueError, match="键顺序"):
        runner.validate_exact_key_alignment(base_rows, candidate_rows)


def test_cache_fingerprint_changes_with_metadata_and_file_signature(
    local_tmp_path: Path,
) -> None:
    runner = _load_runner()
    cache_path = local_tmp_path / "candidate_feature_cache.parquet"
    metadata_path = local_tmp_path / "meta.json"
    cache_path.write_bytes(b"first-cache")
    metadata_path.write_text(
        json.dumps({"rows": 2, "feature_columns": DIRECT_CANDIDATE_COLUMNS}),
        encoding="utf-8",
    )

    first = runner.build_candidate_cache_fingerprint(cache_path, metadata_path)
    metadata_path.write_text(
        json.dumps({"rows": 3, "feature_columns": DIRECT_CANDIDATE_COLUMNS}),
        encoding="utf-8",
    )
    metadata_changed = runner.build_candidate_cache_fingerprint(
        cache_path,
        metadata_path,
    )
    cache_path.write_bytes(b"second-cache-with-different-size")
    file_changed = runner.build_candidate_cache_fingerprint(
        cache_path,
        metadata_path,
    )

    assert metadata_changed != first
    assert file_changed != metadata_changed
