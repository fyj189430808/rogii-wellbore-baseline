from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f05b_internal_stats import F05B_FEATURE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


BUILDER_PATH = CLEAN_ROOT / "scripts" / "build_f05b_internal_stats_cache.py"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_f05b_internal_stats_cv.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "f05b_internal_stats_v1.json"
F05B_COLUMNS = list(F05B_FEATURE_COLUMNS)


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_f05b_pipeline_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_b00_plus_exactly_ten_f05b_features() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert len(F05B_COLUMNS) == 10
    assert config["experiment_id"] == "F05b_internal_stats_v1"
    assert config["feature_count"] == 22
    assert config["feature_columns"] == [*FEATURE_COLUMNS, *F05B_COLUMNS]
    assert config["internal_stat_feature_columns"] == F05B_COLUMNS
    assert config["cache_rebuild_allowed"] is False
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"


def test_builder_reads_only_legal_columns_uses_seed_42_and_resumes(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_script(BUILDER_PATH, "test_build_f05b_internal_stats_cache")
    horizontal_path = local_tmp_path / "w1__horizontal_well.csv"
    typewell_path = local_tmp_path / "w1__typewell.csv"
    cache_path = local_tmp_path / "per_well" / "w1.parquet"
    pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "Z": [10.0, 10.1, 10.2, 10.3],
            "GR": [50.0, 51.0, 52.0, 53.0],
            "TVT_input": [100.0, 100.1, np.nan, np.nan],
            "TVT": [100.0, 100.1, 9999.0, -9999.0],
            "ANCC": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(horizontal_path, index=False)
    pd.DataFrame(
        {
            "TVT": [99.0, 100.0, 101.0],
            "GR": [49.0, 50.0, 51.0],
            "Geology": ["A", "B", "C"],
        }
    ).to_csv(typewell_path, index=False)

    calls: list[int] = []

    def fake_generator(horizontal: pd.DataFrame, typewell: pd.DataFrame, seed: int):
        assert list(horizontal.columns) == ["MD", "Z", "GR", "TVT_input"]
        assert list(typewell.columns) == ["TVT", "GR"]
        assert seed == 42
        calls.append(seed)
        output = {"row_index": [2, 3]}
        for feature_index, feature_name in enumerate(F05B_FEATURE_COLUMNS):
            output[feature_name] = [feature_index + 0.25, feature_index + 0.50]
        return pd.DataFrame(output)

    monkeypatch.setattr(builder, "build_internal_stats_features", fake_generator)
    first = builder.build_f05b_well_cache(
        well_id="w1",
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint="f05b-w1",
        expected_hidden_rows=2,
    )
    assert first["status"] == "generated"
    assert calls == [42]

    def must_not_run(*args, **kwargs):
        raise AssertionError("匹配指纹的单井 F05b 缓存不应重新生成")

    monkeypatch.setattr(builder, "build_internal_stats_features", must_not_run)
    second = builder.build_f05b_well_cache(
        well_id="w1",
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint="f05b-w1",
        expected_hidden_rows=2,
    )
    assert second["status"] == "reused"


def test_runner_rejects_wrong_key_order_and_never_rebuilds_missing_cache(
    local_tmp_path: Path,
) -> None:
    runner = _load_script(RUNNER_PATH, "test_run_f05b_internal_stats_cv")
    base_rows = pd.DataFrame(
        {"well_id": ["well_a", "well_b"], "row_index": [10, 20]}
    )
    stats_rows = pd.DataFrame(
        {"well_id": ["well_b", "well_a"], "row_index": [20, 10]}
    )
    with pytest.raises(ValueError, match="键顺序"):
        runner.validate_exact_key_alignment(base_rows, stats_rows)

    cache_path = local_tmp_path / "internal_stats_cache.parquet"
    metadata_path = local_tmp_path / "meta.json"
    with pytest.raises(FileNotFoundError, match="不会重建"):
        runner.require_internal_stats_cache(cache_path, metadata_path)
    assert not cache_path.exists()
    assert not metadata_path.exists()
