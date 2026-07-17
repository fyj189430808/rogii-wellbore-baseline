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

from src.lgbm_features import FEATURE_COLUMNS


BUILDER_PATH = CLEAN_ROOT / "scripts" / "build_f06_prefix_holdout_cache.py"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_f06_prefix_holdout_cv.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "f06_prefix_holdout_v1.json"


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_f06_pipeline_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_files_and_frozen_config_exist() -> None:
    from src.f06_prefix_holdout import F06_FEATURE_COLUMNS

    assert BUILDER_PATH.is_file()
    assert RUNNER_PATH.is_file()
    assert CONFIG_PATH.is_file()

    f06_columns = list(F06_FEATURE_COLUMNS)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert f06_columns
    assert config["experiment_id"] == "F06_prefix_holdout_v1"
    assert config["feature_columns"] == [*FEATURE_COLUMNS, *f06_columns]
    assert config["prefix_holdout_feature_columns"] == f06_columns
    assert config["feature_count"] == len(FEATURE_COLUMNS) + len(f06_columns)
    assert config["cache_rebuild_allowed"] is False
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"


def test_builder_reads_only_legal_columns_uses_seed_42_and_resumes(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.f06_prefix_holdout import F06_FEATURE_COLUMNS

    builder = _load_script(BUILDER_PATH, "test_build_f06_prefix_holdout_cache")
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
        output: dict[str, list[float]] = {}
        for feature_index, feature_name in enumerate(F06_FEATURE_COLUMNS):
            # 不支持的 horizon 按 core 合同输出 NaN；缓存必须保留它交给 LightGBM。
            feature_value = (
                np.nan if feature_index == 0 else float(feature_index) + 0.25
            )
            output[feature_name] = [feature_value]
        return pd.DataFrame(output)

    monkeypatch.setattr(builder, "build_prefix_holdout_features", fake_generator)
    first = builder.build_f06_well_cache(
        well_id="w1",
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint="f06-w1",
        expected_hidden_rows=2,
    )
    assert first["status"] == "generated"
    assert calls == [42]

    cached = pd.read_parquet(cache_path)
    for feature_name in F06_FEATURE_COLUMNS:
        assert cached[feature_name].nunique(dropna=False) == 1

    def must_not_run(*args, **kwargs):
        raise AssertionError("匹配指纹的单井 F06 缓存不应重新生成")

    monkeypatch.setattr(builder, "build_prefix_holdout_features", must_not_run)
    second = builder.build_f06_well_cache(
        well_id="w1",
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint="f06-w1",
        expected_hidden_rows=2,
    )
    assert second["status"] == "reused"


def test_runner_rejects_wrong_key_order_and_never_rebuilds_missing_cache(
    local_tmp_path: Path,
) -> None:
    runner = _load_script(RUNNER_PATH, "test_run_f06_prefix_holdout_cv")
    base_rows = pd.DataFrame(
        {"well_id": ["well_a", "well_b"], "row_index": [10, 20]}
    )
    holdout_rows = pd.DataFrame(
        {"well_id": ["well_b", "well_a"], "row_index": [20, 10]}
    )
    with pytest.raises(ValueError, match="键顺序"):
        runner.validate_exact_key_alignment(base_rows, holdout_rows)

    cache_path = local_tmp_path / "prefix_holdout_cache.parquet"
    metadata_path = local_tmp_path / "meta.json"
    with pytest.raises(FileNotFoundError, match="不会重建"):
        runner.require_prefix_holdout_cache(cache_path, metadata_path)
    assert not cache_path.exists()
    assert not metadata_path.exists()


def test_runner_preserves_nan_as_native_lightgbm_missing_value(
    local_tmp_path: Path,
) -> None:
    from src.f06_prefix_holdout import F06_FEATURE_COLUMNS

    runner = _load_script(RUNNER_PATH, "test_run_f06_prefix_holdout_cv_nan")
    base_path = local_tmp_path / "base.parquet"
    cache_path = local_tmp_path / "f06.parquet"
    base_rows = pd.DataFrame(
        {"well_id": ["well_a", "well_a"], "row_index": [10, 11]}
    )
    feature_data: dict[str, list[float] | list[str] | list[int]] = {
        "well_id": ["well_a", "well_a"],
        "row_index": [10, 11],
    }
    for feature_index, feature_name in enumerate(F06_FEATURE_COLUMNS):
        feature_value = np.nan if feature_index == 0 else float(feature_index)
        feature_data[feature_name] = [feature_value, feature_value]
    base_rows.to_parquet(base_path, index=False)
    pd.DataFrame(feature_data).to_parquet(cache_path, index=False)

    loaded = runner.load_aligned_feature_table(base_path, cache_path, 2)

    assert loaded[F06_FEATURE_COLUMNS[0]].isna().all()


def test_runner_does_not_copy_the_complete_f06_matrix_during_alignment(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全量 174 列不能为有限值检查或重置索引再复制一次。"""

    from src.f06_prefix_holdout import F06_FEATURE_COLUMNS

    runner = _load_script(RUNNER_PATH, "test_run_f06_prefix_holdout_cv_memory")
    base_path = local_tmp_path / "base.parquet"
    cache_path = local_tmp_path / "f06.parquet"
    base_rows = pd.DataFrame(
        {"well_id": ["well_a", "well_a"], "row_index": [10, 11]}
    )
    feature_data: dict[str, list[float] | list[str] | list[int]] = {
        "well_id": ["well_a", "well_a"],
        "row_index": [10, 11],
    }
    for feature_index, feature_name in enumerate(F06_FEATURE_COLUMNS):
        feature_data[feature_name] = [float(feature_index), float(feature_index)]
    base_rows.to_parquet(base_path, index=False)
    pd.DataFrame(feature_data).to_parquet(cache_path, index=False)

    original_to_numpy = pd.DataFrame.to_numpy
    original_reset_index = pd.DataFrame.reset_index

    def guarded_to_numpy(frame: pd.DataFrame, *args, **kwargs):
        if list(frame.columns) == list(F06_FEATURE_COLUMNS):
            raise AssertionError("禁止把完整 F06 特征块复制为 NumPy 数组")
        return original_to_numpy(frame, *args, **kwargs)

    def guarded_reset_index(frame: pd.DataFrame, *args, **kwargs):
        if list(frame.columns) == list(F06_FEATURE_COLUMNS):
            raise AssertionError("禁止为完整 F06 特征块 reset_index 深拷贝")
        return original_reset_index(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_numpy", guarded_to_numpy)
    monkeypatch.setattr(pd.DataFrame, "reset_index", guarded_reset_index)

    loaded = runner.load_aligned_feature_table(base_path, cache_path, 2)

    assert len(loaded) == 2
    assert list(loaded.columns[-len(F06_FEATURE_COLUMNS) :]) == list(
        F06_FEATURE_COLUMNS
    )
