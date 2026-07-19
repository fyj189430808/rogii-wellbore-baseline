"""二阶段 P2-CV00 按井五折与缓存折号重映射测试。"""

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

from src.fold_split import (  # noqa: E402
    add_balanced_well_fold_ids,
    build_balanced_well_fold_registry,
)


TRAIN_DIR = CLEAN_ROOT.parent / "input" / "data" / "raw" / "train"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_cv00_group5_c01_v1.json"
RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_p2_cv00_group5_c01.py"
FOLD_GENERATOR_PATH = CLEAN_ROOT / "scripts" / "make_balanced_well_folds.py"


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    """把临时目录建在工作区旁，避开当前 Windows 用户临时目录权限问题。"""

    with tempfile.TemporaryDirectory(
        prefix=".pytest_p2_cv00_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _synthetic_well_summary() -> pd.DataFrame:
    """构造一个只含分折必需字段的小型井级表。"""

    return pd.DataFrame(
        {
            "well_id": ["well_c", "well_a", "well_e", "well_b", "well_d"],
            "representative_x": [30.0, 10.0, 50.0, 20.0, 40.0],
            "representative_y": [3.0, 1.0, 5.0, 2.0, 4.0],
            "total_rows": [15, 11, 18, 13, 17],
            "visible_rows": [5, 5, 5, 5, 5],
            "hidden_rows": [10, 6, 13, 8, 12],
        }
    )


def test_balanced_well_folds_are_deterministic_and_do_not_mutate_input() -> None:
    """输入行序变化不能改变逐井折号，函数也不能修改调用者的表。"""

    original = _synthetic_well_summary()
    original_before_call = original.copy(deep=True)
    shuffled = original.sample(frac=1.0, random_state=19).reset_index(drop=True)

    first = add_balanced_well_fold_ids(original, n_splits=3)
    second = add_balanced_well_fold_ids(shuffled, n_splits=3)

    pd.testing.assert_frame_equal(original, original_before_call)
    first_mapping = first.set_index("well_id")["fold"].sort_index()
    second_mapping = second.set_index("well_id")["fold"].sort_index()
    pd.testing.assert_series_equal(first_mapping, second_mapping)
    assert first["pad_id"].astype(str).equals(first["well_id"].astype(str))
    assert first.groupby("well_id")["fold"].nunique().max() == 1
    assert set(first["fold"].astype(int)) == {0, 1, 2}


@pytest.mark.parametrize("n_splits", [1, 6])
def test_balanced_well_folds_reject_invalid_split_count(n_splits: int) -> None:
    """折数必须至少为二且不能超过井数。"""

    with pytest.raises(ValueError, match="n_splits"):
        add_balanced_well_fold_ids(_synthetic_well_summary(), n_splits=n_splits)


def test_balanced_well_folds_reject_duplicate_well_ids() -> None:
    """一井多行会破坏验证单位，必须显式拒绝。"""

    duplicated = pd.concat(
        [_synthetic_well_summary(), _synthetic_well_summary().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="well_id"):
        add_balanced_well_fold_ids(duplicated, n_splits=3)


def test_full_balanced_well_registry_invariants() -> None:
    """从原始 773 井重建注册表并冻结二阶段五折规模。"""

    registry = build_balanced_well_fold_registry(TRAIN_DIR, n_splits=5)

    assert len(registry) == 773
    assert registry["well_id"].nunique() == 773
    assert registry["pad_id"].nunique() == 773
    assert registry["pad_id"].astype(str).equals(registry["well_id"].astype(str))
    assert int(registry["hidden_rows"].sum()) == 3_783_989
    assert set(registry["fold"].astype(int)) == {0, 1, 2, 3, 4}

    actual = (
        registry.groupby("fold", as_index=True)
        .agg(wells=("well_id", "count"), hidden_rows=("hidden_rows", "sum"))
        .astype(int)
        .to_dict(orient="index")
    )
    expected = {
        0: {"wells": 155, "hidden_rows": 757_738},
        1: {"wells": 155, "hidden_rows": 756_650},
        2: {"wells": 154, "hidden_rows": 756_255},
        3: {"wells": 155, "hidden_rows": 757_101},
        4: {"wells": 154, "hidden_rows": 756_245},
    }
    assert actual == expected


def _load_module(module_name: str, module_path: Path):
    """从指定脚本路径加载模块，供命令行脚本的纯函数测试使用。"""

    assert module_path.is_file(), f"脚本尚未实现：{module_path.name}"
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_fold_generator_writes_registry_metadata_and_matching_sha(
    local_tmp_path: Path,
) -> None:
    """注册表 CSV、说明字段和文件 SHA 必须在一次写入中保持一致。"""

    generator = _load_module("make_balanced_well_folds", FOLD_GENERATOR_PATH)
    registry = add_balanced_well_fold_ids(_synthetic_well_summary(), n_splits=3)
    output_path = local_tmp_path / "balanced_well_test.csv"

    metadata = generator.write_registry_artifacts(
        registry,
        output_path=output_path,
        train_dir=local_tmp_path,
        n_splits=3,
    )

    metadata_path = output_path.with_suffix(".meta.json")
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert output_path.is_file()
    assert saved_metadata == metadata
    assert metadata["config_version"] == "balanced_well_5fold_v1"
    assert metadata["validation_unit"] == "complete_well"
    assert metadata["grouping"] == "none"
    assert metadata["pad_id_semantics"] == "well_id_compatibility_alias"
    assert metadata["registry_sha256"] == generator.sha256_file(output_path)
    assert metadata["wells"] == 5
    assert metadata["hidden_rows"] == 49


def _load_runner():
    """按文件路径加载独立 P2 runner，避免把 scripts 变成新框架。"""

    return _load_module("run_p2_cv00_group5_c01", RUNNER_PATH)


def test_remap_replaces_only_stale_fold_and_preserves_row_order() -> None:
    """新注册表必须覆盖旧折号，同时保持键、特征和行序不变。"""

    runner = _load_runner()
    feature_table = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_b"],
            "row_index": [0, 0, 1],
            "fold": [4, 4, 4],
            "target_delta": [1.0, 2.0, 3.0],
            "feature_value": [10.0, 20.0, 30.0],
        }
    )
    registry = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b"],
            "pad_id": ["well_a", "well_b"],
            "fold": [0, 1],
            "hidden_rows": [1, 2],
        }
    )

    remapped = runner.remap_feature_table_folds(feature_table, registry)

    assert remapped["fold"].tolist() == [1, 0, 1]
    pd.testing.assert_frame_equal(
        remapped.drop(columns="fold"),
        feature_table.drop(columns="fold"),
    )
    assert feature_table["fold"].tolist() == [4, 4, 4]


@pytest.mark.parametrize(
    ("feature_wells", "registry_wells", "message"),
    [
        (["well_a", "well_c"], ["well_a", "well_b"], "井集合"),
        (["well_a", "well_b"], ["well_a", "well_a"], "重复 well_id"),
    ],
)
def test_remap_rejects_incompatible_well_registry(
    feature_wells: list[str],
    registry_wells: list[str],
    message: str,
) -> None:
    """漏井、多井或重复注册不能被静默映射。"""

    runner = _load_runner()
    feature_table = pd.DataFrame(
        {
            "well_id": feature_wells,
            "row_index": [0, 0],
            "fold": [4, 4],
        }
    )
    registry = pd.DataFrame(
        {
            "well_id": registry_wells,
            "pad_id": registry_wells,
            "fold": [0, 1],
            "hidden_rows": [1, 1],
        }
    )

    with pytest.raises(ValueError, match=message):
        runner.remap_feature_table_folds(feature_table, registry)


def test_remap_rejects_per_well_row_count_mismatch() -> None:
    """缓存中每井评价行数必须与注册表冻结值一致。"""

    runner = _load_runner()
    feature_table = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b", "well_b"],
            "row_index": [0, 0, 1],
            "fold": [4, 4, 4],
        }
    )
    registry = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b"],
            "pad_id": ["well_a", "well_b"],
            "fold": [0, 1],
            "hidden_rows": [2, 1],
        }
    )

    with pytest.raises(ValueError, match="评价行数"):
        runner.remap_feature_table_folds(feature_table, registry)


def test_p2_config_keeps_exact_c01_features_and_changes_only_cv() -> None:
    """二阶段配置必须冻结 C01 特征和模型，只替换 fold 注册表。"""

    assert CONFIG_PATH.is_file(), "P2-CV00 配置尚未实现"
    p2_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    old_config_path = CLEAN_ROOT / "configs" / "f05a_deterministic_candidates_v1.json"
    old_config = json.loads(old_config_path.read_text(encoding="utf-8"))

    assert p2_config["experiment_id"] == "P2_CV00_group5_c01_v1"
    assert p2_config["feature_columns"] == old_config["feature_columns"]
    assert p2_config["candidate_feature_columns"] == old_config[
        "candidate_feature_columns"
    ]
    assert p2_config["model_config"] == old_config["model_config"]
    assert p2_config["base_feature_cache"] == old_config["base_feature_cache"]
    assert p2_config["candidate_feature_cache"] == old_config[
        "candidate_feature_cache"
    ]
    assert p2_config["fold_registry"] == (
        "artifacts/folds/balanced_well_5fold_v1.csv"
    )
    assert p2_config["fold_remapped_by_well_id"] is True
    assert p2_config["cached_fold_column_ignored"] is True
