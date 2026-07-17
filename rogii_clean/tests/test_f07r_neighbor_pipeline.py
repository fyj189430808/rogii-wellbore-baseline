from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_f07r_neighbor_cache import (
    LINEAGE_COLUMNS,
    build_fold_feature_cache,
)
from scripts.run_f07r_neighbor_cv import (
    EXPECTED_CACHE_COLUMNS,
    load_aligned_feature_table,
    mode_to_fold,
    validate_cache_metadata,
)
from src.f07r_neighbor_prior import F07R_FEATURE_COLUMNS
from src.lgbm_features import FEATURE_COLUMNS


def _write_well(
    train_dir: Path,
    well_id: str,
    y_offset: float,
    visible_count: int = 41,
) -> int:
    x = np.arange(0.0, 2500.1, 25.0, dtype=np.float64)
    z = np.full(len(x), 1000.0, dtype=np.float64)
    u = 300.0 + 0.02 * x + 2.0e-6 * x**2 + y_offset * 0.001
    tvt = u - z
    tvt_input = tvt.copy()
    tvt_input[visible_count:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": np.arange(len(x), dtype=np.float64) * 25.0,
            "X": x,
            "Y": np.full(len(x), y_offset),
            "Z": z,
            "GR": np.full(len(x), 60.0),
            "TVT": tvt,
            "TVT_input": tvt_input,
            "ANCC": np.linspace(1.0e6, 2.0e6, len(x)),
            "contact": np.arange(len(x), dtype=np.float64),
        }
    )
    frame.to_csv(train_dir / f"{well_id}__horizontal_well.csv", index=False)
    return int(len(frame) - visible_count)


def _small_registry(train_dir: Path) -> pd.DataFrame:
    definitions = [
        ("val_a", "pad_a", 0, 0.0),
        ("val_b", "pad_d", 0, 300.0),
        ("train_a", "pad_b", 1, 100.0),
        ("train_a2", "pad_b", 1, 125.0),
        ("train_b", "pad_c", 1, 200.0),
    ]
    rows = []
    for well_id, pad_id, fold, y_offset in definitions:
        hidden_rows = _write_well(train_dir, well_id, y_offset)
        rows.append(
            {
                "well_id": well_id,
                "pad_id": pad_id,
                "fold": fold,
                "hidden_rows": hidden_rows,
            }
        )
    return pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_fold0_cache_is_invariant_to_validation_labels_and_records_sources(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    registry = _small_registry(train_dir)

    first = build_fold_feature_cache(
        registry_df=registry,
        raw_train_dir=train_dir,
        artifact_dir=tmp_path / "first",
        outer_fold=0,
        config_fingerprint="unit-test-config",
    )
    assert first["completed"] is True
    first_cache = pd.read_parquet(first["cache_path"])
    first_lineage = pd.read_parquet(first["lineage_path"])

    validation_path = train_dir / "val_a__horizontal_well.csv"
    validation = pd.read_csv(validation_path)
    hidden = validation["TVT_input"].isna()
    validation.loc[hidden, "TVT"] = np.linspace(-1.0e12, 1.0e12, int(hidden.sum()))
    validation.loc[hidden, "ANCC"] *= -9999.0
    validation["contact"] += 1.0e9
    validation.to_csv(validation_path, index=False)

    second = build_fold_feature_cache(
        registry_df=registry,
        raw_train_dir=train_dir,
        artifact_dir=tmp_path / "second",
        outer_fold=0,
        config_fingerprint="unit-test-config",
    )
    second_cache = pd.read_parquet(second["cache_path"])
    second_lineage = pd.read_parquet(second["lineage_path"])

    pd.testing.assert_frame_equal(first_cache, second_cache, check_exact=True)
    pd.testing.assert_frame_equal(first_lineage, second_lineage, check_exact=True)
    assert list(first_cache.columns) == EXPECTED_CACHE_COLUMNS
    assert list(first_lineage.columns) == LINEAGE_COLUMNS
    assert not first_cache.duplicated(["well_id", "row_index"]).any()

    lineage = first_lineage.set_index("target_well_id")
    assert json.loads(lineage.loc["val_a", "allowed_source_well_ids_json"]) == [
        "train_a",
        "train_a2",
        "train_b",
    ]
    assert json.loads(lineage.loc["train_a", "allowed_source_well_ids_json"]) == [
        "train_b"
    ]
    assert lineage.loc["val_a", "target_role"] == "validation"
    assert lineage.loc["train_a", "target_role"] == "outer_train"
    assert set(first_lineage["source_fold_ids_json"]) == {"[1]"}


def test_builder_supports_fold1_and_excludes_its_validation_pad(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    registry = _small_registry(train_dir)

    result = build_fold_feature_cache(
        registry_df=registry,
        raw_train_dir=train_dir,
        artifact_dir=tmp_path / "artifact",
        outer_fold=1,
        config_fingerprint="unit-test-config",
    )

    metadata = _read_json(result["metadata_path"])
    lineage = pd.read_parquet(result["lineage_path"]).set_index("target_well_id")
    assert metadata["outer_fold"] == 1
    assert metadata["source_fold_excluded"] == 1
    assert metadata["validation_pad_ids"] == ["pad_b", "pad_c"]
    assert json.loads(lineage.loc["train_a", "allowed_source_well_ids_json"]) == [
        "val_a",
        "val_b",
    ]
    assert set(lineage["source_fold_ids_json"]) == {"[0]"}


def test_partial_build_resumes_only_matching_per_well_cache(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    registry = _small_registry(train_dir)
    artifact_dir = tmp_path / "artifact"

    first = build_fold_feature_cache(
        registry_df=registry,
        raw_train_dir=train_dir,
        artifact_dir=artifact_dir,
        outer_fold=0,
        config_fingerprint="unit-test-config",
        limit=3,
    )
    second = build_fold_feature_cache(
        registry_df=registry,
        raw_train_dir=train_dir,
        artifact_dir=artifact_dir,
        outer_fold=0,
        config_fingerprint="unit-test-config",
        limit=3,
    )

    assert first["completed"] is False
    assert first["generated_wells"] == 3
    assert second["generated_wells"] == 0
    assert second["reused_wells"] == 3


def test_runner_alignment_accepts_lineage_and_key_dtype_difference(
    tmp_path: Path,
) -> None:
    base = pd.DataFrame(
        {
            "well_id": pd.Series(["001", "001", "002"], dtype="string"),
            "row_index": np.asarray([4, 5, 3], dtype=np.int32),
            "fold": [0, 0, 1],
            "target_delta": [1.0, 2.0, 3.0],
            "true_tvt": [100.0, 101.0, 102.0],
            "last_visible_tvt": [99.0, 99.0, 99.0],
        }
    )
    for feature in FEATURE_COLUMNS:
        if feature not in base:
            base[feature] = 1.0
    neighbor = pd.DataFrame(
        {
            "well_id": ["001", "001", "002"],
            "row_index": np.asarray([4, 5, 3], dtype=np.int64),
            "outer_fold": np.asarray([0, 0, 0], dtype=np.int8),
            "target_pad_id": ["pad_a", "pad_a", "pad_b"],
            "target_role": ["validation", "validation", "outer_train"],
        }
    )
    for feature in F07R_FEATURE_COLUMNS:
        neighbor[feature] = np.asarray([1.0, np.nan, 3.0], dtype=np.float32)
    base_path = tmp_path / "base.parquet"
    neighbor_path = tmp_path / "neighbor.parquet"
    base.to_parquet(base_path, index=False)
    neighbor.to_parquet(neighbor_path, index=False)

    combined = load_aligned_feature_table(
        base_path,
        neighbor_path,
        expected_rows=3,
        expected_outer_fold=0,
    )
    assert len(combined) == 3
    assert list(combined.columns[-len(F07R_FEATURE_COLUMNS) :]) == list(
        F07R_FEATURE_COLUMNS
    )

    neighbor.loc[0, "row_index"] = 999
    neighbor.to_parquet(neighbor_path, index=False)
    with pytest.raises(ValueError, match="键顺序"):
        load_aligned_feature_table(
            base_path,
            neighbor_path,
            expected_rows=3,
            expected_outer_fold=0,
        )


def test_mode_mapping_and_config_freeze_all_fixed_folds() -> None:
    assert mode_to_fold("smoke") is None
    assert [mode_to_fold(f"fold{fold}") for fold in range(5)] == list(range(5))
    with pytest.raises(ValueError, match="未知运行模式"):
        mode_to_fold("all")

    clean_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (clean_root / "configs" / "f07r_neighbor_relative_u_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["feature_columns"] == [*FEATURE_COLUMNS, *F07R_FEATURE_COLUMNS]
    assert config["n_estimators"] == 1734
    assert config["model_seed"] == 29
    assert config["early_stopping"] is False
    assert config["target"] == "target_delta"
    assert config["supported_modes"] == [
        "smoke",
        "fold0",
        "fold1",
        "fold2",
        "fold3",
        "fold4",
    ]
    assert config["legal_target_columns"] == [
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
        "TVT_input",
    ]
    assert "TVT" in config["legal_source_columns"]
    assert "ANCC" not in config["legal_source_columns"]
    assert "surface" not in " ".join(config["legal_source_columns"]).lower()
    assert "{fold}" in config["neighbor_cache_pattern"]
    assert "{fold}" in config["cache_metadata_pattern"]


def test_cache_metadata_is_validated_against_requested_fold() -> None:
    valid_metadata = {
        "experiment_id": "F07R_neighbor_cache_v1",
        "completed": True,
        "outer_fold": 3,
        "wells": 773,
        "rows": 3_783_989,
        "keys_unique": True,
        "feature_columns": list(F07R_FEATURE_COLUMNS),
        "feature_count": len(F07R_FEATURE_COLUMNS),
        "lineage_columns": LINEAGE_COLUMNS,
        "source_fold_excluded": 3,
        "same_well_excluded": True,
        "same_pad_excluded": True,
        "cache_sha256": "abc",
        "lineage_sha256": "ghi",
        "source_set_fingerprint": "def",
        "family_fingerprint": "family",
    }
    validate_cache_metadata(
        valid_metadata,
        expected_wells=773,
        expected_rows=3_783_989,
        expected_outer_fold=3,
    )
    invalid = dict(valid_metadata)
    invalid["outer_fold"] = 2
    with pytest.raises(ValueError, match="请求的 outer fold"):
        validate_cache_metadata(
            invalid,
            expected_wells=773,
            expected_rows=3_783_989,
            expected_outer_fold=3,
        )
