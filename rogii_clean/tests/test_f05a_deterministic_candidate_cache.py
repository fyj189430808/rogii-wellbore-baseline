from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.build_f05a_deterministic_candidate_cache import (
    build_base_lineage,
    build_well_cache,
    build_well_lineage,
    merge_per_well_caches,
    per_well_cache_path,
)


def test_lineage_fingerprint_tracks_code_parameters_seed_and_raw_files(
    tmp_path: Path,
) -> None:
    generator_path = tmp_path / "generator.py"
    parameter_path = tmp_path / "parameters.json"
    registry_path = tmp_path / "folds.csv"
    raw_train_dir = tmp_path / "raw"
    raw_train_dir.mkdir()
    generator_path.write_text("VERSION = 1\n", encoding="utf-8")
    parameter_path.write_text('{"particle_count": 600}\n', encoding="utf-8")
    registry_path.write_text("well_id,hidden_rows\nw1,2\n", encoding="utf-8")

    horizontal_path = raw_train_dir / "w1__horizontal_well.csv"
    typewell_path = raw_train_dir / "w1__typewell.csv"
    horizontal_path.write_text("MD,Z,GR,TVT_input\n0,1,2,3\n", encoding="utf-8")
    typewell_path.write_text("TVT,GR\n1,2\n", encoding="utf-8")

    base = build_base_lineage(
        generator_path,
        parameter_path,
        registry_path,
        raw_train_dir,
        seed=42,
    )
    repeated = build_base_lineage(
        generator_path,
        parameter_path,
        registry_path,
        raw_train_dir,
        seed=42,
    )
    different_seed = build_base_lineage(
        generator_path,
        parameter_path,
        registry_path,
        raw_train_dir,
        seed=43,
    )

    assert base == repeated
    assert base["fingerprint"] != different_seed["fingerprint"]
    assert base["generator_code_sha256"]
    assert base["parameter_json_sha256"]
    assert base["seed"] == 42
    assert base["raw_path_signature"]["path"] == str(raw_train_dir.resolve())

    first_well_lineage = build_well_lineage(base, horizontal_path, typewell_path)
    horizontal_path.write_text(
        "MD,Z,GR,TVT_input\n0,1,2,3\n1,2,3,4\n",
        encoding="utf-8",
    )
    changed_raw_lineage = build_well_lineage(base, horizontal_path, typewell_path)
    assert first_well_lineage["fingerprint"] != changed_raw_lineage["fingerprint"]

    generator_path.write_text("VERSION = 2\n", encoding="utf-8")
    changed_code = build_base_lineage(
        generator_path,
        parameter_path,
        registry_path,
        raw_train_dir,
        seed=42,
    )
    assert base["fingerprint"] != changed_code["fingerprint"]

    parameter_path.write_text('{"particle_count": 601}\n', encoding="utf-8")
    changed_parameters = build_base_lineage(
        generator_path,
        parameter_path,
        registry_path,
        raw_train_dir,
        seed=42,
    )
    assert changed_code["fingerprint"] != changed_parameters["fingerprint"]


def test_well_cache_uses_only_legal_columns_seed_42_and_resumes(
    tmp_path: Path,
) -> None:
    horizontal_path = tmp_path / "w1__horizontal_well.csv"
    typewell_path = tmp_path / "w1__typewell.csv"
    cache_path = tmp_path / "per_well" / "w1.parquet"
    pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "Z": [10.0, 10.1, 10.2, 10.3],
            "GR": [50.0, 51.0, 52.0, 53.0],
            "TVT_input": [100.0, 100.1, None, None],
            "TVT": [100.0, 100.1, 999.0, 999.0],
            "ANCC": [1.0, 1.0, 1.0, 1.0],
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
        return pd.DataFrame({"row_index": [2, 3], "candidate": [0.25, 0.50]})

    first = build_well_cache(
        well_id="w1",
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint="fingerprint-w1",
        expected_hidden_rows=2,
        feature_columns=["candidate"],
        seed=42,
        rebuild=False,
        generator=fake_generator,
    )
    assert first["status"] == "generated"
    assert calls == [42]
    cached = pd.read_parquet(cache_path)
    assert list(cached.columns) == [
        "well_id",
        "row_index",
        "candidate",
        "_cache_fingerprint",
    ]

    def must_not_run(*args, **kwargs):
        raise AssertionError("匹配指纹的单井缓存不应重新生成")

    second = build_well_cache(
        well_id="w1",
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint="fingerprint-w1",
        expected_hidden_rows=2,
        feature_columns=["candidate"],
        seed=42,
        rebuild=False,
        generator=must_not_run,
    )
    assert second["status"] == "reused"


def test_merge_uses_registry_order_and_validates_unique_keys(tmp_path: Path) -> None:
    cache_dir = tmp_path / "per_well"
    cache_dir.mkdir()
    registry = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a"],
            "hidden_rows": [2, 3],
        }
    )
    fingerprints = {"well_b": "fp-b", "well_a": "fp-a"}

    pd.DataFrame(
        {
            "well_id": ["well_a"] * 3,
            "row_index": [7, 8, 9],
            "candidate": [0.7, 0.8, 0.9],
            "_cache_fingerprint": ["fp-a"] * 3,
        }
    ).to_parquet(per_well_cache_path(cache_dir, "well_a"), index=False)
    pd.DataFrame(
        {
            "well_id": ["well_b"] * 2,
            "row_index": [4, 5],
            "candidate": [0.4, 0.5],
            "_cache_fingerprint": ["fp-b"] * 2,
        }
    ).to_parquet(per_well_cache_path(cache_dir, "well_b"), index=False)

    output_path = tmp_path / "candidate_feature_cache.parquet"
    summary = merge_per_well_caches(
        registry=registry,
        cache_dir=cache_dir,
        expected_fingerprints=fingerprints,
        output_path=output_path,
        feature_columns=["candidate"],
        expected_wells=2,
        expected_rows=5,
    )
    merged = pd.read_parquet(output_path)

    assert merged["well_id"].tolist() == ["well_b", "well_b", "well_a", "well_a", "well_a"]
    assert merged["row_index"].tolist() == [4, 5, 7, 8, 9]
    assert list(merged.columns) == ["well_id", "row_index", "candidate"]
    assert summary == {"wells": 2, "rows": 5, "keys_unique": True}

    broken = pd.read_parquet(per_well_cache_path(cache_dir, "well_a"))
    broken["row_index"] = [7, 7, 9]
    broken.to_parquet(per_well_cache_path(cache_dir, "well_a"), index=False)
    with pytest.raises(ValueError, match="重复"):
        merge_per_well_caches(
            registry=registry,
            cache_dir=cache_dir,
            expected_fingerprints=fingerprints,
            output_path=output_path,
            feature_columns=["candidate"],
            expected_wells=2,
            expected_rows=5,
        )
