"""P3-PFM02 Task 1: no-label three-mode PF cache contract."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

# 无论 pytest 从仓库根目录还是 rogii_clean 目录启动，都只导入本项目的 scripts/src。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import generate_p3_pfm02_mode_path_cache as cache_generator
from src.p3_pfm01_ordered_pf_modes import cluster_seed_descriptors, summarize_ordered_modes


SHARED_FINGERPRINT = "91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0"
EXPECTED_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    "pf_mode_low_delta",
    "pf_mode_middle_delta",
    "pf_mode_high_delta",
    "_cache_fingerprint",
]


def test_cli_entrypoint_resolves_src_from_workspace_parent() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    script_path = workspace_root / "rogii_clean" / "scripts" / "generate_p3_pfm02_mode_path_cache.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-wells" in result.stdout


def _paths_for(well_offset: float, rows: int = 5) -> np.ndarray:
    bases = np.repeat(np.asarray([-10.0, 0.0, 10.0]), repeats=[43, 42, 43])
    depth = np.linspace(0.0, 1.0, rows)
    return bases[:, None] + well_offset + depth[None, :] + (np.arange(128) % 3)[:, None] * 0.01


def _write_shared(path: Path, *, offset: float = 0.0, rows: int = 5, seed_ids: np.ndarray | None = None, row_index: np.ndarray | None = None, final_ll: np.ndarray | None = None, finite: bool = True) -> None:
    seed_delta = _paths_for(offset, rows)
    if not finite:
        seed_delta[0, 0] = np.nan
    np.savez(
        path,
        seed_delta=seed_delta,
        final_ll=np.linspace(-2.0, 2.0, 128) if final_ll is None else final_ll,
        seed_ids=np.arange(128, dtype=np.int64) if seed_ids is None else seed_ids,
        row_index=np.arange(rows, dtype=np.int64) if row_index is None else row_index,
        last_tvt=np.asarray([100.0 + offset]),
    )


def _make_inputs(tmp_path: Path, wells: tuple[str, ...] = ("well-a", "well-b")) -> tuple[Path, Path, Path]:
    folds = pd.DataFrame(
        {
            "well_id": list(wells) + ["shadow-well"],
            "fold": [0, 1, 2],
            "total_rows": [9, 9, 9],
            "visible_rows": [4, 4, 4],
            "hidden_rows": [5, 5, 5],
        }
    )
    folds_path = tmp_path / "folds.csv"
    folds.to_csv(folds_path, index=False)
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame({"well_id": ["shadow-well"], "is_shadow": [True]}).to_csv(shadow_path, index=False)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    for index, well_id in enumerate((*wells, "shadow-well")):
        _write_shared(
            shared_dir / f"{well_id}.npz",
            offset=float(index),
            row_index=np.arange(1442, 1447, dtype=np.int64),
        )
    return folds_path, shadow_path, shared_dir


def _generate(tmp_path: Path, *, max_wells: int | None = None) -> Path:
    folds_path, shadow_path, shared_dir = _make_inputs(tmp_path)
    output_dir = tmp_path / "output"
    effective_max_wells = 2 if max_wells is None else max_wells
    cache_generator.generate_mode_path_cache(
        folds_path=folds_path,
        shadow_path=shadow_path,
        shared_dir=shared_dir,
        output_dir=output_dir,
        max_wells=effective_max_wells,
        workers=1,
    )
    return output_dir / f"smoke_{effective_max_wells}"


def test_synthetic_paths_match_pfm01_core_bit_for_bit(tmp_path: Path) -> None:
    artifact_dir = _generate(tmp_path)
    cache = pq.read_table(artifact_dir / "legal_cache" / "well-a.parquet").to_pandas()
    source = np.load(tmp_path / "shared" / "well-a.npz", allow_pickle=False)
    modes = summarize_ordered_modes(
        source["seed_delta"],
        source["final_ll"],
        source["seed_ids"],
        cluster_seed_descriptors(source["seed_delta"]),
    )
    assert cache.columns.tolist() == EXPECTED_COLUMNS
    for name in ("low", "middle", "high"):
        np.testing.assert_allclose(cache[f"pf_mode_{name}_delta"].to_numpy(), modes[name]["center_path"], rtol=1e-12, atol=1e-12)


def test_extreme_cross_mode_likelihoods_produce_finite_three_mode_centers(tmp_path: Path) -> None:
    folds_path, shadow_path, shared_dir = _make_inputs(tmp_path)
    extreme_ll = np.concatenate(
        [
            np.full(43, -6861.414),
            np.full(42, -3000.0),
            np.full(43, -693.582),
        ]
    )
    _write_shared(
        shared_dir / "well-a.npz",
        row_index=np.arange(1442, 1447, dtype=np.int64),
        final_ll=extreme_ll,
    )
    output_dir = tmp_path / "output"
    cache_generator.generate_mode_path_cache(
        folds_path,
        shadow_path,
        shared_dir,
        output_dir,
        max_wells=2,
        workers=1,
    )
    cache = pq.read_table(output_dir / "smoke_2" / "legal_cache" / "well-a.parquet").to_pandas()
    assert np.isfinite(cache[["pf_mode_low_delta", "pf_mode_middle_delta", "pf_mode_high_delta"]].to_numpy()).all()


def test_schema_is_exact_and_has_no_forbidden_fields(tmp_path: Path) -> None:
    artifact_dir = _generate(tmp_path)
    schema = pq.read_schema(artifact_dir / "legal_cache" / "well-a.parquet")
    assert schema.names == EXPECTED_COLUMNS
    assert [str(field.type) for field in schema] == ["string", "int64", "int64", "double", "double", "double", "double", "string"]
    forbidden = ("target", "true", "residual", "error", "rmse", "oracle", "best", "mass", "direction", "position", "separation", "seed_count")
    assert not any(token in column.lower() for token in forbidden for column in schema.names)


def test_hidden_rows_not_total_rows_and_nonzero_natural_row_index_are_preserved(tmp_path: Path) -> None:
    artifact_dir = _generate(tmp_path)
    cache = pq.read_table(artifact_dir / "legal_cache" / "well-a.parquet").to_pandas()
    assert len(cache) == 5
    np.testing.assert_array_equal(cache["row_index"].to_numpy(), np.arange(1442, 1447))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda shared: (shared / "well-b.npz").unlink(), "missing"),
        (lambda shared: _write_shared(shared / "well-a.npz", rows=5, seed_ids=np.arange(127)), "128"),
        (lambda shared: _write_shared(shared / "well-a.npz", rows=5, seed_ids=np.r_[np.arange(127), 0]), "seed_ids"),
        (lambda shared: _write_shared(shared / "well-a.npz", rows=5, row_index=np.asarray([0, 1, 1, 3, 4])), "row_index"),
        (lambda shared: _write_shared(shared / "well-a.npz", rows=5, finite=False), "finite"),
    ],
)
def test_invalid_shared_npz_is_rejected_before_cache_generation(tmp_path: Path, mutator: object, message: str) -> None:
    folds_path, shadow_path, shared_dir = _make_inputs(tmp_path)
    mutator(shared_dir)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        cache_generator.generate_mode_path_cache(folds_path, shadow_path, shared_dir, tmp_path / "output", max_wells=2, workers=1)


@pytest.mark.parametrize(
    ("seed_ids", "row_index", "message"),
    [
        (np.r_[0.5, np.arange(1, 128, dtype=float)], None, "seed_ids"),
        (np.r_[np.nan, np.arange(1, 128, dtype=float)], None, "finite"),
        (None, np.asarray([1442.5, 1443.0, 1444.0, 1445.0, 1446.0]), "row_index"),
        (None, np.asarray([np.nan, 1443.0, 1444.0, 1445.0, 1446.0]), "finite"),
    ],
)
def test_raw_seed_ids_and_row_index_reject_fractional_or_nonfinite_values(
    tmp_path: Path,
    seed_ids: np.ndarray | None,
    row_index: np.ndarray | None,
    message: str,
) -> None:
    folds_path, shadow_path, shared_dir = _make_inputs(tmp_path)
    _write_shared(
        shared_dir / "well-a.npz",
        row_index=np.arange(1442, 1447) if row_index is None else row_index,
        seed_ids=np.arange(128) if seed_ids is None else seed_ids,
    )
    with pytest.raises(ValueError, match=message):
        cache_generator.generate_mode_path_cache(folds_path, shadow_path, shared_dir, tmp_path / "output", max_wells=2, workers=1)


def test_wrong_row_count_fold_and_shadow_cache_are_rejected(tmp_path: Path) -> None:
    artifact_dir = _generate(tmp_path)
    cache_path = artifact_dir / "legal_cache" / "well-a.parquet"
    runtime_path = artifact_dir / "legal_runtime" / "well-a.json"
    source_path = tmp_path / "shared" / "well-a.npz"
    source = np.load(source_path, allow_pickle=False)
    row_index = source["row_index"]
    source.close()
    source_sha = cache_generator.file_sha256(source_path)
    frame = pq.read_table(cache_path).to_pandas()
    frame.loc[0, "fold"] = 4
    cache_generator.write_parquet_atomic(frame, cache_path)
    with pytest.raises(ValueError, match="fold"):
        cache_generator.validate_cache_hit(cache_path, runtime_path, "well-a", 0, 5, "fingerprint", row_index, source_sha)
    frame.loc[0, "fold"] = 0
    cache_generator.write_parquet_atomic(frame.iloc[:-1], cache_path)
    with pytest.raises(ValueError, match="rows"):
        cache_generator.validate_cache_hit(cache_path, runtime_path, "well-a", 0, 5, "fingerprint", row_index, source_sha)
    with pytest.raises(ValueError, match="shadow"):
        cache_generator.validate_cache_hit(cache_path, runtime_path, "shadow-well", 2, 5, "fingerprint", row_index, source_sha, shadow_wells={"shadow-well"})


def test_cache_hit_requires_exact_fingerprint_and_runtime(tmp_path: Path) -> None:
    artifact_dir = _generate(tmp_path)
    cache_path = artifact_dir / "legal_cache" / "well-a.parquet"
    runtime_path = artifact_dir / "legal_runtime" / "well-a.json"
    fingerprint = json.loads(runtime_path.read_text(encoding="utf-8"))["experiment_fingerprint"]
    source_path = tmp_path / "shared" / "well-a.npz"
    source = np.load(source_path, allow_pickle=False)
    row_index = source["row_index"]
    source.close()
    source_sha = cache_generator.file_sha256(source_path)
    with pytest.raises(ValueError, match="fingerprint"):
        cache_generator.validate_cache_hit(cache_path, runtime_path, "well-a", 0, 5, "wrong", row_index, source_sha)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.pop("experiment_fingerprint")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        cache_generator.validate_cache_hit(cache_path, runtime_path, "well-a", 0, 5, fingerprint, row_index, source_sha)


def test_changed_shared_npz_sha_prevents_cache_hit(tmp_path: Path) -> None:
    folds_path, shadow_path, shared_dir = _make_inputs(tmp_path)
    output_dir = tmp_path / "output"
    first_summary = cache_generator.generate_mode_path_cache(folds_path, shadow_path, shared_dir, output_dir, max_wells=2, workers=1)
    assert first_summary["total_elapsed_seconds"] >= 0.0
    runtime_path = output_dir / "smoke_2" / "legal_runtime" / "well-a.json"
    old_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    _write_shared(
        shared_dir / "well-a.npz",
        offset=20.0,
        row_index=np.arange(1442, 1447, dtype=np.int64),
    )
    cache_generator.generate_mode_path_cache(folds_path, shadow_path, shared_dir, output_dir, max_wells=2, workers=1)
    new_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert new_runtime["shared_npz_sha256"] != old_runtime["shared_npz_sha256"]
    assert new_runtime["cache_hit"] is False


def test_formal_call_has_no_bypass_and_rejects_nonfrozen_contract(tmp_path: Path) -> None:
    parameters = inspect.signature(cache_generator.generate_mode_path_cache).parameters
    assert "formal_expected_wells" not in parameters
    assert "formal_expected_rows" not in parameters
    folds_path, shadow_path, shared_dir = _make_inputs(tmp_path)
    with pytest.raises(ValueError, match="773"):
        cache_generator.generate_mode_path_cache(folds_path, shadow_path, shared_dir, tmp_path / "output", workers=1)


def test_smoke_artifacts_are_physically_separate_from_formal_directory(tmp_path: Path) -> None:
    output_dir = _generate(tmp_path, max_wells=1).parent
    assert (output_dir / "smoke_1" / "legal_cache" / "well-a.parquet").exists()
    assert not (output_dir / "legal_cache" / "well-a.parquet").exists()


def test_generation_never_calls_target_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_target_reader(*args: object, **kwargs: object) -> object:
        raise AssertionError("target read is forbidden")

    monkeypatch.setattr(cache_generator.pd, "read_parquet", forbidden_target_reader)
    artifact_dir = _generate(tmp_path)
    assert (artifact_dir / "legal_cache" / "well-a.parquet").exists()
