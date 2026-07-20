"""P3-PFM03 无标签合法缓存的自然键和指纹测试。"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import generate_p3_pfm03_shape_mode_cache as generator


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """Windows 系统临时目录权限不稳定，因此把测试文件放在项目 artifacts 下。"""

    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pfm03_cache_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_seed_npz(path: Path, row_index: np.ndarray, hidden_md: np.ndarray) -> None:
    rows = len(row_index)
    progress = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    reference = 2.0 * progress
    groups = np.repeat(np.arange(3), [43, 42, 43])
    shapes = np.stack(
        [
            -4.0 * np.sin(2.0 * np.pi * progress),
            np.zeros(rows),
            4.0 * np.sin(2.0 * np.pi * progress),
        ]
    )
    seed_delta = reference[None, :] + shapes[groups]
    np.savez(
        path,
        seed_delta=seed_delta.astype(np.float32),
        row_index=row_index.astype(np.int32),
        hidden_md=hidden_md.astype(np.float32),
        last_tvt=np.asarray([111.0], dtype=np.float64),
        final_ll=np.linspace(-4.0, 2.0, 128, dtype=np.float64),
        seed_ids=np.arange(128, dtype=np.int32),
        _cache_fingerprint=np.asarray([generator.SHARED_FINGERPRINT]),
        _format_version=np.asarray([1], dtype=np.int64),
    )


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    wells = ("00001234", "well-b", "shadow-well")
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "well_id": wells,
            "fold": [1, 2, 3],
            "hidden_rows": [9, 9, 9],
        }
    ).to_csv(folds_path, index=False)
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame({"well_id": ["shadow-well"]}).to_csv(shadow_path, index=False)
    shared_dir = tmp_path / "shared"
    reference_dir = tmp_path / "reference"
    shared_dir.mkdir()
    reference_dir.mkdir()
    row_index = np.arange(1442, 1451, dtype=np.int64)
    hidden_md = np.linspace(10_000.0, 10_800.0, 9)
    reference = np.linspace(0.0, 2.0, 9)
    for well_id in wells:
        _write_seed_npz(shared_dir / f"{well_id}.npz", row_index, hidden_md)
        pd.DataFrame(
            {
                "well_id": [well_id] * 9,
                "row_index": row_index,
                "last_visible_tvt": np.full(9, 111.0),
                "pf128_scale_8_delta": reference,
                "_cache_fingerprint": [generator.P01_FINGERPRINT] * 9,
            }
        ).to_parquet(reference_dir / f"{well_id}.parquet", index=False)
    return folds_path, shadow_path, shared_dir, reference_dir


def test_generator_signature_and_source_do_not_accept_or_read_hidden_truth(workspace_tmp_path: Path) -> None:
    parameters = inspect.signature(generator.generate_shape_mode_cache).parameters
    assert {"target", "target_tvt", "true_tvt", "oracle"}.isdisjoint(parameters)
    source = Path(generator.__file__).read_text(encoding="utf-8").lower()
    assert '"target_tvt"' not in source
    assert '"true_tvt"' not in source


def test_cache_preserves_natural_keys_and_binds_both_source_fingerprints(workspace_tmp_path: Path) -> None:
    folds_path, shadow_path, shared_dir, reference_dir = _make_inputs(workspace_tmp_path)
    output_dir = workspace_tmp_path / "output"

    summary = generator.generate_shape_mode_cache(
        folds_path,
        shadow_path,
        shared_dir,
        reference_dir,
        output_dir,
        max_wells=2,
        workers=1,
    )

    run_dir = output_dir / "smoke_2"
    cache = pq.read_table(run_dir / "legal_cache" / "00001234.parquet").to_pandas()
    runtime = json.loads(
        (run_dir / "legal_runtime" / "00001234.json").read_text(encoding="utf-8")
    )
    assert cache.columns.tolist() == generator.CACHE_COLUMNS
    assert cache["well_id"].unique().tolist() == ["00001234"]
    np.testing.assert_array_equal(cache["row_index"], np.arange(1442, 1451))
    assert cache["_cache_fingerprint"].nunique() == 1
    assert runtime["shared_fingerprint"] == generator.SHARED_FINGERPRINT
    assert runtime["p01_fingerprint"] == generator.P01_FINGERPRINT
    assert runtime["hidden_tvt_read"] is False
    assert summary["wells"] == 2
    assert not (run_dir / "legal_cache" / "shadow-well.parquet").exists()


def test_wrong_seed_or_reference_fingerprint_is_rejected(workspace_tmp_path: Path) -> None:
    folds_path, shadow_path, shared_dir, reference_dir = _make_inputs(workspace_tmp_path)
    seed_path = shared_dir / "00001234.npz"
    with np.load(seed_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["_cache_fingerprint"] = np.asarray(["wrong"])
    np.savez(seed_path, **payload)
    with pytest.raises(ValueError, match="fingerprint"):
        generator.generate_shape_mode_cache(
            folds_path,
            shadow_path,
            shared_dir,
            reference_dir,
            workspace_tmp_path / "bad-seed",
            max_wells=2,
            workers=1,
        )

    fresh_folds, fresh_shadow, fresh_shared, fresh_reference = _make_inputs(
        workspace_tmp_path / "fresh"
    )
    reference_path = fresh_reference / "00001234.parquet"
    frame = pd.read_parquet(reference_path)
    frame["_cache_fingerprint"] = "wrong"
    frame.to_parquet(reference_path, index=False)
    with pytest.raises(ValueError, match="fingerprint"):
        generator.generate_shape_mode_cache(
            fresh_folds,
            fresh_shadow,
            fresh_shared,
            fresh_reference,
            workspace_tmp_path / "bad-reference",
            max_wells=2,
            workers=1,
        )


def test_reference_anchor_accepts_only_float32_rounding_error(workspace_tmp_path: Path) -> None:
    """真实 P01 anchor 是 float32，约 3e-4 ft 的量化差不能误判为错井。"""

    reference_path = workspace_tmp_path / "reference.parquet"
    row_index = np.arange(100, 103, dtype=np.int64)
    pd.DataFrame(
        {
            "well_id": ["well-a"] * 3,
            "row_index": row_index,
            "last_visible_tvt": np.full(3, 11604.82, dtype=np.float32),
            "pf128_scale_8_delta": np.arange(3, dtype=np.float64),
            "_cache_fingerprint": [generator.P01_FINGERPRINT] * 3,
        }
    ).to_parquet(reference_path, index=False)

    observed = generator._read_reference_cache(
        reference_path,
        "well-a",
        3,
        row_index,
        11604.820000000002,
    )

    np.testing.assert_array_equal(observed, np.arange(3, dtype=np.float64))
