"""Generate the P3-PFM02 legal three-mode PF path cache without labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_pfm01_ordered_pf_modes import (
    FORMAL_LIKELIHOOD_SCALE,
    FORMAL_NUMBER_OF_SEEDS,
    MODE_NAMES,
    cluster_seed_descriptors,
)


PROJECT_ROOT = CLEAN_ROOT
SHARED_FINGERPRINT = "91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0"
FORMAL_WELLS = 657
FORMAL_ROWS = 3_211_872
CACHE_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    "pf_mode_low_delta",
    "pf_mode_middle_delta",
    "pf_mode_high_delta",
    "_cache_fingerprint",
]
CACHE_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("row_index", pa.int64()),
        pa.field("last_visible_tvt", pa.float64()),
        pa.field("pf_mode_low_delta", pa.float64()),
        pa.field("pf_mode_middle_delta", pa.float64()),
        pa.field("pf_mode_high_delta", pa.float64()),
        pa.field("_cache_fingerprint", pa.string()),
    ]
)
FORBIDDEN_TOKENS = (
    "target", "true", "residual", "error", "rmse", "oracle", "best", "mass",
    "direction", "position", "separation", "seed_count",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_run_artifact_dir(output_dir: Path, max_wells: int | None) -> Path:
    if max_wells is None:
        return output_dir
    if max_wells not in (1, 2, 3):
        raise ValueError("--max-wells must be one of 1, 2, 3")
    return output_dir / f"smoke_{max_wells}"


def _assert_legal_column_names(columns: list[str]) -> None:
    if columns != CACHE_COLUMNS:
        raise ValueError("cache schema columns are not exact")
    unsafe = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    if unsafe:
        raise ValueError(f"forbidden cache columns: {unsafe}")


def _load_selection(folds_path: Path, shadow_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    folds = pd.read_csv(folds_path)
    required = {"well_id", "fold", "hidden_rows"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"fold registry missing columns: {sorted(missing)}")
    if folds["well_id"].isna().any() or folds["well_id"].duplicated().any():
        raise ValueError("fold registry has invalid well_id")
    if folds["fold"].isna().any() or not np.all(np.equal(folds["fold"], np.floor(folds["fold"]))):
        raise ValueError("fold registry has invalid fold")
    if not folds["fold"].between(0, 4).all():
        raise ValueError("fold registry has invalid fold")
    if not np.all(np.equal(folds["hidden_rows"], np.floor(folds["hidden_rows"]))) or (folds["hidden_rows"] <= 0).any():
        raise ValueError("fold registry has invalid hidden_rows")
    shadow = pd.read_csv(shadow_path)
    if "well_id" not in shadow.columns:
        raise ValueError("shadow registry missing well_id")
    shadow_wells = set(shadow["well_id"].astype(str))
    selected = folds.loc[~folds["well_id"].astype(str).isin(shadow_wells), ["well_id", "fold", "hidden_rows"]].copy()
    selected["well_id"] = selected["well_id"].astype(str)
    selected["fold"] = selected["fold"].astype(np.int64)
    selected["hidden_rows"] = selected["hidden_rows"].astype(np.int64)
    if selected.empty:
        raise ValueError("shadow selection leaves no development wells")
    return folds, selected.sort_values("well_id", kind="stable").reset_index(drop=True), shadow_wells


def _validate_formal_contract(folds: pd.DataFrame, selected: pd.DataFrame, shadow_wells: set[str]) -> None:
    fold_wells = set(folds["well_id"].astype(str))
    if len(folds) != 773:
        raise ValueError("formal fold registry must contain 773 wells")
    if len(shadow_wells) != 116:
        raise ValueError("formal shadow registry must contain 116 wells")
    if not shadow_wells.issubset(fold_wells):
        raise ValueError("formal shadow wells must all be in fold registry")
    if set(selected["well_id"]).intersection(shadow_wells):
        raise ValueError("formal development wells intersect shadow wells")
    if len(selected) != FORMAL_WELLS:
        raise ValueError("formal development selection must contain 657 wells")
    if int(selected["hidden_rows"].sum()) != FORMAL_ROWS:
        raise ValueError("formal development selection must contain 3211872 hidden rows")


def _validate_integer_vector(raw_values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(raw_values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not (np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)):
        raise ValueError(f"{name} must be numeric")
    numeric = raw.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} values must be finite")
    limits = np.iinfo(np.int64)
    if np.any(numeric < limits.min) or np.any(numeric > limits.max):
        raise ValueError(f"{name} values are outside int64 range")
    if not np.array_equal(numeric, np.trunc(numeric)):
        raise ValueError(f"{name} values must be integers")
    return numeric.astype(np.int64)


def _load_and_validate_shared(npz_path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            required = {"seed_delta", "final_ll", "seed_ids", "row_index", "last_tvt"}
            if missing := required.difference(source.files):
                raise ValueError(f"shared NPZ missing fields: {sorted(missing)}")
            data = {name: np.asarray(source[name]) for name in required}
    except FileNotFoundError as exc:
        raise ValueError(f"missing shared NPZ: {npz_path.name}") from exc
    paths = np.asarray(data["seed_delta"], dtype=np.float64)
    final_ll = np.asarray(data["final_ll"], dtype=np.float64)
    seed_ids = _validate_integer_vector(data["seed_ids"], "seed_ids")
    row_index = _validate_integer_vector(data["row_index"], "row_index")
    last_tvt = np.asarray(data["last_tvt"], dtype=np.float64)
    if paths.shape != (FORMAL_NUMBER_OF_SEEDS, expected_rows):
        raise ValueError("seed_delta must have exactly 128 seeds and expected rows")
    if final_ll.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("final_ll must have exactly 128 values")
    if seed_ids.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("seed_ids must contain exactly 128 values")
    if not np.array_equal(seed_ids, np.arange(FORMAL_NUMBER_OF_SEEDS)):
        raise ValueError("seed_ids must be exactly 0..127")
    if row_index.shape != (expected_rows,) or np.any(np.diff(row_index) <= 0):
        raise ValueError("row_index must be strictly increasing and unique")
    if last_tvt.shape != (1,):
        raise ValueError("last_tvt must have shape [1]")
    all_values = np.concatenate((paths.ravel(), final_ll, last_tvt))
    if not np.isfinite(all_values).all():
        raise ValueError("shared NPZ values must be finite")
    return {
        "seed_delta": paths,
        "final_ll": final_ll,
        "seed_ids": seed_ids,
        "row_index": row_index,
        "last_tvt": last_tvt,
    }


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    _assert_legal_column_names(frame.columns.tolist())
    arrays = [pa.array(frame[column].tolist(), type=field.type) for column, field in zip(CACHE_COLUMNS, CACHE_SCHEMA, strict=True)]
    table = pa.Table.from_arrays(arrays, schema=CACHE_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_cache_hit(
    cache_path: Path,
    runtime_path: Path,
    well_id: str,
    fold: int,
    rows: int,
    experiment_fingerprint: str,
    expected_row_index: np.ndarray,
    shared_npz_sha256: str,
    *,
    shadow_wells: set[str] | None = None,
) -> None:
    if shadow_wells and well_id in shadow_wells:
        raise ValueError("shadow well cache is forbidden")
    if not cache_path.exists() or not runtime_path.exists():
        raise ValueError("cache or runtime missing")
    schema = pq.read_schema(cache_path)
    if schema != CACHE_SCHEMA:
        raise ValueError("cache schema is invalid")
    table = pq.read_table(cache_path)
    if table.num_rows != rows:
        raise ValueError("cache rows are invalid")
    frame = table.to_pandas()
    _assert_legal_column_names(frame.columns.tolist())
    if frame["well_id"].nunique() != 1 or frame["well_id"].iloc[0] != well_id:
        raise ValueError("cache well_id is invalid")
    if frame["fold"].nunique() != 1 or int(frame["fold"].iloc[0]) != fold:
        raise ValueError("cache fold is invalid")
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), expected_row_index):
        raise ValueError("cache row_index is invalid")
    if frame["_cache_fingerprint"].nunique() != 1 or frame["_cache_fingerprint"].iloc[0] != experiment_fingerprint:
        raise ValueError("cache fingerprint is invalid")
    if not np.isfinite(frame[["last_visible_tvt", "pf_mode_low_delta", "pf_mode_middle_delta", "pf_mode_high_delta"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("cache contains non-finite values")
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("runtime is invalid") from exc
    runtime_required = {"well_id", "fold", "rows", "experiment_fingerprint", "shared_npz_sha256", "shared_fingerprint", "generator_sha256", "pfm01_core_sha256", "hidden_tvt_read", "cache_hit", "elapsed_seconds"}
    if not runtime_required.issubset(runtime):
        raise ValueError("runtime is incomplete")
    if runtime["well_id"] != well_id or int(runtime["fold"]) != fold or int(runtime["rows"]) != rows or runtime["experiment_fingerprint"] != experiment_fingerprint or runtime["shared_npz_sha256"] != shared_npz_sha256:
        raise ValueError("runtime does not match cache")
    if runtime["shared_fingerprint"] != SHARED_FINGERPRINT or runtime["hidden_tvt_read"] is not False:
        raise ValueError("runtime violates legal input contract")


def _stable_ordered_mode_centers(
    seed_delta: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """PFM02-only stable equivalent of PFM01's within-mode scale-8 centers."""

    paths = np.asarray(seed_delta, dtype=np.float64)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    raw_labels = cluster_seed_descriptors(paths)
    records: list[tuple[float, float, int, np.ndarray]] = []
    for raw_label in sorted(np.unique(raw_labels).tolist()):
        member_indices = np.flatnonzero(raw_labels == raw_label)
        member_ll = likelihoods[member_indices]
        shifted = (member_ll - float(np.max(member_ll))) / FORMAL_LIKELIHOOD_SCALE
        unnormalized = np.exp(shifted)
        weights = unnormalized / float(np.sum(unnormalized))
        center_path = np.sum(paths[member_indices] * weights[:, None], axis=0, dtype=np.float64)
        records.append(
            (
                float(np.mean(center_path)),
                float(center_path[-1]),
                int(np.min(ids[member_indices])),
                center_path,
            )
        )
    records.sort(key=lambda record: record[:3])
    return {name: record[3] for name, record in zip(MODE_NAMES, records, strict=True)}


def _build_legal_frame(well_id: str, fold: int, shared: dict[str, np.ndarray], fingerprint: str) -> pd.DataFrame:
    paths = shared["seed_delta"]
    modes = _stable_ordered_mode_centers(paths, shared["final_ll"], shared["seed_ids"])
    rows = paths.shape[1]
    return pd.DataFrame(
        {
            "well_id": np.repeat(well_id, rows),
            "fold": np.repeat(fold, rows).astype(np.int64),
            "row_index": shared["row_index"],
            "last_visible_tvt": np.repeat(float(shared["last_tvt"][0]), rows),
            "pf_mode_low_delta": modes["low"],
            "pf_mode_middle_delta": modes["middle"],
            "pf_mode_high_delta": modes["high"],
            "_cache_fingerprint": np.repeat(fingerprint, rows),
        }
    )


def generate_mode_path_cache(
    folds_path: Path,
    shadow_path: Path,
    shared_dir: Path,
    output_dir: Path,
    *,
    max_wells: int | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    if workers <= 0:
        raise ValueError("workers must be positive")
    folds, selected, shadow_wells = _load_selection(folds_path, shadow_path)
    if max_wells is None:
        _validate_formal_contract(folds, selected, shadow_wells)
    else:
        if max_wells not in (1, 2, 3):
            raise ValueError("--max-wells must be one of 1, 2, 3")
        selected = selected.head(max_wells).copy()
    missing_paths = [well_id for well_id in selected["well_id"] if not (shared_dir / f"{well_id}.npz").is_file()]
    if missing_paths:
        raise ValueError(f"missing shared NPZ for wells: {missing_paths}")
    run_dir = resolve_run_artifact_dir(output_dir, max_wells)
    generator_sha = file_sha256(Path(__file__))
    core_sha = file_sha256(PROJECT_ROOT / "src" / "p3_pfm01_ordered_pf_modes.py")
    fingerprint = _canonical_fingerprint(
        {
            "version": "P3_PFM02_mode_paths_v1",
            "shared_fingerprint": SHARED_FINGERPRINT,
            "generator_sha256": generator_sha,
            "pfm01_core_sha256": core_sha,
            "mode_definition": "mean_endpoint_ward_k3_scale8_ordered_low_middle_high",
        }
    )
    config = {
        "experiment_id": "P3_PFM02_mode_paths_v1",
        "max_wells": max_wells,
        "workers": workers,
        "selected_wells": int(len(selected)),
        "selected_rows": int(selected["hidden_rows"].sum()),
        "shadow_wells": 0,
        "shared_fingerprint": SHARED_FINGERPRINT,
        "generator_sha256": generator_sha,
        "pfm01_core_sha256": core_sha,
        "experiment_fingerprint": fingerprint,
        "hidden_tvt_read": False,
    }
    _write_json_atomic(config, run_dir / "config.json")
    _write_json_atomic(["pf_mode_low_delta", "pf_mode_middle_delta", "pf_mode_high_delta"], run_dir / "feature_list.json")
    _write_json_atomic({"ward_k": 3, "descriptor": ["mean_delta", "end_delta"], "standardize": False, "likelihood_scale": 8.0}, run_dir / "parameter_list.json")

    def build_one(row: Any) -> dict[str, Any]:
        started = time.perf_counter()
        well_id, fold, rows = str(row.well_id), int(row.fold), int(row.hidden_rows)
        cache_path = run_dir / "legal_cache" / f"{well_id}.parquet"
        runtime_path = run_dir / "legal_runtime" / f"{well_id}.json"
        shared_path = shared_dir / f"{well_id}.npz"
        shared = _load_and_validate_shared(shared_path, rows)
        shared_sha = file_sha256(shared_path)
        try:
            validate_cache_hit(
                cache_path,
                runtime_path,
                well_id,
                fold,
                rows,
                fingerprint,
                shared["row_index"],
                shared_sha,
                shadow_wells=shadow_wells,
            )
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["cache_hit"] = True
            runtime["elapsed_seconds"] = float(time.perf_counter() - started)
            _write_json_atomic(runtime, runtime_path)
            print(f"{well_id}: cache hit", flush=True)
            return {"well_id": well_id, "fold": fold, "rows": rows, "cache_hit": True}
        except ValueError:
            pass
        frame = _build_legal_frame(well_id, fold, shared, fingerprint)
        write_parquet_atomic(frame, cache_path)
        runtime = {
            "well_id": well_id,
            "fold": fold,
            "rows": rows,
            "shared_npz_sha256": shared_sha,
            "shared_fingerprint": SHARED_FINGERPRINT,
            "generator_sha256": generator_sha,
            "pfm01_core_sha256": core_sha,
            "experiment_fingerprint": fingerprint,
            "hidden_tvt_read": False,
            "cache_hit": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json_atomic(runtime, runtime_path)
        validate_cache_hit(
            cache_path,
            runtime_path,
            well_id,
            fold,
            rows,
            fingerprint,
            shared["row_index"],
            shared_sha,
            shadow_wells=shadow_wells,
        )
        print(f"{well_id}: generated", flush=True)
        return {"well_id": well_id, "fold": fold, "rows": rows, "cache_hit": False}

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(build_one, selected.itertuples(index=False)):
            records.append(record)
    per_well = pd.DataFrame(records).sort_values("well_id", kind="stable")
    (run_dir / "legal").mkdir(parents=True, exist_ok=True)
    per_well.to_csv(run_dir / "legal" / "per_well.csv", index=False)
    runtime_summary = {
        "wells": int(len(per_well)),
        "rows": int(per_well["rows"].sum()),
        "shadow_wells": 0,
        "cache_hits": int(per_well["cache_hit"].sum()),
        "cache_misses": int((~per_well["cache_hit"]).sum()),
        "experiment_fingerprint": fingerprint,
        "hidden_tvt_read": False,
        "total_elapsed_seconds": float(time.perf_counter() - run_started),
    }
    _write_json_atomic(runtime_summary, run_dir / "runtime.json")
    return runtime_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wells", type=int, choices=(1, 2, 3))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "P3_PFM02_mode_paths_v1")
    args = parser.parse_args()
    generate_mode_path_cache(
        folds_path=PROJECT_ROOT / "artifacts" / "folds" / "balanced_well_5fold_v1.csv",
        shadow_path=PROJECT_ROOT / "artifacts" / "P3_shadow_holdout_v1" / "shadow_holdout.csv",
        shared_dir=PROJECT_ROOT / "artifacts" / "P3_shared_pf_seed_paths_v1",
        output_dir=args.output_dir,
        max_wells=args.max_wells,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
