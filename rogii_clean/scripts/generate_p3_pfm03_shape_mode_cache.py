"""生成 P3-PFM03 三条形状模式路径的无标签合法缓存。"""

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

from src.p3_pfm03_shape_aware_modes import (  # noqa: E402
    FORMAL_FEATURE_COLUMNS,
    FORMAL_LIKELIHOOD_SCALE,
    FORMAL_NUMBER_OF_MODES,
    FORMAL_NUMBER_OF_SEEDS,
    FORMAL_SAMPLE_POINTS,
    build_shape_mode_paths,
)


PROJECT_ROOT = CLEAN_ROOT
EXPERIMENT_ID = "P3_PFM03_shape_aware_mode_paths_v1"
SHARED_FINGERPRINT = "91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0"
P01_FINGERPRINT = "ac1dbefd59923585671156b7d3e8b4fc7faca95c20f5f1ae6fdab92a8665954a"
FORMAL_WELLS = 657
FORMAL_ROWS = 3_211_872
FORMAT_VERSION = 1

CACHE_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    *FORMAL_FEATURE_COLUMNS,
    "_cache_fingerprint",
]
CACHE_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("row_index", pa.int64()),
        pa.field("last_visible_tvt", pa.float64()),
        pa.field("shape_mode_low_delta", pa.float64()),
        pa.field("shape_mode_middle_delta", pa.float64()),
        pa.field("shape_mode_high_delta", pa.float64()),
        pa.field("_cache_fingerprint", pa.string()),
    ]
)


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大型 npz/parquet 一次读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    """把固定配置和代码哈希转成跨进程稳定的实验指纹。"""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(payload: Any, path: Path) -> None:
    """先写临时文件再原子替换，防止中断留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """按照冻结 Arrow schema 原子保存一口井的逐行路径。"""

    if frame.columns.tolist() != CACHE_COLUMNS:
        raise ValueError("PFM03 缓存列与冻结合同不一致")
    arrays = [
        pa.array(frame[column].tolist(), type=field.type)
        for column, field in zip(CACHE_COLUMNS, CACHE_SCHEMA, strict=True)
    ]
    table = pa.Table.from_arrays(arrays, schema=CACHE_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_development_selection(
    folds_path: Path,
    shadow_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """只用井号、fold 和隐藏行数移除冻结影子井，不读取任何标签。"""

    folds = pd.read_csv(folds_path, dtype={"well_id": str})
    required = {"well_id", "fold", "hidden_rows"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"fold 表缺列：{sorted(missing)}")
    if folds["well_id"].isna().any() or folds["well_id"].duplicated().any():
        raise ValueError("fold 表井号为空或重复")
    shadow = pd.read_csv(shadow_path, dtype={"well_id": str})
    if "well_id" not in shadow.columns:
        raise ValueError("影子表缺少 well_id")
    shadow_wells = set(shadow["well_id"].astype(str))
    selected = folds.loc[
        ~folds["well_id"].astype(str).isin(shadow_wells),
        ["well_id", "fold", "hidden_rows"],
    ].copy()
    selected["well_id"] = selected["well_id"].astype(str)
    selected["fold"] = selected["fold"].astype(np.int64)
    selected["hidden_rows"] = selected["hidden_rows"].astype(np.int64)
    if selected.empty or (selected["hidden_rows"] <= 0).any():
        raise ValueError("开发井选择为空或隐藏行数非法")
    return (
        folds,
        selected.sort_values("well_id", kind="stable").reset_index(drop=True),
        shadow_wells,
    )


def _validate_formal_selection(
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    shadow_wells: set[str],
) -> None:
    """正式生成时锁定 773/116/657 口井和 3,211,872 行。"""

    if len(folds) != 773:
        raise ValueError("正式 fold 表必须有 773 口井")
    if len(shadow_wells) != 116:
        raise ValueError("正式影子集必须有 116 口井")
    if len(selected) != FORMAL_WELLS:
        raise ValueError("正式开发集必须有 657 口井")
    if int(selected["hidden_rows"].sum()) != FORMAL_ROWS:
        raise ValueError("正式开发集必须有 3211872 个隐藏行")


def _read_seed_cache(path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    """读取一井 PF128 路径，并验证来源指纹、格式和所有自然键。"""

    required = {
        "seed_delta",
        "row_index",
        "hidden_md",
        "last_tvt",
        "final_ll",
        "seed_ids",
        "_cache_fingerprint",
        "_format_version",
    }
    try:
        with np.load(path, allow_pickle=False) as source:
            if missing := required.difference(source.files):
                raise ValueError(f"seed NPZ 缺字段：{sorted(missing)}")
            arrays = {name: np.asarray(source[name]) for name in required}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少 seed NPZ：{path}") from exc

    fingerprint = arrays["_cache_fingerprint"]
    if fingerprint.shape != (1,) or str(fingerprint[0]) != SHARED_FINGERPRINT:
        raise ValueError("seed NPZ fingerprint 与冻结共享缓存不一致")
    format_version = arrays["_format_version"]
    if format_version.shape != (1,) or int(format_version[0]) != FORMAT_VERSION:
        raise ValueError("seed NPZ format version 不一致")

    paths = np.asarray(arrays["seed_delta"], dtype=np.float64)
    row_index = np.asarray(arrays["row_index"])
    hidden_md = np.asarray(arrays["hidden_md"], dtype=np.float64)
    last_tvt = np.asarray(arrays["last_tvt"], dtype=np.float64)
    final_ll = np.asarray(arrays["final_ll"], dtype=np.float64)
    seed_ids = np.asarray(arrays["seed_ids"])
    if paths.shape != (FORMAL_NUMBER_OF_SEEDS, expected_rows):
        raise ValueError("seed_delta 不是冻结的 [128, 隐藏行数]")
    if row_index.shape != (expected_rows,) or not np.issubdtype(row_index.dtype, np.integer):
        raise ValueError("seed NPZ row_index 形状或类型非法")
    row_index = row_index.astype(np.int64, copy=False)
    if np.any(np.diff(row_index) <= 0):
        raise ValueError("seed NPZ row_index 必须严格递增")
    if hidden_md.shape != (expected_rows,) or np.any(np.diff(hidden_md) < 0.0):
        raise ValueError("seed NPZ hidden_md 形状非法或不单调")
    if last_tvt.shape != (1,) or final_ll.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("seed NPZ last_tvt/final_ll 形状非法")
    if seed_ids.shape != (FORMAL_NUMBER_OF_SEEDS,) or not np.issubdtype(seed_ids.dtype, np.integer):
        raise ValueError("seed NPZ seed_ids 形状或类型非法")
    seed_ids = seed_ids.astype(np.int64, copy=False)
    if not np.array_equal(seed_ids, np.arange(FORMAL_NUMBER_OF_SEEDS)):
        raise ValueError("seed NPZ seed_ids 必须严格为 0～127")
    if not np.isfinite(paths).all() or not np.isfinite(hidden_md).all():
        raise ValueError("seed NPZ 路径或 MD 含 NaN/Inf")
    if not np.isfinite(last_tvt).all() or not np.isfinite(final_ll).all():
        raise ValueError("seed NPZ last_tvt 或 final_ll 含 NaN/Inf")
    return {
        "seed_delta": paths,
        "row_index": row_index,
        "hidden_md": hidden_md,
        "last_tvt": last_tvt,
        "final_ll": final_ll,
        "seed_ids": seed_ids,
    }


def _read_reference_cache(
    path: Path,
    well_id: str,
    expected_rows: int,
    expected_row_index: np.ndarray,
    expected_last_tvt: float,
) -> np.ndarray:
    """按井号+原始 row_index 读取 P01 scale8 delta，并绑定其冻结指纹。"""

    columns = [
        "well_id",
        "row_index",
        "last_visible_tvt",
        "pf128_scale_8_delta",
        "_cache_fingerprint",
    ]
    if not path.is_file():
        raise FileNotFoundError(f"缺少 P01 参考缓存：{path}")
    schema_names = pq.read_schema(path).names
    if missing := set(columns).difference(schema_names):
        raise ValueError(f"P01 参考缓存缺列：{sorted(missing)}")
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != expected_rows:
        raise ValueError("P01 参考缓存行数不一致")
    if not frame["well_id"].astype(str).eq(well_id).all():
        raise ValueError("P01 参考缓存井号不一致")
    row_index = frame["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_index, expected_row_index):
        raise ValueError("P01 参考缓存自然 row_index 不一致")
    fingerprints = frame["_cache_fingerprint"].astype(str).unique().tolist()
    if fingerprints != [P01_FINGERPRINT]:
        raise ValueError("P01 reference fingerprint 与冻结来源不一致")
    anchors = frame["last_visible_tvt"].to_numpy(dtype=np.float64)
    # P01 为节省空间把 anchor 保存成 float32；容差只放宽到该数值的一格 float32 ULP。
    anchor_tolerance = max(
        1e-6,
        abs(float(np.spacing(np.float32(expected_last_tvt)))),
    )
    if not np.isfinite(anchors).all() or not np.allclose(
        anchors,
        expected_last_tvt,
        rtol=0.0,
        atol=anchor_tolerance,
    ):
        raise ValueError("P01 参考缓存 last_visible_tvt 与 seed 缓存不一致")
    reference = frame["pf128_scale_8_delta"].to_numpy(dtype=np.float64)
    if not np.isfinite(reference).all():
        raise ValueError("P01 scale8 参考路径含 NaN/Inf")
    return reference


def _run_dir(output_dir: Path, max_wells: int | None) -> Path:
    """smoke 与正式缓存物理分开，避免三井文件混入 657 井目录。"""

    if max_wells is None:
        return output_dir
    if max_wells not in (1, 2, 3):
        raise ValueError("--max-wells 只允许 1、2、3")
    return output_dir / f"smoke_{max_wells}"


def _validate_cache_hit(
    cache_path: Path,
    runtime_path: Path,
    *,
    well_id: str,
    fold: int,
    rows: int,
    row_index: np.ndarray,
    fingerprint: str,
    seed_sha256: str,
    reference_sha256: str,
) -> None:
    """命中缓存前同时核对自然键、实验指纹和两份逐井来源文件。"""

    if not cache_path.is_file() or not runtime_path.is_file():
        raise ValueError("缓存或 runtime 不存在")
    if not pq.read_schema(cache_path).equals(CACHE_SCHEMA):
        raise ValueError("缓存 schema 不一致")
    frame = pq.read_table(cache_path).to_pandas()
    if len(frame) != rows or frame.columns.tolist() != CACHE_COLUMNS:
        raise ValueError("缓存行数或列顺序不一致")
    if not frame["well_id"].eq(well_id).all() or not frame["fold"].eq(fold).all():
        raise ValueError("缓存井号或 fold 不一致")
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), row_index):
        raise ValueError("缓存自然 row_index 不一致")
    if frame["_cache_fingerprint"].astype(str).unique().tolist() != [fingerprint]:
        raise ValueError("缓存 fingerprint 不一致")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    expected = {
        "well_id": well_id,
        "fold": fold,
        "rows": rows,
        "experiment_fingerprint": fingerprint,
        "seed_npz_sha256": seed_sha256,
        "reference_parquet_sha256": reference_sha256,
        "shared_fingerprint": SHARED_FINGERPRINT,
        "p01_fingerprint": P01_FINGERPRINT,
        "hidden_tvt_read": False,
    }
    if any(runtime.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime 与当前缓存来源不一致")


def generate_shape_mode_cache(
    folds_path: Path,
    shadow_path: Path,
    shared_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    max_wells: int | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """生成 1～3 井 smoke 或 657 井正式无标签 PFM03 路径缓存。"""

    started = time.perf_counter()
    if workers <= 0:
        raise ValueError("workers 必须为正整数")
    folds, selected, shadow_wells = _load_development_selection(
        folds_path,
        shadow_path,
    )
    if max_wells is None:
        _validate_formal_selection(folds, selected, shadow_wells)
    else:
        if max_wells not in (1, 2, 3):
            raise ValueError("--max-wells 只允许 1、2、3")
        selected = selected.head(max_wells).copy()

    generator_sha256 = file_sha256(Path(__file__).resolve())
    core_path = PROJECT_ROOT / "src" / "p3_pfm03_shape_aware_modes.py"
    core_sha256 = file_sha256(core_path)
    experiment_fingerprint = _stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "shared_fingerprint": SHARED_FINGERPRINT,
            "p01_fingerprint": P01_FINGERPRINT,
            "generator_sha256": generator_sha256,
            "core_sha256": core_sha256,
            "sample_points": FORMAL_SAMPLE_POINTS,
            "ward_k": FORMAL_NUMBER_OF_MODES,
            "per_dimension_standardization": False,
            "reference_path": "pf128_scale_8_delta",
            "representative": "final_ll_scale8_weighted_center",
            "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
            "ordering": "mean_relative_to_reference_then_endpoint_then_minimum_seed",
        }
    )
    run_dir = _run_dir(output_dir, max_wells)
    _write_json_atomic(
        {
            "experiment_id": EXPERIMENT_ID,
            "selected_wells": int(len(selected)),
            "selected_rows": int(selected["hidden_rows"].sum()),
            "max_wells": max_wells,
            "workers": workers,
            "experiment_fingerprint": experiment_fingerprint,
            "shared_fingerprint": SHARED_FINGERPRINT,
            "p01_fingerprint": P01_FINGERPRINT,
            "generator_sha256": generator_sha256,
            "core_sha256": core_sha256,
            "hidden_tvt_read": False,
        },
        run_dir / "config.json",
    )
    _write_json_atomic(list(FORMAL_FEATURE_COLUMNS), run_dir / "feature_list.json")
    _write_json_atomic(
        {
            "sample_points": FORMAL_SAMPLE_POINTS,
            "ward_k": FORMAL_NUMBER_OF_MODES,
            "standardize": False,
            "reference": "pf128_scale_8_delta",
            "representative": "final_ll_scale8_weighted_center",
            "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
        },
        run_dir / "parameter_list.json",
    )

    def build_one(registry_row: Any) -> dict[str, Any]:
        well_started = time.perf_counter()
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        rows = int(registry_row.hidden_rows)
        if well_id in shadow_wells:
            raise RuntimeError("影子井不能进入 PFM03 生成")
        seed_path = shared_dir / f"{well_id}.npz"
        reference_path = reference_dir / f"{well_id}.parquet"
        seed = _read_seed_cache(seed_path, rows)
        reference = _read_reference_cache(
            reference_path,
            well_id,
            rows,
            seed["row_index"],
            float(seed["last_tvt"][0]),
        )
        seed_sha256 = file_sha256(seed_path)
        reference_sha256 = file_sha256(reference_path)
        cache_path = run_dir / "legal_cache" / f"{well_id}.parquet"
        runtime_path = run_dir / "legal_runtime" / f"{well_id}.json"
        diagnostics_path = run_dir / "legal_diagnostics" / f"{well_id}.json"
        try:
            _validate_cache_hit(
                cache_path,
                runtime_path,
                well_id=well_id,
                fold=fold,
                rows=rows,
                row_index=seed["row_index"],
                fingerprint=experiment_fingerprint,
                seed_sha256=seed_sha256,
                reference_sha256=reference_sha256,
            )
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["cache_hit"] = True
            runtime["elapsed_seconds"] = float(time.perf_counter() - well_started)
            _write_json_atomic(runtime, runtime_path)
            print(f"{well_id}: cache hit", flush=True)
            return {"well_id": well_id, "fold": fold, "rows": rows, "cache_hit": True}
        except (ValueError, json.JSONDecodeError):
            pass

        features, diagnostics = build_shape_mode_paths(
            seed["seed_delta"],
            seed["hidden_md"],
            seed["final_ll"],
            seed["seed_ids"],
            reference,
        )
        frame = pd.DataFrame(
            {
                "well_id": np.repeat(well_id, rows),
                "fold": np.repeat(fold, rows).astype(np.int64),
                "row_index": seed["row_index"],
                "last_visible_tvt": np.repeat(float(seed["last_tvt"][0]), rows),
                "shape_mode_low_delta": features["shape_mode_low_delta"],
                "shape_mode_middle_delta": features["shape_mode_middle_delta"],
                "shape_mode_high_delta": features["shape_mode_high_delta"],
                "_cache_fingerprint": np.repeat(experiment_fingerprint, rows),
            },
            columns=CACHE_COLUMNS,
        )
        _write_parquet_atomic(frame, cache_path)
        _write_json_atomic(diagnostics, diagnostics_path)
        runtime = {
            "well_id": well_id,
            "fold": fold,
            "rows": rows,
            "experiment_fingerprint": experiment_fingerprint,
            "seed_npz_sha256": seed_sha256,
            "reference_parquet_sha256": reference_sha256,
            "shared_fingerprint": SHARED_FINGERPRINT,
            "p01_fingerprint": P01_FINGERPRINT,
            "generator_sha256": generator_sha256,
            "core_sha256": core_sha256,
            "hidden_tvt_read": False,
            "cache_hit": False,
            "elapsed_seconds": float(time.perf_counter() - well_started),
        }
        _write_json_atomic(runtime, runtime_path)
        _validate_cache_hit(
            cache_path,
            runtime_path,
            well_id=well_id,
            fold=fold,
            rows=rows,
            row_index=seed["row_index"],
            fingerprint=experiment_fingerprint,
            seed_sha256=seed_sha256,
            reference_sha256=reference_sha256,
        )
        print(f"{well_id}: generated", flush=True)
        return {"well_id": well_id, "fold": fold, "rows": rows, "cache_hit": False}

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(build_one, selected.itertuples(index=False)):
            records.append(record)
    per_well = pd.DataFrame(records).sort_values("well_id", kind="stable")
    run_dir.mkdir(parents=True, exist_ok=True)
    per_well.to_csv(run_dir / "per_well.csv", index=False)
    runtime_summary = {
        "wells": int(len(per_well)),
        "rows": int(per_well["rows"].sum()),
        "cache_hits": int(per_well["cache_hit"].sum()),
        "cache_misses": int((~per_well["cache_hit"]).sum()),
        "experiment_fingerprint": experiment_fingerprint,
        "hidden_tvt_read": False,
        "total_elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json_atomic(runtime_summary, run_dir / "runtime.json")
    return runtime_summary


def main() -> None:
    """CLI 默认写正式目录；传 --max-wells 3 时只写独立 smoke_3。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wells", type=int, choices=(1, 2, 3))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / EXPERIMENT_ID,
    )
    args = parser.parse_args()
    summary = generate_shape_mode_cache(
        PROJECT_ROOT / "artifacts" / "folds" / "balanced_well_5fold_v1.csv",
        PROJECT_ROOT / "artifacts" / "P3_shadow_holdout_v1" / "shadow_holdout.csv",
        PROJECT_ROOT / "artifacts" / "P3_shared_pf_seed_paths_v1",
        PROJECT_ROOT / "artifacts" / "P2_P01_multiseed_pf_mean_v1" / "legal_cache",
        args.output_dir,
        max_wells=args.max_wells,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
