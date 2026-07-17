"""并行生成 F06 前缀伪 holdout 井级特征，并复制到自然隐藏行。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.build_f05a_deterministic_candidate_cache import (  # noqa: E402
    _json_fingerprint,
    _load_existing_runtime,
    _write_json_atomic,
    _write_runtime_table,
    build_base_lineage,
    build_well_lineage,
    file_sha256,
    FINGERPRINT_COLUMN,
    HORIZONTAL_INPUT_COLUMNS,
    load_registry,
    per_well_cache_path,
    TYPEWELL_INPUT_COLUMNS,
)
from src.f06_prefix_holdout import (  # noqa: E402
    F06_FEATURE_COLUMNS,
    build_prefix_holdout_features,
)


EXPERIMENT_ID = "F06_prefix_holdout_cache_v1"
F06_COLUMNS = list(F06_FEATURE_COLUMNS)
FIXED_SEED = 42
DEFAULT_WORKERS = 4
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
EXPECTED_REGISTRY_SHA256 = (
    "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab"
)

DEFAULT_RAW_TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
DEFAULT_REGISTRY_PATH = (
    CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv"
)
DEFAULT_GENERATOR_PATH = CLEAN_ROOT / "src" / "f06_prefix_holdout.py"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "f06_prefix_holdout_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID


def _expand_well_features_to_hidden_rows(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """把生成器返回的一行井级特征，原样复制给该井每个自然隐藏行。"""

    well_features = build_prefix_holdout_features(
        horizontal_df,
        typewell_df,
        seed=seed,
    )
    if list(well_features.columns) != F06_COLUMNS:
        raise ValueError("F06 井级生成器的列或顺序不正确")
    if len(well_features) != 1:
        raise ValueError("F06 井级生成器必须恰好返回一行")

    hidden_row_indices = horizontal_df.index[
        horizontal_df["TVT_input"].isna()
    ].to_numpy(dtype=np.int64)
    if len(hidden_row_indices) == 0:
        raise ValueError("F06 输入井没有自然隐藏行")

    output: dict[str, np.ndarray] = {"row_index": hidden_row_indices}
    for feature_name in F06_COLUMNS:
        feature_value = np.float32(well_features.iloc[0][feature_name])
        if np.isinf(feature_value):
            raise ValueError(f"F06 井级特征 {feature_name} 是 Inf")
        output[feature_name] = np.full(
            len(hidden_row_indices),
            feature_value,
            dtype=np.float32,
        )
    return pd.DataFrame(output)


def _validate_f06_well_cache_frame(
    frame: pd.DataFrame,
    well_id: str,
    cache_fingerprint: str,
    expected_hidden_rows: int,
) -> None:
    """验证单井缓存；F06 允许 NaN 表示不支持的 cut，但拒绝 Inf。"""

    expected_columns = [
        "well_id",
        "row_index",
        *F06_COLUMNS,
        FINGERPRINT_COLUMN,
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{well_id} 单井 F06 缓存列或顺序错误")
    if len(frame) != int(expected_hidden_rows):
        raise ValueError(f"{well_id} 单井 F06 缓存行数错误")
    if not frame["well_id"].astype(str).eq(str(well_id)).all():
        raise ValueError(f"{well_id} 单井 F06 缓存含其他井号")
    if not frame[FINGERPRINT_COLUMN].astype(str).eq(str(cache_fingerprint)).all():
        raise ValueError(f"{well_id} 单井 F06 缓存指纹不匹配")

    row_indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError(f"{well_id} 单井 F06 缓存含重复 row_index")
    if len(row_indices) > 1 and not bool(np.all(np.diff(row_indices) > 0)):
        raise ValueError(f"{well_id} 单井 F06 缓存 row_index 未严格递增")

    feature_values = frame[F06_COLUMNS].to_numpy(dtype=np.float32)
    if np.isinf(feature_values).any():
        raise ValueError(f"{well_id} 单井 F06 缓存含 Inf")
    if frame[F06_COLUMNS].nunique(dropna=False).gt(1).any():
        raise ValueError(f"{well_id} F06 井级特征没有逐行保持相同")


def is_reusable_f06_well_cache(
    cache_path: Path,
    well_id: str,
    cache_fingerprint: str,
    expected_hidden_rows: int,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        frame = pd.read_parquet(cache_path)
        _validate_f06_well_cache_frame(
            frame,
            well_id,
            cache_fingerprint,
            expected_hidden_rows,
        )
    except Exception:
        return False
    return True


def build_f06_well_cache(
    well_id: str,
    horizontal_path: Path,
    typewell_path: Path,
    cache_path: Path,
    cache_fingerprint: str,
    expected_hidden_rows: int,
    rebuild: bool = False,
) -> dict:
    """只读取合法列，以 seed 42 生成或复用一口井的 F06 缓存。"""

    started = time.perf_counter()
    if not rebuild and is_reusable_f06_well_cache(
        cache_path,
        well_id,
        cache_fingerprint,
        expected_hidden_rows,
    ):
        return {
            "well_id": str(well_id),
            "status": "reused",
            "rows": int(expected_hidden_rows),
            "seconds": time.perf_counter() - started,
            "cache_path": str(cache_path.resolve()),
            "cache_fingerprint": str(cache_fingerprint),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    horizontal = pd.read_csv(horizontal_path, usecols=HORIZONTAL_INPUT_COLUMNS)
    typewell = pd.read_csv(typewell_path, usecols=TYPEWELL_INPUT_COLUMNS)
    generated = _expand_well_features_to_hidden_rows(
        horizontal[HORIZONTAL_INPUT_COLUMNS].copy(),
        typewell[TYPEWELL_INPUT_COLUMNS].copy(),
        seed=FIXED_SEED,
    )
    expected_columns = ["row_index", *F06_COLUMNS]
    if list(generated.columns) != expected_columns:
        raise ValueError(f"{well_id} F06 生成器输出列或顺序错误")
    expected_row_indices = horizontal.index[
        horizontal["TVT_input"].isna()
    ].to_numpy(dtype=np.int64)
    if not np.array_equal(
        generated["row_index"].to_numpy(dtype=np.int64),
        expected_row_indices,
    ):
        raise ValueError(f"{well_id} F06 row_index 与自然隐藏 mask 不一致")
    if len(generated) != int(expected_hidden_rows):
        raise ValueError(f"{well_id} F06 生成行数错误")

    output = generated.copy()
    output.insert(0, "well_id", str(well_id))
    output[FINGERPRINT_COLUMN] = str(cache_fingerprint)
    _validate_f06_well_cache_frame(
        output,
        well_id,
        cache_fingerprint,
        expected_hidden_rows,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f"{cache_path.name}.{os.getpid()}.building"
    )
    try:
        output.to_parquet(temporary_path, index=False, compression="zstd")
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "well_id": str(well_id),
        "status": "generated",
        "rows": int(len(output)),
        "seconds": time.perf_counter() - started,
        "cache_path": str(cache_path.resolve()),
        "cache_fingerprint": str(cache_fingerprint),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def merge_f06_per_well_caches(
    registry: pd.DataFrame,
    cache_dir: Path,
    expected_fingerprints: dict[str, str],
    output_path: Path,
    expected_wells: int,
    expected_rows: int,
) -> dict:
    """按固定注册表顺序流式合并 773 份单井缓存。"""

    if len(registry) != expected_wells:
        raise ValueError("F06 合并前注册表井数错误")
    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("F06 合并前注册表含重复 well_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.building")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    written_wells = 0
    try:
        for row in registry.itertuples(index=False):
            well_id = str(row.well_id)
            cache_path = per_well_cache_path(cache_dir, well_id)
            frame = pd.read_parquet(cache_path)
            _validate_f06_well_cache_frame(
                frame,
                well_id,
                expected_fingerprints[well_id],
                int(row.hidden_rows),
            )
            output_frame = frame[["well_id", "row_index", *F06_COLUMNS]].copy()
            table = pa.Table.from_pandas(output_frame, preserve_index=False)
            table = table.replace_schema_metadata(None)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression="zstd",
                )
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total_rows += len(output_frame)
            written_wells += 1

        if writer is None:
            raise ValueError("没有可合并的 F06 单井缓存")
        writer.close()
        writer = None
        if written_wells != expected_wells:
            raise ValueError("F06 合并井数错误")
        if total_rows != expected_rows:
            raise ValueError("F06 合并行数错误")
        os.replace(temporary_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    return {"wells": written_wells, "rows": total_rows, "keys_unique": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 F06 前缀伪 holdout 特征缓存")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_RAW_TRAIN_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def _report_progress(done: int, total: int, generated: int, reused: int) -> None:
    if done % 25 == 0 or done == total:
        print(
            f"F06 缓存进度：{done}/{total} 井，生成={generated}，复用={reused}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers 必须至少为 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须至少为 1")

    raw_train_dir = args.train_dir.resolve()
    registry_path = args.registry.resolve()
    config_path = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    per_well_dir = artifact_dir / "per_well"
    final_cache_path = artifact_dir / "prefix_holdout_cache.parquet"
    metadata_path = artifact_dir / "meta.json"
    runtime_path = artifact_dir / "per_well_runtime.csv"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    per_well_dir.mkdir(parents=True, exist_ok=True)

    registry_hash = file_sha256(registry_path)
    if registry_hash != EXPECTED_REGISTRY_SHA256:
        raise ValueError("固定 fold 注册表 SHA-256 不一致")
    registry = load_registry(registry_path)
    if len(registry) != EXPECTED_WELLS:
        raise ValueError(f"固定注册表井数错误：{len(registry)} != {EXPECTED_WELLS}")
    if int(registry["hidden_rows"].sum()) != EXPECTED_ROWS:
        raise ValueError("固定注册表隐藏行数错误")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("prefix_holdout_feature_columns") != F06_COLUMNS:
        raise ValueError("配置中的 F06 特征列或顺序与生成器不一致")
    if int(config.get("cache_seed", -1)) != FIXED_SEED:
        raise ValueError("F06 缓存 seed 必须固定为 42")

    base_lineage = build_base_lineage(
        DEFAULT_GENERATOR_PATH,
        config_path,
        registry_path,
        raw_train_dir,
        seed=FIXED_SEED,
    )
    expected_fingerprints: dict[str, str] = {}
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        lineage = build_well_lineage(
            base_lineage,
            raw_train_dir / f"{well_id}__horizontal_well.csv",
            raw_train_dir / f"{well_id}__typewell.csv",
        )
        expected_fingerprints[well_id] = str(lineage["fingerprint"])

    selected = registry
    if args.limit is not None:
        selected = registry.iloc[: min(int(args.limit), len(registry))]
    registry_order = registry["well_id"].astype(str).tolist()
    runtime_by_well = _load_existing_runtime(runtime_path)

    print(
        f"开始 {EXPERIMENT_ID}：{len(selected)}/{len(registry)} 井，"
        f"workers={args.workers}，rebuild={bool(args.rebuild)}",
        flush=True,
    )
    pending: list[tuple[str, Path, Path, Path, str, int]] = []
    done = 0
    generated_count = 0
    reused_count = 0
    for row in selected.itertuples(index=False):
        well_id = str(row.well_id)
        cache_path = per_well_cache_path(per_well_dir, well_id)
        fingerprint = expected_fingerprints[well_id]
        hidden_rows = int(row.hidden_rows)
        if not args.rebuild and is_reusable_f06_well_cache(
            cache_path,
            well_id,
            fingerprint,
            hidden_rows,
        ):
            runtime_by_well[well_id] = {
                "well_id": well_id,
                "status": "reused",
                "rows": hidden_rows,
                "seconds": 0.0,
                "cache_path": str(cache_path.resolve()),
                "cache_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            done += 1
            reused_count += 1
            _report_progress(done, len(selected), generated_count, reused_count)
        else:
            pending.append(
                (
                    well_id,
                    raw_train_dir / f"{well_id}__horizontal_well.csv",
                    raw_train_dir / f"{well_id}__typewell.csv",
                    cache_path,
                    fingerprint,
                    hidden_rows,
                )
            )

    if pending:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    build_f06_well_cache,
                    well_id,
                    horizontal_path,
                    typewell_path,
                    cache_path,
                    fingerprint,
                    hidden_rows,
                    bool(args.rebuild),
                ): well_id
                for (
                    well_id,
                    horizontal_path,
                    typewell_path,
                    cache_path,
                    fingerprint,
                    hidden_rows,
                ) in pending
            }
            for future in as_completed(futures):
                well_id = futures[future]
                runtime = future.result()
                runtime_by_well[well_id] = runtime
                done += 1
                if runtime["status"] == "generated":
                    generated_count += 1
                else:
                    reused_count += 1
                _write_runtime_table(runtime_path, runtime_by_well, registry_order)
                _report_progress(done, len(selected), generated_count, reused_count)

    _write_runtime_table(runtime_path, runtime_by_well, registry_order)
    missing_or_stale: list[str] = []
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        if not is_reusable_f06_well_cache(
            per_well_cache_path(per_well_dir, well_id),
            well_id,
            expected_fingerprints[well_id],
            int(row.hidden_rows),
        ):
            missing_or_stale.append(well_id)
    if missing_or_stale:
        print(
            f"仍缺少或过期 {len(missing_or_stale)} 口井；保留单井缓存，暂不合并。",
            flush=True,
        )
        return

    summary = merge_f06_per_well_caches(
        registry=registry,
        cache_dir=per_well_dir,
        expected_fingerprints=expected_fingerprints,
        output_path=final_cache_path,
        expected_wells=EXPECTED_WELLS,
        expected_rows=EXPECTED_ROWS,
    )
    ordered_fingerprints = [
        {"well_id": well_id, "fingerprint": expected_fingerprints[well_id]}
        for well_id in registry_order
    ]
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "wells": int(summary["wells"]),
        "rows": int(summary["rows"]),
        "keys_unique": bool(summary["keys_unique"]),
        "registry_order_preserved": True,
        "seed": FIXED_SEED,
        "workers": int(args.workers),
        "feature_columns": F06_COLUMNS,
        "feature_count": len(F06_COLUMNS),
        "cache_path": str(final_cache_path),
        "cache_sha256": file_sha256(final_cache_path),
        "per_well_cache_dir": str(per_well_dir),
        "base_lineage": base_lineage,
        "cache_fingerprint": _json_fingerprint(
            {
                "base_lineage": base_lineage,
                "ordered_well_fingerprints": ordered_fingerprints,
            }
        ),
    }
    _write_json_atomic(metadata_path, metadata)
    print(
        f"F06 完整缓存完成：{summary['wells']} 井，{summary['rows']:,} 行",
        flush=True,
    )


if __name__ == "__main__":
    main()
