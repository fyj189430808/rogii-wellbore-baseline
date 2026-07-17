"""从 773 口原始训练井确定性重建 F05a 的 24 个候选特征。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.f05a_candidate_reproduction import build_candidate_features  # noqa: E402
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F05a_deterministic_candidate_cache_v1"
FIXED_SEED = 42
DEFAULT_WORKERS = 4
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
EXPECTED_REGISTRY_SHA256 = (
    "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab"
)
HORIZONTAL_INPUT_COLUMNS = ["MD", "Z", "GR", "TVT_input"]
TYPEWELL_INPUT_COLUMNS = ["TVT", "GR"]
FINGERPRINT_COLUMN = "_cache_fingerprint"

DEFAULT_RAW_TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
DEFAULT_REGISTRY_PATH = (
    CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv"
)
DEFAULT_GENERATOR_PATH = CLEAN_ROOT / "src" / "f05a_candidate_reproduction.py"
DEFAULT_PARAMETER_PATH = CLEAN_ROOT / "configs" / "f05a_candidate_generator_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID


def file_sha256(path: Path) -> str:
    """分块计算文件 SHA-256，避免把文件整体读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_signature(path: Path) -> dict:
    """返回路径、类型、大小和纳秒修改时间组成的轻量签名。"""

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_base_lineage(
    generator_code_path: Path,
    parameter_json_path: Path,
    registry_path: Path,
    raw_train_dir: Path,
    seed: int,
) -> dict:
    """构造所有井共享的代码、参数、fold 和原始目录血缘。"""

    payload = {
        "generator_code_path": str(generator_code_path.resolve()),
        "generator_code_sha256": file_sha256(generator_code_path.resolve()),
        "parameter_json_path": str(parameter_json_path.resolve()),
        "parameter_json_sha256": file_sha256(parameter_json_path.resolve()),
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": file_sha256(registry_path.resolve()),
        "seed": int(seed),
        "raw_path_signature": path_signature(raw_train_dir),
    }
    payload["fingerprint"] = _json_fingerprint(payload)
    return payload


def build_well_lineage(
    base_lineage: dict,
    horizontal_path: Path,
    typewell_path: Path,
) -> dict:
    """在共享血缘上加入当前井两份原始 CSV 的路径签名。"""

    payload = {
        "base_fingerprint": str(base_lineage["fingerprint"]),
        "horizontal_signature": path_signature(horizontal_path),
        "typewell_signature": path_signature(typewell_path),
    }
    payload["fingerprint"] = _json_fingerprint(payload)
    return payload


def per_well_cache_path(cache_dir: Path, well_id: str) -> Path:
    """返回单井 parquet 路径，同时拒绝带目录分隔符的井号。"""

    normalized = str(well_id)
    if Path(normalized).name != normalized or "/" in normalized or "\\" in normalized:
        raise ValueError(f"非法 well_id：{well_id!r}")
    return cache_dir / f"{normalized}.parquet"


def _validate_well_cache_frame(
    frame: pd.DataFrame,
    well_id: str,
    cache_fingerprint: str,
    expected_hidden_rows: int,
    feature_columns: list[str],
) -> None:
    expected_columns = [
        "well_id",
        "row_index",
        *feature_columns,
        FINGERPRINT_COLUMN,
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{well_id} 单井缓存列或顺序错误")
    if len(frame) != int(expected_hidden_rows):
        raise ValueError(
            f"{well_id} 单井缓存行数错误：{len(frame)} != {expected_hidden_rows}"
        )
    if not frame["well_id"].astype(str).eq(str(well_id)).all():
        raise ValueError(f"{well_id} 单井缓存含其他井号")
    if not frame[FINGERPRINT_COLUMN].astype(str).eq(str(cache_fingerprint)).all():
        raise ValueError(f"{well_id} 单井缓存指纹不匹配")
    row_indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError(f"{well_id} 单井缓存含重复 row_index")
    if len(row_indices) > 1 and not bool(np.all(np.diff(row_indices) > 0)):
        raise ValueError(f"{well_id} 单井缓存 row_index 未严格递增")
    feature_values = frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        raise ValueError(f"{well_id} 单井缓存候选特征含 NaN 或 Inf")


def is_reusable_well_cache(
    cache_path: Path,
    well_id: str,
    cache_fingerprint: str,
    expected_hidden_rows: int,
    feature_columns: list[str],
) -> bool:
    """只有文件结构、行数、键和指纹全部匹配时才允许续跑复用。"""

    if not cache_path.is_file():
        return False
    try:
        frame = pd.read_parquet(cache_path)
        _validate_well_cache_frame(
            frame,
            well_id,
            cache_fingerprint,
            expected_hidden_rows,
            feature_columns,
        )
    except Exception:
        return False
    return True


def build_well_cache(
    well_id: str,
    horizontal_path: Path,
    typewell_path: Path,
    cache_path: Path,
    cache_fingerprint: str,
    expected_hidden_rows: int,
    feature_columns: list[str],
    seed: int = FIXED_SEED,
    rebuild: bool = False,
    generator: Callable | None = None,
) -> dict:
    """读取合法列、生成一口井的候选特征并原子写入独立 parquet。"""

    started = time.perf_counter()
    if int(seed) != FIXED_SEED:
        raise ValueError(f"候选缓存只允许固定 seed={FIXED_SEED}")
    if not rebuild and is_reusable_well_cache(
        cache_path,
        well_id,
        cache_fingerprint,
        expected_hidden_rows,
        feature_columns,
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
    horizontal_legal = horizontal[HORIZONTAL_INPUT_COLUMNS].copy()
    typewell_legal = typewell[TYPEWELL_INPUT_COLUMNS].copy()

    if generator is None:
        generated = build_candidate_features(
            horizontal_legal,
            typewell_legal,
            seed=42,
        )
    else:
        generated = generator(horizontal_legal, typewell_legal, seed=42)

    expected_generated_columns = ["row_index", *feature_columns]
    if list(generated.columns) != expected_generated_columns:
        raise ValueError(f"{well_id} 生成器输出列或顺序错误")
    expected_row_indices = horizontal_legal.index[
        horizontal_legal["TVT_input"].isna()
    ].to_numpy(dtype=np.int64)
    generated_row_indices = generated["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(generated_row_indices, expected_row_indices):
        raise ValueError(f"{well_id} 生成 row_index 与原始隐藏 mask 不一致")
    if len(generated) != int(expected_hidden_rows):
        raise ValueError(
            f"{well_id} 生成行数错误：{len(generated)} != {expected_hidden_rows}"
        )

    output = generated.copy()
    output.insert(0, "well_id", str(well_id))
    output[FINGERPRINT_COLUMN] = str(cache_fingerprint)
    _validate_well_cache_frame(
        output,
        well_id,
        cache_fingerprint,
        expected_hidden_rows,
        feature_columns,
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


def merge_per_well_caches(
    registry: pd.DataFrame,
    cache_dir: Path,
    expected_fingerprints: dict[str, str],
    output_path: Path,
    feature_columns: list[str],
    expected_wells: int,
    expected_rows: int,
) -> dict:
    """按注册表原顺序流式合并单井缓存，并验证行数和复合键唯一。"""

    if len(registry) != int(expected_wells):
        raise ValueError(f"注册表井数错误：{len(registry)} != {expected_wells}")
    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("注册表含重复 well_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.building")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    written_wells = 0
    try:
        for registry_row in registry.itertuples(index=False):
            well_id = str(registry_row.well_id)
            hidden_rows = int(registry_row.hidden_rows)
            if well_id not in expected_fingerprints:
                raise ValueError(f"{well_id} 缺少期望缓存指纹")
            cache_path = per_well_cache_path(cache_dir, well_id)
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            frame = pd.read_parquet(cache_path)
            _validate_well_cache_frame(
                frame,
                well_id,
                expected_fingerprints[well_id],
                hidden_rows,
                feature_columns,
            )
            output_frame = frame[["well_id", "row_index", *feature_columns]].copy()
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
            raise ValueError("没有可合并的单井缓存")
        writer.close()
        writer = None
        if written_wells != int(expected_wells):
            raise ValueError(f"合并井数错误：{written_wells} != {expected_wells}")
        if total_rows != int(expected_rows):
            raise ValueError(f"合并行数错误：{total_rows} != {expected_rows}")
        os.replace(temporary_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return {"wells": written_wells, "rows": total_rows, "keys_unique": True}


def load_registry(path: Path) -> pd.DataFrame:
    """按文件原顺序读取固定 fold 注册表，不额外排序。"""

    registry = pd.read_csv(path, dtype={"well_id": str, "pad_id": str})
    required_columns = {"well_id", "hidden_rows", "fold"}
    missing_columns = required_columns - set(registry.columns)
    if missing_columns:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing_columns)}")
    if registry["well_id"].duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")
    registry = registry.copy()
    registry["hidden_rows"] = pd.to_numeric(
        registry["hidden_rows"], errors="raise"
    ).astype(np.int64)
    return registry


def _write_json_atomic(path: Path, value: object) -> None:
    temporary_path = path.with_name(f"{path.name}.building")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _write_runtime_table(
    path: Path,
    runtime_by_well: dict[str, dict],
    registry_order: list[str],
) -> None:
    rows = [runtime_by_well[well_id] for well_id in registry_order if well_id in runtime_by_well]
    frame = pd.DataFrame(rows)
    temporary_path = path.with_name(f"{path.name}.building")
    frame.to_csv(temporary_path, index=False, lineterminator="\n")
    # Windows 的杀毒/索引进程偶尔会在 CSV 刚写完时短暂占用目标文件。
    # 这里只重试完全相同的原子替换，不改变任何候选特征或实验结果。
    for retry_index in range(10):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if retry_index == 9:
                raise
            time.sleep(0.2)


def _load_existing_runtime(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype={"well_id": str})
    if "well_id" not in frame.columns:
        return {}
    return {
        str(row["well_id"]): row
        for row in frame.to_dict(orient="records")
    }


def _report_progress(done: int, total: int, generated: int, reused: int) -> None:
    if done % 25 == 0 or done == total:
        print(
            f"候选缓存进度：{done}/{total} 井，生成={generated}，复用={reused}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建 F05a 773 井确定性候选缓存")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_RAW_TRAIN_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--generator-config", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须至少为 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须至少为 1")

    raw_train_dir = args.train_dir.resolve()
    registry_path = args.registry.resolve()
    parameter_path = args.generator_config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    per_well_dir = artifact_dir / "per_well"
    final_cache_path = artifact_dir / "candidate_feature_cache.parquet"
    metadata_path = artifact_dir / "meta.json"
    runtime_path = artifact_dir / "per_well_runtime.csv"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    per_well_dir.mkdir(parents=True, exist_ok=True)

    registry_hash = file_sha256(registry_path)
    if registry_hash != EXPECTED_REGISTRY_SHA256:
        raise ValueError(
            "固定 fold 注册表 SHA-256 不匹配："
            f"{registry_hash} != {EXPECTED_REGISTRY_SHA256}"
        )
    registry = load_registry(registry_path)
    if len(registry) != EXPECTED_WELLS:
        raise ValueError(f"固定注册表井数错误：{len(registry)} != {EXPECTED_WELLS}")
    if int(registry["hidden_rows"].sum()) != EXPECTED_ROWS:
        raise ValueError(
            "固定注册表隐藏行数错误："
            f"{int(registry['hidden_rows'].sum())} != {EXPECTED_ROWS}"
        )

    parameter_config = json.loads(parameter_path.read_text(encoding="utf-8"))
    if int(parameter_config["randomness"]["seed"]) != FIXED_SEED:
        raise ValueError("候选参数 JSON 的 seed 不是 42")
    if parameter_config["output_feature_columns"] != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("候选参数 JSON 的特征顺序与生成器不一致")

    base_lineage = build_base_lineage(
        DEFAULT_GENERATOR_PATH,
        parameter_path,
        registry_path,
        raw_train_dir,
        seed=FIXED_SEED,
    )
    well_lineages: dict[str, dict] = {}
    expected_fingerprints: dict[str, str] = {}
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
        lineage = build_well_lineage(base_lineage, horizontal_path, typewell_path)
        well_lineages[well_id] = lineage
        expected_fingerprints[well_id] = str(lineage["fingerprint"])

    selected_registry = registry
    if args.limit is not None:
        selected_registry = registry.iloc[: min(int(args.limit), len(registry))]
    selected_well_ids = selected_registry["well_id"].astype(str).tolist()
    registry_order = registry["well_id"].astype(str).tolist()
    runtime_by_well = _load_existing_runtime(runtime_path)

    print(
        f"开始 {EXPERIMENT_ID}：选择 {len(selected_registry)}/{len(registry)} 井，"
        f"workers={args.workers}，rebuild={bool(args.rebuild)}",
        flush=True,
    )
    pending: list[tuple] = []
    done = 0
    generated_count = 0
    reused_count = 0
    for registry_row in selected_registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        cache_path = per_well_cache_path(per_well_dir, well_id)
        fingerprint = expected_fingerprints[well_id]
        if not args.rebuild and is_reusable_well_cache(
            cache_path,
            well_id,
            fingerprint,
            int(registry_row.hidden_rows),
            DIRECT_CANDIDATE_COLUMNS,
        ):
            runtime_by_well[well_id] = {
                "well_id": well_id,
                "status": "reused",
                "rows": int(registry_row.hidden_rows),
                "seconds": 0.0,
                "cache_path": str(cache_path.resolve()),
                "cache_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            done += 1
            reused_count += 1
            _report_progress(done, len(selected_registry), generated_count, reused_count)
        else:
            pending.append(
                (
                    well_id,
                    raw_train_dir / f"{well_id}__horizontal_well.csv",
                    raw_train_dir / f"{well_id}__typewell.csv",
                    cache_path,
                    fingerprint,
                    int(registry_row.hidden_rows),
                )
            )

    if pending:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    build_well_cache,
                    well_id,
                    horizontal_path,
                    typewell_path,
                    cache_path,
                    fingerprint,
                    hidden_rows,
                    DIRECT_CANDIDATE_COLUMNS,
                    FIXED_SEED,
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
                try:
                    runtime = future.result()
                except Exception:
                    _write_runtime_table(runtime_path, runtime_by_well, registry_order)
                    raise
                runtime_by_well[well_id] = runtime
                done += 1
                if runtime["status"] == "generated":
                    generated_count += 1
                else:
                    reused_count += 1
                _write_runtime_table(runtime_path, runtime_by_well, registry_order)
                _report_progress(
                    done,
                    len(selected_registry),
                    generated_count,
                    reused_count,
                )

    _write_runtime_table(runtime_path, runtime_by_well, registry_order)

    missing_or_stale: list[str] = []
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        if not is_reusable_well_cache(
            per_well_cache_path(per_well_dir, well_id),
            well_id,
            expected_fingerprints[well_id],
            int(registry_row.hidden_rows),
            DIRECT_CANDIDATE_COLUMNS,
        ):
            missing_or_stale.append(well_id)

    if missing_or_stale:
        print(
            f"当前仍缺少或过期 {len(missing_or_stale)} 口井；"
            "单井缓存已保留，本次不覆盖最终合并缓存。",
            flush=True,
        )
        return

    merge_summary = merge_per_well_caches(
        registry=registry,
        cache_dir=per_well_dir,
        expected_fingerprints=expected_fingerprints,
        output_path=final_cache_path,
        feature_columns=DIRECT_CANDIDATE_COLUMNS,
        expected_wells=EXPECTED_WELLS,
        expected_rows=EXPECTED_ROWS,
    )
    ordered_fingerprints = [
        {"well_id": well_id, "fingerprint": expected_fingerprints[well_id]}
        for well_id in registry_order
    ]
    raw_manifest_signature = _json_fingerprint(ordered_fingerprints)
    final_fingerprint = _json_fingerprint(
        {
            "base_lineage": base_lineage,
            "ordered_well_fingerprints": ordered_fingerprints,
        }
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "wells": int(merge_summary["wells"]),
        "rows": int(merge_summary["rows"]),
        "keys_unique": bool(merge_summary["keys_unique"]),
        "registry_order_preserved": True,
        "seed": FIXED_SEED,
        "workers": int(args.workers),
        "feature_columns": DIRECT_CANDIDATE_COLUMNS,
        "feature_count": len(DIRECT_CANDIDATE_COLUMNS),
        "candidate_cache_path": str(final_cache_path),
        "candidate_cache_sha256": file_sha256(final_cache_path),
        "per_well_cache_dir": str(per_well_dir),
        "per_well_runtime_path": str(runtime_path),
        "base_lineage": base_lineage,
        "raw_manifest_signature": raw_manifest_signature,
        "cache_fingerprint": final_fingerprint,
    }
    _write_json_atomic(metadata_path, metadata)
    print(
        f"完整候选缓存完成：{final_cache_path}，"
        f"井={merge_summary['wells']}，行={merge_summary['rows']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
