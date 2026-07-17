"""运行 F05a：B00 单模 LightGBM 加冻结候选路径的汇总几何特征。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.par
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f05a_precomputed_candidate_geometry import (  # noqa: E402
    F05A_FEATURE_COLUMNS,
    SOURCE_USE_COLUMNS,
    build_candidate_geometry,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "f05a_precomputed_candidate_geometry_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F05a 候选路径几何特征实验")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rebuild-candidate-cache", action="store_true")
    return parser.parse_args()


def _source_signature(source_path: Path) -> dict[str, object]:
    file_stat = source_path.stat()
    return {
        "path": str(source_path.resolve()),
        "size": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
    }


def build_candidate_cache_fingerprint(source_path: Path) -> str:
    module_path = CLEAN_ROOT / "src" / "f05a_precomputed_candidate_geometry.py"
    payload = {
        "source": _source_signature(source_path),
        "source_use_columns": SOURCE_USE_COLUMNS,
        "feature_columns": F05A_FEATURE_COLUMNS,
        "formula_code_sha256": file_sha256(module_path),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_candidate_chunks(source_path: Path, chunk_size: int):
    numeric_columns = [name for name in SOURCE_USE_COLUMNS if name not in {"well", "id"}]
    dtypes: dict[str, object] = {name: np.float32 for name in numeric_columns}
    dtypes["well"] = str
    dtypes["id"] = str
    return pd.read_csv(
        source_path,
        usecols=SOURCE_USE_COLUMNS,
        dtype=dtypes,
        chunksize=int(chunk_size),
    )


def _validate_key_alignment(
    base_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> None:
    if len(base_rows) != len(candidate_rows):
        raise ValueError("候选几何缓存与 B00 缓存行数不一致")
    base_wells = base_rows["well_id"].astype(str).to_numpy()
    candidate_wells = candidate_rows["well_id"].astype(str).to_numpy()
    if not np.array_equal(base_wells, candidate_wells):
        raise ValueError("候选几何缓存与 B00 缓存的 well_id 顺序不一致")
    base_indices = base_rows["row_index"].to_numpy(dtype=np.int32)
    candidate_indices = candidate_rows["row_index"].to_numpy(dtype=np.int32)
    if not np.array_equal(base_indices, candidate_indices):
        raise ValueError("候选几何缓存与 B00 缓存的 row_index 顺序不一致")


def run_smoke(source_path: Path, base_cache_path: Path, chunk_size: int) -> None:
    first_chunk = next(iter(_read_candidate_chunks(source_path, chunk_size)))
    smoke_wells = first_chunk["well"].drop_duplicates().astype(str).iloc[:3].tolist()
    if len(smoke_wells) != 3:
        raise ValueError("首个候选数据分块不足三口井")
    smoke_source = first_chunk.loc[first_chunk["well"].isin(smoke_wells)].copy()
    smoke_features = build_candidate_geometry(smoke_source)

    base_keys = pd.read_parquet(base_cache_path, columns=["well_id", "row_index"])
    base_keys["well_id"] = base_keys["well_id"].astype(str)
    smoke_base = base_keys.loc[base_keys["well_id"].isin(smoke_wells)].reset_index(
        drop=True
    )
    _validate_key_alignment(smoke_base, smoke_features)

    feature_values = smoke_features[F05A_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        raise ValueError("smoke 特征含 NaN 或 Inf")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_features):,}，"
        f"新增特征={len(F05A_FEATURE_COLUMNS)}，finite=100%",
        flush=True,
    )
    print(smoke_features.head(3).to_string(index=False), flush=True)


def _write_feature_batch(
    source_rows: pd.DataFrame,
    writer: pq.ParquetWriter | None,
    output_path: Path,
) -> tuple[pq.ParquetWriter, int]:
    features = build_candidate_geometry(source_rows)
    feature_values = features[F05A_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        raise ValueError("候选几何特征含 NaN 或 Inf")
    table = pa.Table.from_pandas(features, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer, len(features)


def build_candidate_feature_cache(
    source_path: Path,
    cache_path: Path,
    metadata_path: Path,
    chunk_size: int,
    expected_rows: int,
    fingerprint: str,
    rebuild: bool,
) -> pd.DataFrame:
    if cache_path.is_file() and metadata_path.is_file() and not rebuild:
        metadata = read_json(metadata_path)
        if metadata.get("fingerprint") == fingerprint:
            print(f"复用候选几何缓存：{cache_path}", flush=True)
            return pd.read_parquet(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".building")
    if temporary_path.exists():
        temporary_path.unlink()

    writer: pq.ParquetWriter | None = None
    pending_rows: pd.DataFrame | None = None
    written_rows = 0
    chunk_count = 0
    start_time = time.perf_counter()
    try:
        for chunk_count, chunk in enumerate(
            _read_candidate_chunks(source_path, chunk_size), start=1
        ):
            combined = chunk if pending_rows is None else pd.concat(
                [pending_rows, chunk], ignore_index=True
            )
            well_values = combined["well"].astype(str)
            if not well_values.is_monotonic_increasing:
                raise ValueError("候选源文件没有按 well 排序，无法安全分块")

            last_well = str(well_values.iloc[-1])
            complete_mask = well_values != last_well
            complete_rows = combined.loc[complete_mask]
            pending_rows = combined.loc[~complete_mask].copy()
            if not complete_rows.empty:
                writer, batch_rows = _write_feature_batch(
                    complete_rows, writer, temporary_path
                )
                written_rows += batch_rows
            if chunk_count % 5 == 0:
                print(
                    f"候选缓存进度：chunk={chunk_count}，已写={written_rows:,}",
                    flush=True,
                )

        if pending_rows is None or pending_rows.empty:
            raise ValueError("候选源文件为空")
        writer, batch_rows = _write_feature_batch(pending_rows, writer, temporary_path)
        written_rows += batch_rows
    finally:
        if writer is not None:
            writer.close()

    if written_rows != int(expected_rows):
        raise ValueError(
            f"候选缓存行数不匹配：实际 {written_rows:,}，预期 {expected_rows:,}"
        )
    temporary_path.replace(cache_path)
    write_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "source_signature": _source_signature(source_path),
            "source_use_columns": SOURCE_USE_COLUMNS,
            "feature_columns": F05A_FEATURE_COLUMNS,
            "rows": written_rows,
            "chunks": chunk_count,
            "seconds": time.perf_counter() - start_time,
        },
    )
    print(f"候选几何缓存完成：{cache_path}，行={written_rows:,}", flush=True)
    return pd.read_parquet(cache_path)


def _model_fingerprint(
    experiment_config: dict,
    model_params: dict,
    registry_hash: str,
    candidate_cache_fingerprint: str,
) -> str:
    payload = {
        "experiment_config": experiment_config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "candidate_cache_fingerprint": candidate_cache_fingerprint,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    experiment_config = read_json(config_path)
    expected_feature_columns = [*FEATURE_COLUMNS, *F05A_FEATURE_COLUMNS]
    if experiment_config["feature_columns"] != expected_feature_columns:
        raise ValueError("F05a 配置中的特征顺序与代码不一致")

    source_path = PROJECT_ROOT / experiment_config["candidate_source"]
    base_cache_path = CLEAN_ROOT / experiment_config["base_feature_cache"]
    if not source_path.is_file() or not base_cache_path.is_file():
        raise FileNotFoundError("缺少候选源文件或 B00 特征缓存")

    chunk_size = int(experiment_config["candidate_chunk_size"])
    if args.mode == "smoke":
        run_smoke(source_path, base_cache_path, chunk_size)
        return

    artifact_dir = CLEAN_ROOT / "artifacts" / experiment_config["experiment_id"]
    candidate_cache_path = artifact_dir / "candidate_feature_cache.parquet"
    candidate_metadata_path = artifact_dir / "candidate_feature_cache.meta.json"
    candidate_fingerprint = build_candidate_cache_fingerprint(source_path)
    candidate_features = build_candidate_feature_cache(
        source_path=source_path,
        cache_path=candidate_cache_path,
        metadata_path=candidate_metadata_path,
        chunk_size=chunk_size,
        expected_rows=int(experiment_config["expected_rows"]),
        fingerprint=candidate_fingerprint,
        rebuild=bool(args.rebuild_candidate_cache),
    )

    base_features = pd.read_parquet(base_cache_path)
    _validate_key_alignment(base_features, candidate_features)
    feature_table = pd.concat(
        [
            base_features.reset_index(drop=True),
            candidate_features[F05A_FEATURE_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )

    registry_path = CLEAN_ROOT / experiment_config["fold_registry"]
    registry_hash = file_sha256(registry_path)
    if registry_hash != experiment_config["fold_registry_sha256"]:
        raise ValueError("fold 注册表 SHA-256 与冻结配置不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(experiment_config["expected_wells"]),
        expected_rows=int(experiment_config["expected_rows"]),
    )
    model_params = read_json(CLEAN_ROOT / experiment_config["model_config"])["params"]
    fingerprint = _model_fingerprint(
        experiment_config, model_params, registry_hash, candidate_fingerprint
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", experiment_config)
    write_json(
        artifact_dir / "feature_list.json",
        {
            "feature_count": len(expected_feature_columns),
            "features": expected_feature_columns,
        },
    )

    fold_id = int(args.mode[-1])
    runtime = train_fold(
        feature_table=feature_table,
        registry=registry,
        fold_id=fold_id,
        model_params=model_params,
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
        feature_columns=expected_feature_columns,
    )
    metrics_path = artifact_dir / "metrics.csv"
    existing = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    existing = existing.loc[existing.get("fold", pd.Series(dtype=int)) != fold_id]
    pd.concat([existing, pd.DataFrame([runtime])], ignore_index=True).sort_values(
        "fold"
    ).to_csv(metrics_path, index=False)

    finalize_complete_cv(
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
        expected_rows=int(experiment_config["expected_rows"]),
        experiment_config=experiment_config,
        model_params=model_params,
        feature_columns=expected_feature_columns,
    )


if __name__ == "__main__":
    main()

