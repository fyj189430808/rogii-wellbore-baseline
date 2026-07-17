"""为 F07R 逐 outer fold 生成严格防泄漏的邻井特征缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.f07r_neighbor_prior import (  # noqa: E402
    F07R_FEATURE_COLUMNS,
    SOURCE_INPUT_COLUMNS,
    NeighborProfile,
    allowed_source_ids,
    build_neighbor_prior_features,
    build_source_profile,
)
from src.lgbm_data import load_and_validate_registry  # noqa: E402


EXPERIMENT_ID = "F07R_neighbor_cache_v1"
EXPECTED_REGISTRY_SHA256 = (
    "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab"
)
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
DEFAULT_RAW_TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
DEFAULT_REGISTRY_PATH = CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "f07r_neighbor_relative_u_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

# 目标井只读取测试期可见列；GR 当前不进入公式，但保留在合法输入合同中。
TARGET_INPUT_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
CACHE_LINEAGE_COLUMNS = ("outer_fold", "target_pad_id", "target_role")
LINEAGE_COLUMNS = [
    "outer_fold",
    "target_well_id",
    "target_pad_id",
    "target_role",
    "target_legal_sha256",
    "allowed_source_well_count",
    "allowed_source_well_ids_json",
    "allowed_source_set_sha256",
    "candidate_source_well_count",
    "candidate_source_well_ids_json",
    "source_fold_ids_json",
    "same_well_excluded",
    "same_pad_excluded",
    "validation_fold_excluded",
]
FINGERPRINT_COLUMN = "__cache_fingerprint"
MAX_BBOX_CANDIDATES = 64
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    hasher = hashlib.sha256()
    hasher.update(json.dumps(list(frame.columns), ensure_ascii=False).encode("utf-8"))
    hasher.update(json.dumps([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    hasher.update(payload.tobytes())
    return hasher.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.building")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.building")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _per_well_path(per_well_dir: Path, well_id: str) -> Path:
    return per_well_dir / f"{well_id}.parquet"


def _validate_registry(registry_df: pd.DataFrame, outer_fold: int) -> pd.DataFrame:
    required = {"well_id", "pad_id", "fold", "hidden_rows"}
    missing = required - set(registry_df.columns)
    if missing:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing)}")
    registry = registry_df.copy()
    registry["well_id"] = registry["well_id"].astype(str)
    registry["pad_id"] = registry["pad_id"].astype(str)
    registry["fold"] = registry["fold"].astype(int)
    if registry["well_id"].duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")
    if int(registry.groupby("pad_id")["fold"].nunique().max()) != 1:
        raise ValueError("同一个 pad 出现在多个 fold")
    if int(outer_fold) not in set(registry["fold"].tolist()):
        raise ValueError(f"outer fold {outer_fold} 不在注册表中")
    return registry.sort_values("well_id").reset_index(drop=True)


def _load_source_profiles(
    registry: pd.DataFrame,
    raw_train_dir: Path,
    outer_fold: int,
) -> tuple[dict[str, NeighborProfile], str, list[int]]:
    """每个 outer fold 只预载并下采样一次全部合法 source 井。"""

    source_rows = registry.loc[registry["fold"] != int(outer_fold)]
    profiles: dict[str, NeighborProfile] = {}
    lineage: list[dict[str, str]] = []
    for row in source_rows.itertuples(index=False):
        well_id = str(row.well_id)
        path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        source = pd.read_csv(path, usecols=list(SOURCE_INPUT_COLUMNS))
        source = source.loc[:, list(SOURCE_INPUT_COLUMNS)]
        source_hash = _stable_frame_fingerprint(source)
        profiles[well_id] = NeighborProfile(
            well_id=well_id,
            pad_id=str(row.pad_id),
            rows=build_source_profile(source),
        )
        lineage.append(
            {
                "well_id": well_id,
                "pad_id": str(row.pad_id),
                "legal_source_sha256": source_hash,
            }
        )
    if not profiles:
        raise ValueError("当前 outer fold 没有可用的训练 source 井")
    source_fold_ids = sorted(source_rows["fold"].astype(int).unique().tolist())
    return profiles, _json_fingerprint(lineage), source_fold_ids


def _bbox_distance(target: pd.DataFrame, profile: NeighborProfile) -> float:
    target_min_x = float(target["X"].min())
    target_max_x = float(target["X"].max())
    target_min_y = float(target["Y"].min())
    target_max_y = float(target["Y"].max())
    source = profile.rows
    source_min_x = float(source["X"].min())
    source_max_x = float(source["X"].max())
    source_min_y = float(source["Y"].min())
    source_max_y = float(source["Y"].max())
    dx = max(target_min_x - source_max_x, source_min_x - target_max_x, 0.0)
    dy = max(target_min_y - source_max_y, source_min_y - target_max_y, 0.0)
    return float(np.hypot(dx, dy))


def _candidate_profiles(
    target: pd.DataFrame,
    allowed_ids: list[str],
    source_profiles: dict[str, NeighborProfile],
) -> list[NeighborProfile]:
    available = [source_profiles[well_id] for well_id in allowed_ids]
    ranked = sorted(
        available,
        key=lambda profile: (_bbox_distance(target, profile), profile.well_id),
    )
    return ranked[:MAX_BBOX_CANDIDATES]


def _expected_per_well_columns() -> list[str]:
    return [
        "well_id",
        "row_index",
        *CACHE_LINEAGE_COLUMNS,
        *F07R_FEATURE_COLUMNS,
        FINGERPRINT_COLUMN,
    ]


def _validate_per_well_cache(
    frame: pd.DataFrame,
    well_id: str,
    fingerprint: str,
    expected_hidden_rows: int,
    outer_fold: int,
    target_pad_id: str,
) -> None:
    if list(frame.columns) != _expected_per_well_columns():
        raise ValueError(f"{well_id} F07R 单井缓存列或顺序错误")
    if len(frame) != int(expected_hidden_rows):
        raise ValueError(f"{well_id} F07R 单井缓存行数错误")
    if not frame["well_id"].astype(str).eq(str(well_id)).all():
        raise ValueError(f"{well_id} F07R 单井缓存混入其他井")
    if not frame[FINGERPRINT_COLUMN].astype(str).eq(str(fingerprint)).all():
        raise ValueError(f"{well_id} F07R 单井缓存指纹错误")
    if not frame["outer_fold"].astype(int).eq(int(outer_fold)).all():
        raise ValueError(f"{well_id} F07R 单井缓存 outer fold 错误")
    if not frame["target_pad_id"].astype(str).eq(str(target_pad_id)).all():
        raise ValueError(f"{well_id} F07R 单井缓存 target pad 错误")
    row_index = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(np.int64)
    if len(np.unique(row_index)) != len(row_index):
        raise ValueError(f"{well_id} F07R 单井缓存含重复 row_index")
    if len(row_index) > 1 and not np.all(np.diff(row_index) > 0):
        raise ValueError(f"{well_id} F07R 单井缓存 row_index 未递增")
    values = frame[list(F07R_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{well_id} F07R 单井缓存含非有限特征")


def _merge_per_well(
    registry: pd.DataFrame,
    per_well_dir: Path,
    fingerprints: dict[str, str],
    output_path: Path,
    outer_fold: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.building")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for row in registry.itertuples(index=False):
            well_id = str(row.well_id)
            frame = pd.read_parquet(_per_well_path(per_well_dir, well_id))
            _validate_per_well_cache(
                frame,
                well_id,
                fingerprints[well_id],
                int(row.hidden_rows),
                outer_fold,
                str(row.pad_id),
            )
            output = frame[
                [
                    "well_id",
                    "row_index",
                    *CACHE_LINEAGE_COLUMNS,
                    *F07R_FEATURE_COLUMNS,
                ]
            ].copy()
            output[list(F07R_FEATURE_COLUMNS)] = output[
                list(F07R_FEATURE_COLUMNS)
            ].astype(np.float32)
            table = pa.Table.from_pandas(output, preserve_index=False)
            table = table.replace_schema_metadata(None)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total_rows += len(output)
        if writer is None:
            raise ValueError("没有可合并的 F07R 单井缓存")
        writer.close()
        writer = None
        os.replace(temporary, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return int(total_rows)


def build_fold_feature_cache(
    registry_df: pd.DataFrame,
    raw_train_dir: Path,
    artifact_dir: Path,
    outer_fold: int,
    config_fingerprint: str,
    limit: int | None = None,
) -> dict[str, object]:
    """生成或续跑一个 outer fold；limit 非空时只保留可续跑单井缓存。"""

    outer_fold = int(outer_fold)
    registry = _validate_registry(registry_df, outer_fold)
    raw_train_dir = Path(raw_train_dir)
    fold_dir = Path(artifact_dir) / f"fold_{outer_fold}"
    per_well_dir = fold_dir / "per_well"
    per_well_dir.mkdir(parents=True, exist_ok=True)
    source_profiles, source_set_fingerprint, source_fold_ids = _load_source_profiles(
        registry,
        raw_train_dir,
        outer_fold,
    )
    family_fingerprint = _json_fingerprint(
        {
            "config": str(config_fingerprint),
            "generator_sha256": file_sha256(CLEAN_ROOT / "src" / "f07r_neighbor_prior.py"),
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "outer_fold": outer_fold,
        }
    )
    selected = registry if limit is None else registry.iloc[: int(limit)]
    fingerprints: dict[str, str] = {}
    lineage_rows: list[dict[str, object]] = []
    generated_wells = 0
    reused_wells = 0

    for number, row in enumerate(selected.itertuples(index=False), start=1):
        well_id = str(row.well_id)
        target_pad_id = str(row.pad_id)
        target_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        target = pd.read_csv(target_path, usecols=list(TARGET_INPUT_COLUMNS))
        target = target.loc[:, list(TARGET_INPUT_COLUMNS)]
        target_hash = _stable_frame_fingerprint(target)
        allowed_ids = allowed_source_ids(registry, outer_fold, well_id)
        candidates = _candidate_profiles(target, allowed_ids, source_profiles)
        candidate_ids = [profile.well_id for profile in candidates]
        allowed_hash = _json_fingerprint(allowed_ids)
        fingerprint = _json_fingerprint(
            {
                "family": family_fingerprint,
                "source_set": source_set_fingerprint,
                "target_legal": target_hash,
                "allowed_sources": allowed_ids,
                "candidate_sources": candidate_ids,
            }
        )
        fingerprints[well_id] = fingerprint
        target_role = "validation" if int(row.fold) == outer_fold else "outer_train"
        lineage_rows.append(
            {
                "outer_fold": outer_fold,
                "target_well_id": well_id,
                "target_pad_id": target_pad_id,
                "target_role": target_role,
                "target_legal_sha256": target_hash,
                "allowed_source_well_count": len(allowed_ids),
                "allowed_source_well_ids_json": json.dumps(allowed_ids, separators=(",", ":")),
                "allowed_source_set_sha256": allowed_hash,
                "candidate_source_well_count": len(candidate_ids),
                "candidate_source_well_ids_json": json.dumps(candidate_ids, separators=(",", ":")),
                "source_fold_ids_json": json.dumps(source_fold_ids, separators=(",", ":")),
                "same_well_excluded": well_id not in allowed_ids,
                "same_pad_excluded": all(
                    source_profiles[source_id].pad_id != target_pad_id
                    for source_id in allowed_ids
                ),
                # source_profiles 只由 fold != outer_fold 的井预载；成员检查就是直接的排除证据。
                "validation_fold_excluded": all(
                    source_id in source_profiles for source_id in allowed_ids
                ),
            }
        )

        cache_path = _per_well_path(per_well_dir, well_id)
        reusable = False
        if cache_path.is_file():
            try:
                existing = pd.read_parquet(cache_path)
                _validate_per_well_cache(
                    existing,
                    well_id,
                    fingerprint,
                    int(row.hidden_rows),
                    outer_fold,
                    target_pad_id,
                )
                reusable = True
            except Exception:
                reusable = False
        if reusable:
            reused_wells += 1
        else:
            features = build_neighbor_prior_features(
                target,
                candidates,
                target_well_id=well_id,
                target_pad_id=target_pad_id,
            )
            expected_indices = target.index[
                target["TVT_input"].isna()
            ].to_numpy(dtype=np.int64)
            if not np.array_equal(
                features["row_index"].to_numpy(dtype=np.int64), expected_indices
            ):
                raise ValueError(f"{well_id} F07R row_index 与自然隐藏 mask 不一致")
            output = features.copy()
            output[list(F07R_FEATURE_COLUMNS)] = output[
                list(F07R_FEATURE_COLUMNS)
            ].astype(np.float32)
            output.insert(0, "well_id", well_id)
            output.insert(2, "outer_fold", np.int8(outer_fold))
            output.insert(3, "target_pad_id", target_pad_id)
            output.insert(4, "target_role", target_role)
            output[FINGERPRINT_COLUMN] = fingerprint
            _validate_per_well_cache(
                output,
                well_id,
                fingerprint,
                int(row.hidden_rows),
                outer_fold,
                target_pad_id,
            )
            _write_parquet_atomic(cache_path, output)
            generated_wells += 1

        if number % 25 == 0 or number == len(selected):
            print(
                f"F07R fold{outer_fold} 缓存：{number}/{len(selected)}，"
                f"生成={generated_wells}，复用={reused_wells}",
                flush=True,
            )

    completed = limit is None or int(limit) >= len(registry)
    if not completed:
        return {
            "experiment_id": EXPERIMENT_ID,
            "completed": False,
            "outer_fold": outer_fold,
            "generated_wells": generated_wells,
            "reused_wells": reused_wells,
            "per_well_dir": str(per_well_dir),
            "source_set_fingerprint": source_set_fingerprint,
        }

    cache_path = fold_dir / "neighbor_feature_cache.parquet"
    lineage_path = fold_dir / "source_exclusion_lineage.parquet"
    metadata_path = fold_dir / "meta.json"
    total_rows = _merge_per_well(
        registry,
        per_well_dir,
        fingerprints,
        cache_path,
        outer_fold,
    )
    lineage = pd.DataFrame(lineage_rows, columns=LINEAGE_COLUMNS)
    if not lineage[["same_well_excluded", "same_pad_excluded", "validation_fold_excluded"]].all().all():
        raise ValueError("F07R source exclusion lineage 检查失败")
    _write_parquet_atomic(lineage_path, lineage)
    validation_pad_ids = sorted(
        registry.loc[registry["fold"] == outer_fold, "pad_id"].astype(str).unique().tolist()
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "outer_fold": outer_fold,
        "wells": int(len(registry)),
        "rows": total_rows,
        "keys_unique": True,
        "feature_columns": list(F07R_FEATURE_COLUMNS),
        "feature_count": len(F07R_FEATURE_COLUMNS),
        "cache_lineage_columns": list(CACHE_LINEAGE_COLUMNS),
        "lineage_columns": LINEAGE_COLUMNS,
        "source_fold_excluded": outer_fold,
        "validation_pad_ids": validation_pad_ids,
        "same_well_excluded": True,
        "same_pad_excluded": True,
        "source_set_fingerprint": source_set_fingerprint,
        "family_fingerprint": family_fingerprint,
        "cache_path": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "lineage_path": str(lineage_path),
        "lineage_sha256": file_sha256(lineage_path),
    }
    _write_json_atomic(metadata_path, metadata)
    return {
        **metadata,
        "generated_wells": generated_wells,
        "reused_wells": reused_wells,
        "metadata_path": str(metadata_path),
    }


def mode_to_fold(mode: str) -> int:
    if mode == "smoke":
        return 0
    if mode.startswith("fold") and mode[4:].isdigit():
        fold = int(mode[4:])
        if 0 <= fold <= 4:
            return fold
    raise ValueError(f"未知运行模式：{mode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 F07R 分 fold 邻井特征缓存")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_RAW_TRAIN_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry_path = args.registry.resolve()
    if file_sha256(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("固定 fold 注册表 SHA-256 不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=EXPECTED_WELLS,
        expected_rows=EXPECTED_ROWS,
    )
    outer_fold = mode_to_fold(args.mode)
    result = build_fold_feature_cache(
        registry_df=registry,
        raw_train_dir=args.train_dir.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        outer_fold=outer_fold,
        config_fingerprint=file_sha256(args.config.resolve()),
        limit=3 if args.mode == "smoke" else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
