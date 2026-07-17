"""读取确定性候选缓存，运行 B00 12 列加 DIRECT 24 列的固定 LightGBM CV。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f05a_direct_physical_candidates import (  # noqa: E402
    DIRECT_CANDIDATE_COLUMNS,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F05a_deterministic_candidates_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "f05a_deterministic_candidates_v1.json"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
EXPECTED_CANDIDATE_COLUMNS = [
    "well_id",
    "row_index",
    *DIRECT_CANDIDATE_COLUMNS,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke 或单折运行参数；本 runner 故意不提供重建缓存选项。"""

    parser = argparse.ArgumentParser(description="运行确定性候选特征单模 CV")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _file_signature(path: Path) -> dict[str, object]:
    """返回进入实验指纹的文件路径、大小和纳秒修改时间。"""

    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def require_candidate_cache_files(cache_path: Path, metadata_path: Path) -> None:
    """要求确定性缓存及元数据已存在；这里只读，绝不会自行重建。"""

    missing_paths = [
        str(path.resolve())
        for path in (cache_path, metadata_path)
        if not path.is_file()
    ]
    if missing_paths:
        joined = "\n".join(missing_paths)
        raise FileNotFoundError(
            "缺少确定性候选缓存。请先运行缓存生成脚本；CV runner 不会重建缓存：\n"
            f"{joined}"
        )


def read_candidate_cache_provenance(
    cache_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """读取完整缓存元数据，并加入缓存文件和元数据文件的轻量签名。"""

    require_candidate_cache_files(cache_path, metadata_path)
    return {
        "metadata": read_json(metadata_path),
        "cache_file_signature": _file_signature(cache_path),
        "metadata_file_signature": _file_signature(metadata_path),
    }


def build_candidate_cache_fingerprint(
    cache_path: Path,
    metadata_path: Path,
) -> str:
    """用缓存元数据和两个文件签名构造稳定 SHA-256 指纹。"""

    provenance = read_candidate_cache_provenance(cache_path, metadata_path)
    encoded = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_exact_key_alignment(
    base_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> None:
    """候选缓存必须与 B00 的 well_id、row_index 逐行同序。"""

    if len(base_rows) != len(candidate_rows):
        raise ValueError("确定性候选缓存与 B00 的键顺序不一致：行数不同")
    base_wells = base_rows["well_id"].astype(str).to_numpy()
    candidate_wells = candidate_rows["well_id"].astype(str).to_numpy()
    base_indices = pd.to_numeric(base_rows["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    candidate_indices = pd.to_numeric(
        candidate_rows["row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(base_wells, candidate_wells) or not np.array_equal(
        base_indices,
        candidate_indices,
    ):
        raise ValueError("确定性候选缓存与 B00 的键顺序不一致")


def _validate_config(experiment_config: dict[str, object]) -> list[str]:
    """确认配置仍是 B00 12 列加冻结 DIRECT 24 列。"""

    model_features = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
    if experiment_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("实验 ID 与确定性候选 runner 不一致")
    if experiment_config.get("feature_columns") != model_features:
        raise ValueError("配置特征顺序不是 B00 12 列加 DIRECT 24 列")
    if experiment_config.get("candidate_feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("配置中的候选特征顺序已变化")
    if experiment_config.get("candidate_cache_rebuild_allowed") is not False:
        raise ValueError("确定性候选 CV 禁止自行重建缓存")
    if experiment_config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("配置中的运行模式与 runner 不一致")
    return model_features


def _validate_cache_metadata(
    metadata: dict[str, object],
    expected_wells: int,
    expected_rows: int,
) -> None:
    """拒绝未完成、行数错误或特征顺序错误的候选缓存。"""

    if metadata.get("experiment_id") != "F05a_deterministic_candidate_cache_v1":
        raise ValueError("候选缓存 experiment_id 不正确")
    if metadata.get("completed") is not True:
        raise ValueError("候选缓存尚未完整生成")
    if int(metadata.get("wells", -1)) != int(expected_wells):
        raise ValueError("候选缓存井数不正确")
    if int(metadata.get("rows", -1)) != int(expected_rows):
        raise ValueError("候选缓存行数不正确")
    if metadata.get("keys_unique") is not True:
        raise ValueError("候选缓存键不唯一")
    if metadata.get("registry_order_preserved") is not True:
        raise ValueError("候选缓存没有保持 fold 注册表顺序")
    if metadata.get("feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("候选缓存的 24 列或顺序不正确")
    if int(metadata.get("feature_count", -1)) != len(DIRECT_CANDIDATE_COLUMNS):
        raise ValueError("候选缓存特征数不正确")
    if int(metadata.get("seed", -1)) != 42:
        raise ValueError("候选缓存不是 seed=42 的冻结版本")
    if not str(metadata.get("candidate_cache_sha256", "")):
        raise ValueError("候选缓存元数据缺少文件 SHA-256")
    if not str(metadata.get("cache_fingerprint", "")):
        raise ValueError("候选缓存元数据缺少生成指纹")


def load_aligned_feature_table(
    base_cache_path: Path,
    candidate_cache_path: Path,
    expected_rows: int,
) -> pd.DataFrame:
    """读取两个只读缓存，严格对齐后返回 12+24 特征表。"""

    if not base_cache_path.is_file():
        raise FileNotFoundError(f"缺少 B00 特征缓存：{base_cache_path.resolve()}")
    candidate_rows = pd.read_parquet(candidate_cache_path)
    if list(candidate_rows.columns) != EXPECTED_CANDIDATE_COLUMNS:
        raise ValueError("确定性候选缓存列或顺序不正确")
    if len(candidate_rows) != int(expected_rows):
        raise ValueError("确定性候选缓存行数与固定评价行不一致")
    if candidate_rows.duplicated(["well_id", "row_index"]).any():
        raise ValueError("确定性候选缓存含重复键")
    candidate_values = candidate_rows[DIRECT_CANDIDATE_COLUMNS].to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(candidate_values).all():
        raise ValueError("确定性候选缓存含 NaN 或 Inf")

    base_rows = pd.read_parquet(base_cache_path)
    if len(base_rows) != int(expected_rows):
        raise ValueError("B00 特征缓存行数与固定评价行不一致")
    validate_exact_key_alignment(base_rows, candidate_rows)

    feature_table = pd.concat(
        [
            base_rows.reset_index(drop=True),
            candidate_rows[DIRECT_CANDIDATE_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
    return feature_table


def _experiment_fingerprint(
    experiment_config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    cache_provenance: dict[str, object],
) -> str:
    """构造模型产物指纹，其中保留完整缓存 meta 和文件签名。"""

    payload = {
        "experiment_config": experiment_config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "candidate_cache_provenance": cache_provenance,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_smoke(feature_table: pd.DataFrame) -> None:
    """只检查前三口井的候选列和键，不训练模型。"""

    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    smoke_rows = feature_table.loc[
        feature_table["well_id"].astype(str).isin(smoke_wells)
    ]
    values = smoke_rows[DIRECT_CANDIDATE_COLUMNS].to_numpy(dtype=np.float32)
    if len(smoke_wells) != 3 or not np.isfinite(values).all():
        raise ValueError("确定性候选 smoke 检查失败")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，候选特征=24",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """加载冻结缓存并运行 smoke 或一个指定 fold。"""

    args = parse_args(argv)
    experiment_config = read_json(args.config.resolve())
    model_features = _validate_config(experiment_config)
    expected_wells = int(experiment_config["expected_wells"])
    expected_rows = int(experiment_config["expected_rows"])

    candidate_cache_path = CLEAN_ROOT / str(
        experiment_config["candidate_feature_cache"]
    )
    metadata_path = CLEAN_ROOT / str(experiment_config["candidate_cache_metadata"])
    require_candidate_cache_files(candidate_cache_path, metadata_path)
    cache_provenance = read_candidate_cache_provenance(
        candidate_cache_path,
        metadata_path,
    )
    metadata = cache_provenance["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("候选缓存 meta 必须是 JSON 对象")
    _validate_cache_metadata(metadata, expected_wells, expected_rows)

    base_cache_path = CLEAN_ROOT / str(experiment_config["base_feature_cache"])
    feature_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    if args.mode == "smoke":
        _run_smoke(feature_table)
        return

    registry_path = CLEAN_ROOT / str(experiment_config["fold_registry"])
    registry_hash = file_sha256(registry_path)
    if registry_hash != experiment_config["fold_registry_sha256"]:
        raise ValueError("fold 注册表 SHA-256 与冻结配置不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    model_params = read_json(
        CLEAN_ROOT / str(experiment_config["model_config"])
    )["params"]
    fingerprint = _experiment_fingerprint(
        experiment_config,
        model_params,
        registry_hash,
        cache_provenance,
    )

    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", experiment_config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(artifact_dir / "candidate_cache_provenance.json", cache_provenance)

    fold_id = int(args.mode[-1])
    runtime = train_fold(
        feature_table,
        registry,
        fold_id,
        model_params,
        artifact_dir,
        fingerprint,
        model_features,
    )
    metrics_path = artifact_dir / "metrics.csv"
    previous = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    if not previous.empty:
        previous = previous.loc[previous["fold"] != fold_id]
    pd.concat([previous, pd.DataFrame([runtime])], ignore_index=True).sort_values(
        "fold"
    ).to_csv(metrics_path, index=False)

    finalize_complete_cv(
        artifact_dir,
        fingerprint,
        expected_rows,
        experiment_config,
        model_params,
        model_features,
    )


if __name__ == "__main__":
    main()
