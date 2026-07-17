"""运行 B00 12 列加 F07R 10 列邻井特征的固定单模 CV。"""

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

from scripts.build_f07r_neighbor_cache import (  # noqa: E402
    CACHE_LINEAGE_COLUMNS,
    LINEAGE_COLUMNS,
    file_sha256,
)
from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f07r_neighbor_prior import F07R_FEATURE_COLUMNS  # noqa: E402
from src.lgbm_data import load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F07R_neighbor_relative_u_v1"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
EXPECTED_CACHE_COLUMNS = [
    "well_id",
    "row_index",
    *CACHE_LINEAGE_COLUMNS,
    *F07R_FEATURE_COLUMNS,
]
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "f07r_neighbor_relative_u_v1.json"


def mode_to_fold(mode: str) -> int | None:
    if mode == "smoke":
        return None
    if mode.startswith("fold") and mode[4:].isdigit():
        fold = int(mode[4:])
        if 0 <= fold <= 4:
            return fold
    raise ValueError(f"未知运行模式：{mode}")


def validate_cache_metadata(
    metadata: dict,
    expected_wells: int,
    expected_rows: int,
    expected_outer_fold: int,
) -> None:
    if metadata.get("experiment_id") != "F07R_neighbor_cache_v1":
        raise ValueError("F07R 缓存 experiment_id 错误")
    if metadata.get("completed") is not True:
        raise ValueError("F07R 缓存没有完整生成")
    if int(metadata.get("outer_fold", -1)) != int(expected_outer_fold):
        raise ValueError("F07R 缓存不是请求的 outer fold")
    if int(metadata.get("wells", -1)) != int(expected_wells):
        raise ValueError("F07R 缓存井数错误")
    if int(metadata.get("rows", -1)) != int(expected_rows):
        raise ValueError("F07R 缓存行数错误")
    if metadata.get("keys_unique") is not True:
        raise ValueError("F07R 缓存复合键不唯一")
    if metadata.get("feature_columns") != list(F07R_FEATURE_COLUMNS):
        raise ValueError("F07R 缓存特征列或顺序错误")
    if int(metadata.get("feature_count", -1)) != len(F07R_FEATURE_COLUMNS):
        raise ValueError("F07R 缓存特征数量错误")
    if metadata.get("lineage_columns") != LINEAGE_COLUMNS:
        raise ValueError("F07R source exclusion lineage 列错误")
    if int(metadata.get("source_fold_excluded", -1)) != int(expected_outer_fold):
        raise ValueError("F07R 缓存没有排除请求的验证 fold")
    if metadata.get("same_well_excluded") is not True:
        raise ValueError("F07R 缓存没有排除同井")
    if metadata.get("same_pad_excluded") is not True:
        raise ValueError("F07R 缓存没有排除同 pad")
    for key in (
        "cache_sha256",
        "lineage_sha256",
        "source_set_fingerprint",
        "family_fingerprint",
    ):
        if not str(metadata.get(key, "")):
            raise ValueError(f"F07R 缓存缺少 {key}")


def _validate_key_alignment(base_rows: pd.DataFrame, neighbor_rows: pd.DataFrame) -> None:
    if len(base_rows) != len(neighbor_rows):
        raise ValueError("F07R 缓存与 B00 键顺序不一致：行数不同")
    base_wells = base_rows["well_id"].astype(str).to_numpy()
    neighbor_wells = neighbor_rows["well_id"].astype(str).to_numpy()
    base_indices = pd.to_numeric(base_rows["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    neighbor_indices = pd.to_numeric(
        neighbor_rows["row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(base_wells, neighbor_wells) or not np.array_equal(
        base_indices, neighbor_indices
    ):
        raise ValueError("F07R 缓存与 B00 键顺序不一致")


def load_aligned_feature_table(
    base_cache_path: Path,
    neighbor_cache_path: Path,
    expected_rows: int,
    expected_outer_fold: int,
) -> pd.DataFrame:
    base_rows = pd.read_parquet(base_cache_path)
    neighbor_rows = pd.read_parquet(neighbor_cache_path)
    if list(neighbor_rows.columns) != EXPECTED_CACHE_COLUMNS:
        raise ValueError("F07R 缓存列或顺序错误")
    if len(base_rows) != int(expected_rows) or len(neighbor_rows) != int(expected_rows):
        raise ValueError("F07R 或 B00 缓存评价行数错误")
    if neighbor_rows.duplicated(["well_id", "row_index"]).any():
        raise ValueError("F07R 缓存含重复键")
    if not neighbor_rows["outer_fold"].astype(int).eq(int(expected_outer_fold)).all():
        raise ValueError("F07R 行级 lineage 的 outer fold 错误")
    expected_roles = np.where(
        base_rows["fold"].astype(int).to_numpy() == int(expected_outer_fold),
        "validation",
        "outer_train",
    )
    if not np.array_equal(neighbor_rows["target_role"].astype(str), expected_roles):
        raise ValueError("F07R 行级 lineage 的 target role 错误")
    values = neighbor_rows[list(F07R_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("F07R 缓存含 Inf")
    _validate_key_alignment(base_rows, neighbor_rows)
    return pd.concat(
        [
            base_rows.reset_index(drop=True),
            neighbor_rows[list(F07R_FEATURE_COLUMNS)].reset_index(drop=True),
        ],
        axis=1,
    )


def _validate_config(config: dict) -> list[str]:
    model_features = [*FEATURE_COLUMNS, *F07R_FEATURE_COLUMNS]
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("F07R 配置 experiment_id 错误")
    if config.get("feature_columns") != model_features:
        raise ValueError("F07R 正式特征必须是 B00 加固定 10 列")
    if int(config.get("feature_count", -1)) != len(model_features):
        raise ValueError("F07R 正式特征数量错误")
    if int(config.get("n_estimators", -1)) != 1734:
        raise ValueError("F07R 必须固定 1734 棵树")
    if int(config.get("model_seed", -1)) != 29:
        raise ValueError("F07R 必须固定模型 seed 29")
    if config.get("early_stopping") is not False:
        raise ValueError("F07R 禁止 early stopping")
    if config.get("target") != "target_delta":
        raise ValueError("F07R target 必须保持不变")
    if config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("F07R 运行模式配置错误")
    for key in (
        "neighbor_cache_pattern",
        "cache_metadata_pattern",
        "lineage_manifest_pattern",
    ):
        if "{fold}" not in str(config.get(key, "")):
            raise ValueError(f"F07R 配置 {key} 必须按 fold 分离")
    return model_features


def _experiment_fingerprint(
    config: dict,
    model_params: dict,
    registry_hash: str,
) -> str:
    """所有 fold 共用一个模型实验指纹；每折 cache SHA 另存到 runtime。"""

    payload = {
        "config": config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "generator_sha256": file_sha256(CLEAN_ROOT / "src" / "f07r_neighbor_prior.py"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_clean_path(pattern: str, fold: int) -> Path:
    return CLEAN_ROOT / pattern.format(fold=int(fold))


def _smoke(config: dict) -> None:
    fold = 0
    cache_path = _format_clean_path(config["neighbor_cache_pattern"], fold)
    if cache_path.is_file():
        smoke = pd.read_parquet(cache_path).head(10_000)
    else:
        per_well_dir = cache_path.parent / "per_well"
        files = sorted(per_well_dir.glob("*.parquet"))[:3]
        if len(files) != 3:
            raise FileNotFoundError("F07R smoke 需要先生成 3 口单井缓存")
        smoke = pd.concat(
            [pd.read_parquet(path)[EXPECTED_CACHE_COLUMNS] for path in files],
            ignore_index=True,
        )
    if list(smoke.columns) != EXPECTED_CACHE_COLUMNS:
        raise ValueError("F07R smoke 缓存列错误")
    values = smoke[list(F07R_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("F07R smoke 特征含非有限值")
    print(
        f"F07R smoke 完成：{smoke['well_id'].nunique()} 井，"
        f"{len(smoke):,} 行，新增 {len(F07R_FEATURE_COLUMNS)} 列",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F07R smoke/fold0..4")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config.resolve())
    model_features = _validate_config(config)
    if args.mode == "smoke":
        _smoke(config)
        return

    fold_id = mode_to_fold(args.mode)
    if fold_id is None:
        raise AssertionError("fold 模式不应解析为空")
    expected_wells = int(config["expected_wells"])
    expected_rows = int(config["expected_rows"])
    cache_path = _format_clean_path(config["neighbor_cache_pattern"], fold_id)
    metadata_path = _format_clean_path(config["cache_metadata_pattern"], fold_id)
    lineage_path = _format_clean_path(config["lineage_manifest_pattern"], fold_id)
    if not cache_path.is_file() or not metadata_path.is_file() or not lineage_path.is_file():
        raise FileNotFoundError(f"F07R fold{fold_id} 缓存或 lineage 未完整生成")

    metadata = read_json(metadata_path)
    validate_cache_metadata(metadata, expected_wells, expected_rows, fold_id)
    if file_sha256(cache_path) != metadata["cache_sha256"]:
        raise ValueError("F07R 缓存实际 SHA-256 与 metadata 不一致")
    if file_sha256(lineage_path) != metadata["lineage_sha256"]:
        raise ValueError("F07R lineage 实际 SHA-256 与 metadata 不一致")
    feature_table = load_aligned_feature_table(
        CLEAN_ROOT / str(config["base_feature_cache"]),
        cache_path,
        expected_rows,
        fold_id,
    )

    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    registry_hash = file_sha256(registry_path)
    if registry_hash != config["fold_registry_sha256"]:
        raise ValueError("F07R fold 注册表 SHA-256 不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    model_params = read_json(CLEAN_ROOT / str(config["model_config"]))["params"]
    fingerprint = _experiment_fingerprint(config, model_params, registry_hash)
    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(artifact_dir / f"fold_{fold_id}_cache_provenance.json", metadata)

    runtime_path = artifact_dir / f"fold_{fold_id}_runtime.json"
    if runtime_path.is_file():
        previous = read_json(runtime_path)
        previous_cache_hash = previous.get("f07r_cache_sha256")
        if previous_cache_hash not in (None, metadata["cache_sha256"]):
            raise ValueError("已有 fold 结果来自另一份 F07R 缓存，拒绝静默复用")
    runtime = train_fold(
        feature_table,
        registry,
        fold_id,
        model_params,
        artifact_dir,
        fingerprint,
        model_features,
    )
    runtime["f07r_cache_sha256"] = metadata["cache_sha256"]
    runtime["f07r_lineage_sha256"] = metadata["lineage_sha256"]
    write_json(runtime_path, runtime)
    pd.DataFrame([runtime]).to_csv(
        artifact_dir / f"fold_{fold_id}_metrics.csv",
        index=False,
    )
    metrics = finalize_complete_cv(
        artifact_dir,
        fingerprint,
        expected_rows,
        config,
        model_params,
        model_features,
    )
    if metrics is None:
        print(f"F07R fold{fold_id} 已保存；尚未完成五折。", flush=True)


if __name__ == "__main__":
    main()
