"""运行 B00 12 列加 F06 前缀伪 holdout 特征的固定单模 CV。"""

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
from src.f06_prefix_holdout import F06_FEATURE_COLUMNS  # noqa: E402
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F06_prefix_holdout_v1"
F06_COLUMNS = list(F06_FEATURE_COLUMNS)
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "f06_prefix_holdout_v1.json"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
EXPECTED_CACHE_COLUMNS = ["well_id", "row_index", *F06_COLUMNS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F06 前缀伪 holdout 固定 CV")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def require_prefix_holdout_cache(cache_path: Path, metadata_path: Path) -> None:
    """CV 只读取完整缓存；缺失时明确报错，不在训练阶段自动重建。"""

    missing = [
        str(path.resolve())
        for path in (cache_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少 F06 缓存；请先运行缓存生成脚本，CV 不会重建：\n"
            + "\n".join(missing)
        )


def validate_exact_key_alignment(
    base_rows: pd.DataFrame,
    holdout_rows: pd.DataFrame,
) -> None:
    """F06 缓存必须与 B00 的 well_id、row_index 逐行同序。"""

    if len(base_rows) != len(holdout_rows):
        raise ValueError("F06 缓存与 B00 键顺序不一致：行数不同")
    base_wells = base_rows["well_id"].astype(str).to_numpy()
    holdout_wells = holdout_rows["well_id"].astype(str).to_numpy()
    base_indices = pd.to_numeric(base_rows["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    holdout_indices = pd.to_numeric(
        holdout_rows["row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(base_wells, holdout_wells) or not np.array_equal(
        base_indices,
        holdout_indices,
    ):
        raise ValueError("F06 缓存与 B00 键顺序不一致")


def _file_signature(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def read_cache_provenance(cache_path: Path, metadata_path: Path) -> dict:
    require_prefix_holdout_cache(cache_path, metadata_path)
    return {
        "metadata": read_json(metadata_path),
        "cache_file_signature": _file_signature(cache_path),
        "metadata_file_signature": _file_signature(metadata_path),
    }


def _validate_config(config: dict) -> list[str]:
    model_features = [*FEATURE_COLUMNS, *F06_COLUMNS]
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("配置 experiment_id 与 F06 runner 不一致")
    if config.get("feature_columns") != model_features:
        raise ValueError("正式特征必须是 B00 12 列加 F06 全部列")
    if config.get("prefix_holdout_feature_columns") != F06_COLUMNS:
        raise ValueError("配置中的 F06 特征列或顺序已变化")
    if int(config.get("feature_count", -1)) != len(model_features):
        raise ValueError("F06 正式模型特征数量不正确")
    if config.get("model_config") != "configs/lgbm_feature_baseline_v1.json":
        raise ValueError("F06 必须使用冻结的单模 LightGBM 配置")
    if int(config.get("n_estimators", -1)) != 1734:
        raise ValueError("F06 必须固定使用 1734 棵树")
    if int(config.get("model_seed", -1)) != 29:
        raise ValueError("F06 必须固定使用 LightGBM seed 29")
    if config.get("early_stopping") is not False:
        raise ValueError("F06 禁止 early stopping")
    if config.get("target") != "target_delta":
        raise ValueError("F06 target 必须保持 TVT 减最后可见 TVT")
    if config.get("cache_rebuild_allowed") is not False:
        raise ValueError("F06 CV 禁止自行重建缓存")
    if config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("配置支持的运行模式与 runner 不一致")
    return model_features


def _validate_cache_metadata(
    metadata: dict,
    expected_wells: int,
    expected_rows: int,
) -> None:
    if metadata.get("experiment_id") != "F06_prefix_holdout_cache_v1":
        raise ValueError("F06 缓存 experiment_id 不正确")
    if metadata.get("completed") is not True:
        raise ValueError("F06 缓存尚未完整生成")
    if int(metadata.get("wells", -1)) != expected_wells:
        raise ValueError("F06 缓存井数不正确")
    if int(metadata.get("rows", -1)) != expected_rows:
        raise ValueError("F06 缓存行数不正确")
    if metadata.get("keys_unique") is not True:
        raise ValueError("F06 缓存复合键不唯一")
    if metadata.get("registry_order_preserved") is not True:
        raise ValueError("F06 缓存没有保持固定 fold 注册表顺序")
    if metadata.get("feature_columns") != F06_COLUMNS:
        raise ValueError("F06 缓存特征列或顺序不正确")
    if int(metadata.get("feature_count", -1)) != len(F06_COLUMNS):
        raise ValueError("F06 缓存特征数不正确")
    if int(metadata.get("seed", -1)) != 42:
        raise ValueError("F06 缓存不是固定 seed=42")
    if not str(metadata.get("cache_sha256", "")):
        raise ValueError("F06 缓存元数据缺少 SHA-256")
    if not str(metadata.get("cache_fingerprint", "")):
        raise ValueError("F06 缓存元数据缺少生成指纹")


def load_aligned_feature_table(
    base_cache_path: Path,
    prefix_holdout_cache_path: Path,
    expected_rows: int,
) -> pd.DataFrame:
    if not base_cache_path.is_file():
        raise FileNotFoundError(f"缺少 B00 特征缓存：{base_cache_path.resolve()}")
    holdout_rows = pd.read_parquet(prefix_holdout_cache_path)
    if list(holdout_rows.columns) != EXPECTED_CACHE_COLUMNS:
        raise ValueError("F06 缓存列或顺序不正确")
    if len(holdout_rows) != expected_rows:
        raise ValueError("F06 缓存行数与固定评价行不一致")
    if holdout_rows.duplicated(["well_id", "row_index"]).any():
        raise ValueError("F06 缓存含重复键")
    # 逐列检查 Inf，避免把 174×378 万的完整特征块再复制成 2.45 GiB 数组。
    for feature_name in F06_COLUMNS:
        feature_values = holdout_rows[feature_name].to_numpy(copy=False)
        if np.isinf(feature_values).any():
            raise ValueError(f"F06 缓存含 Inf：{feature_name}")

    base_rows = pd.read_parquet(base_cache_path)
    if len(base_rows) != expected_rows:
        raise ValueError("B00 特征缓存行数与固定评价行不一致")
    validate_exact_key_alignment(base_rows, holdout_rows)

    # 键已经逐行验证相同，因此可以直接共享 RangeIndex，不需要 reset_index 深拷贝。
    holdout_rows.pop("well_id")
    holdout_rows.pop("row_index")
    holdout_rows.index = base_rows.index

    # copy=False 保留两个表已有的数据块，只拼接列索引，降低全量运行的峰值内存。
    return pd.concat([base_rows, holdout_rows], axis=1, copy=False)


def _experiment_fingerprint(
    config: dict,
    model_params: dict,
    registry_hash: str,
    cache_provenance: dict,
) -> str:
    payload = {
        "config": config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "cache_provenance": cache_provenance,
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
    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    smoke_rows = feature_table.loc[
        feature_table["well_id"].astype(str).isin(smoke_wells)
    ]
    values = smoke_rows[F06_COLUMNS].to_numpy(dtype=np.float32)
    if len(smoke_wells) != 3 or not np.isfinite(values).all():
        raise ValueError("F06 smoke 检查失败")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，"
        f"新增特征={len(F06_COLUMNS)}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config.resolve())
    model_features = _validate_config(config)
    expected_wells = int(config["expected_wells"])
    expected_rows = int(config["expected_rows"])

    cache_path = CLEAN_ROOT / str(config["prefix_holdout_cache"])
    metadata_path = CLEAN_ROOT / str(config["cache_metadata"])
    provenance = read_cache_provenance(cache_path, metadata_path)
    metadata = provenance["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("F06 缓存 meta 必须是 JSON 对象")
    _validate_cache_metadata(metadata, expected_wells, expected_rows)

    feature_table = load_aligned_feature_table(
        CLEAN_ROOT / str(config["base_feature_cache"]),
        cache_path,
        expected_rows,
    )
    if args.mode == "smoke":
        _run_smoke(feature_table)
        return

    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    registry_hash = file_sha256(registry_path)
    if registry_hash != config["fold_registry_sha256"]:
        raise ValueError("fold 注册表 SHA-256 与冻结配置不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    model_params = read_json(CLEAN_ROOT / str(config["model_config"]))["params"]
    if int(model_params.get("n_estimators", -1)) != 1734:
        raise ValueError("冻结 LightGBM 必须使用 1734 棵树")
    if int(model_params.get("random_state", -1)) != 29:
        raise ValueError("冻结 LightGBM 必须使用 seed 29")
    fingerprint = _experiment_fingerprint(
        config,
        model_params,
        registry_hash,
        provenance,
    )

    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(artifact_dir / "cache_provenance.json", provenance)

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
        config,
        model_params,
        model_features,
    )


if __name__ == "__main__":
    main()
