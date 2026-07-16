"""运行 B00：12 个基础特征、单个 LightGBM、固定 spatial-pad 五折。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# 当前文件位于 rogii_clean/scripts，因此父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 主项目根目录默认包含未提交到 Git 的 input 数据目录。
PROJECT_ROOT = CLEAN_ROOT.parent

# 让直接运行脚本时可以导入 rogii_clean/src。
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_data import build_feature_table, file_sha256, load_and_validate_registry
from src.f01b_features import ALL_F01B_FEATURE_COLUMNS, build_f01b_lgbm_rows
from src.f01c_features import ALL_F01C_FEATURE_COLUMNS, build_f01c_lgbm_rows
from src.f01_features import ALL_FEATURE_COLUMNS, build_f01_lgbm_rows
from src.f02_features import ALL_F02_FEATURE_COLUMNS, build_f02_lgbm_rows
from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows
from src.metrics import (
    build_per_well_metrics,
    paired_well_bootstrap,
    summarize_by_fold,
    summarize_per_well_metrics,
)


# 读取 JSON 配置并返回普通字典，不做隐式覆盖。
def read_json(path: Path) -> dict:
    """读取 UTF-8 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


# 统一保存 UTF-8 JSON，保证实验配置和指标易于人工检查。
def write_json(path: Path, value: dict | list) -> None:
    """创建父目录并写入格式化 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# 根据 DataFrame 中冻结的 fold 列返回训练行和验证行的位置。
def fold_indices(
    feature_table: pd.DataFrame,
    validation_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回互斥的训练与验证整数行索引。"""

    fold_values = feature_table["fold"].to_numpy(dtype=np.int8)
    validation_indices = np.flatnonzero(fold_values == int(validation_fold))
    train_indices = np.flatnonzero(fold_values != int(validation_fold))

    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("训练行或验证行为空")

    return train_indices, validation_indices


# 模型预测的是相对可见末值的 delta，这里还原绝对 TVT。
def restore_absolute_tvt(
    anchor: np.ndarray,
    predicted_delta: np.ndarray,
) -> np.ndarray:
    """逐行计算 last_visible_tvt + predicted_delta。"""

    anchor_values = np.asarray(anchor, dtype=np.float64)
    delta_values = np.asarray(predicted_delta, dtype=np.float64)

    if anchor_values.shape != delta_values.shape:
        raise ValueError("锚点与 delta 预测 shape 不一致")

    return anchor_values + delta_values


# 配置指纹覆盖 fold、特征、模型参数和当前三份核心代码。
def build_experiment_fingerprint(
    experiment_config: dict,
    model_params: dict,
    registry_hash: str,
) -> str:
    """返回用于缓存和 fold 复用的 SHA-256 指纹。"""

    source_paths = [
        Path(__file__).resolve(),
        CLEAN_ROOT / "src" / "lgbm_features.py",
        CLEAN_ROOT / "src" / "f01_features.py",
        CLEAN_ROOT / "src" / "f01b_features.py",
        CLEAN_ROOT / "src" / "f01c_features.py",
        CLEAN_ROOT / "src" / "f02_features.py",
        CLEAN_ROOT / "src" / "lgbm_data.py",
    ]
    source_hashes = {path.name: file_sha256(path) for path in source_paths}

    payload = {
        "experiment_config": experiment_config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "source_hashes": source_hashes,
    }
    encoded_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


# 将大表压缩成适合内存和 LightGBM 使用的 dtype。
def optimize_feature_table(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """下调模型列 dtype，同时保留评分列的 float64 精度。"""

    optimized = feature_table.copy()
    optimized["well_id"] = optimized["well_id"].astype("category")
    optimized["fold"] = optimized["fold"].astype(np.int8)
    optimized["row_index"] = optimized["row_index"].astype(np.int32)
    optimized["target_delta"] = optimized["target_delta"].astype(np.float32)

    for feature_name in feature_columns:
        optimized[feature_name] = optimized[feature_name].astype(np.float32)

    return optimized


# 在训练前集中检查行数、键、fold 和非有限值，错误时立即停止。
def validate_feature_table(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    expected_rows: int,
    feature_columns: list[str],
    allow_nan_features: list[str] | None = None,
) -> None:
    """验证完整特征缓存与冻结注册表一致。"""

    if list(feature_table[feature_columns].columns) != feature_columns:
        raise ValueError("模型特征列表与冻结配置不一致")

    if len(feature_table) != int(expected_rows):
        raise ValueError("完整特征表评价行数不匹配")

    if feature_table.duplicated(["well_id", "row_index"]).any():
        raise ValueError("完整特征表含重复井号与行号")

    if set(feature_table["fold"].astype(int).unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("完整特征表不是固定五折")

    # gr_raw 原本就允许缺失；额外允许列必须由实验配置逐列声明，禁止静默放宽检查。
    allowed_nan_feature_set = {"gr_raw"}
    if allow_nan_features is not None:
        allowed_nan_feature_set.update(allow_nan_features)

    unknown_allowed_features = allowed_nan_feature_set - set(feature_columns)
    if unknown_allowed_features:
        raise ValueError(
            f"允许 NaN 的特征不在模型输入中：{sorted(unknown_allowed_features)}"
        )

    finite_required_features = [
        name for name in feature_columns if name not in allowed_nan_feature_set
    ]
    if not np.isfinite(
        feature_table[finite_required_features].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("未声明允许缺失的模型特征含 NaN 或 Inf")

    # 允许缺失只代表可以含 NaN，正负无穷仍然表示公式或数据发生错误。
    for feature_name in sorted(allowed_nan_feature_set):
        feature_values = feature_table[feature_name].to_numpy(dtype=np.float64)
        if np.isinf(feature_values).any():
            raise ValueError(f"允许缺失的特征 {feature_name} 含 Inf")

    for column_name in ["target_tvt", "target_delta", "carry_tvt"]:
        if not np.isfinite(
            feature_table[column_name].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(f"{column_name} 含 NaN 或 Inf")

    actual_fold_rows = (
        feature_table.groupby("fold", observed=True).size().astype(int).to_dict()
    )
    expected_fold_rows = (
        registry.groupby("fold")["hidden_rows"].sum().astype(int).to_dict()
    )
    if actual_fold_rows != expected_fold_rows:
        raise ValueError("特征表逐折行数与注册表不一致")


# 构造或读取完整的本地特征缓存，缓存不进入 Git。
def load_or_build_feature_cache(
    registry: pd.DataFrame,
    train_dir: Path,
    artifact_dir: Path,
    fingerprint: str,
    expected_rows: int,
    rebuild_cache: bool,
    feature_columns: list[str],
    row_builder,
    allow_nan_features: list[str] | None = None,
) -> pd.DataFrame:
    """返回完整 773 井隐藏行特征表。"""

    cache_path = artifact_dir / "feature_cache.parquet"
    metadata_path = artifact_dir / "feature_cache.meta.json"

    stored_metadata = read_json(metadata_path) if metadata_path.is_file() else None
    stored_fingerprint = (
        stored_metadata.get("fingerprint") if stored_metadata is not None else None
    )
    cache_matches = (
        cache_path.is_file()
        and stored_metadata is not None
        and stored_fingerprint == fingerprint
    )

    print(f"缓存文件：{cache_path}", flush=True)
    print(
        f"缓存创建时间：{stored_metadata.get('created_at') if stored_metadata else None}",
        flush=True,
    )
    print(f"缓存配置指纹：{stored_fingerprint}", flush=True)
    print(f"当前配置指纹：{fingerprint}", flush=True)
    print(f"缓存完全匹配：{cache_matches and not rebuild_cache}", flush=True)

    if cache_matches and not rebuild_cache:
        feature_table = pd.read_parquet(cache_path)
        feature_table["well_id"] = feature_table["well_id"].astype("category")
    else:
        build_start = time.perf_counter()
        feature_table = build_feature_table(
            registry,
            train_dir,
            progress_interval=50,
            row_builder=row_builder,
        )
        feature_table = optimize_feature_table(feature_table, feature_columns)
        validate_feature_table(
            feature_table,
            registry,
            expected_rows,
            feature_columns,
            allow_nan_features,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        feature_table.to_parquet(cache_path, index=False, compression="zstd")
        write_json(
            metadata_path,
            {
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rows": int(len(feature_table)),
                "wells": int(feature_table["well_id"].nunique()),
                "build_seconds": time.perf_counter() - build_start,
            },
        )

    validate_feature_table(
        feature_table,
        registry,
        expected_rows,
        feature_columns,
        allow_nan_features,
    )
    return feature_table


# LightGBM 没有 early stopping，因此单独打印树训练进度供长任务观察。
def make_progress_callback(fold_id: int, interval: int = 100):
    """返回每隔固定树数打印一次进度的 LightGBM callback。"""

    def callback(environment: lgb.callback.CallbackEnv) -> None:
        tree_number = int(environment.iteration) + 1
        final_tree = int(environment.end_iteration)
        if tree_number == 1 or tree_number % interval == 0 or tree_number == final_tree:
            print(
                f"fold {fold_id} 训练进度：{tree_number}/{final_tree} 棵树",
                flush=True,
            )

    callback.order = 10
    callback.before_iteration = False
    return callback


# 一折只训练一个固定参数模型，并在完成后立刻保存可恢复产物。
def train_fold(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    fold_id: int,
    model_params: dict,
    artifact_dir: Path,
    fingerprint: str,
    feature_columns: list[str],
) -> dict:
    """训练或复用指定 fold，返回该折运行摘要。"""

    fold_dir = artifact_dir / f"fold_{fold_id}"
    prediction_path = fold_dir / "predictions.parquet"
    model_path = fold_dir / "model.txt"
    metadata_path = fold_dir / "runtime.json"

    if prediction_path.is_file() and model_path.is_file() and metadata_path.is_file():
        stored_runtime = read_json(metadata_path)
        if stored_runtime.get("fingerprint") == fingerprint:
            print(f"fold {fold_id}：复用完全匹配的已完成结果", flush=True)
            return stored_runtime

    train_indices, validation_indices = fold_indices(feature_table, fold_id)

    train_wells = set(
        feature_table.iloc[train_indices]["well_id"].astype(str).unique()
    )
    validation_wells = set(
        feature_table.iloc[validation_indices]["well_id"].astype(str).unique()
    )
    if train_wells & validation_wells:
        raise ValueError(f"fold {fold_id} 的训练井和验证井有交集")

    train_pads = set(
        registry.loc[registry["fold"] != fold_id, "pad_id"].astype(str).unique()
    )
    validation_pads = set(
        registry.loc[registry["fold"] == fold_id, "pad_id"].astype(str).unique()
    )
    if train_pads & validation_pads:
        raise ValueError(f"fold {fold_id} 的训练 pad 和验证 pad 有交集")

    print(
        f"fold {fold_id}：训练井 {len(train_wells)}，验证井 {len(validation_wells)}，"
        f"训练行 {len(train_indices):,}，验证行 {len(validation_indices):,}，"
        f"特征 {len(feature_columns)}，树 {model_params['n_estimators']}",
        flush=True,
    )

    fold_start = time.perf_counter()
    x_train = feature_table.iloc[train_indices][feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y_train = feature_table.iloc[train_indices]["target_delta"].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    x_validation = feature_table.iloc[validation_indices][feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        x_train,
        y_train,
        callbacks=[make_progress_callback(fold_id)],
    )
    predicted_delta = model.predict(x_validation)

    validation_anchor = feature_table.iloc[validation_indices][
        "last_visible_tvt"
    ].to_numpy(dtype=np.float64)
    predicted_tvt = restore_absolute_tvt(validation_anchor, predicted_delta)
    if not np.isfinite(predicted_tvt).all():
        raise ValueError(f"fold {fold_id} 的预测含 NaN 或 Inf")

    validation_metadata = feature_table.iloc[validation_indices][
        ["well_id", "fold", "row_index", "md", "target_tvt", "carry_tvt"]
    ].copy()
    validation_metadata["well_id"] = validation_metadata["well_id"].astype(str)
    validation_metadata["pred_tvt"] = predicted_tvt
    validation_metadata = validation_metadata.sort_values(
        ["well_id", "row_index"]
    ).reset_index(drop=True)

    fold_dir.mkdir(parents=True, exist_ok=True)
    validation_metadata.to_parquet(prediction_path, index=False, compression="zstd")
    model.booster_.save_model(str(model_path))

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(fold_dir / "feature_importance.csv", index=False)

    error = (
        validation_metadata["target_tvt"].to_numpy(dtype=np.float64)
        - validation_metadata["pred_tvt"].to_numpy(dtype=np.float64)
    )
    carry_error = (
        validation_metadata["target_tvt"].to_numpy(dtype=np.float64)
        - validation_metadata["carry_tvt"].to_numpy(dtype=np.float64)
    )

    runtime = {
        "fold": int(fold_id),
        "fingerprint": fingerprint,
        "training_wells": int(len(train_wells)),
        "validation_wells": int(len(validation_wells)),
        "training_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "features": int(len(feature_columns)),
        "trees": int(model_params["n_estimators"]),
        "seconds": time.perf_counter() - fold_start,
        "micro_rmse": float(np.sqrt(np.mean(error * error))),
        "carry_micro_rmse": float(np.sqrt(np.mean(carry_error * carry_error))),
    }
    write_json(metadata_path, runtime)

    print(
        f"fold {fold_id} 完成：RMSE={runtime['micro_rmse']:.6f}，"
        f"carry={runtime['carry_micro_rmse']:.6f}，耗时={runtime['seconds']:.1f} 秒",
        flush=True,
    )

    del x_train, y_train, x_validation, model, predicted_delta
    gc.collect()
    return runtime


# 五折结果都存在时合并 OOF，并调用项目统一指标实现。
def finalize_complete_cv(
    artifact_dir: Path,
    fingerprint: str,
    expected_rows: int,
    experiment_config: dict,
    model_params: dict,
    feature_columns: list[str],
) -> dict | None:
    """若五折齐全则生成完整指标，否则返回 None。"""

    fold_predictions: list[pd.DataFrame] = []
    fold_runtimes: list[dict] = []

    for fold_id in range(5):
        fold_dir = artifact_dir / f"fold_{fold_id}"
        prediction_path = fold_dir / "predictions.parquet"
        runtime_path = fold_dir / "runtime.json"
        if not prediction_path.is_file() or not runtime_path.is_file():
            return None
        fold_runtime = read_json(runtime_path)
        if fold_runtime.get("fingerprint") != fingerprint:
            return None
        fold_predictions.append(pd.read_parquet(prediction_path))
        fold_runtimes.append(fold_runtime)

    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions = predictions.sort_values(["well_id", "row_index"]).reset_index(
        drop=True
    )
    if len(predictions) != int(expected_rows):
        raise ValueError("完整 OOF 行数不匹配")
    if predictions.duplicated(["well_id", "row_index"]).any():
        raise ValueError("完整 OOF 含重复评价行")
    if not np.isfinite(predictions["pred_tvt"].to_numpy(dtype=np.float64)).all():
        raise ValueError("完整 OOF 预测含 NaN 或 Inf")

    prediction_path = artifact_dir / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False, compression="zstd")

    per_well = build_per_well_metrics(predictions)
    overall = summarize_per_well_metrics(per_well)
    folds = summarize_by_fold(per_well)
    bootstrap = paired_well_bootstrap(per_well, n_resamples=2000, seed=42)
    per_well.to_csv(artifact_dir / "per_well.csv", index=False)

    metrics = {
        "prediction_file": str(prediction_path.resolve()),
        "overall": overall,
        "folds": folds,
        "paired_well_bootstrap": bootstrap,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "config.json", experiment_config)
    write_json(artifact_dir / "parameter_list.json", model_params)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(feature_columns), "features": feature_columns},
    )
    write_json(
        artifact_dir / "runtime.json",
        {
            "fingerprint": fingerprint,
            "folds": fold_runtimes,
            "total_fold_seconds": float(
                sum(float(item["seconds"]) for item in fold_runtimes)
            ),
        },
    )

    importance_tables = []
    for fold_id in range(5):
        fold_importance = pd.read_csv(
            artifact_dir / f"fold_{fold_id}" / "feature_importance.csv"
        )
        fold_importance["fold"] = fold_id
        importance_tables.append(fold_importance)
    all_importance = pd.concat(importance_tables, ignore_index=True)
    mean_importance = (
        all_importance.groupby("feature", as_index=False)[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    mean_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    print("完整五折指标：", json.dumps(overall, ensure_ascii=False, indent=2))
    return metrics


# 解析最少命令行参数；所有实验定义仍来自冻结 JSON。
def parse_args() -> argparse.Namespace:
    """返回命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 B00 单模 LightGBM 固定五折")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=PROJECT_ROOT / "input" / "data" / "raw" / "train",
    )
    parser.add_argument("--fold", choices=["0", "1", "2", "3", "4", "all"], required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "b00_simple_lgbm_v1.json",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


# 配置只允许选择已经显式实现的特征 builder，禁止按列名自动猜测。
def select_feature_definition(experiment_config: dict):
    """返回当前实验的固定特征列表和逐井构造函数。"""

    feature_version = str(experiment_config["feature_version"])
    if feature_version == "simple_horizontal_12_v1":
        return FEATURE_COLUMNS, build_simple_lgbm_rows
    if feature_version == "prefix_u_slope_22_v1":
        return ALL_FEATURE_COLUMNS, build_f01_lgbm_rows
    if feature_version == "prefix_u_slope_19_v1":
        return ALL_F01B_FEATURE_COLUMNS, build_f01b_lgbm_rows
    if feature_version == "prefix_u_gated_projection_22_v1":
        return ALL_F01C_FEATURE_COLUMNS, build_f01c_lgbm_rows
    if feature_version == "prefix_u_stability_25_v1":
        return ALL_F02_FEATURE_COLUMNS, build_f02_lgbm_rows
    raise ValueError(f"未知 feature_version：{feature_version}")


# 主流程严格按照：注册表 → 特征缓存 → 独立 fold 模型 → 完整 OOF。
def main() -> None:
    """运行指定 fold 或完整五折。"""

    args = parse_args()
    experiment_config = read_json(args.config.resolve())
    model_config_path = CLEAN_ROOT / experiment_config["model_config"]
    model_params = read_json(model_config_path)["params"]
    feature_columns, row_builder = select_feature_definition(experiment_config)

    if experiment_config["feature_columns"] != feature_columns:
        raise ValueError("实验配置中的特征顺序与代码不一致")

    registry_path = CLEAN_ROOT / experiment_config["fold_registry"]
    registry_hash = file_sha256(registry_path)
    if registry_hash != experiment_config["fold_registry_sha256"]:
        raise ValueError(
            "固定 fold 注册表 SHA-256 不匹配："
            f"current={registry_hash}, expected={experiment_config['fold_registry_sha256']}"
        )

    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(experiment_config["expected_wells"]),
        expected_rows=int(experiment_config["expected_rows"]),
    )
    fingerprint = build_experiment_fingerprint(
        experiment_config,
        model_params,
        registry_hash,
    )
    artifact_dir = CLEAN_ROOT / "artifacts" / experiment_config["experiment_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    feature_table = load_or_build_feature_cache(
        registry=registry,
        train_dir=args.train_dir.resolve(),
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
        expected_rows=int(experiment_config["expected_rows"]),
        rebuild_cache=bool(args.rebuild_cache),
        feature_columns=feature_columns,
        row_builder=row_builder,
        allow_nan_features=experiment_config.get("allow_nan_features", []),
    )

    requested_folds = range(5) if args.fold == "all" else [int(args.fold)]
    for fold_id in requested_folds:
        train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            artifact_dir,
            fingerprint,
            feature_columns,
        )

    metrics = finalize_complete_cv(
        artifact_dir,
        fingerprint,
        int(experiment_config["expected_rows"]),
        experiment_config,
        model_params,
        feature_columns,
    )
    if metrics is None:
        print("尚未完成五折；已保存当前 fold，可用 --fold all 继续。", flush=True)


if __name__ == "__main__":
    main()
