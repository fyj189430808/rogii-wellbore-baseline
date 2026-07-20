"""严格影子评分：每个影子折仅由其余四个非影子开发折训练。

严格候选生成不读取影子目标，也不读取标准影子评分结果；五折基础模型、
UP01 和 PFS 修正均使用打开前冻结的合同。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p2_p02_multiscale_pf_paths_cv import FROZEN_MODEL_PARAMS  # noqa: E402
from scripts.run_p3_pf02_target_ess_lgbm_cv import (  # noqa: E402
    OLD_PF_FEATURES,
    P3B00_FEATURES,
    load_development_feature_table,
    load_development_registry,
    merge_existing_p3b00_pf_cache,
)
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402
from scripts.run_simple_lgbm_cv import make_progress_callback  # noqa: E402
from scripts.run_p3_final_candidate_shadow import (  # noqa: E402
    B00_FEATURE_CACHE,
    FOLD_REGISTRY,
    PREDICTION_PATH,
    SHADOW_REGISTRY,
    build_legal_candidate,
    file_sha256,
    load_shadow_pfs,
    load_shadow_registry,
    load_targets_after_candidate_landed,
    score_shadow,
)


EXPERIMENT_ID = "P4_FINAL00_UP03_strict_shadow_v1"
CANDIDATE_ID = "P4_FINAL00_UP03_v1"
UP01_DEGREE = 2
UP01_BLEND = 0.5
PFS_CORRECTION = 0.25
P01_FINGERPRINT = "ac1dbefd59923585671156b7d3e8b4fc7faca95c20f5f1ae6fdab92a8665954a"
BASE_FEATURE_CACHE = B00_FEATURE_CACHE
CANDIDATE_FEATURE_CACHE = CLEAN_ROOT / "artifacts/F05a_deterministic_candidate_cache_v1/candidate_feature_cache.parquet"
P01_CACHE_DIR = CLEAN_ROOT / "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache"
STANDARD_SHADOW_ARTIFACT = CLEAN_ROOT / "artifacts/P4_FINAL00_UP03_shadow_v1"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts/P4_FINAL00_UP03_strict_shadow_v1"
KEYS = ["well_id", "fold", "row_index"]


def strict_well_split(
    development: pd.DataFrame,
    shadow: pd.DataFrame,
    *,
    fold: int,
) -> tuple[set[str], set[str]]:
    train_ids = set(
        development.loc[development["fold"].astype(int).ne(int(fold)), "well_id"].astype(str)
    )
    validation_ids = set(
        shadow.loc[shadow["fold"].astype(int).eq(int(fold)), "well_id"].astype(str)
    )
    shadow_ids = set(shadow["well_id"].astype(str))
    if train_ids.intersection(shadow_ids):
        raise RuntimeError("严格训练集混入影子井")
    if train_ids.intersection(validation_ids):
        raise RuntimeError("严格训练井与验证井重叠")
    if not train_ids or not validation_ids:
        raise RuntimeError(f"严格 fold {fold} 的训练或验证井为空")
    return train_ids, validation_ids


def restore_strict_tvt(anchor: np.ndarray, predicted_delta: np.ndarray) -> np.ndarray:
    anchor_values = np.asarray(anchor, dtype=np.float64)
    delta_values = np.asarray(predicted_delta, dtype=np.float64)
    if anchor_values.shape != delta_values.shape:
        raise ValueError("锚点与预测增量形状不一致")
    result = anchor_values + delta_values
    if not np.isfinite(result).all():
        raise ValueError("严格绝对 TVT 含 NaN/Inf")
    return result


def validate_feature_matrix(values: np.ndarray) -> None:
    """保持冻结 LightGBM 的 NaN 缺失语义，只拒绝正负无穷。"""

    matrix = np.asarray(values)
    if np.isinf(matrix).any():
        raise ValueError("冻结 LightGBM 特征含 Inf")


def remap_frozen_folds(frame: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """用冻结的按井五折表覆盖旧缓存中可能过期的 fold 元数据。"""

    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("冻结折表含重复井号")
    fold_by_well = registry.assign(
        well_id=registry["well_id"].astype(str)
    ).set_index("well_id")["fold"]
    result = frame.copy()
    result["well_id"] = result["well_id"].astype(str)
    frozen_fold = result["well_id"].map(fold_by_well)
    if frozen_fold.isna().any():
        missing = sorted(result.loc[frozen_fold.isna(), "well_id"].unique())
        raise ValueError(f"冻结折表缺少井号：{missing[:5]}")
    result["fold"] = frozen_fold.astype(np.int8)
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _read_filtered(path: Path, well_ids: set[str], columns: list[str]) -> pd.DataFrame:
    dataset = arrow_dataset.dataset(path, format="parquet")
    missing = set(columns).difference(dataset.schema.names)
    if missing:
        raise ValueError(f"{path.name} 缺列：{sorted(missing)}")
    frame = dataset.to_table(
        columns=columns,
        filter=arrow_dataset.field("well_id").isin(sorted(well_ids)),
    ).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    if set(frame["well_id"]).difference(well_ids):
        raise RuntimeError(f"{path.name} 的井过滤失败")
    return frame


def load_shadow_feature_table(shadow_registry: pd.DataFrame) -> pd.DataFrame:
    """物理排除 target_tvt/target_delta，只加载严格预测所需的41列。"""

    well_ids = set(shadow_registry["well_id"].astype(str))
    base_columns = _unique(
        ["well_id", "fold", "row_index", "md", *list(FEATURE_COLUMNS)]
    )
    candidate_columns = ["well_id", "row_index", *list(DIRECT_CANDIDATE_COLUMNS)]
    base = _read_filtered(BASE_FEATURE_CACHE, well_ids, base_columns)
    base = remap_frozen_folds(base, shadow_registry)
    candidate = _read_filtered(CANDIDATE_FEATURE_CACHE, well_ids, candidate_columns)
    if "target_tvt" in base.columns or "target_delta" in base.columns:
        raise RuntimeError("严格影子特征表意外加载了目标列")
    if base.duplicated(["well_id", "row_index"]).any() or candidate.duplicated(
        ["well_id", "row_index"]
    ).any():
        raise RuntimeError("严格影子基础缓存含重复行键")
    base["_order"] = np.arange(len(base), dtype=np.int64)
    table = base.merge(
        candidate,
        on=["well_id", "row_index"],
        how="inner",
        validate="one_to_one",
    ).sort_values("_order", kind="stable").drop(columns="_order")
    if len(table) != int(shadow_registry["hidden_rows"].sum()):
        raise RuntimeError("严格影子基础特征行数不匹配")
    table = merge_existing_p3b00_pf_cache(
        table,
        shadow_registry,
        P01_CACHE_DIR,
        P01_FINGERPRINT,
    )
    missing_features = set(P3B00_FEATURES).difference(table.columns)
    if missing_features:
        raise RuntimeError(f"严格影子特征缺失：{sorted(missing_features)}")
    forbidden = {"target_tvt", "target_delta", "TVT", "oracle"}.intersection(table.columns)
    if forbidden:
        raise RuntimeError(f"严格影子特征含目标信息：{sorted(forbidden)}")
    validate_feature_matrix(table[P3B00_FEATURES].to_numpy(dtype=np.float64))
    return table.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)


def load_development_training_table() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    registry, shadow_ids = load_development_registry(FOLD_REGISTRY, SHADOW_REGISTRY)
    table = load_development_feature_table(
        BASE_FEATURE_CACHE,
        CANDIDATE_FEATURE_CACHE,
        registry,
    )
    table = merge_existing_p3b00_pf_cache(
        table,
        registry,
        P01_CACHE_DIR,
        P01_FINGERPRINT,
    )
    if set(table["well_id"].astype(str)).intersection(shadow_ids):
        raise RuntimeError("严格开发训练表混入影子井")
    if set(P3B00_FEATURES).difference(table.columns):
        raise RuntimeError("严格开发训练表缺少冻结41列")
    return table, registry, shadow_ids


def train_strict_fold(
    development_table: pd.DataFrame,
    shadow_table: pd.DataFrame,
    development_registry: pd.DataFrame,
    shadow_registry: pd.DataFrame,
    fold: int,
    output_dir: Path,
) -> dict[str, Any]:
    fold_dir = output_dir / "base_models" / f"fold_{fold}"
    model_path = fold_dir / "model.txt"
    prediction_path = fold_dir / "predictions.parquet"
    runtime_path = fold_dir / "runtime.json"
    train_ids, validation_ids = strict_well_split(
        development_registry, shadow_registry, fold=fold
    )
    fingerprint = _stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "fold": int(fold),
            "train_wells": sorted(train_ids),
            "validation_wells": sorted(validation_ids),
            "features": P3B00_FEATURES,
            "model": FROZEN_MODEL_PARAMS,
            "base_cache_sha256": file_sha256(BASE_FEATURE_CACHE),
            "candidate_cache_sha256": file_sha256(CANDIDATE_FEATURE_CACHE),
        }
    )
    if model_path.is_file() and prediction_path.is_file() and runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("fingerprint") == fingerprint:
            predictions = pd.read_parquet(prediction_path)
            prediction_ids = set(predictions["well_id"].astype(str))
            if prediction_ids != validation_ids:
                raise RuntimeError(f"strict fold {fold} 缓存预测井集不匹配")
            expected_rows = int(
                shadow_registry.loc[
                    shadow_registry["fold"].astype(int).eq(fold), "hidden_rows"
                ].sum()
            )
            if len(predictions) != expected_rows:
                raise RuntimeError(f"strict fold {fold} 缓存预测行数不匹配")
            if {"target_tvt", "target_delta"}.intersection(predictions.columns):
                raise RuntimeError(f"strict fold {fold} 缓存预测意外含目标")
            remapped = remap_frozen_folds(predictions, shadow_registry)
            if not remapped["fold"].equals(predictions["fold"].astype(np.int8)):
                _write_parquet(prediction_path, remapped)
            runtime["predictions_sha256"] = file_sha256(prediction_path)
            runtime["prediction_fold_source"] = "balanced_well_5fold_v1"
            _write_json(runtime_path, runtime)
            print(f"strict fold {fold}: 复用完整匹配缓存", flush=True)
            return runtime

    train = development_table.loc[
        development_table["well_id"].astype(str).isin(train_ids)
    ].copy()
    validation = shadow_table.loc[
        shadow_table["well_id"].astype(str).isin(validation_ids)
    ].copy()
    if set(train["fold"].astype(int)) == {int(fold)} or train["fold"].astype(int).eq(fold).any():
        raise RuntimeError(f"strict fold {fold} 训练行混入当前折")
    if len(validation) != int(
        shadow_registry.loc[shadow_registry["fold"].astype(int).eq(fold), "hidden_rows"].sum()
    ):
        raise RuntimeError(f"strict fold {fold} 验证行数不匹配")
    print(
        f"strict fold {fold}: 训练{len(train_ids)}井/{len(train):,}行，"
        f"预测{len(validation_ids)}影子井/{len(validation):,}行，41列/1734树",
        flush=True,
    )
    started = time.perf_counter()
    x_train = train[P3B00_FEATURES].to_numpy(dtype=np.float32, copy=True)
    y_train = train["target_delta"].to_numpy(dtype=np.float32, copy=True)
    x_validation = validation[P3B00_FEATURES].to_numpy(dtype=np.float32, copy=True)
    model = lgb.LGBMRegressor(**FROZEN_MODEL_PARAMS)
    model.fit(x_train, y_train, callbacks=[make_progress_callback(fold)])
    predicted_delta = model.predict(x_validation)
    predicted_tvt = restore_strict_tvt(
        validation["last_visible_tvt"].to_numpy(dtype=np.float64),
        predicted_delta,
    )
    predictions = validation[
        ["well_id", "fold", "row_index", "md", "last_visible_tvt", "z_current"]
    ].copy()
    predictions["pred_tvt"] = predicted_tvt
    predictions = predictions.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(prediction_path, predictions)
    model.booster_.save_model(str(model_path))
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "fold": int(fold),
        "fingerprint": fingerprint,
        "training_wells": len(train_ids),
        "validation_wells": len(validation_ids),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "features": len(P3B00_FEATURES),
        "trees": int(FROZEN_MODEL_PARAMS["n_estimators"]),
        "seed": int(FROZEN_MODEL_PARAMS["random_state"]),
        "shadow_target_read": False,
        "model_sha256": file_sha256(model_path),
        "predictions_sha256": file_sha256(prediction_path),
        "prediction_fold_source": "balanced_well_5fold_v1",
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(runtime_path, runtime)
    del x_train, y_train, x_validation, model, predicted_delta, train, validation
    gc.collect()
    return runtime


def land_strict_candidate(
    output_dir: Path,
    shadow_registry: pd.DataFrame,
    runtimes: list[dict[str, Any]],
) -> tuple[Path, Path]:
    prediction_parts = [
        pd.read_parquet(output_dir / "base_models" / f"fold_{fold}" / "predictions.parquet")
        for fold in range(5)
    ]
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["well_id"] = predictions["well_id"].astype(str)
    if predictions.duplicated(KEYS).any():
        raise RuntimeError("严格基础预测存在重复行键")
    if len(predictions) != int(shadow_registry["hidden_rows"].sum()):
        raise RuntimeError("严格基础预测没有完整覆盖116口影子井")
    if "target_tvt" in predictions.columns or "target_delta" in predictions.columns:
        raise RuntimeError("严格基础预测意外包含影子目标")
    pfs = load_shadow_pfs(shadow_registry, STANDARD_SHADOW_ARTIFACT)
    legal_input = predictions[
        ["well_id", "fold", "row_index", "md", "pred_tvt", "z_current"]
    ].copy()
    legal = build_legal_candidate(legal_input, pfs)
    candidate_path = output_dir / "legal_candidates.parquet"
    manifest_path = output_dir / "legal_generation_manifest.json"
    _write_parquet(candidate_path, legal)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID,
        "generation_stage": "deterministic_pipeline_strict_shadow_pending_score",
        "candidate_generation_complete": True,
        "formal_shadow_complete": True,
        "hidden_target_read": False,
        "standard_shadow_metrics_read": False,
        "wells": int(legal["well_id"].nunique()),
        "rows": int(len(legal)),
        "fold_counts": {
            str(k): int(v) for k, v in shadow_registry.groupby("fold").size().to_dict().items()
        },
        "strict_training_rule": "for shadow fold f train only non-shadow development wells with fold != f",
        "features": P3B00_FEATURES,
        "model_params": FROZEN_MODEL_PARAMS,
        "postprocess": {
            "up01_degree": UP01_DEGREE,
            "up01_blend": UP01_BLEND,
            "pfs_lag_ft": 1000,
            "pfs_correction_fraction": PFS_CORRECTION,
        },
        "model_runtimes": runtimes,
        "legal_candidates_sha256": file_sha256(candidate_path),
        "paths_scored": ["strict_P3B00", CANDIDATE_ID],
        "parameter_search": False,
    }
    _write_json(manifest_path, manifest)
    _write_parquet(output_dir / "strict_base_predictions.parquet", predictions)
    return candidate_path, manifest_path


def run_generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_registry = load_shadow_registry(FOLD_REGISTRY, SHADOW_REGISTRY)
    development_table, development_registry, shadow_ids = load_development_training_table()
    if shadow_ids != set(shadow_registry["well_id"].astype(str)):
        raise RuntimeError("严格开发表排除的影子井与冻结清单不一致")
    shadow_table = load_shadow_feature_table(shadow_registry)
    print(
        f"严格影子候选：开发657井/{len(development_table):,}行，"
        f"影子116井/{len(shadow_table):,}行；目标只存在于开发训练表",
        flush=True,
    )
    runtimes: list[dict[str, Any]] = []
    for fold in range(5):
        runtimes.append(
            train_strict_fold(
                development_table,
                shadow_table,
                development_registry,
                shadow_registry,
                fold,
                output_dir,
            )
        )
    candidate_path, manifest_path = land_strict_candidate(
        output_dir, shadow_registry, runtimes
    )
    _write_json(
        output_dir / "config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_id": CANDIDATE_ID,
            "source_contract": "configs/p4_final00_up03_shadow_v1.json",
            "features": P3B00_FEATURES,
            "model_params": FROZEN_MODEL_PARAMS,
            "hidden_target_read": False,
            "standard_shadow_metrics_read": False,
        },
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate_path),
                "manifest": str(manifest_path),
                "wells": 116,
                "rows": int(shadow_registry["hidden_rows"].sum()),
                "hidden_target_read": False,
                "standard_shadow_metrics_read": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def run_score(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    candidate_path = output_dir / "legal_candidates.parquet"
    manifest_path = output_dir / "legal_generation_manifest.json"
    target = load_targets_after_candidate_landed(
        candidate_path, manifest_path, PREDICTION_PATH
    )
    legal = pd.read_parquet(candidate_path)
    metrics, per_fold, per_well, scored = score_shadow(legal, target)
    metrics["experiment_id"] = EXPERIMENT_ID
    metrics["score_type"] = "strict_shadow"
    metrics["strict_training_rule"] = (
        "for shadow fold f train only non-shadow development wells with fold != f"
    )
    metrics["result_class"] = (
        "shadow_confirmed_pipeline"
        if metrics["gate"]["passed"]
        else "deterministic_pipeline_development"
    )
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    _write_parquet(output_dir / "scored_predictions.parquet", scored)
    _write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "score"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.stage == "generate":
        run_generate(output_dir)
    else:
        run_score(output_dir)


if __name__ == "__main__":
    main()
