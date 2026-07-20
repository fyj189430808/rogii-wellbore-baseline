"""把三条 PFM03 形状模式路径加入原 41 列，运行冻结单模 LightGBM。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pf02_target_ess_lgbm_cv as common_runner  # noqa: E402
from scripts.run_p2_cv00_group5_c01 import save_fold_runtime_row  # noqa: E402
from scripts.run_p2_p02_multiscale_pf_paths_cv import (  # noqa: E402
    FROZEN_MODEL_FEATURES as _P3B00_FEATURES,
    FROZEN_MODEL_PARAMS,
)
from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json  # noqa: E402


EXPERIMENT_ID = "P3_PFM03_shape_aware_modes_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pfm03_shape_aware_modes_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
P3B00_FEATURES = list(_P3B00_FEATURES)
NEW_SHAPE_FEATURES = [
    "shape_mode_low_delta",
    "shape_mode_middle_delta",
    "shape_mode_high_delta",
]
SHAPE_CACHE_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    *NEW_SHAPE_FEATURES,
    "_cache_fingerprint",
]
SHAPE_CACHE_SCHEMA = pa.schema(
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

# 这些函数已经用于 P3B00/PF02，直接复用可确保读取、训练和评分口径不变。
resolve_clean_path = common_runner.resolve_clean_path
file_sha256 = common_runner.file_sha256
stable_hash = common_runner.stable_hash
load_development_registry = common_runner.load_development_registry
load_development_feature_table = common_runner.load_development_feature_table
merge_existing_p3b00_pf_cache = common_runner.merge_existing_p3b00_pf_cache
validate_feature_values = common_runner.validate_feature_values
validate_source_hashes = common_runner.validate_source_hashes
read_baseline_predictions = common_runner.read_baseline_predictions
read_fold_predictions = common_runner.read_fold_predictions


FROZEN_TRAINING_POLICY = {
    "early_stopping": False,
    "uniform_row_weight": True,
    "native_missing_values": True,
    "well_id_is_feature": False,
    "row_id_is_feature": False,
    "test_overlap_wells_removed_from_training": True,
}


def build_formal_feature_names() -> list[str]:
    """返回顺序冻结的原 41 列加三条形状路径，共 44 列。"""

    features = [*P3B00_FEATURES, *NEW_SHAPE_FEATURES]
    if len(features) != 44 or len(set(features)) != 44:
        raise ValueError("PFM03 正式特征必须是无重复的 41+3=44 列")
    return features


def parse_fold_spec(value: str) -> list[int]:
    """正式只允许先跑 folds 1～2，或通过后继续到 folds 3～4。"""

    normalized = str(value).replace(" ", "")
    if normalized == "1,2":
        return [1, 2]
    if normalized == "1,2,3,4":
        return [1, 2, 3, 4]
    raise ValueError("--folds 只允许 1,2 或 1,2,3,4；fold 0 不参与正式门槛")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析正式 runner 的配置、输出目录和确认折范围。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="先传 1,2；晋级后传 1,2,3,4")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def validate_frozen_contract(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """核对 41 列、三条新增路径、1734 棵树和 seed29 均未改变。"""

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PFM03 experiment_id 不一致")
    expected_scalars = {
        "baseline_id": "P3B00_group5_p2p02_v1",
        "fold_version": "balanced_well_5fold_v1",
        "baseline_feature_count": 41,
        "formal_feature_count": 44,
        "total_wells": 773,
        "shadow_wells": 116,
        "development_wells": 657,
        "development_hidden_rows": 3_211_872,
        "model_training": True,
        "shadow_target_access": False,
        "confirmation_folds": [1, 2, 3, 4],
        "screen_folds": [1, 2],
        "new_feature_names": NEW_SHAPE_FEATURES,
    }
    changed = [key for key, value in expected_scalars.items() if config.get(key) != value]
    if changed:
        raise ValueError(f"PFM03 冻结配置发生变化：{changed}")

    manifest = read_json(resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41 or manifest.get("features") != P3B00_FEATURES:
        raise ValueError("P3B00 原 41 列清单发生变化")
    model_config = read_json(resolve_clean_path(config["model_config"]))
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("正式模型不是单模 LightGBM")
    if model_config.get("params") != FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 不再是冻结的 1734 树 seed29 参数")
    if model_config.get("training_policy") != FROZEN_TRAINING_POLICY:
        raise ValueError("LightGBM 训练策略发生变化")
    return build_formal_feature_names(), dict(FROZEN_MODEL_PARAMS)


def merge_pfm03_legal_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    runtime_dir: Path,
    shadow_ids: set[str],
) -> tuple[pd.DataFrame, str]:
    """逐井验证无标签缓存，再按 well_id+row_index 一对一追加三列。"""

    if not cache_dir.is_dir() or not runtime_dir.is_dir():
        raise FileNotFoundError("PFM03 legal_cache 或 legal_runtime 不存在")
    cache_wells = {path.stem for path in cache_dir.glob("*.parquet")}
    if overlap := cache_wells.intersection(shadow_ids):
        raise ValueError(f"PFM03 legal_cache 含影子井：{sorted(overlap)[:3]}")

    path_frames: list[pd.DataFrame] = []
    common_fingerprint: str | None = None
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        expected_fold = int(registry_row.fold)
        expected_rows = int(registry_row.hidden_rows)
        cache_path = cache_dir / f"{well_id}.parquet"
        if not cache_path.is_file():
            raise FileNotFoundError(f"缺少 PFM03 井缓存：{cache_path}")
        if not pq.read_schema(cache_path).equals(SHAPE_CACHE_SCHEMA):
            raise ValueError(f"PFM03 {well_id} schema 不等于冻结八列合同")
        cache = pd.read_parquet(cache_path, columns=SHAPE_CACHE_COLUMNS)
        if len(cache) != expected_rows:
            raise ValueError(f"PFM03 {well_id} 行数不匹配")
        if cache.duplicated(["well_id", "row_index"]).any():
            raise ValueError(f"PFM03 {well_id} 含重复自然键")
        if not cache["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"PFM03 {well_id} 井号不匹配")
        if not cache["fold"].astype(np.int64).eq(expected_fold).all():
            raise ValueError(f"PFM03 {well_id} fold 不匹配")
        fingerprint_values = (
            cache["_cache_fingerprint"].astype(str).str.strip().unique().tolist()
        )
        if len(fingerprint_values) != 1 or not fingerprint_values[0]:
            raise ValueError(f"PFM03 {well_id} 缓存指纹为空或不唯一")
        fingerprint = fingerprint_values[0]
        if common_fingerprint is None:
            common_fingerprint = fingerprint
        elif fingerprint != common_fingerprint:
            raise ValueError("PFM03 各井缓存指纹不统一")
        runtime_path = runtime_dir / f"{well_id}.json"
        if not runtime_path.is_file():
            raise FileNotFoundError(f"缺少 PFM03 井运行记录：{runtime_path}")
        runtime = read_json(runtime_path)
        if runtime.get("experiment_fingerprint") != fingerprint:
            raise ValueError(f"PFM03 {well_id} runtime 与缓存指纹不一致")
        if runtime.get("hidden_tvt_read") is not False:
            raise ValueError(f"PFM03 {well_id} 路径生成读取了隐藏 TVT")
        if (
            str(runtime.get("well_id")) != well_id
            or int(runtime.get("fold", -1)) != expected_fold
            or int(runtime.get("rows", -1)) != expected_rows
        ):
            raise ValueError(f"PFM03 {well_id} runtime 井/fold/行数不一致")
        path_frames.append(
            cache[["well_id", "row_index", "last_visible_tvt", *NEW_SHAPE_FEATURES]]
        )

    if common_fingerprint is None:
        raise ValueError("PFM03 没有可合并的开发井缓存")
    path_table = pd.concat(path_frames, ignore_index=True)
    merged = common_runner._merge_path_table(
        feature_table,
        path_table,
        NEW_SHAPE_FEATURES,
        anchor_name="pfm03_shape_anchor",
    )
    return merged, common_fingerprint


def folds12_gate(improvement_by_fold: dict[int, float]) -> dict[str, Any]:
    """执行路线图冻结的 folds1～2 平均改善与单折退化门槛。"""

    if set(improvement_by_fold) != {1, 2}:
        raise ValueError("folds12_gate 必须恰好收到 fold 1 和 fold 2")
    average_improvement = float(np.mean(list(improvement_by_fold.values())))
    worst_improvement = float(min(improvement_by_fold.values()))
    checks: dict[str, Any] = {
        "average_fold_improvement_ft": average_improvement,
        "minimum_average_improvement_ft": 0.15,
        "worst_fold_improvement_ft": worst_improvement,
        "maximum_single_fold_degradation_ft": 0.15,
    }
    checks["folds12_pass"] = bool(
        average_improvement >= 0.15 and worst_improvement >= -0.15
    )
    return checks


def _metric_config(config: dict[str, Any]) -> dict[str, Any]:
    """把新阶段门槛映射到冻结评分器需要的键名；只影响判定，不改指标。"""

    mapped = dict(config)
    mapped["success_conditions"] = {
        "fold01_minimum_combined_improvement_ft": 0.0,
        "fold01_maximum_single_fold_degradation_ft": 999.0,
        "full5_minimum_improvement_ft": 0.10,
        "full5_minimum_improved_folds": 3,
        "full5_minimum_improved_folds_2_to_4": 0,
        "full5_maximum_single_fold_degradation_ft": 999.0,
        "full5_maximum_bootstrap_ci_upper": 0.0,
        "full5_minimum_well_win_rate": 0.0,
        "full5_maximum_p90_degradation_ft": 0.20,
        "full5_maximum_top5_percent_positive_gain_share": 0.60,
    }
    return mapped


def compare_stage(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """沿用统一逐井评分；仅把正式阶段名和门槛改为 folds12/confirmed4。"""

    if stage == "folds12":
        metrics, per_well, per_fold = common_runner.compare_with_p3b00(
            candidate,
            baseline,
            _metric_config(config),
            "folds01",
        )
        improvements = {
            int(row.fold): float(row.improvement_ft)
            for row in per_fold.itertuples(index=False)
        }
        metrics["stage"] = "folds12"
        metrics["success_checks"] = folds12_gate(improvements)
    elif stage == "confirmed4":
        metrics, per_well, per_fold = common_runner.compare_with_p3b00(
            candidate,
            baseline,
            _metric_config(config),
            "full5",
        )
        old_checks = metrics["success_checks"]
        old_checks["confirmed4_pass"] = bool(old_checks.pop("full5_pass"))
        old_checks["confirmed4_pooled_improvement_ft"] = float(
            old_checks.pop("full5_improvement_ft")
        )
        metrics["stage"] = "confirmed4"
    else:
        raise ValueError("stage 只允许 folds12 或 confirmed4")
    metrics["experiment_id"] = EXPERIMENT_ID
    return metrics, per_well, per_fold


def remaining_folds_after_screen(
    requested_folds: list[int],
    metrics_folds12: dict[str, Any],
) -> list[int]:
    """只有请求四个确认折且 folds1～2 通过时才返回 folds3～4。"""

    if requested_folds == [1, 2]:
        return []
    if requested_folds != [1, 2, 3, 4]:
        raise ValueError("正式请求折必须是 1,2 或 1,2,3,4")
    if metrics_folds12["success_checks"]["folds12_pass"]:
        return [3, 4]
    return []


def _save_stage(
    output_dir: Path,
    stage: str,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    per_well: pd.DataFrame,
    per_fold: pd.DataFrame,
) -> None:
    """分别保存两折筛查和四折确认产物，避免名称暗示 fold0 已参与。"""

    suffix = "folds12" if stage == "folds12" else "confirmed4"
    predictions.sort_values(["well_id", "row_index"]).to_parquet(
        output_dir / f"predictions_{suffix}.parquet",
        index=False,
        compression="zstd",
    )
    write_json(output_dir / f"metrics_{suffix}.json", metrics)
    per_well.to_csv(output_dir / f"per_well_{suffix}.csv", index=False)
    per_fold.to_csv(output_dir / f"per_fold_{suffix}.csv", index=False)


def _save_feature_importance(output_dir: Path, fold_ids: list[int]) -> None:
    """只汇总真正运行的确认折，不伪造或要求 fold0 文件。"""

    tables: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        table = pd.read_csv(output_dir / f"fold_{fold_id}" / "feature_importance.csv")
        table["fold"] = fold_id
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    summary = (
        combined.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False, kind="stable")
    )
    summary.to_csv(output_dir / "feature_importance_confirmed4.csv", index=False)


def _write_conclusion(output_dir: Path, metrics: dict[str, Any]) -> None:
    """结论严格限定为形状三路径直接入模这一种实现。"""

    stage = str(metrics["stage"])
    checks = metrics["success_checks"]
    if stage == "folds12":
        improvement = float(checks["average_fold_improvement_ft"])
        passed = bool(checks["folds12_pass"])
    else:
        improvement = float(checks["confirmed4_pooled_improvement_ft"])
        passed = bool(checks["confirmed4_pass"])
    text = f"""# P3-PFM03 形状模式路径结论

数据直接证明的事实：当前阶段改善为 `{improvement:.6f} ft`，预登记门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：这里只检验三条形状聚类路径对原 41 列的直接增量价值。

仍然没有验证的猜测：这些模式是否需要沿井深动态可靠性权重。

当前实验只能否定的具体实现：形状 Ward K=3 的三条似然加权中心直接进入冻结 LightGBM。

下一步最便宜的验证：按路线图门槛停止或继续，不调整模型参数。
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """加载完整开发特征一次，然后固定顺序训练 fold1、2，必要时训练 3、4。"""

    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config = read_json(args.config.resolve())
    model_features, model_params = validate_frozen_contract(config)
    observed_hashes = validate_source_hashes(config)
    registry, shadow_ids = load_development_registry(
        resolve_clean_path(config["fold_registry"]),
        resolve_clean_path(config["shadow_registry"]),
        config,
    )
    feature_table = load_development_feature_table(
        resolve_clean_path(config["base_feature_cache"]),
        resolve_clean_path(config["candidate_feature_cache"]),
        registry,
    )
    feature_table = merge_existing_p3b00_pf_cache(
        feature_table,
        registry,
        resolve_clean_path(config["source_p01_legal_cache_dir"]),
        str(config["source_p01_fingerprint"]),
    )
    feature_table, generator_fingerprint = merge_pfm03_legal_cache(
        feature_table,
        registry,
        resolve_clean_path(config["source_pfm03_legal_cache_dir"]),
        resolve_clean_path(config["source_pfm03_runtime_dir"]),
        shadow_ids,
    )
    if set(feature_table["well_id"].astype(str).unique()).intersection(shadow_ids):
        raise RuntimeError("正式训练表含影子井")
    validate_feature_values(feature_table, model_features)

    cv_fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "features": model_features,
            "model_params": model_params,
            "generator_fingerprint": generator_fingerprint,
            "observed_source_hashes": observed_hashes,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"),
        }
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "parameter_list.json", model_params)
    write_json(
        output_dir / "feature_list.json",
        {"feature_count": 44, "features": model_features},
    )
    write_json(
        output_dir / "leakage_audit.json",
        {
            "shadow_target_access": False,
            "shadow_wells": len(shadow_ids),
            "development_wells": int(feature_table["well_id"].nunique()),
            "development_hidden_rows": len(feature_table),
            "validation_unit": "complete_well",
            "confirmation_folds": [1, 2, 3, 4],
            "fold0_used_for_gate": False,
            "old_41_features_preserved": model_features[:41] == P3B00_FEATURES,
            "new_features": NEW_SHAPE_FEATURES,
            "path_generator_fingerprint": generator_fingerprint,
            "hidden_tvt_read_for_path_generation": False,
            "cv_fingerprint": cv_fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )
    print(
        f"P3-PFM03：开发井={len(registry)}，行={len(feature_table):,}，"
        f"特征=44，树=1734，正式折={requested_folds}",
        flush=True,
    )

    for fold_id in (1, 2):
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            cv_fingerprint,
            model_features,
        )
        save_fold_runtime_row(output_dir, runtime)
    registry12 = registry.loc[registry["fold"].astype(int).isin([1, 2])].copy()
    candidate12 = read_fold_predictions(output_dir, [1, 2])
    baseline12 = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]),
        registry12,
    )
    metrics12, per_well12, per_fold12 = compare_stage(
        candidate12,
        baseline12,
        config,
        "folds12",
    )
    metrics12["cv_fingerprint"] = cv_fingerprint
    _save_stage(output_dir, "folds12", candidate12, metrics12, per_well12, per_fold12)
    _write_conclusion(output_dir, metrics12)
    print(json.dumps(metrics12["success_checks"], ensure_ascii=False, indent=2), flush=True)

    late_folds = remaining_folds_after_screen(requested_folds, metrics12)
    if not late_folds:
        if requested_folds == [1, 2, 3, 4]:
            print("PFM03 folds1～2 未晋级，不训练 folds3～4。", flush=True)
        return
    for fold_id in late_folds:
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            cv_fingerprint,
            model_features,
        )
        save_fold_runtime_row(output_dir, runtime)
    registry_confirmed = registry.loc[registry["fold"].astype(int).isin([1, 2, 3, 4])].copy()
    candidate_confirmed = read_fold_predictions(output_dir, [1, 2, 3, 4])
    baseline_confirmed = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]),
        registry_confirmed,
    )
    metrics_confirmed, per_well_confirmed, per_fold_confirmed = compare_stage(
        candidate_confirmed,
        baseline_confirmed,
        config,
        "confirmed4",
    )
    metrics_confirmed["cv_fingerprint"] = cv_fingerprint
    _save_stage(
        output_dir,
        "confirmed4",
        candidate_confirmed,
        metrics_confirmed,
        per_well_confirmed,
        per_fold_confirmed,
    )
    _save_feature_importance(output_dir, [1, 2, 3, 4])
    fold_runtime = [
        read_json(output_dir / f"fold_{fold_id}" / "runtime.json")
        for fold_id in (1, 2, 3, 4)
    ]
    write_json(
        output_dir / "runtime.json",
        {
            "cv_fingerprint": cv_fingerprint,
            "folds": fold_runtime,
            "total_fold_seconds": float(
                sum(float(record["seconds"]) for record in fold_runtime)
            ),
        },
    )
    _write_conclusion(output_dir, metrics_confirmed)
    print(
        json.dumps(metrics_confirmed["success_checks"], ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()

