"""把三条无标签模式路径加入 P3B00，运行开发集单模 LightGBM。

只使用 657 口开发井。原 P3B00 的 41 列逐位保留，三条模式路径追加
在末尾，总计 44 列。CLI 只允许 ``--folds 0,1`` 或 ``--folds all``；
``all`` 仍先重算 folds 0～1，未通过预登记门槛便自动停止。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pf02_target_ess_lgbm_cv as pf02_runner  # noqa: E402
from scripts.run_p2_cv00_group5_c01 import save_fold_runtime_row  # noqa: E402
from scripts.run_p2_p02_multiscale_pf_paths_cv import (  # noqa: E402
    FROZEN_MODEL_FEATURES as _P3B00_FEATURES,
    FROZEN_MODEL_PARAMS,
)
from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json  # noqa: E402


EXPERIMENT_ID = "P3_PFM02_direct_mode_paths_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pfm02_direct_mode_paths_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

P3B00_FEATURES = list(_P3B00_FEATURES)
NEW_MODE_FEATURES = [
    "pf_mode_low_delta",
    "pf_mode_middle_delta",
    "pf_mode_high_delta",
]
MODE_CACHE_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    *NEW_MODE_FEATURES,
    "_cache_fingerprint",
]
MODE_CACHE_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("row_index", pa.int64()),
        pa.field("last_visible_tvt", pa.float64()),
        pa.field("pf_mode_low_delta", pa.float64()),
        pa.field("pf_mode_middle_delta", pa.float64()),
        pa.field("pf_mode_high_delta", pa.float64()),
        pa.field("_cache_fingerprint", pa.string()),
    ]
)

# 直接复用已冻结 PF02 正式训练骨架中不依赖候选特征名的实现。
resolve_clean_path = pf02_runner.resolve_clean_path
file_sha256 = pf02_runner.file_sha256
stable_hash = pf02_runner.stable_hash
load_development_registry = pf02_runner.load_development_registry
read_development_parquet = pf02_runner.read_development_parquet
load_development_feature_table = pf02_runner.load_development_feature_table
merge_existing_p3b00_pf_cache = pf02_runner.merge_existing_p3b00_pf_cache
validate_feature_values = pf02_runner.validate_feature_values
validate_source_hashes = pf02_runner.validate_source_hashes
read_baseline_predictions = pf02_runner.read_baseline_predictions
read_fold_predictions = pf02_runner.read_fold_predictions
save_stage_artifacts = pf02_runner.save_stage_artifacts
save_mean_feature_importance = pf02_runner.save_mean_feature_importance


FROZEN_TRAINING_POLICY = {
    "early_stopping": False,
    "uniform_row_weight": True,
    "native_missing_values": True,
    "well_id_is_feature": False,
    "row_id_is_feature": False,
    "test_overlap_wells_removed_from_training": True,
}
FROZEN_SUCCESS_CONDITIONS = {
    "fold01_minimum_combined_improvement_ft": 0.20,
    "fold01_maximum_single_fold_degradation_ft": 0.10,
    "full5_minimum_improvement_ft": 0.10,
    "full5_minimum_improved_folds": 4,
    "full5_minimum_improved_folds_2_to_4": 2,
    "full5_maximum_single_fold_degradation_ft": 0.25,
    "full5_maximum_bootstrap_ci_upper": 0.0,
    "full5_minimum_well_win_rate": 0.55,
    "full5_maximum_p90_degradation_ft": 0.20,
    "full5_maximum_top5_percent_positive_gain_share": 0.60,
}
FROZEN_CONFIG_CONTRACT: dict[str, Any] = {
    "experiment_id": EXPERIMENT_ID,
    "baseline_id": "P3B00_group5_p2p02_v1",
    "fold_version": "balanced_well_5fold_v1",
    "fold_registry": "artifacts/folds/balanced_well_5fold_v1.csv",
    "fold_registry_sha256": "a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c",
    "shadow_registry": "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    "shadow_registry_sha256": "7fde7c16895e02f74c96a70a1ca3d9935e0e9bfb24c7744a1992c025aad483c1",
    "base_feature_cache": "artifacts/B00_simple_lgbm_v1/feature_cache.parquet",
    "base_feature_cache_sha256": "8801493752e03f40a3957843e8efa25c091d04244dd341c64c9329b50d5de625",
    "candidate_feature_cache": "artifacts/F05a_deterministic_candidate_cache_v1/candidate_feature_cache.parquet",
    "candidate_feature_cache_sha256": "66b32f8ed790ea53184d43bc02d6899031348deec418d7933366bbef64105076",
    "source_p01_legal_cache_dir": "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache",
    "source_p01_fingerprint": "ac1dbefd59923585671156b7d3e8b4fc7faca95c20f5f1ae6fdab92a8665954a",
    "source_pfm02_legal_cache_dir": "artifacts/P3_PFM02_mode_paths_v1/legal_cache",
    "source_pfm02_runtime_dir": "artifacts/P3_PFM02_mode_paths_v1/legal_runtime",
    "source_p3b00_predictions": "artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet",
    "source_p3b00_predictions_sha256": "8109514eb125f8beda2559f39e1d9f54396d41fd38fa76114423c8dd6bca4d37",
    "source_p3b00_feature_list": "artifacts/P2_P02_multiscale_pf_paths_v1/feature_list.json",
    "source_p3b00_feature_list_sha256": "2bf5992b45fe95e6ae38862da2408b306c0303806291d112ff8fa1ecf4b00172",
    "model_config": "configs/lgbm_feature_baseline_v1.json",
    "model_config_sha256": "02f6c4cb737132641c2dbbb6076d0e248371786559f2cd6f8581ce72146cb838",
    "new_feature_names": NEW_MODE_FEATURES,
    "baseline_feature_count": 41,
    "formal_feature_count": 44,
    "total_wells": 773,
    "shadow_wells": 116,
    "development_wells": 657,
    "development_hidden_rows": 3_211_872,
    "development_fold_well_counts": {
        "0": 131,
        "1": 132,
        "2": 131,
        "3": 132,
        "4": 131,
    },
    "development_fold_hidden_row_counts": {
        "0": 651_881,
        "1": 630_395,
        "2": 645_557,
        "3": 649_717,
        "4": 634_322,
    },
    "baseline_development_micro_rmse": 10.272146267501086,
    "model_training": True,
    "shadow_target_access": False,
    "success_conditions": FROZEN_SUCCESS_CONDITIONS,
}


def build_formal_feature_names() -> list[str]:
    """返回无重复且顺序冻结的 41+3=44 列正式特征。"""

    features = [*P3B00_FEATURES, *NEW_MODE_FEATURES]
    if len(features) != 44 or len(set(features)) != 44:
        raise ValueError("PFM02 正式特征必须是无重复的 41+3=44 列")
    return features


def parse_fold_spec(value: str) -> list[int]:
    """只允许冻结预筛模式或完整模式。"""

    normalized = str(value).strip().lower()
    if normalized == "0,1":
        return [0, 1]
    if normalized == "all":
        return [0, 1, 2, 3, 4]
    raise ValueError("--folds 只允许 0,1 或 all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="预筛传 0,1；晋级后传 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def validate_frozen_contract(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """验证配置、41 列清单以及冻结 LightGBM 参数和训练策略。"""

    if config != FROZEN_CONFIG_CONTRACT:
        changed = sorted(
            key
            for key in set(config).union(FROZEN_CONFIG_CONTRACT)
            if config.get(key) != FROZEN_CONFIG_CONTRACT.get(key)
        )
        raise ValueError(f"PFM02 冻结配置合同发生变化：{changed}")

    manifest = read_json(resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41 or manifest.get("features") != P3B00_FEATURES:
        raise ValueError("P3B00 41 列清单发生变化")

    model_config = read_json(resolve_clean_path(config["model_config"]))
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("模型不是冻结的单模 LightGBM")
    if model_config.get("params") != FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 参数不是冻结的 1734 树 seed29 参数")
    if model_config.get("training_policy") != FROZEN_TRAINING_POLICY:
        raise ValueError("LightGBM training policy 不是冻结的无早停、行等权策略")

    features = build_formal_feature_names()
    forbidden = (
        "direction",
        "mass",
        "seed_count",
        "p2_position",
        "separation",
        "oracle",
        "target",
        "surface",
        "geology",
    )
    bad = [name for name in features if any(token in name.lower() for token in forbidden)]
    if features[:41] != P3B00_FEATURES or bad:
        raise ValueError(f"PFM02 正式 44 列合同非法：{bad}")
    return features, dict(FROZEN_MODEL_PARAMS)


def merge_pfm02_legal_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    runtime_dir: Path,
    shadow_ids: set[str],
) -> tuple[pd.DataFrame, str]:
    """严格验证 Task1 八列缓存及 runtime 后，按自然键追加三列。"""

    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)
    if not runtime_dir.is_dir():
        raise FileNotFoundError(runtime_dir)
    shadow_overlap = {
        path.stem for path in cache_dir.glob("*.parquet")
    }.intersection(shadow_ids)
    if shadow_overlap:
        raise ValueError(f"PFM02 legal_cache 出现影子井：{sorted(shadow_overlap)[:3]}")

    rows: list[pd.DataFrame] = []
    common_fingerprint: str | None = None
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        expected_fold = int(registry_row.fold)
        expected_rows = int(registry_row.hidden_rows)
        cache_path = cache_dir / f"{well_id}.parquet"
        if not cache_path.is_file():
            raise FileNotFoundError(f"缺少 PFM02 井缓存：{cache_path}")
        observed_schema = parquet.ParquetFile(cache_path).schema_arrow
        if not observed_schema.equals(MODE_CACHE_SCHEMA):
            raise ValueError(
                f"PFM02 {well_id} schema 不等于冻结八列合同：{observed_schema}"
            )
        cache = pd.read_parquet(cache_path, columns=MODE_CACHE_COLUMNS)
        if len(cache) != expected_rows:
            raise ValueError(f"PFM02 {well_id} 行数不匹配")
        if cache.duplicated(["well_id", "row_index"]).any():
            raise ValueError(f"PFM02 {well_id} 含重复行键")
        if not cache["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"PFM02 {well_id} 井号不匹配")
        if not cache["fold"].astype(np.int64).eq(expected_fold).all():
            raise ValueError(f"PFM02 {well_id} fold 不匹配")
        if cache["_cache_fingerprint"].isna().any():
            raise ValueError(f"PFM02 {well_id} 缓存指纹为空")
        fingerprint_values = (
            cache["_cache_fingerprint"].astype(str).str.strip().unique().tolist()
        )
        if len(fingerprint_values) != 1 or not fingerprint_values[0]:
            raise ValueError(f"PFM02 {well_id} 缓存指纹不唯一或为空")
        fingerprint = fingerprint_values[0]
        if common_fingerprint is None:
            common_fingerprint = fingerprint
        elif fingerprint != common_fingerprint:
            raise ValueError("PFM02 各井缓存指纹不统一")

        runtime_path = runtime_dir / f"{well_id}.json"
        if not runtime_path.is_file():
            raise FileNotFoundError(f"缺少 PFM02 井运行记录：{runtime_path}")
        runtime = read_json(runtime_path)
        if runtime.get("experiment_fingerprint") != fingerprint:
            raise ValueError(f"PFM02 {well_id} 运行记录与缓存指纹不一致")
        if runtime.get("hidden_tvt_read") is not False:
            raise ValueError(f"PFM02 {well_id} 路径生成读取了隐藏 TVT")
        if (
            str(runtime.get("well_id")) != well_id
            or int(runtime.get("fold", -1)) != expected_fold
            or int(runtime.get("rows", -1)) != expected_rows
        ):
            raise ValueError(f"PFM02 {well_id} 运行记录井/fold/行数不匹配")
        rows.append(
            cache[["well_id", "row_index", "last_visible_tvt", *NEW_MODE_FEATURES]]
        )

    if common_fingerprint is None:
        raise ValueError("PFM02 没有开发井缓存")
    paths = pd.concat(rows, ignore_index=True)
    merged = pf02_runner._merge_path_table(
        feature_table,
        paths,
        NEW_MODE_FEATURES,
        anchor_name="pfm02_anchor",
    )
    return merged, common_fingerprint


def compare_with_p3b00(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """复用冻结配对评分实现，仅改写本实验编号。"""

    metrics, per_well, per_fold = pf02_runner.compare_with_p3b00(
        candidate, baseline, config, stage
    )
    metrics["experiment_id"] = EXPERIMENT_ID
    return metrics, per_well, per_fold


def remaining_folds_after_screen(
    requested_folds: list[int], metrics_folds01: dict[str, Any]
) -> list[int]:
    """只有 ``all`` 且 folds01 晋级时才返回 folds 2～4。"""

    if requested_folds == [0, 1]:
        return []
    if requested_folds != [0, 1, 2, 3, 4]:
        raise ValueError("请求折必须是 0,1 或 all")
    return [2, 3, 4] if metrics_folds01["success_checks"]["folds01_pass"] else []


def write_conclusion(output_dir: Path, metrics: dict[str, Any]) -> None:
    """写入严格限定到当前三路径直接追加实现的结论。"""

    stage = str(metrics["stage"])
    checks = metrics["success_checks"]
    if stage == "folds01":
        improvement = float(checks["combined_improvement_ft"])
        passed = bool(checks["folds01_pass"])
    else:
        improvement = float(checks["full5_improvement_ft"])
        passed = bool(checks["full5_pass"])
    text = f"""# P3-PFM02 单模 LightGBM 结论

数据直接证明的事实：本阶段相对 P3B00 的 micro RMSE 改善为 `{improvement:.6f} ft`，预登记门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：这里只检验三条无标签模式路径对原 41 列的直接增量价值。

仍然没有验证的猜测：模式路径是否需要可靠性权重或沿井深动态加权。

当前实验只能否定的具体实现：将 low/middle/high 三条整井路径直接追加到冻结单模 LightGBM。

下一步最便宜的验证：{'按预登记路线继续。' if passed else '停止当前三路径直接追加实现，不据此否定模式路径信息源。'}
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
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
    feature_table, generator_fingerprint = merge_pfm02_legal_cache(
        feature_table,
        registry,
        resolve_clean_path(config["source_pfm02_legal_cache_dir"]),
        resolve_clean_path(config["source_pfm02_runtime_dir"]),
        shadow_ids,
    )
    if set(feature_table["well_id"].astype(str).unique()).intersection(shadow_ids):
        raise RuntimeError("正式训练表含影子井")
    validate_feature_values(feature_table, model_features)

    fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "model_features": model_features,
            "model_params": model_params,
            "pfm02_generator_fingerprint": generator_fingerprint,
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
            "shadow_feature_overlap": 0,
            "arrow_filter_before_target_to_pandas": True,
            "validation_unit": "complete_well",
            "fold_version": config["fold_version"],
            "old_41_features_preserved": model_features[:41] == P3B00_FEATURES,
            "new_features": NEW_MODE_FEATURES,
            "pfm02_generator_fingerprint": generator_fingerprint,
            "hidden_tvt_read": False,
            "cv_fingerprint": fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )
    print(
        f"P3-PFM02 模型：开发井={len(registry)}，行={len(feature_table):,}，"
        f"特征=44，树=1734，折={requested_folds}，输出={output_dir}",
        flush=True,
    )

    for fold_id in (0, 1):
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            model_features,
        )
        save_fold_runtime_row(output_dir, runtime)
    candidate_folds01 = read_fold_predictions(output_dir, [0, 1])
    registry_folds01 = registry.loc[registry["fold"].astype(int).isin([0, 1])].copy()
    baseline_folds01 = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]), registry_folds01
    )
    metrics01, per_well01, per_fold01 = compare_with_p3b00(
        candidate_folds01, baseline_folds01, config, "folds01"
    )
    metrics01["cv_fingerprint"] = fingerprint
    save_stage_artifacts(
        output_dir,
        "folds01",
        candidate_folds01,
        metrics01,
        per_well01,
        per_fold01,
    )
    write_conclusion(output_dir, metrics01)
    print(json.dumps(metrics01["success_checks"], ensure_ascii=False, indent=2), flush=True)

    late_folds = remaining_folds_after_screen(requested_folds, metrics01)
    if not late_folds:
        if requested_folds == [0, 1, 2, 3, 4]:
            print("PFM02 模型 folds 0～1 未晋级，停止，不训练 folds 2～4。", flush=True)
        return

    for fold_id in late_folds:
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            model_features,
        )
        save_fold_runtime_row(output_dir, runtime)
    candidate_full = read_fold_predictions(output_dir, [0, 1, 2, 3, 4])
    baseline_full = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]), registry
    )
    metrics_full, per_well_full, per_fold_full = compare_with_p3b00(
        candidate_full, baseline_full, config, "full5"
    )
    metrics_full["cv_fingerprint"] = fingerprint
    save_stage_artifacts(
        output_dir,
        "full5",
        candidate_full,
        metrics_full,
        per_well_full,
        per_fold_full,
    )
    save_mean_feature_importance(output_dir)
    fold_runtimes = [
        read_json(output_dir / f"fold_{fold_id}" / "runtime.json")
        for fold_id in range(5)
    ]
    write_json(
        output_dir / "runtime.json",
        {
            "cv_fingerprint": fingerprint,
            "folds": fold_runtimes,
            "total_fold_seconds": float(
                sum(float(row["seconds"]) for row in fold_runtimes)
            ),
        },
    )
    write_conclusion(output_dir, metrics_full)
    print(json.dumps(metrics_full["success_checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
