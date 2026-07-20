"""在 P3B00 原 41 列后只追加三个 PFM04 井级摘要并运行冻结 LightGBM。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

# PFM03 已经实现并验证了同一批 41 列的加载、按折训练、评分和落盘流程。
from scripts import run_p3_pfm03_shape_aware_modes_cv as base_runner  # noqa: E402
from scripts import run_p3_pf02_target_ess_lgbm_cv as common_runner  # noqa: E402
from scripts.run_p2_cv00_group5_c01 import save_fold_runtime_row  # noqa: E402
from scripts.run_p2_p02_multiscale_pf_paths_cv import (  # noqa: E402
    FROZEN_MODEL_FEATURES as _P3B00_FEATURES,
    FROZEN_MODEL_PARAMS,
)
from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json  # noqa: E402


EXPERIMENT_ID = "P3_PFM04_mode_direction_summary_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pfm04_mode_direction_summary_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
P3B00_FEATURES = list(_P3B00_FEATURES)
NEW_SUMMARY_FEATURES = [
    "direction_score",
    "p2_position_raw",
    "high_minus_low_separation",
]
SUMMARY_COLUMNS = [
    "well_id",
    "fold",
    *NEW_SUMMARY_FEATURES,
    "source_seed_sha256",
    "source_p2_sha256",
    "_cache_fingerprint",
]

# 公开这些冻结工具，测试和正式入口都通过同一实现核对 SHA、路径与 CV 血缘。
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
    """严格返回原 41 列加三个摘要，新增列固定追加在末尾。"""

    features = [*P3B00_FEATURES, *NEW_SUMMARY_FEATURES]
    if len(features) != 44 or len(set(features)) != 44:
        raise ValueError("PFM04 正式特征必须是无重复的 41+3=44 列")
    return features


def parse_fold_spec(value: str) -> list[int]:
    """第一入口固定 folds1～2；只有它通过后才允许请求 folds1～4。"""

    normalized = str(value).replace(" ", "")
    if normalized == "1,2":
        return [1, 2]
    if normalized == "1,2,3,4":
        return [1, 2, 3, 4]
    raise ValueError("--folds 只允许 1,2 或 1,2,3,4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析折范围、冻结配置和产物目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def validate_frozen_contract(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """核对实验名、三列、原 41 列和 LightGBM 全部参数均未变化。"""

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PFM04 experiment_id 不一致")
    expected = {
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
        "new_feature_names": NEW_SUMMARY_FEATURES,
    }
    changed = [key for key, value in expected.items() if config.get(key) != value]
    if changed:
        raise ValueError(f"PFM04 冻结配置发生变化：{changed}")
    manifest = read_json(resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41 or manifest.get("features") != P3B00_FEATURES:
        raise ValueError("P3B00 原 41 列清单发生变化")
    model_path = resolve_clean_path(config["model_config"])
    if file_sha256(model_path) != str(config["model_config_sha256"]):
        raise ValueError("冻结 LightGBM 配置文件 SHA 发生变化")
    model_config = read_json(model_path)
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("正式模型不是单模 LightGBM")
    if model_config.get("params") != FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 不再是冻结的 1734 树 seed29 参数")
    if model_config.get("training_policy") != FROZEN_TRAINING_POLICY:
        raise ValueError("LightGBM 训练策略发生变化")
    return build_formal_feature_names(), dict(FROZEN_MODEL_PARAMS)


def broadcast_mode_summaries(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    summary_path: Path,
    shadow_ids: set[str],
    runtime_path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """验证一井一行摘要，再只按自然 well_id 广播到该井所有隐藏行。"""

    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少 PFM04 合法摘要：{summary_path}")
    summaries = pd.read_parquet(summary_path)
    if summaries.columns.tolist() != SUMMARY_COLUMNS:
        raise ValueError("PFM04 摘要列不等于冻结合同")
    summaries["well_id"] = summaries["well_id"].astype(str)
    if summaries["well_id"].duplicated().any():
        raise ValueError("PFM04 摘要不是一井一行")
    if overlap := set(summaries["well_id"]).intersection(shadow_ids):
        raise ValueError(f"PFM04 摘要含影子井：{sorted(overlap)[:3]}")

    expected = registry[["well_id", "fold", "hidden_rows"]].copy()
    expected["well_id"] = expected["well_id"].astype(str)
    if set(summaries["well_id"]) != set(expected["well_id"]):
        raise ValueError("PFM04 摘要井集合与开发 registry 不一致")
    fold_check = expected[["well_id", "fold"]].merge(
        summaries[["well_id", "fold"]], on="well_id", how="left", suffixes=("_expected", "_summary")
    )
    if not fold_check["fold_expected"].astype(np.int64).equals(
        fold_check["fold_summary"].astype(np.int64)
    ):
        raise ValueError("PFM04 摘要 fold 与冻结 registry 不一致")
    if not np.isfinite(summaries[NEW_SUMMARY_FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError("PFM04 摘要特征含 NaN/Inf")
    fingerprint_values = summaries["_cache_fingerprint"].astype(str).unique().tolist()
    if len(fingerprint_values) != 1 or not fingerprint_values[0]:
        raise ValueError("PFM04 摘要指纹为空或不统一")
    fingerprint = fingerprint_values[0]
    if summaries[["source_seed_sha256", "source_p2_sha256"]].isna().any().any():
        raise ValueError("PFM04 摘要缺来源指纹")

    if runtime_path is not None:
        runtime = read_json(runtime_path)
        if runtime.get("experiment_fingerprint") != fingerprint:
            raise ValueError("PFM04 runtime 与摘要指纹不一致")
        if runtime.get("hidden_tvt_read") is not False:
            raise ValueError("PFM04 摘要生成读取了隐藏 TVT")
        if int(runtime.get("wells", -1)) != len(expected):
            raise ValueError("PFM04 runtime 开发井数不一致")
        if int(runtime.get("rows", -1)) != int(expected["hidden_rows"].sum()):
            raise ValueError("PFM04 runtime 隐藏行数不一致")

    original_keys = feature_table[["well_id", "row_index"]].copy()
    merged = feature_table.merge(
        summaries[["well_id", *NEW_SUMMARY_FEATURES]],
        on="well_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if not merged[["well_id", "row_index"]].equals(original_keys):
        raise RuntimeError("PFM04 广播改变了原始行顺序或自然键")
    if merged[NEW_SUMMARY_FEATURES].isna().any().any():
        raise ValueError("PFM04 广播后有开发行缺摘要")
    return merged, fingerprint


def _compare_stage(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """复用 PFM03 的冻结评分门槛，只把实验编号改为 PFM04。"""

    metrics, per_well, per_fold = base_runner.compare_stage(
        candidate, baseline, config, stage
    )
    metrics["experiment_id"] = EXPERIMENT_ID
    return metrics, per_well, per_fold


def _write_conclusion(output_dir: Path, metrics: dict[str, Any]) -> None:
    """结论只覆盖三个模式摘要直接进入冻结 LightGBM 这一种实现。"""

    checks = metrics["success_checks"]
    if metrics["stage"] == "folds12":
        improvement = float(checks["average_fold_improvement_ft"])
        passed = bool(checks["folds12_pass"])
    else:
        improvement = float(checks["confirmed4_pooled_improvement_ft"])
        passed = bool(checks["confirmed4_pass"])
    text = f"""# P3-PFM04 模式方向摘要结论

数据直接证明的事实：当前阶段改善为 `{improvement:.6f} ft`，预登记门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：这里只检验三个井级 PF 模式摘要对原 41 列的直接增量价值。

仍然没有验证的猜测：这些摘要是否适合独立的严格嵌套残差分类器。

当前实验只能否定的具体实现：三个冻结摘要直接进入单模 LightGBM。

下一步最便宜的验证：按预登记门槛停止或继续，不调整特征和模型参数。
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """生成正式训练表，固定先训练 folds1～2，通过门槛后才训练 folds3～4。"""

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
    feature_table, generator_fingerprint = broadcast_mode_summaries(
        feature_table,
        registry,
        resolve_clean_path(config["source_pfm04_summary"]),
        shadow_ids,
        resolve_clean_path(config["source_pfm04_runtime"]),
    )
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
    write_json(output_dir / "feature_list.json", {"feature_count": 44, "features": model_features})
    write_json(
        output_dir / "leakage_audit.json",
        {
            "shadow_target_access": False,
            "development_wells": int(feature_table["well_id"].nunique()),
            "development_hidden_rows": len(feature_table),
            "fold0_used_for_gate": False,
            "old_41_features_preserved": model_features[:41] == P3B00_FEATURES,
            "new_features": NEW_SUMMARY_FEATURES,
            "summary_generator_fingerprint": generator_fingerprint,
            "hidden_tvt_read_for_summary_generation": False,
            "cv_fingerprint": cv_fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )
    print(
        f"P3-PFM04：开发井={len(registry)}，行={len(feature_table):,}，"
        f"特征=44，树=1734，正式折={requested_folds}",
        flush=True,
    )

    for fold_id in (1, 2):
        runtime = train_fold(
            feature_table, registry, fold_id, model_params, output_dir, cv_fingerprint, model_features
        )
        save_fold_runtime_row(output_dir, runtime)
    registry12 = registry.loc[registry["fold"].astype(int).isin([1, 2])].copy()
    candidate12 = read_fold_predictions(output_dir, [1, 2])
    baseline12 = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]), registry12
    )
    metrics12, per_well12, per_fold12 = _compare_stage(
        candidate12, baseline12, config, "folds12"
    )
    metrics12["cv_fingerprint"] = cv_fingerprint
    base_runner._save_stage(
        output_dir, "folds12", candidate12, metrics12, per_well12, per_fold12
    )
    _write_conclusion(output_dir, metrics12)
    print(json.dumps(metrics12["success_checks"], ensure_ascii=False, indent=2), flush=True)

    late_folds = base_runner.remaining_folds_after_screen(requested_folds, metrics12)
    if not late_folds:
        if requested_folds == [1, 2, 3, 4]:
            print("PFM04 folds1～2 未晋级，不训练 folds3～4。", flush=True)
        return
    for fold_id in late_folds:
        runtime = train_fold(
            feature_table, registry, fold_id, model_params, output_dir, cv_fingerprint, model_features
        )
        save_fold_runtime_row(output_dir, runtime)
    registry4 = registry.loc[registry["fold"].astype(int).isin([1, 2, 3, 4])].copy()
    candidate4 = read_fold_predictions(output_dir, [1, 2, 3, 4])
    baseline4 = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]), registry4
    )
    metrics4, per_well4, per_fold4 = _compare_stage(
        candidate4, baseline4, config, "confirmed4"
    )
    metrics4["cv_fingerprint"] = cv_fingerprint
    base_runner._save_stage(output_dir, "confirmed4", candidate4, metrics4, per_well4, per_fold4)
    base_runner._save_feature_importance(output_dir, [1, 2, 3, 4])
    _write_conclusion(output_dir, metrics4)
    print(json.dumps(metrics4["success_checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
