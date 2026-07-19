"""用四条固定 ESS 路径一一替换 P3B00 的四条固定 scale 路径。

本实验不增加特征数量：P3B00 的其余 37 列原样保留，
``scale 3/5/8/12`` 分别替换成 ``ESS 2/8/32/96``，总数仍为 41。
训练、折分、评价行和 LightGBM 参数全部复用冻结基线合同。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pf02_target_ess_lgbm_cv as common  # noqa: E402


EXPERIMENT_ID = "P3_PF02b_replace_fixed_scales_lgbm_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pf02b_replace_fixed_scales_lgbm_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

REPLACEMENT_MAPPING = {
    "pf128_scale_3_delta": "pf128_ess2_delta",
    "pf128_scale_5_delta": "pf128_ess8_delta",
    "pf128_scale_8_delta": "pf128_ess32_delta",
    "pf128_scale_12_delta": "pf128_ess96_delta",
}


def parse_fold_spec(value: str) -> list[int]:
    """只允许预注册的 folds 0～1 筛查或完整五折。"""

    return common.parse_fold_spec(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="筛查用 0,1；晋级后用 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def build_formal_feature_names() -> list[str]:
    """按原位置替换四条路径，返回等维的 41 列正式特征。"""

    features = [
        REPLACEMENT_MAPPING.get(feature_name, feature_name)
        for feature_name in common.P3B00_FEATURES
    ]
    if len(features) != 41 or len(set(features)) != 41:
        raise ValueError("PF02b 正式特征必须是无重复的 41 列")
    if any(old_name in features for old_name in REPLACEMENT_MAPPING):
        raise ValueError("PF02b 仍残留被替换的固定 scale 路径")
    if list(REPLACEMENT_MAPPING.values()) != features[-4:]:
        raise ValueError("PF02b 四条 ESS 路径没有按冻结顺序替换")
    return features


def validate_frozen_contract(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """确认唯一变化是四条固定 scale 路径被等维 ESS 路径替换。"""

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PF02b 配置实验编号错误")
    if config.get("baseline_id") != "P3B00_group5_p2p02_v1":
        raise ValueError("PF02b 基线不是冻结 P3B00")
    if config.get("fold_version") != "balanced_well_5fold_v1":
        raise ValueError("PF02b fold 版本错误")
    if config.get("shadow_target_access") is not False:
        raise ValueError("PF02b 禁止打开影子集目标")
    if config.get("model_training") is not True:
        raise ValueError("PF02b 未声明单模训练")
    if config.get("replacement_mapping") != REPLACEMENT_MAPPING:
        raise ValueError("PF02b 替换关系不是预注册的一一映射")
    if int(config.get("formal_feature_count", -1)) != 41:
        raise ValueError("PF02b 正式特征数必须固定为 41")

    manifest = common.read_json(common.resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41:
        raise ValueError("P3B00 特征数不再是 41")
    if manifest.get("features") != common.P3B00_FEATURES:
        raise ValueError("P3B00 冻结特征清单发生变化")

    model_config = common.read_json(common.resolve_clean_path(config["model_config"]))
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("模型不是冻结的单模 LightGBM")
    if model_config.get("params") != common.FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 参数不是冻结的 1734 树、seed29 参数")

    formal_features = build_formal_feature_names()
    unchanged_features = [
        feature_name
        for feature_name in common.P3B00_FEATURES
        if feature_name not in REPLACEMENT_MAPPING
    ]
    if len(unchanged_features) != 37:
        raise ValueError("PF02b 没有精确保留 P3B00 的其他 37 列")
    for feature_name in unchanged_features:
        original_position = common.P3B00_FEATURES.index(feature_name)
        if formal_features[original_position] != feature_name:
            raise ValueError(f"PF02b 改动了冻结列 {feature_name}")

    forbidden = [
        feature_name
        for feature_name in formal_features
        if any(word in feature_name.lower() for word in ("target", "truth", "oracle", "surface"))
    ]
    if forbidden:
        raise ValueError(f"PF02b 正式特征含禁止列：{forbidden}")
    return formal_features, dict(common.FROZEN_MODEL_PARAMS)


def write_conclusion(output_dir: Path, metrics: dict[str, Any]) -> None:
    """按三阶段模板记录事实、边界和下一步。"""

    stage = str(metrics["stage"])
    checks = metrics["success_checks"]
    if stage == "folds01":
        improvement = float(checks["combined_improvement_ft"])
        passed = bool(checks["folds01_pass"])
    else:
        improvement = float(checks["full5_improvement_ft"])
        passed = bool(checks["full5_pass"])

    next_step = (
        "按预注册流程继续下一阶段。"
        if passed
        else "停止当前等维替换实现；不根据本次 fold 结果挑选 ESS 子集。"
    )
    text = f"""# P3-PF02b 等维路径替换结论

数据直接证明的事实：本阶段相对 P3B00 的 micro RMSE 改善为 `{improvement:.6f} ft`，预注册门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：这里只检验固定 ESS 路径能否在不增加维度时，比固定 scale 路径提供更有用的整井路径表示。

仍然没有验证的猜测：PF 路径可靠性是否需要沿井深动态变化。

当前实验只能否定的具体实现：把 scale 3/5/8/12 一一替换为 ESS 2/8/32/96 的 41 列单模 LightGBM。

下一步最便宜的验证：{next_step}
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _score_stage(
    output_dir: Path,
    fold_ids: list[int],
    registry,
    config: dict[str, Any],
    fingerprint: str,
    stage: str,
) -> dict[str, Any]:
    """读取指定折预测，并与同评价行的 P3B00 开发集 OOF 配对评分。"""

    candidate = common.read_fold_predictions(output_dir, fold_ids)
    stage_registry = registry.loc[registry["fold"].astype(int).isin(fold_ids)].copy()
    baseline = common.read_baseline_predictions(
        common.resolve_clean_path(config["source_p3b00_predictions"]),
        stage_registry,
    )
    metrics, per_well, per_fold = common.compare_with_p3b00(
        candidate,
        baseline,
        config,
        stage,
    )
    metrics["experiment_id"] = EXPERIMENT_ID
    metrics["cv_fingerprint"] = fingerprint
    common.save_stage_artifacts(
        output_dir,
        stage,
        candidate,
        metrics,
        per_well,
        per_fold,
    )
    write_conclusion(output_dir, metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config = common.read_json(args.config.resolve())
    model_features, model_params = validate_frozen_contract(config)
    observed_hashes = common.validate_source_hashes(config)
    registry, shadow_ids = common.load_development_registry(
        common.resolve_clean_path(config["fold_registry"]),
        common.resolve_clean_path(config["shadow_registry"]),
        config,
    )

    feature_table = common.load_development_feature_table(
        common.resolve_clean_path(config["base_feature_cache"]),
        common.resolve_clean_path(config["candidate_feature_cache"]),
        registry,
    )
    feature_table = common.merge_existing_p3b00_pf_cache(
        feature_table,
        registry,
        common.resolve_clean_path(config["source_p01_legal_cache_dir"]),
        str(config["source_p01_fingerprint"]),
    )
    feature_table, pf02_fingerprint = common.merge_pf02_legal_cache(
        feature_table,
        registry,
        common.resolve_clean_path(config["source_pf02_legal_cache_dir"]),
        common.resolve_clean_path(config["source_pf02_runtime_dir"]),
        shadow_ids,
        require_runtime=True,
    )
    if set(feature_table["well_id"].astype(str).unique()).intersection(shadow_ids):
        raise RuntimeError("PF02b 正式训练表含影子井")
    common.validate_feature_values(feature_table, model_features)

    fingerprint = common.stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "model_features": model_features,
            "model_params": model_params,
            "pf02_generator_fingerprint": pf02_fingerprint,
            "observed_source_hashes": observed_hashes,
            "runner_sha256": common.file_sha256(Path(__file__).resolve()),
            "shared_runner_sha256": common.file_sha256(
                CLEAN_ROOT / "scripts" / "run_p3_pf02_target_ess_lgbm_cv.py"
            ),
            "trainer_sha256": common.file_sha256(
                CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"
            ),
        }
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(output_dir / "config.json", config)
    common.write_json(output_dir / "parameter_list.json", model_params)
    common.write_json(
        output_dir / "feature_list.json",
        {
            "feature_count": 41,
            "features": model_features,
            "replacement_mapping": REPLACEMENT_MAPPING,
        },
    )
    common.write_json(
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
            "baseline_other_37_features_preserved": True,
            "replaced_features": REPLACEMENT_MAPPING,
            "formal_feature_count": 41,
            "pf02_generator_fingerprint": pf02_fingerprint,
            "cv_fingerprint": fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )
    print(
        f"P3-PF02b：开发井={len(registry)}，行={len(feature_table):,}，"
        f"特征=41，树=1734，折={requested_folds}，输出={output_dir}",
        flush=True,
    )

    for fold_id in (0, 1):
        runtime = common.train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            model_features,
        )
        common.save_fold_runtime_row(output_dir, runtime)

    metrics01 = _score_stage(
        output_dir,
        [0, 1],
        registry,
        config,
        fingerprint,
        "folds01",
    )
    print(json.dumps(metrics01["success_checks"], ensure_ascii=False, indent=2), flush=True)
    if requested_folds == [0, 1]:
        return
    if not metrics01["success_checks"]["folds01_pass"]:
        print("PF02b folds 0～1 未晋级，停止，不训练 folds 2～4。", flush=True)
        return

    for fold_id in (2, 3, 4):
        runtime = common.train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            model_features,
        )
        common.save_fold_runtime_row(output_dir, runtime)

    metrics_full = _score_stage(
        output_dir,
        [0, 1, 2, 3, 4],
        registry,
        config,
        fingerprint,
        "full5",
    )
    common.save_mean_feature_importance(output_dir)
    fold_runtimes = [
        common.read_json(output_dir / f"fold_{fold_id}" / "runtime.json")
        for fold_id in range(5)
    ]
    common.write_json(
        output_dir / "runtime.json",
        {
            "cv_fingerprint": fingerprint,
            "folds": fold_runtimes,
            "total_fold_seconds": float(
                sum(float(runtime["seconds"]) for runtime in fold_runtimes)
            ),
        },
    )
    print(json.dumps(metrics_full["success_checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
