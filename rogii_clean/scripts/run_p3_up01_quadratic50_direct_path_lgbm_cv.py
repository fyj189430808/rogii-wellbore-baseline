"""将 UP01 的二次投影 50% 单路径加入 P3B00，运行冻结 LightGBM。

候选已经由 folds1-2 的路径自身结果选定，因此本入口先且只用 folds3-4 做独立
确认。确认达到“平均改善至少 0.15 ft、任一折恶化不超过 0.15 ft”后，才补 fold0，
最后训练 folds1-2 仅用于拼接完整开发集 OOF，并明确标记它们参与过候选选择。
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


EXPERIMENT_ID = "P3_UP01_quadratic50_direct_path_lgbm_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up01_quadratic50_direct_path_lgbm_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
FROZEN_CONFIG_FILE_SHA256 = "7b4e79e23289d76d32d2708c2eb0b8fc1349ae51bf556944d14a13c3bc8d3c7d"

P3B00_FEATURES = list(_P3B00_FEATURES)
NEW_UP01_FEATURES = ["up01_degree2_blend50_delta"]
SELECTED_CANDIDATE = "degree2_blend50"
SELECTED_CANDIDATE_COLUMN = "degree2_blend50_pred_tvt"
CONFIRMATION_FOLDS = [3, 4]
SELECTION_FOLDS = [1, 2]

EXPECTED_UP01_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("row_index", pa.int32()),
        pa.field("md", pa.float64()),
        pa.field("p2_pred_tvt", pa.float64()),
        pa.field("degree2_blend25_pred_tvt", pa.float64()),
        pa.field("degree2_blend50_pred_tvt", pa.float64()),
        pa.field("degree2_blend75_pred_tvt", pa.float64()),
        pa.field("degree3_blend25_pred_tvt", pa.float64()),
        pa.field("degree3_blend50_pred_tvt", pa.float64()),
        pa.field("degree3_blend75_pred_tvt", pa.float64()),
    ]
)
FROZEN_TRAINING_POLICY = {
    "early_stopping": False,
    "uniform_row_weight": True,
    "native_missing_values": True,
    "well_id_is_feature": False,
    "row_id_is_feature": False,
    "test_overlap_wells_removed_from_training": True,
}
CONFIRMATION_GATE = {
    "minimum_arithmetic_mean_fold_improvement_ft": 0.15,
    "maximum_any_fold_degradation_ft": 0.15,
}
# 这三个大 parquet 已属于冻结 P3B00，UP01 不重复顺序读取整文件算 SHA；随后仍会
# 实际检查 321 万行、自然键、fold 和目标配对。新生成的 UP01 候选不在此集合中，
# 它必须在 ``validate_legal_candidate_cache`` 中重新逐字节核对 SHA。
LARGE_REGISTERED_SOURCE_KEYS = {
    "base_feature_cache",
    "candidate_feature_cache",
    "source_p3b00_predictions",
}

resolve_clean_path = pf02_runner.resolve_clean_path
file_sha256 = pf02_runner.file_sha256
stable_hash = pf02_runner.stable_hash
load_development_registry = pf02_runner.load_development_registry
read_development_parquet = pf02_runner.read_development_parquet
load_development_feature_table = pf02_runner.load_development_feature_table
merge_existing_p3b00_pf_cache = pf02_runner.merge_existing_p3b00_pf_cache
validate_feature_values = pf02_runner.validate_feature_values
read_baseline_predictions = pf02_runner.read_baseline_predictions
read_fold_predictions = pf02_runner.read_fold_predictions
save_stage_artifacts = pf02_runner.save_stage_artifacts
save_mean_feature_importance = pf02_runner.save_mean_feature_importance


def build_formal_feature_names() -> list[str]:
    """返回原 41 列逐名保留、末尾只追加一条 UP01 路径的 42 列合同。"""

    features = [*P3B00_FEATURES, *NEW_UP01_FEATURES]
    if len(features) != 42 or len(set(features)) != 42:
        raise ValueError("UP01 正式特征必须是无重复的 41+1=42 列")
    return features


def validate_frozen_contract(
    config: dict[str, Any],
    enforce_config_file_hash: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """在读开发目标前锁死候选、证据折、模型参数和唯一新增特征。"""

    if enforce_config_file_hash and file_sha256(DEFAULT_CONFIG) != FROZEN_CONFIG_FILE_SHA256:
        raise ValueError("UP01 模型冻结配置文件 SHA-256 已变化")
    exact_values = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "P3B00_group5_p2p02_v1",
        "fold_version": "balanced_well_5fold_v1",
        "selected_candidate": SELECTED_CANDIDATE,
        "selected_candidate_column": SELECTED_CANDIDATE_COLUMN,
        "new_feature_names": NEW_UP01_FEATURES,
        "selection_folds": SELECTION_FOLDS,
        "independent_confirmation_folds": CONFIRMATION_FOLDS,
        "post_confirmation_fold": [0],
        "selection_folds_full5_only": SELECTION_FOLDS,
        "baseline_feature_count": 41,
        "formal_feature_count": 42,
        "development_wells": 657,
        "development_hidden_rows": 3_211_872,
        "model_training": True,
        "shadow_target_access": False,
        "touch_pfs_legal_cache": False,
        "confirmation_gate": CONFIRMATION_GATE,
    }
    changed = [key for key, expected in exact_values.items() if config.get(key) != expected]
    if changed:
        raise ValueError(f"UP01 冻结候选/模型合同发生变化：{changed}")

    manifest = read_json(resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41 or manifest.get("features") != P3B00_FEATURES:
        raise ValueError("P3B00 原 41 列清单发生变化")
    model = read_json(resolve_clean_path(config["model_config"]))
    if model.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("冻结模型不是单模 LightGBM")
    if model.get("params") != FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 参数、1734 棵树或 seed29 发生变化")
    if model.get("training_policy") != FROZEN_TRAINING_POLICY:
        raise ValueError("LightGBM 训练策略发生变化")
    features = build_formal_feature_names()
    if [name for name in features if name.startswith("up01_")] != NEW_UP01_FEATURES:
        raise ValueError("正式模型混入了未登记 UP01 候选或摘要")
    return features, dict(FROZEN_MODEL_PARAMS)


def validate_selection_provenance(
    metrics: dict[str, Any],
    selected_candidate: str,
    selection_folds: list[int],
) -> dict[str, Any]:
    """确认 degree2_blend50 确实仅凭 folds1-2 的路径自身平均改善被选中。"""

    if [int(value) for value in metrics.get("folds", [])] != selection_folds:
        raise ValueError("UP01 候选选择指标不是固定 folds1-2")
    if metrics.get("model_training") is not False:
        raise ValueError("UP01 候选选择阶段不应训练模型")
    if metrics.get("candidate_generation_hidden_target_read") is not False:
        raise ValueError("UP01 候选生成阶段读取了隐藏目标")
    if metrics.get("shadow_target_read") is not False:
        raise ValueError("UP01 候选选择读取了影子目标")
    candidates = metrics.get("candidates")
    if not isinstance(candidates, dict) or selected_candidate not in candidates:
        raise ValueError("UP01 候选选择指标缺少锁定候选")
    means = {
        str(name): float(values["arithmetic_mean_fold_improvement_ft"])
        for name, values in candidates.items()
    }
    best_value = max(means.values())
    best_names = [name for name, value in means.items() if value == best_value]
    if best_names != [selected_candidate]:
        raise ValueError(f"锁定候选不是 folds1-2 平均改善最好的唯一选择：{best_names}")
    return {
        "selected_candidate": selected_candidate,
        "selection_folds": selection_folds,
        "selection_used_target": True,
        "selection_metric": "arithmetic_mean_fold_improvement_ft",
        "selected_mean_improvement_ft": means[selected_candidate],
        "independent_confirmation_folds": CONFIRMATION_FOLDS,
    }


def validate_legal_candidate_cache(
    candidate_path: Path,
    manifest: dict[str, Any],
    registry: pd.DataFrame,
    shadow_ids: set[str],
    safe_source_keys: pd.DataFrame,
    expected_sha256: str,
    expected_wells: int,
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只读合法列，核验 657 井、自然键、影子隔离及生成阶段未读目标。"""

    actual_sha = file_sha256(candidate_path)
    if actual_sha != str(expected_sha256):
        raise ValueError("UP01 合法候选 parquet SHA-256 不匹配")
    if manifest.get("experiment_id") != "P3_UP01_robust_u_projection_v1":
        raise ValueError("UP01 合法候选 manifest 实验编号错误")
    if manifest.get("candidate_generation_complete") is not True:
        raise ValueError("UP01 合法候选尚未完整生成")
    if manifest.get("hidden_target_read") is not False:
        raise ValueError("UP01 候选生成阶段读取了隐藏 target")
    if int(manifest.get("shadow_overlap_wells", -1)) != 0:
        raise ValueError("UP01 合法候选 manifest 报告影子井重叠")
    if int(manifest.get("wells", -1)) != int(expected_wells):
        raise ValueError("UP01 合法候选 manifest 井数错误")
    if int(manifest.get("rows", -1)) != int(expected_rows):
        raise ValueError("UP01 合法候选 manifest 行数错误")
    if str(manifest.get("legal_candidates_sha256")) != actual_sha:
        raise ValueError("UP01 manifest 与候选 parquet 哈希不一致")

    observed_schema = parquet.ParquetFile(candidate_path).schema_arrow
    if not observed_schema.equals(EXPECTED_UP01_SCHEMA):
        raise ValueError("UP01 legal_candidates schema 不符合冻结生成合同")
    if any(token in observed_schema.names for token in ("target_tvt", "TVT", "oracle")):
        raise ValueError("UP01 合法候选缓存含隐藏目标或 oracle 列")
    columns = ["well_id", "fold", "row_index", SELECTED_CANDIDATE_COLUMN]
    selected = pd.read_parquet(candidate_path, columns=columns)
    selected["well_id"] = selected["well_id"].astype(str)
    if len(selected) != int(expected_rows):
        raise ValueError("UP01 合法候选实际行数错误")
    if selected["well_id"].nunique() != int(expected_wells):
        raise ValueError("UP01 合法候选实际井数错误")
    if selected.duplicated(["well_id", "row_index"]).any():
        raise ValueError("UP01 合法候选含重复自然行键")
    if set(selected["well_id"].unique()).intersection(shadow_ids):
        raise ValueError("UP01 合法候选混入影子井")
    if not np.isfinite(selected[SELECTED_CANDIDATE_COLUMN].to_numpy(dtype=np.float64)).all():
        raise ValueError("UP01 选定路径含 NaN/Inf，必须全部有限")

    fold_map = registry.set_index("well_id")["fold"].astype(int)
    expected_fold = selected["well_id"].map(fold_map)
    if expected_fold.isna().any() or not np.array_equal(
        expected_fold.to_numpy(dtype=np.int64),
        selected["fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("UP01 合法候选 fold 与冻结注册表不一致")
    registry_rows = registry.set_index("well_id")["hidden_rows"].astype(int)
    observed_rows = selected.groupby("well_id", sort=False).size().astype(int)
    if not observed_rows.sort_index().equals(registry_rows.sort_index()):
        raise ValueError("UP01 合法候选逐井行数不一致")

    left = selected[["well_id", "fold", "row_index"]].copy()
    right = safe_source_keys[["well_id", "fold", "row_index"]].copy()
    for frame in (left, right):
        frame["well_id"] = frame["well_id"].astype(str)
        frame["fold"] = frame["fold"].astype(np.int64)
        frame["row_index"] = frame["row_index"].astype(np.int64)
        frame.sort_values(["well_id", "row_index"], inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)
    if len(left) != len(right) or not left.equals(right):
        raise ValueError("UP01 合法候选自然键与 P2-P02 安全行键不一致")

    audit = {
        "legal_cache_complete": True,
        "hidden_target_read": False,
        "shadow_overlap_wells": 0,
        "natural_key_match": True,
        "development_wells": int(expected_wells),
        "development_hidden_rows": int(expected_rows),
        "legal_candidates_sha256": actual_sha,
        "only_selected_candidate_loaded": True,
        "selected_candidate_column": SELECTED_CANDIDATE_COLUMN,
    }
    return (
        selected[["well_id", "row_index", SELECTED_CANDIDATE_COLUMN]].copy(),
        audit,
    )


def merge_selected_path(
    feature_table: pd.DataFrame,
    selected_path: pd.DataFrame,
) -> pd.DataFrame:
    """按自然键追加唯一绝对路径，并转成相对末个可见 TVT 的一列 delta。"""

    if NEW_UP01_FEATURES[0] in feature_table.columns:
        raise ValueError("UP01 新特征已存在，禁止覆盖")
    base = feature_table.copy()
    base["_original_order"] = np.arange(len(base), dtype=np.int64)
    merged = base.merge(
        selected_path,
        on=["well_id", "row_index"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(base) or not merged["_merge"].eq("both").all():
        raise ValueError("UP01 选定路径与特征表自然键不能一一对齐")
    absolute = merged[SELECTED_CANDIDATE_COLUMN].to_numpy(dtype=np.float64)
    anchor = merged["last_visible_tvt"].to_numpy(dtype=np.float64)
    delta = absolute - anchor
    if not np.isfinite(delta).all():
        raise ValueError("UP01 degree2_blend50 delta 含 NaN/Inf")
    merged[NEW_UP01_FEATURES[0]] = delta.astype(np.float32)
    merged = merged.sort_values("_original_order", kind="stable")
    return merged.drop(columns=[SELECTED_CANDIDATE_COLUMN, "_original_order", "_merge"])


def evaluate_confirmation_gate(per_fold: pd.DataFrame) -> dict[str, Any]:
    """按算术平均折改善和最差折退化评价独立 folds3-4。"""

    if set(per_fold["fold"].astype(int)) != set(CONFIRMATION_FOLDS):
        raise ValueError("独立确认指标必须且只能来自 folds3-4")
    improvements = per_fold["improvement_ft"].to_numpy(dtype=np.float64)
    mean_improvement = float(np.mean(improvements))
    maximum_degradation = float(max(0.0, -float(np.min(improvements))))
    passed = bool(
        mean_improvement
        >= CONFIRMATION_GATE["minimum_arithmetic_mean_fold_improvement_ft"]
        and maximum_degradation
        <= CONFIRMATION_GATE["maximum_any_fold_degradation_ft"]
    )
    return {
        "arithmetic_mean_fold_improvement_ft": mean_improvement,
        "maximum_any_fold_degradation_ft": maximum_degradation,
        "minimum_required_mean_improvement_ft": 0.15,
        "maximum_allowed_fold_degradation_ft": 0.15,
        "confirmation_pass": passed,
    }


def parse_stage(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"confirm", "all"}:
        raise ValueError("--stage 只允许 confirm 或 all")
    return normalized


def continuation_folds(stage: str, gate: dict[str, Any]) -> list[int]:
    """未通过确认绝不打开其他折；通过后先 fold0，再仅为拼接训练 folds1-2。"""

    if stage == "confirm":
        return []
    if stage != "all":
        raise ValueError("stage 只允许 confirm 或 all")
    return [0, 1, 2] if gate.get("confirmation_pass") is True else []


def fold_evidence_labels() -> dict[str, str]:
    """永久记录哪些折独立、哪些折参与过候选选择。"""

    return {
        "0": "post_confirmation_additional_fold",
        "1": "used_for_candidate_selection_full5_only",
        "2": "used_for_candidate_selection_full5_only",
        "3": "independent_confirmation",
        "4": "independent_confirmation",
    }


def _comparison_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["success_conditions"] = {
        "fold01_minimum_combined_improvement_ft": 0.15,
        "fold01_maximum_single_fold_degradation_ft": 0.15,
        **config["full5_success_conditions"],
    }
    return result


def compare_predictions(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metrics, per_well, per_fold = pf02_runner.compare_with_p3b00(
        candidate,
        baseline,
        _comparison_config(config),
        "full5" if stage == "full5" else "folds01",
    )
    metrics["experiment_id"] = EXPERIMENT_ID
    metrics["stage"] = stage
    if stage == "confirmation_folds34":
        metrics["success_checks"] = evaluate_confirmation_gate(per_fold)
    return metrics, per_well, per_fold


def _validate_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    keys = [
        "fold_registry",
        "shadow_registry",
        "base_feature_cache",
        "candidate_feature_cache",
        "source_up01_legal_manifest",
        "source_selection_metrics",
        "source_projection_config",
        "source_p3b00_predictions",
        "source_p3b00_feature_list",
        "model_config",
    ]
    observed: dict[str, str] = {}
    for key in keys:
        expected = str(config[f"{key}_sha256"])
        if key in LARGE_REGISTERED_SOURCE_KEYS:
            observed[key] = f"registered:{expected}"
            continue
        actual = file_sha256(resolve_clean_path(config[key]))
        if actual != expected:
            raise ValueError(f"冻结来源 {key} 的 SHA-256 不匹配")
        observed[key] = actual
    return observed


def _save_confirmation(
    output_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    per_well: pd.DataFrame,
    per_fold: pd.DataFrame,
) -> None:
    predictions.sort_values(["well_id", "row_index"]).to_parquet(
        output_dir / "predictions_confirmation_folds34.parquet",
        index=False,
        compression="zstd",
    )
    write_json(output_dir / "metrics_confirmation_folds34.json", metrics)
    per_well.to_csv(output_dir / "per_well_confirmation_folds34.csv", index=False)
    per_fold.to_csv(output_dir / "per_fold_confirmation_folds34.csv", index=False)


def _write_conclusion(
    output_dir: Path,
    confirmation: dict[str, Any],
    full_metrics: dict[str, Any] | None,
) -> None:
    checks = confirmation["success_checks"]
    passed = bool(checks["confirmation_pass"])
    full_line = (
        f"完整开发集描述性 micro RMSE 为 `{full_metrics['overall']['micro_rmse']:.6f}`；"
        "folds1-2 参与过候选选择，因此该五折结果不是全独立确认。"
        if full_metrics is not None
        else "确认未通过或只请求 confirm，未拼接完整五折。"
    )
    text = f"""# P3-UP01 quadratic50 单路径 LightGBM 结论

数据直接证明的事实：独立 folds3-4 的平均改善为 `{checks['arithmetic_mean_fold_improvement_ft']:.6f} ft`，最差折退化为 `{checks['maximum_any_fold_degradation_ft']:.6f} ft`，确认门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：本实验只检验 degree2_blend50 单路径对原 41 列的增量价值。

仍然没有验证的猜测：其他投影候选、投影摘要或自适应融合进入模型是否有效。

当前实验只能否定的具体实现：原 41 列后只追加 `up01_degree2_blend50_delta` 的冻结单模 LightGBM。

证据边界：folds3-4 是未参与候选选择的独立确认；fold0 在确认通过后才打开；folds1-2 只用于完整 OOF 拼接。{full_line}
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="confirm 或 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stage = parse_stage(args.stage)
    config = read_json(args.config.resolve())
    features, model_params = validate_frozen_contract(config)
    observed_hashes = _validate_source_hashes(config)
    selection_metrics = read_json(resolve_clean_path(config["source_selection_metrics"]))
    selection_provenance = validate_selection_provenance(
        selection_metrics,
        selected_candidate=SELECTED_CANDIDATE,
        selection_folds=SELECTION_FOLDS,
    )

    registry, shadow_ids = load_development_registry(
        resolve_clean_path(config["fold_registry"]),
        resolve_clean_path(config["shadow_registry"]),
        config,
    )
    # Arrow 只读取 P2-P02 的自然键，不读取 target_tvt；用于在真值阶段前复核候选行键。
    safe_source_keys = read_development_parquet(
        resolve_clean_path(config["source_p3b00_predictions"]),
        registry["well_id"].astype(str).tolist(),
        ["well_id", "fold", "row_index"],
    )
    legal_manifest = read_json(resolve_clean_path(config["source_up01_legal_manifest"]))
    selected_path, legal_audit = validate_legal_candidate_cache(
        candidate_path=resolve_clean_path(config["source_up01_legal_candidates"]),
        manifest=legal_manifest,
        registry=registry,
        shadow_ids=shadow_ids,
        safe_source_keys=safe_source_keys,
        expected_sha256=str(config["source_up01_legal_candidates_sha256"]),
        expected_wells=int(config["development_wells"]),
        expected_rows=int(config["development_hidden_rows"]),
    )
    observed_hashes["source_up01_legal_candidates"] = legal_audit[
        "legal_candidates_sha256"
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "legal_cache_audit.json", legal_audit)
    write_json(output_dir / "selection_provenance.json", selection_provenance)
    print(
        "UP01 合法候选门通过：657 口、3,211,872 行、shadow=0、"
        "生成阶段 hidden_target_read=false；现在才读取训练目标。",
        flush=True,
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
    feature_table = merge_selected_path(feature_table, selected_path)
    validate_feature_values(feature_table, features)
    if set(feature_table["well_id"].astype(str).unique()).intersection(shadow_ids):
        raise RuntimeError("UP01 正式训练表混入影子井")

    fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "features": features,
            "model_params": model_params,
            "observed_source_hashes": observed_hashes,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"),
        }
    )
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "feature_list.json", {"feature_count": 42, "features": features})
    write_json(output_dir / "parameter_list.json", model_params)
    write_json(
        output_dir / "leakage_audit.json",
        {
            **legal_audit,
            "shadow_target_access": False,
            "touch_pfs_legal_cache": False,
            "selected_candidate": SELECTED_CANDIDATE,
            "selected_candidate_only_model_feature": True,
            "selection_folds": SELECTION_FOLDS,
            "independent_confirmation_folds": CONFIRMATION_FOLDS,
            "fold_evidence_labels": fold_evidence_labels(),
            "cv_fingerprint": fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )

    print("先运行独立确认 folds3-4：42 列、1734 棵树、seed29。", flush=True)
    for fold_id in CONFIRMATION_FOLDS:
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            features,
        )
        save_fold_runtime_row(output_dir, runtime)
    confirmation_predictions = read_fold_predictions(output_dir, CONFIRMATION_FOLDS)
    confirmation_registry = registry.loc[
        registry["fold"].astype(int).isin(CONFIRMATION_FOLDS)
    ].copy()
    confirmation_baseline = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]),
        confirmation_registry,
    )
    confirmation_metrics, confirmation_per_well, confirmation_per_fold = compare_predictions(
        confirmation_predictions,
        confirmation_baseline,
        config,
        "confirmation_folds34",
    )
    confirmation_metrics["cv_fingerprint"] = fingerprint
    confirmation_metrics["evidence_labels"] = fold_evidence_labels()
    _save_confirmation(
        output_dir,
        confirmation_predictions,
        confirmation_metrics,
        confirmation_per_well,
        confirmation_per_fold,
    )
    print(json.dumps(confirmation_metrics["success_checks"], ensure_ascii=False, indent=2), flush=True)

    remaining = continuation_folds(stage, confirmation_metrics["success_checks"])
    if not remaining:
        _write_conclusion(output_dir, confirmation_metrics, None)
        if stage == "all":
            print("独立 folds3-4 未通过门槛；停止，不运行 fold0/1/2。", flush=True)
        return

    print("确认通过：按预注册顺序补 fold0，再训练 folds1-2 仅用于完整 OOF 拼接。", flush=True)
    for fold_id in remaining:
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            features,
        )
        save_fold_runtime_row(output_dir, runtime)
    full_predictions = read_fold_predictions(output_dir, [0, 1, 2, 3, 4])
    full_baseline = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]),
        registry,
    )
    full_metrics, full_per_well, full_per_fold = compare_predictions(
        full_predictions,
        full_baseline,
        config,
        "full5",
    )
    full_metrics["cv_fingerprint"] = fingerprint
    full_metrics["evidence_labels"] = fold_evidence_labels()
    full_metrics["independent_confirmation"] = confirmation_metrics
    full_metrics["full5_is_fully_independent_confirmation"] = False
    save_stage_artifacts(
        output_dir,
        "full5",
        full_predictions,
        full_metrics,
        full_per_well,
        full_per_fold,
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
            "total_fold_seconds": float(sum(float(row["seconds"]) for row in fold_runtimes)),
            "confirmation_folds_completed_first": CONFIRMATION_FOLDS,
            "fold_evidence_labels": fold_evidence_labels(),
        },
    )
    _write_conclusion(output_dir, confirmation_metrics, full_metrics)
    print(json.dumps(full_metrics["success_checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
