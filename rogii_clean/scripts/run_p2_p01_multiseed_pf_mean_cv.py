"""把 P2-P01 的 128-seed 均值路径作为第 37 列运行固定单模 CV。

本脚本不重新生成粒子路径。它只接受已经通过 P2-P01 程序控制、且逐井
``_cache_fingerprint`` 与当前生成代码完全一致的合法缓存。正式模型输入严格为
P2B00/C01 原 36 列加 ``pf128_mean_delta``；seed0、四条 scale 路径和 seed
标准差只属于路径诊断，永远不会并入模型表。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_f05a_deterministic_candidates_cv import (  # noqa: E402
    _validate_cache_metadata,
    load_aligned_feature_table,
    read_candidate_cache_provenance,
    require_candidate_cache_files,
)
from scripts.run_p2_cv00_group5_c01 import (  # noqa: E402
    remap_feature_table_folds,
    save_fold_runtime_row,
    validate_complete_predictions,
)
from scripts.run_p2_p01_multiseed_pf_mean import (  # noqa: E402
    evaluate_path_success_checks,
    experiment_fingerprint as build_p01_fingerprint,
    validate_config as validate_p01_config,
    validate_legal_cache,
)
from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f05a_direct_physical_candidates import (  # noqa: E402
    DIRECT_CANDIDATE_COLUMNS,
)
from src.lgbm_data import load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402
from src.metrics import (  # noqa: E402
    build_per_well_metrics,
    paired_well_bootstrap,
    summarize_by_fold,
    summarize_per_well_metrics,
)


EXPERIMENT_ID = "P2_P01_multiseed_pf_mean_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_p01_multiseed_pf_mean_v1.json"
DEFAULT_P01_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
DEFAULT_OUTPUT_DIR = DEFAULT_P01_ARTIFACT_DIR / "model_cv"
FORMAL_NEW_FEATURE = "pf128_mean_delta"
FROZEN_BASE_FEATURES = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
FROZEN_MODEL_FEATURES = [*FROZEN_BASE_FEATURES, FORMAL_NEW_FEATURE]
DIAGNOSTIC_ONLY_FEATURES = [
    "pf128_seed0_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
    "pf128_seed_std",
]
# 这个哈希独立写在代码中，不能只相信同一份可编辑 P01 JSON 里的声明。
FROZEN_MODEL_CONFIG_SHA256 = (
    "02f6c4cb737132641c2dbbb6076d0e248371786559f2cd6f8581ce72146cb838"
)
# 完整参数逐项硬冻结；任何额外参数、缺失参数或数值变化都必须停止。
FROZEN_MODEL_PARAMS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "learning_rate": 0.00934485794382918,
    "n_estimators": 1734,
    "num_leaves": 64,
    "min_child_samples": 40,
    "min_child_weight": 0.24081152127177283,
    "subsample": 0.47437582748953966,
    "subsample_freq": 0,
    "colsample_bytree": 0.39283351290380497,
    "reg_alpha": 10.788188919840913,
    "reg_lambda": 95.75401894533888,
    "max_bin": 255,
    "random_state": 29,
    "bagging_seed": 29,
    "feature_fraction_seed": 29,
    "data_random_seed": 29,
    "n_jobs": -1,
    "verbose": -1,
}
HASHED_INPUT_NAMES = [
    "baseline_config",
    "model_config",
    "base_feature_cache",
    "candidate_feature_cache",
    "fold_registry",
    "baseline_feature_list",
    "baseline_predictions",
]


def file_sha256(path: Path) -> str:
    """分块计算文件哈希，避免一次读入数百 MB 的 parquet。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """把字典按稳定顺序编码后计算实验指纹。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_fold_spec(fold_text: str) -> list[int]:
    """只接受预注册的 ``0,1`` 预筛或 ``all`` 完整两阶段运行。"""

    normalized = str(fold_text).strip().lower()
    if normalized == "all":
        return [0, 1, 2, 3, 4]
    if normalized == "0,1":
        return [0, 1]
    raise ValueError("folds 只允许恰好传入 0,1 或 all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析固定配置、折号、P01 缓存目录和独立模型产物目录。"""

    parser = argparse.ArgumentParser(description="运行 P2-P01 固定 37 特征单模 CV")
    parser.add_argument(
        "--folds",
        required=True,
        help="先筛选时传 0,1；通过后传 all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--p01-artifact-dir",
        type=Path,
        default=DEFAULT_P01_ARTIFACT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _contains_forbidden_feature(feature_name: str) -> bool:
    """判断字段名是否属于诊断、标签、地层面或 oracle 信息。"""

    lowered = str(feature_name).lower()
    if feature_name in DIAGNOSTIC_ONLY_FEATURES:
        return True
    if lowered == "tvt" or lowered.startswith("target"):
        return True
    if any(fragment in lowered for fragment in ("oracle", "surface", "geology")):
        return True
    # 六个训练 surface 使用固定地层缩写；这些列测试时不可直接获得。
    if str(feature_name).upper() in {
        "ANCC",
        "ASTNU",
        "ASTNL",
        "EGFDU",
        "EGFDL",
        "BUDA",
    }:
        return True
    return False


def validate_model_feature_names(
    model_features: list[str],
    baseline_features: list[str],
) -> None:
    """确认正式输入恰好是冻结 36 列加一条 P01 均值路径。"""

    forbidden = [name for name in model_features if _contains_forbidden_feature(name)]
    if forbidden:
        raise ValueError(f"正式特征含禁止输入：{forbidden}")
    expected = [*baseline_features, FORMAL_NEW_FEATURE]
    if model_features != expected or len(model_features) != 37:
        raise ValueError("正式特征必须严格等于冻结 C01 36 列加 pf128_mean_delta，共 37 列")
    if len(set(model_features)) != len(model_features):
        raise ValueError("正式 37 特征含重复列")


def validate_declared_model_config_hash(config: dict[str, Any]) -> None:
    """要求 P01 声明的模型配置哈希与 runner 内独立常量完全一致。"""

    declared_hash = str(config.get("model_config_sha256", "")).lower()
    if declared_hash != FROZEN_MODEL_CONFIG_SHA256:
        raise ValueError("P01 model_config SHA 不等于 runner 独立冻结值")


def validate_frozen_contract(
    p01_config: dict[str, Any],
    baseline_config: dict[str, Any],
    model_config: dict[str, Any],
    baseline_feature_manifest: dict[str, Any],
) -> list[str]:
    """同时冻结 P01、P2B00 36 列、按井五折和 1734 树模型合同。"""

    validate_p01_config(p01_config)
    validate_declared_model_config_hash(p01_config)
    manifest_features = baseline_feature_manifest.get("features")
    if manifest_features != FROZEN_BASE_FEATURES:
        raise ValueError("P2B00 feature_list 不是冻结 C01 36 列")
    if int(baseline_feature_manifest.get("feature_count", -1)) != 36:
        raise ValueError("P2B00 feature_count 必须是 36")
    if baseline_config.get("experiment_id") != "P2_CV00_group5_c01_v1":
        raise ValueError("baseline_config 不是冻结 P2B00")
    if baseline_config.get("feature_columns") != FROZEN_BASE_FEATURES:
        raise ValueError("baseline_config 的 C01 36 特征发生变化")
    if baseline_config.get("candidate_feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("baseline_config 的候选 24 特征发生变化")
    if baseline_config.get("fold_registry") != p01_config.get("fold_registry"):
        raise ValueError("P01 与 P2B00 没有使用同一按井五折注册表")
    if p01_config.get("fold_version") != "balanced_well_5fold_v1":
        raise ValueError("P01 fold_version 不是 balanced_well_5fold_v1")
    if p01_config.get("baseline_experiment_id") != "P2_CV00_group5_c01_v1":
        raise ValueError("P01 baseline_experiment_id 不是 P2B00")
    if p01_config.get("formal_feature_name") != FORMAL_NEW_FEATURE:
        raise ValueError("P01 formal_feature_name 发生变化")
    if int(p01_config.get("formal_model_feature_count", -1)) != 37:
        raise ValueError("P01 formal_model_feature_count 必须是 37")
    if p01_config.get("diagnostic_columns_excluded_from_model") != (
        DIAGNOSTIC_ONLY_FEATURES
    ):
        raise ValueError("P01 诊断列排除清单发生变化")

    params = model_config.get("params")
    if not isinstance(params, dict):
        raise ValueError("固定模型 params 缺失")
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("固定模型不再是单模 LightGBM")
    if set(params) != set(FROZEN_MODEL_PARAMS):
        raise ValueError("固定模型完整参数的字段集合发生变化")
    for parameter_name, frozen_value in FROZEN_MODEL_PARAMS.items():
        observed_value = params.get(parameter_name)
        if (
            type(observed_value) is not type(frozen_value)
            or observed_value != frozen_value
        ):
            raise ValueError(
                f"固定模型参数 {parameter_name} 不等于冻结值 {frozen_value}"
            )
    training_policy = model_config.get("training_policy")
    if not isinstance(training_policy, dict):
        raise ValueError("固定模型 training_policy 缺失")
    if training_policy.get("early_stopping") is not False:
        raise ValueError("固定模型不得启用 early stopping")
    if training_policy.get("uniform_row_weight") is not True:
        raise ValueError("固定模型必须保持统一行权重")

    model_features = [*manifest_features, FORMAL_NEW_FEATURE]
    validate_model_feature_names(model_features, manifest_features)
    return model_features


def validate_frozen_file_hashes(
    config: dict[str, Any],
    clean_root: Path = CLEAN_ROOT,
) -> dict[str, str]:
    """在读取训练缓存前核对所有冻结来源文件的 SHA-256。"""

    validate_declared_model_config_hash(config)
    observed: dict[str, str] = {}
    for name in HASHED_INPUT_NAMES:
        path_key = name
        hash_key = f"{name}_sha256"
        if path_key not in config or hash_key not in config:
            raise ValueError(f"P01 配置缺少 {path_key} 或 {hash_key}")
        path = (Path(clean_root) / str(config[path_key])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到冻结输入 {name}：{path}")
        actual_hash = file_sha256(path)
        expected_hash = str(config[hash_key]).lower()
        if name == "model_config":
            expected_hash = FROZEN_MODEL_CONFIG_SHA256
        if actual_hash.lower() != expected_hash:
            raise ValueError(f"{name} SHA-256 与冻结配置不一致")
        observed[name] = actual_hash
    return observed


def validate_p01_generation_artifact(
    p01_artifact_dir: Path,
    expected_fingerprint: str,
    p01_config: dict[str, Any],
) -> None:
    """要求当前指纹的 smoke 程序控制已通过，才允许读取全量缓存。"""

    control_path = p01_artifact_dir / "program_controls_smoke.json"
    if not control_path.is_file():
        raise FileNotFoundError("P01 CV 前必须先完成 smoke 程序控制")
    controls = read_json(control_path)
    if controls.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError("P01 smoke 指纹与当前生成代码不一致")
    checks = evaluate_path_success_checks(controls, p01_config)
    if not checks["feature_generation_supported"]:
        raise ValueError("P01 smoke 程序或合法性控制未通过")


def merge_p01_feature_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    expected_fingerprint: str,
) -> pd.DataFrame:
    """逐井验证 P01 缓存，再按自然隐藏行键只并入均值 delta 一列。"""

    required_base_columns = {"well_id", "row_index", "last_visible_tvt"}
    missing_base = required_base_columns.difference(feature_table.columns)
    if missing_base:
        raise ValueError(f"基础特征表缺少 P01 对齐列：{sorted(missing_base)}")
    required_registry_columns = {"well_id", "hidden_rows"}
    missing_registry = required_registry_columns.difference(registry.columns)
    if missing_registry:
        raise ValueError(f"fold 注册表缺少 P01 对齐列：{sorted(missing_registry)}")
    if bool(feature_table.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("基础特征表含重复自然隐藏行键")
    if bool(registry["well_id"].astype(str).duplicated().any()):
        raise ValueError("fold 注册表含重复井")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"P01 legal_cache 不存在：{cache_dir}")

    expected_wells = registry["well_id"].astype(str).tolist()
    actual_cache_wells = {path.stem for path in cache_dir.glob("*.parquet")}
    if actual_cache_wells != set(expected_wells):
        missing = sorted(set(expected_wells) - actual_cache_wells)
        extra = sorted(actual_cache_wells - set(expected_wells))
        raise ValueError(
            "P01 单井缓存集合与 fold 注册表不一致："
            f"缺少={missing[:5]}，多余={extra[:5]}"
        )

    formal_rows: list[pd.DataFrame] = []
    for completed, registry_row in enumerate(registry.itertuples(index=False), start=1):
        well_id = str(registry_row.well_id)
        expected_rows = int(registry_row.hidden_rows)
        cache_path = cache_dir / f"{well_id}.parquet"
        cache = pd.read_parquet(cache_path)
        validate_legal_cache(cache, well_id, expected_rows)
        fingerprints = cache["_cache_fingerprint"].astype(str).unique().tolist()
        if fingerprints != [expected_fingerprint]:
            raise ValueError(f"P01 {well_id} 缓存指纹与当前生成代码不一致")
        formal_rows.append(
            cache[
                [
                    "well_id",
                    "row_index",
                    "last_visible_tvt",
                    FORMAL_NEW_FEATURE,
                ]
            ].rename(columns={"last_visible_tvt": "p01_last_visible_tvt"})
        )
        if completed % 100 == 0 or completed == len(expected_wells):
            print(f"P01 CV 缓存核对 {completed}/{len(expected_wells)}", flush=True)

    p01_rows = pd.concat(formal_rows, ignore_index=True)
    if len(p01_rows) != len(feature_table):
        raise ValueError("P01 缓存总评价行数与基础特征表不一致")

    base = feature_table.copy()
    base["well_id"] = base["well_id"].astype(str)
    base["_p01_original_order"] = np.arange(len(base), dtype=np.int64)
    p01_rows["well_id"] = p01_rows["well_id"].astype(str)
    merged = base.merge(
        p01_rows,
        on=["well_id", "row_index"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("P01 缓存自然隐藏行键无法与基础特征表一一对应")
    merged = merged.sort_values("_p01_original_order", kind="stable").reset_index(drop=True)

    base_anchor = merged["last_visible_tvt"].to_numpy(dtype=np.float32)
    p01_anchor = merged["p01_last_visible_tvt"].to_numpy(dtype=np.float32)
    if not np.array_equal(base_anchor, p01_anchor):
        raise ValueError("P01 last_visible_tvt 路径起点与基础特征表不一致")
    formal_values = merged[FORMAL_NEW_FEATURE].to_numpy(dtype=np.float64)
    if not np.isfinite(formal_values).all():
        raise ValueError("P01 正式均值路径特征含 NaN 或 Inf")

    merged = merged.drop(
        columns=["p01_last_visible_tvt", "_p01_original_order", "_merge"]
    )
    present_diagnostics = [
        name for name in DIAGNOSTIC_ONLY_FEATURES if name in merged.columns
    ]
    if present_diagnostics:
        raise ValueError(f"P01 诊断列意外进入模型表：{present_diagnostics}")
    return merged


def build_p2b00_comparison(
    candidate_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """严格对齐候选和 P2B00，生成逐井配对指标与 pooled 汇总。"""

    required = {
        "well_id",
        "fold",
        "row_index",
        "target_tvt",
        "carry_tvt",
        "pred_tvt",
    }
    for name, frame in (
        ("候选", candidate_predictions),
        ("P2B00", baseline_predictions),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}预测缺列：{sorted(missing)}")
        if bool(frame.duplicated(["well_id", "row_index"]).any()):
            raise ValueError(f"{name}预测含重复行键")

    candidate = candidate_predictions.copy()
    baseline = baseline_predictions.copy()
    candidate["well_id"] = candidate["well_id"].astype(str)
    baseline["well_id"] = baseline["well_id"].astype(str)
    candidate = candidate.rename(
        columns={
            "fold": "candidate_fold",
            "target_tvt": "candidate_target_tvt",
            "carry_tvt": "candidate_carry_tvt",
            "pred_tvt": "candidate_pred_tvt",
        }
    )
    baseline = baseline.rename(
        columns={
            "fold": "baseline_fold",
            "target_tvt": "baseline_target_tvt",
            "carry_tvt": "baseline_carry_tvt",
            "pred_tvt": "p2b00_tvt",
        }
    )
    paired = candidate.merge(
        baseline,
        on=["well_id", "row_index"],
        how="outer",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(paired) != len(candidate_predictions) or not paired["_merge"].eq("both").all():
        raise ValueError("候选与 P2B00 自然隐藏行键不一致")
    if not np.array_equal(
        paired["candidate_fold"].to_numpy(dtype=np.int64),
        paired["baseline_fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("候选与 P2B00 fold 不一致")
    if not np.array_equal(
        paired["candidate_target_tvt"].to_numpy(dtype=np.float64),
        paired["baseline_target_tvt"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("候选与 P2B00 target 真值不一致")
    if not np.array_equal(
        paired["candidate_carry_tvt"].to_numpy(dtype=np.float64),
        paired["baseline_carry_tvt"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("候选与 P2B00 carry 起点不一致")

    scoring_rows = pd.DataFrame(
        {
            "well_id": paired["well_id"].astype(str),
            "fold": paired["candidate_fold"].astype(int),
            "target_tvt": paired["candidate_target_tvt"].astype(np.float64),
            "pred_tvt": paired["candidate_pred_tvt"].astype(np.float64),
            "p2b00_tvt": paired["p2b00_tvt"].astype(np.float64),
        }
    )
    per_well = build_per_well_metrics(
        scoring_rows,
        baseline_column="p2b00_tvt",
    )
    overall = summarize_per_well_metrics(per_well)
    overall["baseline_p90_well_rmse"] = float(
        per_well["baseline_rmse"].quantile(0.90)
    )
    folds = summarize_by_fold(per_well)
    fold_by_id = {int(row["fold"]): row for row in folds}
    for fold_id, fold_frame in per_well.groupby("fold", sort=True):
        fold_by_id[int(fold_id)]["baseline_p90_well_rmse"] = float(
            fold_frame["baseline_rmse"].quantile(0.90)
        )
    comparison = {
        "comparison": "candidate_minus_P2B00",
        "overall": overall,
        "folds": [fold_by_id[key] for key in sorted(fold_by_id)],
        "paired_well_bootstrap": paired_well_bootstrap(
            per_well,
            n_resamples=2000,
            seed=42,
        ),
    }
    return per_well, comparison


def evaluate_model_success_checks(
    comparison: dict[str, Any],
    config: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """按实验卡冻结门槛评价 folds 0～1 或完整五折。"""

    if stage not in {"folds01", "full5"}:
        raise ValueError(f"未知模型门槛阶段：{stage}")
    conditions = config["model_success_conditions"]
    fold_rows = comparison["folds"]
    fold_by_id = {int(row["fold"]): row for row in fold_rows}
    if 0 not in fold_by_id or 1 not in fold_by_id:
        raise ValueError("folds 0～1 结果不完整，不能判断预筛门槛")

    overall = comparison["overall"]
    combined_improvement = float(
        overall["baseline_micro_rmse"] - overall["micro_rmse"]
    )
    single_fold_degradations = {
        str(fold_id): float(
            fold_by_id[fold_id]["micro_rmse"]
            - fold_by_id[fold_id]["baseline_micro_rmse"]
        )
        for fold_id in (0, 1)
    }
    maximum_degradation = max(single_fold_degradations.values())
    checks: dict[str, Any] = {
        "stage": stage,
        "folds01_combined_improvement_ft": combined_improvement,
        "minimum_required_folds01_improvement_ft": float(
            conditions["minimum_folds01_combined_improvement_ft"]
        ),
        "single_fold_degradation_ft": single_fold_degradations,
        "maximum_single_fold_degradation_ft": maximum_degradation,
        "maximum_allowed_single_fold_degradation_ft": float(
            conditions["maximum_single_fold_degradation_ft"]
        ),
    }
    checks["folds01_pass"] = bool(
        combined_improvement
        >= float(conditions["minimum_folds01_combined_improvement_ft"])
        and maximum_degradation
        <= float(conditions["maximum_single_fold_degradation_ft"])
    )
    if stage == "folds01":
        return checks

    if set(fold_by_id) != {0, 1, 2, 3, 4}:
        raise ValueError("完整五折结果必须含 fold 0～4")
    improved_folds = int(
        sum(
            row["micro_rmse"] < row["baseline_micro_rmse"]
            for row in fold_rows
        )
    )
    p90_degradation = float(
        overall["p90_well_rmse"] - overall["baseline_p90_well_rmse"]
    )
    bootstrap_ci_high = float(
        comparison["paired_well_bootstrap"]["ci95_high"]
    )
    checks.update(
        {
            "full5_improvement_ft": combined_improvement,
            "minimum_required_full5_improvement_ft": float(
                conditions["minimum_full5_improvement_ft"]
            ),
            "improved_folds": improved_folds,
            "minimum_required_improved_folds": int(
                conditions["minimum_improved_folds"]
            ),
            "bootstrap_ci95_high_ft": bootstrap_ci_high,
            "maximum_allowed_bootstrap_ci95_high_ft": float(
                conditions["maximum_bootstrap_ci_upper"]
            ),
            "p90_degradation_ft": p90_degradation,
            "maximum_allowed_p90_degradation_ft": float(
                conditions["maximum_p90_degradation_ft"]
            ),
            "well_win_rate_vs_p2b00": float(overall["well_win_rate"]),
        }
    )
    checks["full5_numeric_pass"] = bool(
        combined_improvement >= float(conditions["minimum_full5_improvement_ft"])
        and improved_folds >= int(conditions["minimum_improved_folds"])
        and bootstrap_ci_high <= float(conditions["maximum_bootstrap_ci_upper"])
        and p90_degradation <= float(conditions["maximum_p90_degradation_ft"])
    )
    # “收益不只来自极少数井”没有冻结数值阈值，保留胜井率供结论人工核查，
    # 不在结果出现后临时发明一个新门槛。
    checks["benefit_concentration_requires_review"] = True
    return checks


def build_cv_fingerprint(
    config: dict[str, Any],
    model_params: dict[str, Any],
    observed_hashes: dict[str, str],
    p01_fingerprint: str,
) -> str:
    """让缓存、路径代码、模型参数、评分器和当前 runner 都进入结果指纹。"""

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "model_params": model_params,
        "observed_input_sha256": observed_hashes,
        "p01_cache_fingerprint": p01_fingerprint,
        "formal_model_features": FROZEN_MODEL_FEATURES,
        "cv_runner_sha256": file_sha256(Path(__file__).resolve()),
        "p01_generator_runner_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_p2_p01_multiseed_pf_mean.py"
        ),
        "p01_core_sha256": file_sha256(
            CLEAN_ROOT / "src" / "p2_p01_multiseed_pf.py"
        ),
        "baseline_runner_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_p2_cv00_group5_c01.py"
        ),
        "trainer_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"
        ),
        "metrics_sha256": file_sha256(CLEAN_ROOT / "src" / "metrics.py"),
    }
    return stable_json_hash(payload)


def _load_candidate_cache_provenance(
    config: dict[str, Any],
) -> dict[str, Any]:
    """复用 C01 的 metadata 审计，禁止把不完整候选缓存并入模型。"""

    candidate_cache_path = (
        CLEAN_ROOT / str(config["candidate_feature_cache"])
    ).resolve()
    metadata_path = (
        CLEAN_ROOT / str(config["candidate_cache_metadata"])
    ).resolve()
    require_candidate_cache_files(candidate_cache_path, metadata_path)
    provenance = read_candidate_cache_provenance(
        candidate_cache_path,
        metadata_path,
    )
    metadata = provenance.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("C01 候选缓存 metadata 必须是 JSON 对象")
    _validate_cache_metadata(
        metadata,
        int(config["expected_wells"]),
        int(config["expected_hidden_rows"]),
    )
    if str(metadata.get("candidate_cache_sha256", "")).lower() != str(
        config["candidate_feature_cache_sha256"]
    ).lower():
        raise ValueError("C01 candidate metadata 与冻结缓存 SHA 不一致")
    return provenance


def _read_completed_fold_predictions(
    output_dir: Path,
    fold_ids: list[int],
) -> pd.DataFrame:
    """读取指定已完成折的候选预测，供与 P2B00 做严格配对。"""

    frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        path = output_dir / f"fold_{fold_id}" / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 fold {fold_id} 候选预测：{path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _save_comparison(
    output_dir: Path,
    stage: str,
    candidate_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    config: dict[str, Any],
    cv_fingerprint: str,
) -> dict[str, Any]:
    """保存候选对 P2B00 的逐井配对表、汇总和预注册门槛。"""

    per_well, comparison = build_p2b00_comparison(
        candidate_predictions,
        baseline_predictions,
    )
    checks = evaluate_model_success_checks(comparison, config, stage)
    comparison["cv_fingerprint"] = cv_fingerprint
    comparison["success_checks"] = checks
    per_well.to_csv(output_dir / f"per_well_vs_p2b00_{stage}.csv", index=False)
    write_json(output_dir / f"comparison_vs_p2b00_{stage}.json", comparison)
    return checks


def load_passed_folds01_comparison(
    output_dir: Path,
    expected_cv_fingerprint: str,
) -> dict[str, Any]:
    """从磁盘重读当前实验已通过的 folds 0～1 门槛凭证。"""

    comparison_path = output_dir / "comparison_vs_p2b00_folds01.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("训练 folds 2～4 前缺少 folds01 comparison")
    comparison = read_json(comparison_path)
    if comparison.get("cv_fingerprint") != expected_cv_fingerprint:
        raise ValueError("folds01 comparison 指纹不是当前 CV 实验指纹")
    checks = comparison.get("success_checks")
    if not isinstance(checks, dict) or checks.get("stage") != "folds01":
        raise ValueError("folds01 comparison 缺少预筛门槛结果")
    if checks.get("folds01_pass") is not True:
        raise ValueError("folds01 comparison 未通过门槛，禁止训练 folds 2～4")
    return comparison


def _write_lineage(
    output_dir: Path,
    config: dict[str, Any],
    observed_hashes: dict[str, str],
    p01_fingerprint: str,
    cv_fingerprint: str,
    candidate_provenance: dict[str, Any],
) -> None:
    """保存模型输入边界和所有来源指纹，供之后复核。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "feature_list.json",
        {"feature_count": 37, "features": FROZEN_MODEL_FEATURES},
    )
    write_json(
        output_dir / "cache_and_fold_provenance.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "cv_fingerprint": cv_fingerprint,
            "p01_cache_fingerprint": p01_fingerprint,
            "observed_input_sha256": observed_hashes,
            "fold_version": "balanced_well_5fold_v1",
            "validation_unit": "complete_well",
            "cached_old_fold_ignored": True,
            "formal_new_feature": FORMAL_NEW_FEATURE,
            "diagnostic_features_excluded": DIAGNOSTIC_ONLY_FEATURES,
            "candidate_cache_provenance": candidate_provenance,
        },
    )


def main(argv: list[str] | None = None) -> None:
    """核对全部来源、合并唯一新特征，并按门槛运行固定 LightGBM。"""

    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config_path = Path(args.config).resolve()
    p01_config = read_json(config_path)

    # 哈希必须在读取训练缓存之前核对；任何旧文件或被改配置都会立即停止。
    observed_hashes = validate_frozen_file_hashes(p01_config, CLEAN_ROOT)
    baseline_config = read_json(
        (CLEAN_ROOT / str(p01_config["baseline_config"])).resolve()
    )
    model_config = read_json(
        (CLEAN_ROOT / str(p01_config["model_config"])).resolve()
    )
    baseline_feature_manifest = read_json(
        (CLEAN_ROOT / str(p01_config["baseline_feature_list"])).resolve()
    )
    model_features = validate_frozen_contract(
        p01_config,
        baseline_config,
        model_config,
        baseline_feature_manifest,
    )
    candidate_provenance = _load_candidate_cache_provenance(p01_config)

    expected_wells = int(p01_config["expected_wells"])
    expected_rows = int(p01_config["expected_hidden_rows"])
    registry_path = (CLEAN_ROOT / str(p01_config["fold_registry"])).resolve()
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )

    base_cache_path = (
        CLEAN_ROOT / str(p01_config["base_feature_cache"])
    ).resolve()
    candidate_cache_path = (
        CLEAN_ROOT / str(p01_config["candidate_feature_cache"])
    ).resolve()
    feature_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    feature_table = remap_feature_table_folds(feature_table, registry)

    p01_fingerprint = build_p01_fingerprint(p01_config)
    p01_artifact_dir = Path(args.p01_artifact_dir).resolve()
    validate_p01_generation_artifact(
        p01_artifact_dir,
        p01_fingerprint,
        p01_config,
    )
    feature_table = merge_p01_feature_cache(
        feature_table,
        registry,
        p01_artifact_dir / "legal_cache",
        p01_fingerprint,
    )
    validate_model_feature_names(model_features, FROZEN_BASE_FEATURES)
    missing_model_features = set(model_features).difference(feature_table.columns)
    if missing_model_features:
        raise ValueError(f"合并后模型缺少正式特征：{sorted(missing_model_features)}")

    model_params = dict(model_config["params"])
    cv_fingerprint = build_cv_fingerprint(
        p01_config,
        model_params,
        observed_hashes,
        p01_fingerprint,
    )
    output_dir = Path(args.output_dir).resolve()
    _write_lineage(
        output_dir,
        p01_config,
        observed_hashes,
        p01_fingerprint,
        cv_fingerprint,
        candidate_provenance,
    )
    print(
        f"P2-P01 CV：折={requested_folds}，井={expected_wells}，"
        f"评价行={expected_rows:,}，正式特征=37，树=1734，输出={output_dir}",
        flush=True,
    )

    # all 也先只完成 0、1 折。若预注册门槛失败，绝不浪费时间训练 2～4 折。
    first_stage_folds = (
        [fold_id for fold_id in requested_folds if fold_id in {0, 1}]
        if requested_folds != [0, 1, 2, 3, 4]
        else [0, 1]
    )
    for fold_id in first_stage_folds:
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

    folds01_checks: dict[str, Any] | None = None
    if {0, 1}.issubset(set(first_stage_folds)):
        folds01_predictions = _read_completed_fold_predictions(output_dir, [0, 1])
        # 训练结束后才读取 P2B00 预测，避免训练矩阵与整份基线预测同时占内存。
        baseline_predictions = pd.read_parquet(
            (CLEAN_ROOT / str(p01_config["baseline_predictions"])).resolve()
        )
        baseline_folds01 = baseline_predictions.loc[
            baseline_predictions["fold"].astype(int).isin([0, 1])
        ].copy()
        folds01_checks = _save_comparison(
            output_dir,
            "folds01",
            folds01_predictions,
            baseline_folds01,
            p01_config,
            cv_fingerprint,
        )
        print(
            "P2-P01 folds 0～1 门槛："
            + json.dumps(folds01_checks, ensure_ascii=False),
            flush=True,
        )
        del folds01_predictions, baseline_folds01, baseline_predictions

    # 预筛命令到这里必须结束。即使目录中碰巧存在旧的 folds 2～4，
    # 也不能由 finalize_complete_cv 把它们拼成一次“完整结果”。
    if requested_folds == [0, 1]:
        return

    if requested_folds == [0, 1, 2, 3, 4]:
        if folds01_checks is None or not folds01_checks["folds01_pass"]:
            print("folds 0～1 未过门槛，自动停止，不训练 folds 2～4。", flush=True)
            return
        # 训练后三折前必须真正从磁盘重读当前指纹的通过凭证，不能只依赖
        # 内存布尔值或目录里已有的 fold 产物。
        load_passed_folds01_comparison(output_dir, cv_fingerprint)
        remaining_folds = [2, 3, 4]
    else:
        # parse_fold_spec 已经保证这里不可达；保留防御式停止，避免未来放宽
        # CLI 时意外产生绕过 folds01 门槛的训练路径。
        raise ValueError("未预注册的折号组合禁止训练")

    for fold_id in remaining_folds:
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

    complete_metrics = finalize_complete_cv(
        output_dir,
        cv_fingerprint,
        expected_rows,
        p01_config,
        model_params,
        model_features,
    )
    if complete_metrics is None:
        return

    validate_complete_predictions(output_dir, registry, expected_rows)
    candidate_all = pd.read_parquet(output_dir / "predictions.parquet")
    baseline_predictions = pd.read_parquet(
        (CLEAN_ROOT / str(p01_config["baseline_predictions"])).resolve()
    )
    full5_checks = _save_comparison(
        output_dir,
        "full5",
        candidate_all,
        baseline_predictions,
        p01_config,
        cv_fingerprint,
    )
    print(
        "P2-P01 完整五折门槛：" + json.dumps(full5_checks, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
