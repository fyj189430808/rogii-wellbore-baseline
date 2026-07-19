"""运行 P2-P02：在 P01 的 37 特征后整体加入四条固定尺度 PF 路径。

正式模型始终是同一个 1,734 树 LightGBM。本脚本只改变输入列：从 P01
已经通过合法性检查的逐井缓存中读取均值路径与 scale 3/5/8/12 四条路径。
seed0、seed 标准差、oracle 和负对照只用于诊断，不会进入正式 41 列。
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
    experiment_fingerprint as build_p01_generator_fingerprint,
    validate_config as validate_p01_generator_config,
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


EXPERIMENT_ID = "P2_P02_multiscale_pf_paths_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_p02_multiscale_pf_paths_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

FORMAL_MEAN_FEATURE = "pf128_mean_delta"
FROZEN_SCALE_FEATURES = [
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
]
DIAGNOSTIC_ONLY_FEATURES = ["pf128_seed0_delta", "pf128_seed_std"]
FROZEN_C01_FEATURES = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
FROZEN_P01_FEATURES = [*FROZEN_C01_FEATURES, FORMAL_MEAN_FEATURE]
FROZEN_MODEL_FEATURES = [*FROZEN_P01_FEATURES, *FROZEN_SCALE_FEATURES]

FROZEN_BASELINE_MICRO_RMSE = 10.93453719961266
FROZEN_BASELINE_FOLD_RMSE = [
    10.823838214365503,
    10.21216180740936,
    9.93405725110749,
    11.824856770444416,
    11.741138844575953,
]

# 哈希独立写在代码里，不能通过同时修改 P02 JSON 和外部文件绕开冻结合同。
FROZEN_EXTERNAL_SHA256 = {
    "fold_registry": "a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c",
    "baseline_predictions": "5d7e5341aa63641c51a8b9f003857036c5086628a8ad46f56d3a070a7ec48d96",
    "baseline_comparison": "6a55ec0bf0a4a7354719391597f3d53bc571e37824ca90c9f3347016582755ea",
    "baseline_feature_list": "0191966ffac4fdc47b667a159390b7fd403782b3d72b62a9a5278aa0ede8d297",
    "p01_generator_config": "041c7e5ac3c40ee683c13d59821dcfe4a675c9632e3c55b609f93dcf9fb4e909",
    "p01_program_controls": "f238ced8fd7a38a577a9712d55433ef8ff1f23be11bfd663016a32ca4dd02dca",
    "model_config": "02f6c4cb737132641c2dbbb6076d0e248371786559f2cd6f8581ce72146cb838",
    "base_feature_cache": "8801493752e03f40a3957843e8efa25c091d04244dd341c64c9329b50d5de625",
    "candidate_feature_cache": "66b32f8ed790ea53184d43bc02d6899031348deec418d7933366bbef64105076",
}
FROZEN_MODEL_CONFIG_SHA256 = FROZEN_EXTERNAL_SHA256["model_config"]
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


def file_sha256(path: Path) -> str:
    """分块计算文件 SHA-256，避免一次读入大型 parquet。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """稳定编码字典并计算实验指纹。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_fold_spec(fold_text: str) -> list[int]:
    """只接受预注册的 ``0,1`` 或 ``all`` 两种运行阶段。"""

    normalized = str(fold_text).strip().lower()
    if normalized == "0,1":
        return [0, 1]
    if normalized == "all":
        return [0, 1, 2, 3, 4]
    raise ValueError("folds 只允许恰好传入 0,1 或 all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析折阶段、冻结配置和独立输出目录。"""

    parser = argparse.ArgumentParser(description="运行 P2-P02 固定四尺度 41 特征 CV")
    parser.add_argument("--folds", required=True, help="预筛传 0,1；晋级后传 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def validate_declared_external_hashes(config: dict[str, Any]) -> None:
    """要求 P02 JSON 中每个来源哈希都等于代码独立冻结值。"""

    for source_name, frozen_hash in FROZEN_EXTERNAL_SHA256.items():
        hash_key = f"{source_name}_sha256"
        declared_hash = str(config.get(hash_key, "")).lower()
        if declared_hash != frozen_hash:
            raise ValueError(f"{source_name} SHA 不等于 P02 runner 冻结值")


def validate_frozen_file_hashes(
    config: dict[str, Any],
    clean_root: Path = CLEAN_ROOT,
) -> dict[str, str]:
    """在加载任何训练缓存前逐个核对全部外部文件。"""

    validate_declared_external_hashes(config)
    observed: dict[str, str] = {}
    for source_name, frozen_hash in FROZEN_EXTERNAL_SHA256.items():
        path = (Path(clean_root) / str(config[source_name])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 P02 冻结输入 {source_name}：{path}")
        actual_hash = file_sha256(path)
        if actual_hash.lower() != frozen_hash:
            raise ValueError(f"{source_name} 实际 SHA 与 P02 冻结值不一致")
        observed[source_name] = actual_hash
    return observed


def _contains_forbidden_feature(feature_name: str) -> bool:
    """判断列名是否属于诊断、标签、surface 或 oracle。"""

    lowered = str(feature_name).lower()
    if feature_name in DIAGNOSTIC_ONLY_FEATURES:
        return True
    if lowered == "tvt" or lowered.startswith("target"):
        return True
    if any(fragment in lowered for fragment in ("oracle", "surface", "geology")):
        return True
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
    p01_features: list[str],
) -> None:
    """确认正式模型恰好是 P01 37 列加预注册四尺度。"""

    forbidden = [name for name in model_features if _contains_forbidden_feature(name)]
    if forbidden:
        raise ValueError(f"P02 正式特征含禁止输入：{forbidden}")
    expected = [*p01_features, *FROZEN_SCALE_FEATURES]
    if model_features != expected or len(model_features) != 41:
        raise ValueError("P02 正式特征必须严格为 P01 37 列加固定四尺度，共 41 列")
    if len(set(model_features)) != 41:
        raise ValueError("P02 正式 41 特征含重复列")


def validate_frozen_contract(
    config: dict[str, Any],
    model_config: dict[str, Any],
    p01_feature_manifest: dict[str, Any],
) -> list[str]:
    """冻结 P01 基线、41 列特征、按井五折和完整 LightGBM 参数。"""

    validate_declared_external_hashes(config)
    expected_top_level = {
        "experiment_id": EXPERIMENT_ID,
        "fold_version": "balanced_well_5fold_v1",
        "baseline_experiment_id": "P2_P01_multiseed_pf_mean_v1",
        "baseline_micro_rmse": FROZEN_BASELINE_MICRO_RMSE,
        "baseline_fold_rmse": FROZEN_BASELINE_FOLD_RMSE,
        "baseline_feature_count": 37,
        "formal_feature_count": 41,
        "new_feature_names": FROZEN_SCALE_FEATURES,
        "diagnostic_columns_excluded_from_model": DIAGNOSTIC_ONLY_FEATURES,
        "expected_wells": 773,
        "expected_hidden_rows": 3_783_989,
    }
    for name, frozen_value in expected_top_level.items():
        if config.get(name) != frozen_value:
            raise ValueError(f"P02 {name} 不等于冻结值 {frozen_value}")

    negative_control = config.get("negative_control")
    if negative_control != {
        "name": "reverse_each_new_scale_path_within_well",
        "run_if_folds01_pass": True,
        "fold": 0,
    }:
        raise ValueError("P02 井内逆序负对照配置发生变化")

    manifest_features = p01_feature_manifest.get("features")
    if manifest_features != FROZEN_P01_FEATURES:
        raise ValueError("P01 feature_list 不是冻结的 37 列")
    if int(p01_feature_manifest.get("feature_count", -1)) != 37:
        raise ValueError("P01 feature_count 必须是 37")

    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("P02 模型不再是单模 LightGBM")
    params = model_config.get("params")
    if not isinstance(params, dict):
        raise ValueError("P02 模型完整参数缺失")
    if set(params) != set(FROZEN_MODEL_PARAMS):
        raise ValueError("P02 模型完整参数字段集合发生变化")
    for parameter_name, frozen_value in FROZEN_MODEL_PARAMS.items():
        observed_value = params.get(parameter_name)
        if (
            type(observed_value) is not type(frozen_value)
            or observed_value != frozen_value
        ):
            raise ValueError(
                f"P02 模型参数 {parameter_name} 不等于冻结值 {frozen_value}"
            )
    training_policy = model_config.get("training_policy")
    if not isinstance(training_policy, dict):
        raise ValueError("P02 模型 training_policy 缺失")
    if training_policy.get("early_stopping") is not False:
        raise ValueError("P02 模型不得 early stopping")
    if training_policy.get("uniform_row_weight") is not True:
        raise ValueError("P02 模型必须保持统一行权重")

    model_features = [*manifest_features, *FROZEN_SCALE_FEATURES]
    validate_model_feature_names(model_features, manifest_features)
    return model_features


def validate_p01_program_controls(
    config: dict[str, Any],
    controls: dict[str, Any],
    computed_generator_fingerprint: str,
) -> None:
    """确认 P01 生成代码、缓存指纹和三个程序控制来自同一版本。"""

    frozen_fingerprint = str(config.get("p01_generator_fingerprint", ""))
    if computed_generator_fingerprint != frozen_fingerprint:
        raise ValueError("当前 P01 生成代码指纹与 P02 冻结指纹不一致")
    if controls.get("experiment_fingerprint") != frozen_fingerprint:
        raise ValueError("P01 program controls 指纹与 P02 冻结指纹不一致")
    for metric_name in (
        "maximum_repeat_difference_ft",
        "maximum_concurrent_difference_ft",
        "maximum_hidden_tvt_mutation_difference_ft",
    ):
        if float(controls.get(metric_name, np.inf)) != 0.0:
            raise ValueError(f"P01 程序控制 {metric_name} 未通过")
    stored_checks = controls.get("success_checks")
    if not isinstance(stored_checks, dict) or (
        stored_checks.get("feature_generation_supported") is not True
    ):
        raise ValueError("P01 程序控制没有通过合法特征生成门槛")


def merge_p01_path_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    expected_fingerprint: str,
) -> pd.DataFrame:
    """逐井验证 P01 合法缓存，并且只并入均值与四尺度五列。"""

    required_base = {"well_id", "row_index", "last_visible_tvt"}
    missing_base = required_base.difference(feature_table.columns)
    if missing_base:
        raise ValueError(f"基础特征表缺少 P01 对齐列：{sorted(missing_base)}")
    if bool(feature_table.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("基础特征表含重复自然隐藏行键")
    if any(name in feature_table.columns for name in [FORMAL_MEAN_FEATURE, *FROZEN_SCALE_FEATURES]):
        raise ValueError("基础特征表已含 P01 路径列，禁止重复合并")
    if bool(registry["well_id"].astype(str).duplicated().any()):
        raise ValueError("fold 注册表含重复井")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"P01 legal_cache 不存在：{cache_dir}")

    expected_wells = registry["well_id"].astype(str).tolist()
    actual_wells = {path.stem for path in cache_dir.glob("*.parquet")}
    if actual_wells != set(expected_wells):
        missing = sorted(set(expected_wells) - actual_wells)
        extra = sorted(actual_wells - set(expected_wells))
        raise ValueError(
            "P01 单井缓存集合与 fold 注册表不一致："
            f"缺少={missing[:5]}，多余={extra[:5]}"
        )

    selected_rows: list[pd.DataFrame] = []
    selected_columns = [
        "well_id",
        "row_index",
        "last_visible_tvt",
        FORMAL_MEAN_FEATURE,
        *FROZEN_SCALE_FEATURES,
    ]
    for completed, registry_row in enumerate(registry.itertuples(index=False), start=1):
        well_id = str(registry_row.well_id)
        expected_rows = int(registry_row.hidden_rows)
        cache = pd.read_parquet(cache_dir / f"{well_id}.parquet")
        # P01 validator 会拒绝 target、surface、oracle、额外列和非有限数。
        validate_legal_cache(cache, well_id, expected_rows)
        fingerprints = cache["_cache_fingerprint"].astype(str).unique().tolist()
        if fingerprints != [expected_fingerprint]:
            raise ValueError(f"P01 {well_id} 缓存指纹与当前生成器不一致")
        selected_rows.append(
            cache[selected_columns].rename(
                columns={"last_visible_tvt": "p01_last_visible_tvt"}
            )
        )
        if completed % 100 == 0 or completed == len(expected_wells):
            print(f"P02 核对 P01 缓存 {completed}/{len(expected_wells)}", flush=True)

    p01_paths = pd.concat(selected_rows, ignore_index=True)
    if len(p01_paths) != len(feature_table):
        raise ValueError("P01 路径缓存总行数与基础特征表不一致")

    base = feature_table.copy()
    base["well_id"] = base["well_id"].astype(str)
    base["_p02_original_order"] = np.arange(len(base), dtype=np.int64)
    p01_paths["well_id"] = p01_paths["well_id"].astype(str)
    merged = base.merge(
        p01_paths,
        on=["well_id", "row_index"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("P01 路径与基础表自然隐藏行键不能一一对应")
    merged = merged.sort_values("_p02_original_order", kind="stable").reset_index(drop=True)

    base_anchor = merged["last_visible_tvt"].to_numpy(dtype=np.float32)
    p01_anchor = merged["p01_last_visible_tvt"].to_numpy(dtype=np.float32)
    if not np.array_equal(base_anchor, p01_anchor):
        raise ValueError("P01 last_visible_tvt 路径起点与基础特征表不一致")
    formal_values = merged[[FORMAL_MEAN_FEATURE, *FROZEN_SCALE_FEATURES]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(formal_values).all():
        raise ValueError("P01 均值或四尺度路径含 NaN/Inf")

    merged = merged.drop(
        columns=["p01_last_visible_tvt", "_p02_original_order", "_merge"]
    )
    diagnostics = [name for name in DIAGNOSTIC_ONLY_FEATURES if name in merged.columns]
    if diagnostics:
        raise ValueError(f"P01 诊断列意外进入 P02 模型表：{diagnostics}")
    return merged


def build_p01_comparison(
    candidate_predictions: pd.DataFrame,
    p01_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """严格配对候选和 P01 OOF，并计算相对 P01 的统一指标。"""

    required = {
        "well_id",
        "fold",
        "row_index",
        "target_tvt",
        "carry_tvt",
        "pred_tvt",
    }
    for name, frame in (("候选", candidate_predictions), ("P01", p01_predictions)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}预测缺列：{sorted(missing)}")
        if bool(frame.duplicated(["well_id", "row_index"]).any()):
            raise ValueError(f"{name}预测含重复行键")

    candidate = candidate_predictions.copy()
    baseline = p01_predictions.copy()
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
            "fold": "p01_fold",
            "target_tvt": "p01_target_tvt",
            "carry_tvt": "p01_carry_tvt",
            "pred_tvt": "p01_pred_tvt",
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
    if len(paired) != len(candidate) or not paired["_merge"].eq("both").all():
        raise ValueError("候选与 P01 自然隐藏行键不一致")
    for candidate_column, p01_column, description, dtype in (
        ("candidate_fold", "p01_fold", "fold", np.int64),
        ("candidate_target_tvt", "p01_target_tvt", "target 真值", np.float64),
        ("candidate_carry_tvt", "p01_carry_tvt", "carry 起点", np.float64),
    ):
        if not np.array_equal(
            paired[candidate_column].to_numpy(dtype=dtype),
            paired[p01_column].to_numpy(dtype=dtype),
        ):
            raise ValueError(f"候选与 P01 {description}不一致")

    scoring_rows = pd.DataFrame(
        {
            "well_id": paired["well_id"].astype(str),
            "fold": paired["candidate_fold"].astype(int),
            "target_tvt": paired["candidate_target_tvt"].astype(np.float64),
            "pred_tvt": paired["candidate_pred_tvt"].astype(np.float64),
            "p01_tvt": paired["p01_pred_tvt"].astype(np.float64),
        }
    )
    per_well = build_per_well_metrics(scoring_rows, baseline_column="p01_tvt")
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
        "comparison": "candidate_minus_P01",
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
    """使用实验卡门槛评价 folds 0～1 或完整五折。"""

    if stage not in {"folds01", "full5"}:
        raise ValueError(f"未知 P02 模型阶段：{stage}")
    conditions = config["model_success_conditions"]
    fold_by_id = {int(row["fold"]): row for row in comparison["folds"]}
    if 0 not in fold_by_id or 1 not in fold_by_id:
        raise ValueError("P02 folds 0～1 结果不完整")

    overall = comparison["overall"]
    improvement = float(overall["baseline_micro_rmse"] - overall["micro_rmse"])
    fold_degradations = {
        str(fold_id): float(
            fold_by_id[fold_id]["micro_rmse"]
            - fold_by_id[fold_id]["baseline_micro_rmse"]
        )
        for fold_id in (0, 1)
    }
    maximum_degradation = max(fold_degradations.values())
    checks: dict[str, Any] = {
        "stage": stage,
        "comparison_baseline": "P2_P01_multiseed_pf_mean_v1",
        "folds01_combined_improvement_ft": improvement,
        "single_fold_degradation_ft": fold_degradations,
        "maximum_single_fold_degradation_ft": maximum_degradation,
    }
    checks["folds01_pass"] = bool(
        improvement >= float(conditions["minimum_folds01_combined_improvement_ft"])
        and maximum_degradation
        <= float(conditions["maximum_single_fold_degradation_ft"])
    )
    if stage == "folds01":
        return checks

    if set(fold_by_id) != {0, 1, 2, 3, 4}:
        raise ValueError("P02 完整五折必须含 fold 0～4")
    improved_folds = int(
        sum(row["micro_rmse"] < row["baseline_micro_rmse"] for row in comparison["folds"])
    )
    p90_degradation = float(
        overall["p90_well_rmse"] - overall["baseline_p90_well_rmse"]
    )
    bootstrap_high = float(comparison["paired_well_bootstrap"]["ci95_high"])
    checks.update(
        {
            "full5_improvement_ft": improvement,
            "improved_folds": improved_folds,
            "bootstrap_ci95_high_ft": bootstrap_high,
            "p90_degradation_ft": p90_degradation,
            "well_win_rate_vs_p01": float(overall["well_win_rate"]),
        }
    )
    checks["full5_numeric_pass"] = bool(
        improvement >= float(conditions["minimum_full5_improvement_ft"])
        and improved_folds >= int(conditions["minimum_improved_folds"])
        and bootstrap_high <= float(conditions["maximum_bootstrap_ci_upper"])
        and p90_degradation <= float(conditions["maximum_p90_degradation_ft"])
    )
    return checks


def load_passed_folds01_comparison(
    output_dir: Path,
    expected_cv_fingerprint: str,
) -> dict[str, Any]:
    """从磁盘读取当前 P02 指纹且已通过的 folds01 凭证。"""

    path = output_dir / "comparison_vs_p01_folds01.json"
    if not path.is_file():
        raise FileNotFoundError("训练 P02 folds 2～4 前缺少 folds01 comparison")
    comparison = read_json(path)
    if comparison.get("cv_fingerprint") != expected_cv_fingerprint:
        raise ValueError("P02 folds01 comparison 指纹不是当前实验指纹")
    checks = comparison.get("success_checks")
    if not isinstance(checks, dict) or checks.get("stage") != "folds01":
        raise ValueError("P02 folds01 comparison 缺少门槛结果")
    if checks.get("folds01_pass") is not True:
        raise ValueError("P02 folds01 comparison 未通过门槛")
    return comparison


def reverse_scale_paths_within_well(feature_table: pd.DataFrame) -> pd.DataFrame:
    """按每口井 row_index 顺序逆转四尺度列，保持其余列和原表不变。"""

    required = {"well_id", "row_index", *FROZEN_SCALE_FEATURES}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"逆序负对照缺列：{sorted(missing)}")
    if bool(feature_table.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("逆序负对照输入含重复行键")

    # 浅拷贝共享未修改列；给四条路径重新赋独立数组，只额外占四列内存。
    reversed_table = feature_table.copy(deep=False)
    row_index = feature_table["row_index"].to_numpy(dtype=np.int64)
    grouped_positions = feature_table.groupby("well_id", sort=False).indices
    for feature_name in FROZEN_SCALE_FEATURES:
        original_values = feature_table[feature_name].to_numpy(copy=True)
        reversed_values = original_values.copy()
        for positions in grouped_positions.values():
            positions_array = np.asarray(positions, dtype=np.int64)
            order = np.argsort(row_index[positions_array], kind="stable")
            ordered_positions = positions_array[order]
            reversed_values[ordered_positions] = original_values[ordered_positions][::-1]
        reversed_table[feature_name] = reversed_values
    return reversed_table


def build_oracle_scale_audit(feature_table: pd.DataFrame) -> dict[str, Any]:
    """事后逐井选择四条裸路径中 RMSE 最低者，仅输出诊断上限。"""

    required = {
        "well_id",
        "target_tvt",
        "last_visible_tvt",
        *FROZEN_SCALE_FEATURES,
    }
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"scale oracle 缺列：{sorted(missing)}")

    scale_names = {
        "pf128_scale_3_delta": "scale_3",
        "pf128_scale_5_delta": "scale_5",
        "pf128_scale_8_delta": "scale_8",
        "pf128_scale_12_delta": "scale_12",
    }
    total_sse = {short_name: 0.0 for short_name in scale_names.values()}
    best_counts: dict[str, int] = {}
    oracle_sse = 0.0
    total_rows = 0
    per_well: list[dict[str, Any]] = []

    for well_id, well_rows in feature_table.groupby("well_id", sort=True):
        truth = well_rows["target_tvt"].to_numpy(dtype=np.float64)
        anchor = well_rows["last_visible_tvt"].to_numpy(dtype=np.float64)
        row_count = int(len(well_rows))
        well_sse: dict[str, float] = {}
        for feature_name, short_name in scale_names.items():
            prediction = anchor + well_rows[feature_name].to_numpy(dtype=np.float64)
            error = prediction - truth
            sse = float(np.sum(error * error))
            well_sse[short_name] = sse
            total_sse[short_name] += sse
        best_scale = min(well_sse, key=well_sse.get)
        best_sse = well_sse[best_scale]
        best_counts[best_scale] = best_counts.get(best_scale, 0) + 1
        oracle_sse += best_sse
        total_rows += row_count
        per_well.append(
            {
                "well_id": str(well_id),
                "rows": row_count,
                "best_scale": best_scale,
                "best_scale_rmse": float(np.sqrt(best_sse / max(row_count, 1))),
                "scale_rmse": {
                    name: float(np.sqrt(sse / max(row_count, 1)))
                    for name, sse in well_sse.items()
                },
            }
        )

    return {
        "diagnostic_only": True,
        "used_for_feature_or_model_selection": False,
        "wells": int(len(per_well)),
        "hidden_rows": int(total_rows),
        "scale_micro_rmse": {
            name: float(np.sqrt(sse / max(total_rows, 1)))
            for name, sse in total_sse.items()
        },
        "oracle_best_per_well_micro_rmse": float(
            np.sqrt(oracle_sse / max(total_rows, 1))
        ),
        "best_scale_counts": best_counts,
        "per_well": per_well,
    }


def build_cv_fingerprint(
    config: dict[str, Any],
    model_params: dict[str, Any],
    observed_hashes: dict[str, str],
    p01_generator_fingerprint: str,
) -> str:
    """让代码、41 列、P01 基线和全部缓存来源进入 P02 指纹。"""

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "model_params": model_params,
        "formal_features": FROZEN_MODEL_FEATURES,
        "observed_sha256": observed_hashes,
        "p01_generator_fingerprint": p01_generator_fingerprint,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "p01_generator_runner_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_p2_p01_multiseed_pf_mean.py"
        ),
        "p01_core_sha256": file_sha256(CLEAN_ROOT / "src" / "p2_p01_multiseed_pf.py"),
        "trainer_sha256": file_sha256(CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"),
        "metrics_sha256": file_sha256(CLEAN_ROOT / "src" / "metrics.py"),
    }
    return stable_json_hash(payload)


def _candidate_cache_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """核对 C01 候选缓存 metadata 完整性。"""

    cache_path = (CLEAN_ROOT / str(config["candidate_feature_cache"])).resolve()
    metadata_path = (CLEAN_ROOT / str(config["candidate_cache_metadata"])).resolve()
    require_candidate_cache_files(cache_path, metadata_path)
    provenance = read_candidate_cache_provenance(cache_path, metadata_path)
    metadata = provenance.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("C01 candidate metadata 必须是 JSON 对象")
    _validate_cache_metadata(
        metadata,
        int(config["expected_wells"]),
        int(config["expected_hidden_rows"]),
    )
    if str(metadata.get("candidate_cache_sha256", "")).lower() != (
        FROZEN_EXTERNAL_SHA256["candidate_feature_cache"]
    ):
        raise ValueError("C01 metadata 的 candidate cache SHA 不一致")
    return provenance


def _read_fold_predictions(output_dir: Path, fold_ids: list[int]) -> pd.DataFrame:
    """读取指定已完成折的正式候选预测。"""

    frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        path = output_dir / f"fold_{fold_id}" / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 P02 fold {fold_id} 预测")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _save_p01_comparison(
    output_dir: Path,
    stage: str,
    candidate_predictions: pd.DataFrame,
    p01_predictions: pd.DataFrame,
    config: dict[str, Any],
    cv_fingerprint: str,
) -> dict[str, Any]:
    """保存相对 P01 的逐井指标、bootstrap 和晋级判断。"""

    per_well, comparison = build_p01_comparison(
        candidate_predictions,
        p01_predictions,
    )
    checks = evaluate_model_success_checks(comparison, config, stage)
    comparison["cv_fingerprint"] = cv_fingerprint
    comparison["success_checks"] = checks
    per_well.to_csv(output_dir / f"per_well_vs_p01_{stage}.csv", index=False)
    write_json(output_dir / f"comparison_vs_p01_{stage}.json", comparison)
    return checks


def _rmse_from_predictions(predictions: pd.DataFrame) -> float:
    """计算一份隐藏行预测的 pooled micro RMSE。"""

    error = (
        predictions["pred_tvt"].to_numpy(dtype=np.float64)
        - predictions["target_tvt"].to_numpy(dtype=np.float64)
    )
    return float(np.sqrt(np.mean(error * error)))


def _run_negative_control_fold0(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    model_params: dict[str, Any],
    model_features: list[str],
    output_dir: Path,
    cv_fingerprint: str,
    p01_fold0: pd.DataFrame,
) -> None:
    """独立训练四尺度井内逆序的 fold0，不让结果参与正式选择。"""

    reversed_table = reverse_scale_paths_within_well(feature_table)
    diagnostic_fingerprint = stable_json_hash(
        {
            "formal_cv_fingerprint": cv_fingerprint,
            "diagnostic": "reverse_each_new_scale_path_within_well",
            "fold": 0,
            "used_for_selection": False,
        }
    )
    diagnostic_dir = output_dir / "negative_control_fold0_model"
    runtime = train_fold(
        reversed_table,
        registry,
        0,
        model_params,
        diagnostic_dir,
        diagnostic_fingerprint,
        model_features,
    )
    del reversed_table

    reversed_predictions = pd.read_parquet(diagnostic_dir / "fold_0" / "predictions.parquet")
    formal_predictions = pd.read_parquet(output_dir / "fold_0" / "predictions.parquet")
    reversed_rmse = _rmse_from_predictions(reversed_predictions)
    formal_rmse = _rmse_from_predictions(formal_predictions)
    p01_rmse = _rmse_from_predictions(p01_fold0)
    write_json(
        output_dir / "negative_control_fold0.json",
        {
            "diagnostic_only": True,
            "used_for_feature_or_model_selection": False,
            "cv_fingerprint": cv_fingerprint,
            "diagnostic_fingerprint": diagnostic_fingerprint,
            "fold": 0,
            "reversed_scale_micro_rmse": reversed_rmse,
            "formal_candidate_micro_rmse": formal_rmse,
            "p01_micro_rmse": p01_rmse,
            "formal_gain_vs_reversed_ft": reversed_rmse - formal_rmse,
            "runtime": runtime,
        },
    )


def _write_lineage(
    output_dir: Path,
    config: dict[str, Any],
    observed_hashes: dict[str, str],
    p01_fingerprint: str,
    cv_fingerprint: str,
    candidate_provenance: dict[str, Any],
) -> None:
    """保存 41 列、排除列和全部来源指纹。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "feature_list.json",
        {"feature_count": 41, "features": FROZEN_MODEL_FEATURES},
    )
    write_json(
        output_dir / "cache_and_fold_provenance.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "cv_fingerprint": cv_fingerprint,
            "p01_generator_fingerprint": p01_fingerprint,
            "observed_external_sha256": observed_hashes,
            "formal_features": FROZEN_MODEL_FEATURES,
            "diagnostic_columns_excluded": DIAGNOSTIC_ONLY_FEATURES,
            "fold_version": "balanced_well_5fold_v1",
            "validation_unit": "complete_well",
            "old_cached_fold_ignored": True,
            "candidate_cache_provenance": candidate_provenance,
        },
    )


def _write_conclusion(
    output_dir: Path,
    stage: str,
    checks: dict[str, Any],
) -> None:
    """按事实、推断、未验证和下一步写简短结论。"""

    if stage == "folds01":
        passed = bool(checks["folds01_pass"])
        fact = (
            f"folds 0～1 相对 P01 改善 "
            f"{checks['folds01_combined_improvement_ft']:.6f} ft，"
            f"晋级={'是' if passed else '否'}。"
        )
        next_step = "运行负对照和 folds 2～4。" if passed else "停止当前四尺度直加实现。"
    else:
        passed = bool(checks["full5_numeric_pass"])
        fact = (
            f"完整五折相对 P01 改善 {checks['full5_improvement_ft']:.6f} ft，"
            f"数值门槛={'通过' if passed else '未通过'}。"
        )
        next_step = "结合负对照和 oracle 解释结果，不事后删除某一 scale。"
    text = (
        "# P2-P02 结论\n\n"
        f"事实：{fact}\n\n"
        "推断：四尺度只有作为一个整体特征组接受同一套门槛。\n\n"
        "仍未验证：重新设计尺度可靠性门控是否有效。\n\n"
        "当前只能否定：把这四条固定尺度路径直接加入 P01 LightGBM。\n\n"
        f"下一步：{next_step}\n"
    )
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """完成来源核对、41 列合并、两阶段训练及独立诊断。"""

    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config = read_json(Path(args.config).resolve())

    # 先核对所有大文件哈希，之后才允许读取缓存或训练。
    observed_hashes = validate_frozen_file_hashes(config, CLEAN_ROOT)
    model_config = read_json((CLEAN_ROOT / str(config["model_config"])).resolve())
    p01_feature_manifest = read_json(
        (CLEAN_ROOT / str(config["baseline_feature_list"])).resolve()
    )
    model_features = validate_frozen_contract(
        config,
        model_config,
        p01_feature_manifest,
    )

    p01_generator_config = read_json(
        (CLEAN_ROOT / str(config["p01_generator_config"])).resolve()
    )
    validate_p01_generator_config(p01_generator_config)
    computed_p01_fingerprint = build_p01_generator_fingerprint(p01_generator_config)
    controls = read_json(
        (CLEAN_ROOT / str(config["p01_program_controls"])).resolve()
    )
    validate_p01_program_controls(config, controls, computed_p01_fingerprint)
    recomputed_checks = evaluate_path_success_checks(controls, p01_generator_config)
    if not recomputed_checks["feature_generation_supported"]:
        raise ValueError("重新计算后的 P01 程序控制未通过")

    expected_wells = int(config["expected_wells"])
    expected_rows = int(config["expected_hidden_rows"])
    registry = load_and_validate_registry(
        (CLEAN_ROOT / str(config["fold_registry"])).resolve(),
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    candidate_provenance = _candidate_cache_provenance(config)
    feature_table = load_aligned_feature_table(
        (CLEAN_ROOT / str(config["base_feature_cache"])).resolve(),
        (CLEAN_ROOT / str(config["candidate_feature_cache"])).resolve(),
        expected_rows,
    )
    feature_table = remap_feature_table_folds(feature_table, registry)
    feature_table = merge_p01_path_cache(
        feature_table,
        registry,
        (CLEAN_ROOT / str(config["p01_legal_cache_dir"])).resolve(),
        computed_p01_fingerprint,
    )
    validate_model_feature_names(model_features, FROZEN_P01_FEATURES)

    model_params = dict(model_config["params"])
    cv_fingerprint = build_cv_fingerprint(
        config,
        model_params,
        observed_hashes,
        computed_p01_fingerprint,
    )
    output_dir = Path(args.output_dir).resolve()
    _write_lineage(
        output_dir,
        config,
        observed_hashes,
        computed_p01_fingerprint,
        cv_fingerprint,
        candidate_provenance,
    )
    print(
        f"P2-P02：折={requested_folds}，井={expected_wells}，行={expected_rows:,}，"
        f"特征=41，树=1734，输出={output_dir}",
        flush=True,
    )

    # all 仍必须先训练或复用当前指纹的 folds 0～1，再重新判断门槛。
    for fold_id in (0, 1):
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

    candidate_folds01 = _read_fold_predictions(output_dir, [0, 1])
    p01_predictions = pd.read_parquet(
        (CLEAN_ROOT / str(config["baseline_predictions"])).resolve()
    )
    p01_folds01 = p01_predictions.loc[
        p01_predictions["fold"].astype(int).isin([0, 1])
    ].copy()
    folds01_checks = _save_p01_comparison(
        output_dir,
        "folds01",
        candidate_folds01,
        p01_folds01,
        config,
        cv_fingerprint,
    )
    _write_conclusion(output_dir, "folds01", folds01_checks)
    del candidate_folds01, p01_folds01

    # 预筛命令必须在这里结束，不能因目录中已有后三折而直接 finalize。
    if requested_folds == [0, 1]:
        return
    if not folds01_checks["folds01_pass"]:
        print("P2-P02 folds 0～1 未晋级，停止，不训练 folds 2～4。", flush=True)
        return
    load_passed_folds01_comparison(output_dir, cv_fingerprint)

    # 负对照在晋级后固定执行，但其结果不会进入任何门槛或特征选择。
    p01_fold0 = p01_predictions.loc[p01_predictions["fold"].astype(int).eq(0)].copy()
    # 负对照只需要 fold0；先释放完整 P01 OOF，避免与训练矩阵同时占内存。
    del p01_predictions
    _run_negative_control_fold0(
        feature_table,
        registry,
        model_params,
        model_features,
        output_dir,
        cv_fingerprint,
        p01_fold0,
    )
    del p01_fold0

    for fold_id in (2, 3, 4):
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
        config,
        model_params,
        model_features,
    )
    if complete_metrics is None:
        raise RuntimeError("P2-P02 all 完成后没有形成完整五折 OOF")
    validate_complete_predictions(output_dir, registry, expected_rows)

    candidate_all = pd.read_parquet(output_dir / "predictions.parquet")
    p01_all = pd.read_parquet(
        (CLEAN_ROOT / str(config["baseline_predictions"])).resolve()
    )
    full5_checks = _save_p01_comparison(
        output_dir,
        "full5",
        candidate_all,
        p01_all,
        config,
        cv_fingerprint,
    )
    _write_conclusion(output_dir, "full5", full5_checks)

    # oracle 最后才读取 target_tvt，只写独立诊断 JSON，不改变模型或结论门槛。
    oracle_audit = build_oracle_scale_audit(feature_table)
    oracle_audit["cv_fingerprint"] = cv_fingerprint
    write_json(output_dir / "oracle_scale_audit.json", oracle_audit)


if __name__ == "__main__":
    main()
