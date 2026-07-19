"""运行 P2-P03：在 P2-P02 的 41 列后只增加逐行 PF seed 标准差。

模型、参数、按井五折、目标和评价行全部冻结。正式新增列来自 P2-P01
已经通过隐藏真值不变性检查的逐井合法缓存，不重新运行粒子滤波。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_f05a_deterministic_candidates_cv import (  # noqa: E402
    load_aligned_feature_table,
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
from scripts.run_p2_p02_multiscale_pf_paths_cv import (  # noqa: E402
    FROZEN_MODEL_FEATURES as FROZEN_P02_FEATURES,
    FROZEN_MODEL_PARAMS,
    _candidate_cache_provenance,
    build_p01_comparison,
    file_sha256,
    parse_fold_spec,
    stable_json_hash,
    validate_p01_program_controls,
)
from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.lgbm_data import load_and_validate_registry  # noqa: E402


EXPERIMENT_ID = "P2_P03_pf_seed_dispersion_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_p03_pf_seed_dispersion_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

FORMAL_DISPERSION_FEATURE = "pf128_seed_std"
P01_CACHE_FORMAL_FEATURES = [
    "pf128_mean_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
    FORMAL_DISPERSION_FEATURE,
]
DIAGNOSTIC_ONLY_FEATURES = ["pf128_seed0_delta"]
FROZEN_MODEL_FEATURES = [*FROZEN_P02_FEATURES, FORMAL_DISPERSION_FEATURE]

FROZEN_BASELINE_MICRO_RMSE = 10.305704992073148
FROZEN_BASELINE_FOLD_RMSE = [
    10.170796687354473,
    9.4624641719035,
    9.181973787681697,
    10.8754729397572,
    11.639196676252652,
]

# 代码内另存一份哈希，避免同时修改 JSON 和输入文件后绕过冻结合同。
FROZEN_EXTERNAL_SHA256 = {
    "fold_registry": "a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c",
    "baseline_predictions": "8109514eb125f8beda2559f39e1d9f54396d41fd38fa76114423c8dd6bca4d37",
    "baseline_comparison": "22425927c695083e873c47382f604ee6cdbbe8e1056e3c4c4eb8912ae92c8aab",
    "baseline_feature_list": "2bf5992b45fe95e6ae38862da2408b306c0303806291d112ff8fa1ecf4b00172",
    "baseline_config": "979e8c713b1a20a7b6c540a632ab126baa87d8b207a495d16fe5f165615615ba",
    "p01_generator_config": "041c7e5ac3c40ee683c13d59821dcfe4a675c9632e3c55b609f93dcf9fb4e909",
    "p01_program_controls": "f238ced8fd7a38a577a9712d55433ef8ff1f23be11bfd663016a32ca4dd02dca",
    "model_config": "02f6c4cb737132641c2dbbb6076d0e248371786559f2cd6f8581ce72146cb838",
    "base_feature_cache": "8801493752e03f40a3957843e8efa25c091d04244dd341c64c9329b50d5de625",
    "candidate_feature_cache": "66b32f8ed790ea53184d43bc02d6899031348deec418d7933366bbef64105076",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """只允许预筛 ``0,1`` 或通过门槛后的 ``all``。"""

    parser = argparse.ArgumentParser(description="运行 P2-P03 seed 分歧 42 特征 CV")
    parser.add_argument("--folds", required=True, help="预筛传 0,1；晋级后传 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def validate_declared_external_hashes(config: dict[str, Any]) -> None:
    """配置声明的每个来源哈希必须等于代码冻结值。"""

    for source_name, frozen_hash in FROZEN_EXTERNAL_SHA256.items():
        declared = str(config.get(f"{source_name}_sha256", "")).lower()
        if declared != frozen_hash:
            raise ValueError(f"{source_name} SHA 不等于 P03 runner 冻结值")


def validate_frozen_file_hashes(
    config: dict[str, Any],
    clean_root: Path = CLEAN_ROOT,
) -> dict[str, str]:
    """训练前逐个复算冻结输入哈希。"""

    validate_declared_external_hashes(config)
    observed: dict[str, str] = {}
    for source_name, frozen_hash in FROZEN_EXTERNAL_SHA256.items():
        path = (Path(clean_root) / str(config[source_name])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 P03 冻结输入 {source_name}：{path}")
        actual = file_sha256(path)
        if actual.lower() != frozen_hash:
            raise ValueError(f"{source_name} 实际 SHA 与 P03 冻结值不一致")
        observed[source_name] = actual
    return observed


def _contains_forbidden_feature(feature_name: str) -> bool:
    """识别诊断列、标签、地层面和 oracle 列。"""

    lowered = str(feature_name).lower()
    if feature_name in DIAGNOSTIC_ONLY_FEATURES:
        return True
    if lowered == "tvt" or lowered.startswith("target"):
        return True
    if any(
        fragment in lowered
        for fragment in ("oracle", "surface", "geology", "marker")
    ):
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
    baseline_features: list[str],
) -> None:
    """正式输入必须严格为 P02 41 列后追加 seed 标准差。"""

    forbidden = [name for name in model_features if _contains_forbidden_feature(name)]
    if forbidden:
        raise ValueError(f"P03 正式特征含禁止输入：{forbidden}")
    expected = [*baseline_features, FORMAL_DISPERSION_FEATURE]
    if model_features != expected or len(model_features) != 42:
        raise ValueError("P03 正式特征必须严格为 P02 41 列加 seed std，共 42 列")
    if len(set(model_features)) != 42:
        raise ValueError("P03 正式 42 特征含重复列")


def validate_frozen_contract(
    config: dict[str, Any],
    model_config: dict[str, Any],
    baseline_feature_manifest: dict[str, Any],
) -> list[str]:
    """冻结 P02 基线、42 列、按井五折和完整 LightGBM 参数。"""

    validate_declared_external_hashes(config)
    expected_values = {
        "experiment_id": EXPERIMENT_ID,
        "fold_version": "balanced_well_5fold_v1",
        "baseline_experiment_id": "P2_P02_multiscale_pf_paths_v1",
        "baseline_micro_rmse": FROZEN_BASELINE_MICRO_RMSE,
        "baseline_fold_rmse": FROZEN_BASELINE_FOLD_RMSE,
        "baseline_feature_count": 41,
        "formal_feature_count": 42,
        "new_feature_names": [FORMAL_DISPERSION_FEATURE],
        "diagnostic_columns_excluded_from_model": DIAGNOSTIC_ONLY_FEATURES,
        "expected_wells": 773,
        "expected_hidden_rows": 3_783_989,
    }
    for field_name, expected in expected_values.items():
        if config.get(field_name) != expected:
            raise ValueError(f"P03 {field_name} 不等于冻结值 {expected}")

    expected_negative_control = {
        "name": "reverse_seed_std_within_well",
        "run_if_folds01_pass": True,
        "fold": 0,
        "minimum_formal_gain_vs_reversed_ft": 0.05,
    }
    if config.get("negative_control") != expected_negative_control:
        raise ValueError("P03 seed std 井内逆序负对照配置发生变化")

    baseline_features = baseline_feature_manifest.get("features")
    if baseline_features != FROZEN_P02_FEATURES:
        raise ValueError("P02 feature_list 不是冻结的 41 列")
    if int(baseline_feature_manifest.get("feature_count", -1)) != 41:
        raise ValueError("P02 feature_count 必须是 41")

    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("P03 模型不再是单模 LightGBM")
    params = model_config.get("params")
    if not isinstance(params, dict) or set(params) != set(FROZEN_MODEL_PARAMS):
        raise ValueError("P03 LightGBM 完整参数字段集合发生变化")
    for parameter_name, frozen_value in FROZEN_MODEL_PARAMS.items():
        observed_value = params.get(parameter_name)
        if type(observed_value) is not type(frozen_value) or observed_value != frozen_value:
            raise ValueError(
                f"P03 模型参数 {parameter_name} 不等于冻结值 {frozen_value}"
            )
    policy = model_config.get("training_policy")
    if not isinstance(policy, dict):
        raise ValueError("P03 模型 training_policy 缺失")
    if policy.get("early_stopping") is not False:
        raise ValueError("P03 模型不得 early stopping")
    if policy.get("uniform_row_weight") is not True:
        raise ValueError("P03 模型必须保持统一行权重")

    model_features = [*baseline_features, FORMAL_DISPERSION_FEATURE]
    validate_model_feature_names(model_features, baseline_features)
    return model_features


def merge_p01_path_and_dispersion_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    expected_fingerprint: str,
) -> pd.DataFrame:
    """一次读取每井 P01 缓存并合并均值、四尺度和 seed 标准差。"""

    required_base = {"well_id", "row_index", "last_visible_tvt"}
    missing_base = required_base.difference(feature_table.columns)
    if missing_base:
        raise ValueError(f"基础特征表缺少 P03 对齐列：{sorted(missing_base)}")
    if bool(feature_table.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("基础特征表含重复自然隐藏行键")
    if any(name in feature_table.columns for name in P01_CACHE_FORMAL_FEATURES):
        raise ValueError("基础特征表已含 P01/P03 路径列，禁止重复合并")
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

    selected_columns = [
        "well_id",
        "row_index",
        "last_visible_tvt",
        *P01_CACHE_FORMAL_FEATURES,
    ]
    selected_rows: list[pd.DataFrame] = []
    for completed, registry_row in enumerate(registry.itertuples(index=False), start=1):
        well_id = str(registry_row.well_id)
        expected_rows = int(registry_row.hidden_rows)
        cache = pd.read_parquet(cache_dir / f"{well_id}.parquet")
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
            print(f"P03 核对 P01 缓存 {completed}/{len(expected_wells)}", flush=True)

    p01_features = pd.concat(selected_rows, ignore_index=True)
    if len(p01_features) != len(feature_table):
        raise ValueError("P01 路径缓存总行数与基础特征表不一致")

    base = feature_table.copy()
    base["well_id"] = base["well_id"].astype(str)
    base["_p03_original_order"] = np.arange(len(base), dtype=np.int64)
    p01_features["well_id"] = p01_features["well_id"].astype(str)
    merged = base.merge(
        p01_features,
        on=["well_id", "row_index"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("P01 路径与基础表自然隐藏行键不能一一对应")
    merged = merged.sort_values("_p03_original_order", kind="stable").reset_index(drop=True)

    base_anchor = merged["last_visible_tvt"].to_numpy(dtype=np.float32)
    cache_anchor = merged["p01_last_visible_tvt"].to_numpy(dtype=np.float32)
    if not np.array_equal(base_anchor, cache_anchor):
        raise ValueError("P01 last_visible_tvt 路径起点与基础特征表不一致")
    formal_values = merged[P01_CACHE_FORMAL_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(formal_values).all():
        raise ValueError("P01 均值、四尺度或 seed std 含 NaN/Inf")
    if np.any(merged[FORMAL_DISPERSION_FEATURE].to_numpy(dtype=np.float64) < 0.0):
        raise ValueError("pf128_seed_std 不得为负数")

    merged = merged.drop(
        columns=["p01_last_visible_tvt", "_p03_original_order", "_merge"]
    )
    diagnostics = [name for name in DIAGNOSTIC_ONLY_FEATURES if name in merged.columns]
    if diagnostics:
        raise ValueError(f"P01 诊断列意外进入 P03 模型表：{diagnostics}")
    return merged


def reverse_seed_std_within_well(feature_table: pd.DataFrame) -> pd.DataFrame:
    """只在每口井内部反转 seed std，保持其他列和原表不变。"""

    if FORMAL_DISPERSION_FEATURE not in feature_table.columns:
        raise ValueError("P03 负对照缺少 pf128_seed_std")
    reversed_table = feature_table.copy(deep=False)
    reversed_values = feature_table[FORMAL_DISPERSION_FEATURE].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    for positions in feature_table.groupby("well_id", sort=False).indices.values():
        index_array = np.asarray(positions, dtype=np.int64)
        reversed_values[index_array] = reversed_values[index_array][::-1]
    reversed_table[FORMAL_DISPERSION_FEATURE] = reversed_values
    return reversed_table


def build_baseline_comparison(
    candidate_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """复用统一配对实现，并把比较对象明确登记为 P02。"""

    per_well, comparison = build_p01_comparison(
        candidate_predictions,
        baseline_predictions,
    )
    comparison["comparison"] = "candidate_minus_P02"
    return per_well, comparison


def evaluate_model_success_checks(
    comparison: dict[str, Any],
    config: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """相对 P02 使用预注册门槛；bootstrap 上界必须严格小于 0。"""

    if stage not in {"folds01", "full5"}:
        raise ValueError(f"未知 P03 模型阶段：{stage}")
    conditions = config["model_success_conditions"]
    fold_by_id = {int(row["fold"]): row for row in comparison["folds"]}
    if 0 not in fold_by_id or 1 not in fold_by_id:
        raise ValueError("P03 folds 0～1 结果不完整")

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
        "comparison_baseline": "P2_P02_multiscale_pf_paths_v1",
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
        raise ValueError("P03 完整五折必须含 fold 0～4")
    improved_folds = int(
        sum(
            row["micro_rmse"] < row["baseline_micro_rmse"]
            for row in comparison["folds"]
        )
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
            "well_win_rate_vs_p02": float(overall["well_win_rate"]),
        }
    )
    checks["full5_numeric_pass"] = bool(
        improvement >= float(conditions["minimum_full5_improvement_ft"])
        and improved_folds >= int(conditions["minimum_improved_folds"])
        and bootstrap_high < float(conditions["maximum_bootstrap_ci_upper"])
        and p90_degradation <= float(conditions["maximum_p90_degradation_ft"])
    )
    return checks


def build_seed_dispersion_oracle_audit(feature_table: pd.DataFrame) -> dict[str, Any]:
    """用隐藏真值诊断 seed 分歧与均值 PF 误差，只返回 JSON 摘要。"""

    required = {
        "well_id",
        "last_visible_tvt",
        "pf128_mean_delta",
        FORMAL_DISPERSION_FEATURE,
        "target_tvt",
    }
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"seed dispersion oracle 缺列：{sorted(missing)}")

    seed_std = feature_table[FORMAL_DISPERSION_FEATURE].to_numpy(dtype=np.float64)
    mean_pf_tvt = (
        feature_table["last_visible_tvt"].to_numpy(dtype=np.float64)
        + feature_table["pf128_mean_delta"].to_numpy(dtype=np.float64)
    )
    target_tvt = feature_table["target_tvt"].to_numpy(dtype=np.float64)
    absolute_error = np.abs(mean_pf_tvt - target_tvt)
    quantile_edges = np.unique(np.quantile(seed_std, np.linspace(0.0, 1.0, 11)))
    quantile_bins: list[dict[str, Any]] = []
    if len(quantile_edges) >= 2:
        bin_ids = np.searchsorted(quantile_edges[1:-1], seed_std, side="right")
        for bin_id in range(len(quantile_edges) - 1):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue
            quantile_bins.append(
                {
                    "bin": int(bin_id),
                    "rows": int(mask.sum()),
                    "seed_std_min": float(seed_std[mask].min()),
                    "seed_std_max": float(seed_std[mask].max()),
                    "mean_pf_absolute_error": float(absolute_error[mask].mean()),
                    "mean_pf_rmse": float(np.sqrt(np.mean((mean_pf_tvt[mask] - target_tvt[mask]) ** 2))),
                }
            )
    correlation = float(pd.Series(seed_std).corr(pd.Series(absolute_error), method="spearman"))
    return {
        "diagnostic_only": True,
        "used_for_feature_or_model_selection": False,
        "rows": int(len(feature_table)),
        "wells": int(feature_table["well_id"].astype(str).nunique()),
        "seed_std_mean": float(seed_std.mean()),
        "seed_std_p90": float(np.quantile(seed_std, 0.90)),
        "seed_std_vs_mean_pf_absolute_error_spearman": correlation,
        "quantile_bins": quantile_bins,
    }


def build_cv_fingerprint(
    config: dict[str, Any],
    model_params: dict[str, Any],
    observed_hashes: dict[str, str],
    p01_generator_fingerprint: str,
) -> str:
    """把代码、42 列、P02 基线和全部缓存来源写入 P03 指纹。"""

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


def _read_fold_predictions(output_dir: Path, fold_ids: list[int]) -> pd.DataFrame:
    """读取当前指纹已完成的指定折预测。"""

    frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        path = output_dir / f"fold_{fold_id}" / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 P03 fold {fold_id} 预测")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _save_baseline_comparison(
    output_dir: Path,
    stage: str,
    candidate_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    config: dict[str, Any],
    cv_fingerprint: str,
) -> dict[str, Any]:
    """保存相对 P02 的逐井比较、bootstrap 和门槛。"""

    per_well, comparison = build_baseline_comparison(
        candidate_predictions,
        baseline_predictions,
    )
    checks = evaluate_model_success_checks(comparison, config, stage)
    comparison["cv_fingerprint"] = cv_fingerprint
    comparison["success_checks"] = checks
    per_well.to_csv(output_dir / f"per_well_vs_p02_{stage}.csv", index=False)
    write_json(output_dir / f"comparison_vs_p02_{stage}.json", comparison)
    return checks


def load_passed_folds01_comparison(
    output_dir: Path,
    expected_cv_fingerprint: str,
) -> dict[str, Any]:
    """读取同一 P03 指纹且已经通过的 folds 0～1 凭证。"""

    path = output_dir / "comparison_vs_p02_folds01.json"
    if not path.is_file():
        raise FileNotFoundError("训练 P03 folds 2～4 前缺少 folds01 comparison")
    comparison = read_json(path)
    if comparison.get("cv_fingerprint") != expected_cv_fingerprint:
        raise ValueError("P03 folds01 comparison 指纹不是当前实验指纹")
    checks = comparison.get("success_checks")
    if not isinstance(checks, dict) or checks.get("folds01_pass") is not True:
        raise ValueError("P03 folds01 comparison 未通过门槛")
    return comparison


def _rmse_from_predictions(predictions: pd.DataFrame) -> float:
    """从自然隐藏行预测计算 pooled micro RMSE。"""

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
    baseline_fold0: pd.DataFrame,
    config: dict[str, Any],
) -> bool:
    """训练只反转 seed std 的 fold 0，并按预注册差值决定是否继续。"""

    reversed_table = reverse_seed_std_within_well(feature_table)
    diagnostic_fingerprint = stable_json_hash(
        {
            "formal_cv_fingerprint": cv_fingerprint,
            "diagnostic": "reverse_seed_std_within_well",
            "fold": 0,
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

    reversed_predictions = pd.read_parquet(
        diagnostic_dir / "fold_0" / "predictions.parquet"
    )
    formal_predictions = pd.read_parquet(output_dir / "fold_0" / "predictions.parquet")
    reversed_rmse = _rmse_from_predictions(reversed_predictions)
    formal_rmse = _rmse_from_predictions(formal_predictions)
    baseline_rmse = _rmse_from_predictions(baseline_fold0)
    gain = float(reversed_rmse - formal_rmse)
    threshold = float(
        config["negative_control"]["minimum_formal_gain_vs_reversed_ft"]
    )
    passed = bool(gain >= threshold)
    write_json(
        output_dir / "negative_control_fold0.json",
        {
            "diagnostic_only": True,
            "used_for_feature_or_model_selection": False,
            "used_as_preregistered_continuation_gate": True,
            "cv_fingerprint": cv_fingerprint,
            "diagnostic_fingerprint": diagnostic_fingerprint,
            "fold": 0,
            "reversed_seed_std_micro_rmse": reversed_rmse,
            "formal_candidate_micro_rmse": formal_rmse,
            "p02_micro_rmse": baseline_rmse,
            "formal_gain_vs_reversed_ft": gain,
            "minimum_required_gain_ft": threshold,
            "negative_control_pass": passed,
            "runtime": runtime,
        },
    )
    return passed


def _write_lineage(
    output_dir: Path,
    config: dict[str, Any],
    observed_hashes: dict[str, str],
    p01_fingerprint: str,
    cv_fingerprint: str,
    candidate_provenance: dict[str, Any],
) -> None:
    """保存 42 列、来源指纹和明确排除列。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "feature_list.json",
        {"feature_count": 42, "features": FROZEN_MODEL_FEATURES},
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
    negative_control_pass: bool | None = None,
) -> None:
    """按固定证据边界写当前阶段结论。"""

    if stage == "folds01":
        passed = bool(checks["folds01_pass"])
        fact = (
            f"folds 0～1 相对 P02 改善 "
            f"{checks['folds01_combined_improvement_ft']:.6f} ft，"
            f"数值晋级={'是' if passed else '否'}。"
        )
        next_step = "运行 fold 0 井内反转负对照。" if passed else "按门槛停止。"
    elif stage == "negative_control":
        fact = f"井内反转负对照门槛={'通过' if negative_control_pass else '未通过'}。"
        next_step = "运行 folds 2～4。" if negative_control_pass else "按门槛停止。"
    else:
        passed = bool(checks["full5_numeric_pass"])
        fact = (
            f"完整五折相对 P02 改善 {checks['full5_improvement_ft']:.6f} ft，"
            f"数值门槛={'通过' if passed else '未通过'}。"
        )
        next_step = "完成收益集中度审计并归档二阶段路线。"
    text = (
        "# P2-P03 结论\n\n"
        f"事实：{fact}\n\n"
        "推断：只有正式顺序优于井内反转时，才能把收益解释为逐行可靠性信息。\n\n"
        "仍未验证：分位宽度、多峰和分歧增长是否比原始标准差更有效。\n\n"
        "当前只能否定：原始 pf128_seed_std 直接加一列的当前实现。\n\n"
        f"下一步：{next_step}\n"
    )
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """核对来源、合并 42 列，并按门槛自动完成或停止 P03。"""

    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config = read_json(Path(args.config).resolve())

    observed_hashes = validate_frozen_file_hashes(config, CLEAN_ROOT)
    model_config = read_json((CLEAN_ROOT / str(config["model_config"])).resolve())
    baseline_manifest = read_json(
        (CLEAN_ROOT / str(config["baseline_feature_list"])).resolve()
    )
    model_features = validate_frozen_contract(
        config,
        model_config,
        baseline_manifest,
    )

    p01_generator_config = read_json(
        (CLEAN_ROOT / str(config["p01_generator_config"])).resolve()
    )
    validate_p01_generator_config(p01_generator_config)
    p01_fingerprint = build_p01_generator_fingerprint(p01_generator_config)
    controls = read_json(
        (CLEAN_ROOT / str(config["p01_program_controls"])).resolve()
    )
    validate_p01_program_controls(config, controls, p01_fingerprint)
    recomputed_controls = evaluate_path_success_checks(controls, p01_generator_config)
    if not recomputed_controls["feature_generation_supported"]:
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
    feature_table = merge_p01_path_and_dispersion_cache(
        feature_table,
        registry,
        (CLEAN_ROOT / str(config["p01_legal_cache_dir"])).resolve(),
        p01_fingerprint,
    )
    validate_model_feature_names(model_features, FROZEN_P02_FEATURES)

    model_params = dict(model_config["params"])
    cv_fingerprint = build_cv_fingerprint(
        config,
        model_params,
        observed_hashes,
        p01_fingerprint,
    )
    output_dir = Path(args.output_dir).resolve()
    _write_lineage(
        output_dir,
        config,
        observed_hashes,
        p01_fingerprint,
        cv_fingerprint,
        candidate_provenance,
    )
    print(
        f"P2-P03：折={requested_folds}，井={expected_wells}，行={expected_rows:,}，"
        f"特征=42，树=1734，输出={output_dir}",
        flush=True,
    )

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
    baseline_predictions = pd.read_parquet(
        (CLEAN_ROOT / str(config["baseline_predictions"])).resolve()
    )
    baseline_folds01 = baseline_predictions.loc[
        baseline_predictions["fold"].astype(int).isin([0, 1])
    ].copy()
    folds01_checks = _save_baseline_comparison(
        output_dir,
        "folds01",
        candidate_folds01,
        baseline_folds01,
        config,
        cv_fingerprint,
    )
    _write_conclusion(output_dir, "folds01", folds01_checks)
    del candidate_folds01, baseline_folds01

    if requested_folds == [0, 1]:
        return
    if not folds01_checks["folds01_pass"]:
        print("P2-P03 folds 0～1 未晋级，停止，不训练 folds 2～4。", flush=True)
        return
    load_passed_folds01_comparison(output_dir, cv_fingerprint)

    baseline_fold0 = baseline_predictions.loc[
        baseline_predictions["fold"].astype(int).eq(0)
    ].copy()
    negative_pass = _run_negative_control_fold0(
        feature_table,
        registry,
        model_params,
        model_features,
        output_dir,
        cv_fingerprint,
        baseline_fold0,
        config,
    )
    _write_conclusion(
        output_dir,
        "negative_control",
        folds01_checks,
        negative_control_pass=negative_pass,
    )
    del baseline_fold0
    if not negative_pass:
        print("P2-P03 井内反转负对照未通过，停止，不训练 folds 2～4。", flush=True)
        return

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
        raise RuntimeError("P2-P03 all 完成后没有形成完整五折 OOF")
    validate_complete_predictions(output_dir, registry, expected_rows)

    candidate_all = pd.read_parquet(output_dir / "predictions.parquet")
    full5_checks = _save_baseline_comparison(
        output_dir,
        "full5",
        candidate_all,
        baseline_predictions,
        config,
        cv_fingerprint,
    )
    _write_conclusion(output_dir, "full5", full5_checks)

    oracle_audit = build_seed_dispersion_oracle_audit(feature_table)
    oracle_audit["cv_fingerprint"] = cv_fingerprint
    write_json(output_dir / "oracle_seed_dispersion_audit.json", oracle_audit)


if __name__ == "__main__":
    main()
