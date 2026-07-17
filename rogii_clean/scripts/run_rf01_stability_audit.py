"""只补 RF01a folds 2–4，并汇总固定五折稳定性诊断。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 当前文件位于 rogii_clean/scripts；父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 直接以文件方式运行时，显式加入父目录以导入 scripts 与 src。
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (
    read_json,
    train_fold,
    validate_feature_table,
    write_json,
)
from src.lgbm_data import file_sha256, load_and_validate_registry
from src.metrics import (
    build_per_well_metrics,
    paired_well_bootstrap,
    summarize_by_fold,
    summarize_per_well_metrics,
)


# folds 0–1 只读取既有 RF01a 预测，任何时候都不得由本脚本重跑。
REUSED_FOLDS = (0, 1)

# folds 2–4 是审计唯一允许交给通用 train_fold 的新折集合。
NEW_FOLDS = (2, 3, 4)

# 行键顺序固定包含 fold，避免相同行号在错误折中仍被当作同一评价行。
ROW_KEY_COLUMNS = ("well_id", "fold", "row_index")

# 下列值逐项抄自已批准的审计卡；脚本用它们拒绝重选、换列或换实验。
AUDIT_EXPERIMENT_ID = "RF01_stability_audit_v1"
SOURCE_EXPERIMENT_ID = "RF01a_huber_slopes_v1"
SOURCE_FEATURE_VERSION = "rf01_huber_slopes_17_v1"
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989

# 固定 spatial-pad 注册表的内容哈希来自审计卡和已批准的机器配置。
EXPECTED_FOLD_REGISTRY_HASH = (
    "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab"
)

# 冻结模型配置使用内容哈希锁定，避免只检查文件名却静默改变任一参数。
EXPECTED_MODEL_CONFIG_HASH = (
    "02f6c4cb737132641c2dbbb6076d0e248371786559f2cd6f8581ce72146cb838"
)

# 17 列顺序就是 LightGBM 的列顺序；不能改为集合比较。
RF01A_FEATURE_COLUMNS = (
    "last_visible_tvt",
    "md_since_visible_end",
    "hidden_fraction",
    "x_current",
    "y_current",
    "z_current",
    "dx_from_visible_end",
    "dy_from_visible_end",
    "dz_from_visible_end",
    "dxy_from_visible_end",
    "gr_raw",
    "gr_missing",
    "u_huber_slope_50",
    "u_huber_slope_100",
    "u_huber_slope_200",
    "u_huber_slope_500",
    "u_huber_slope_1000",
)

# 除通用 runner 已允许的 gr_raw 外，只有五个 Huber slope 可以为 NaN。
RF01A_ALLOWED_NAN_FEATURES = (
    "u_huber_slope_50",
    "u_huber_slope_100",
    "u_huber_slope_200",
    "u_huber_slope_500",
    "u_huber_slope_1000",
)

# 四条选择理由在看到 folds 2–4 前已登记，顺序和文字都不可改写。
PRE_REGISTERED_SELECTION_REASONS = (
    "公式最简单。",
    "fold 1 损失最小。",
    "不含派生程度更高的差值、曲率和无界外推。",
    "选择依据不使用 folds 2–4。",
)

# 路径字符串也属于预登记对象，防止配置转向其他 RF01 版本或 B00 文件。
EXPECTED_SOURCE_CONFIG = "configs/rf01a_huber_slopes_v1.json"
EXPECTED_SOURCE_ARTIFACT_DIR = "artifacts/RF01a_huber_slopes_v1"
EXPECTED_SOURCE_FEATURE_CACHE = (
    "artifacts/RF01a_huber_slopes_v1/feature_cache.parquet"
)
EXPECTED_MODEL_CONFIG = "configs/lgbm_feature_baseline_v1.json"
EXPECTED_FOLD_REGISTRY = "artifacts/folds/spatial_pad_1000_v1.csv"
EXPECTED_B00_PREDICTIONS = "artifacts/B00_simple_lgbm_v1/predictions.parquet"

# AGENTS 6.3 与路线图第 14 节共同锁定 feature_quality 的列名和顺序。
FEATURE_QUALITY_COLUMNS = (
    "feature",
    "dtype",
    "unit",
    "source",
    "finite_rate",
    "unique_count",
    "mean",
    "std",
    "p01",
    "p50",
    "p99",
    "well_constant_rate",
    "correlation_with_existing_feature",
)

# leakage_tests.json 必须逐项保留这些固定检查名，不能用近义词替换。
LEAKAGE_TEST_NAMES = (
    "hidden TVT deletion invariance",
    "hidden TVT mutation invariance",
    "surface deletion invariance",
    "outer-fold source exclusion",
    "PF cache fold match",
    "row hash match",
    "fixed-seed reproducibility",
    "negative-control status",
)

# 完整审计成功后必须生成的汇总文件；fold_2/3/4 子目录由 train_fold 管理。
FINAL_OUTPUT_RELATIVE_PATHS = (
    "predictions.parquet",
    "metrics.json",
    "per_well.csv",
    "feature_list.json",
    "parameter_list.json",
    "feature_definition.json",
    "feature_lineage.json",
    "feature_quality.csv",
    "cache_manifest.json",
    "leakage_tests.json",
    "negative_control_metrics.json",
    "runtime.json",
    "config.json",
    "feature_importance.csv",
    "per_fold.csv",
    "slice_metrics.csv",
    "bootstrap_replicates.parquet",
    "comparison_vs_B00/predictions.parquet",
    "comparison_vs_B00/per_well.csv",
    "comparison_vs_B00/metrics.json",
    "conclusion.md",
)


@dataclass
class AuditInputs:
    """训练前已通过全部合同检查的审计输入和指纹。"""

    # 三份配置保留原始字典，供训练、留档和最终汇总使用。
    audit_config: dict
    source_config: dict
    model_config: dict
    model_params: dict

    # 完整表 shape 分别为 [773, registry列] 与 [3,783,989, 特征/元数据列]。
    registry: pd.DataFrame
    feature_table: pd.DataFrame
    b00_predictions: pd.DataFrame

    # 源 folds 0–1 的既有内容只读保存，绝不复制为审计 fold_0/fold_1。
    reused_fold_predictions: dict[int, pd.DataFrame]
    reused_fold_runtimes: dict[int, dict]
    reused_fold_importance: dict[int, pd.DataFrame]

    # 固定特征顺序和 cache metadata 共同描述源训练血缘。
    feature_columns: list[str]
    cache_metadata: dict
    source_fingerprint: str
    audit_fingerprint: str
    reporting_fingerprint: str
    training_fingerprint_policy: str
    eval_row_hash: str

    # 路径用于新折训练和固定来源汇总；预检本身不会创建 artifact_dir。
    clean_root: Path
    source_artifact_dir: Path
    artifact_dir: Path


def build_feature_contract_documents(
    feature_columns: list[str],
) -> tuple[dict, dict]:
    """返回固定 17 列的完整定义文档和逐列 lineage。"""

    if feature_columns != list(RF01A_FEATURE_COLUMNS):
        raise ValueError("feature contract 只允许固定 RF01a 17 列及顺序")

    # 每列定义直接对应 lgbm_features.py 与 rf01a_features.py 的实际公式。
    definitions = {
        "last_visible_tvt": ("最后一个可见 TVT_input", "TVT_input[last_visible]", "ft", ["TVT_input"]),
        "md_since_visible_end": ("当前隐藏行距可见末端的 MD", "MD_current - MD_last_visible", "ft", ["MD", "TVT_input visibility mask"]),
        "hidden_fraction": ("隐藏后缀内的归一化 MD 进度", "(MD_current - MD_hidden_start) / max(MD_hidden_end - MD_hidden_start, 1)", "ratio", ["MD", "TVT_input visibility mask"]),
        "x_current": ("当前隐藏行 X 坐标", "X_current", "ft", ["X"]),
        "y_current": ("当前隐藏行 Y 坐标", "Y_current", "ft", ["Y"]),
        "z_current": ("当前隐藏行 Z 坐标", "Z_current", "ft", ["Z"]),
        "dx_from_visible_end": ("当前 X 相对可见末端位移", "X_current - X_last_visible", "ft", ["X", "TVT_input visibility mask"]),
        "dy_from_visible_end": ("当前 Y 相对可见末端位移", "Y_current - Y_last_visible", "ft", ["Y", "TVT_input visibility mask"]),
        "dz_from_visible_end": ("当前 Z 相对可见末端位移", "Z_current - Z_last_visible", "ft", ["Z", "TVT_input visibility mask"]),
        "dxy_from_visible_end": ("当前水平位移模长", "sqrt(dx_from_visible_end^2 + dy_from_visible_end^2)", "ft", ["X", "Y", "TVT_input visibility mask"]),
        "gr_raw": ("当前隐藏行原始 GR", "GR_current", "API", ["GR"]),
        "gr_missing": ("当前隐藏行 GR 缺失指示", "isnan(GR_current)", "indicator", ["GR"]),
    }
    for window_ft in (50, 100, 200, 500, 1000):
        feature_name = f"u_huber_slope_{window_ft}"
        definitions[feature_name] = (
            f"可见前缀末端 {window_ft} ft 内 U 对 MD 的 Huber 倾角",
            f"Huber_IRLS_slope(U=TVT_input+Z, MD, visible_window={window_ft}ft)",
            "ft_per_ft",
            ["MD", "TVT_input", "Z"],
        )

    feature_definition_rows: dict[str, dict] = {}
    feature_lineage: dict[str, dict] = {}
    visible_boundary_features = {
        "last_visible_tvt",
        "md_since_visible_end",
        "hidden_fraction",
        "dx_from_visible_end",
        "dy_from_visible_end",
        "dz_from_visible_end",
        "dxy_from_visible_end",
    }
    slope_features = set(RF01A_FEATURE_COLUMNS) - set(RF01A_FEATURE_COLUMNS[:12])
    for feature_name in feature_columns:
        description, formula, unit, raw_source = definitions[feature_name]
        transform_version = (
            SOURCE_FEATURE_VERSION
            if feature_name in slope_features
            else "simple_horizontal_12_v1"
        )
        uses_tvt_input = (
            feature_name in visible_boundary_features or feature_name in slope_features
        )
        uses_hidden_gr = feature_name in {"gr_raw", "gr_missing"}
        feature_definition_rows[feature_name] = {
            "description": description,
            "formula": formula,
            "unit": unit,
            "raw_source": raw_source,
            "transform_version": transform_version,
        }
        feature_lineage[feature_name] = {
            "raw_source": raw_source,
            "uses_TVT_input": uses_tvt_input,
            "uses_hidden_GR": uses_hidden_gr,
            "uses_Typewell": False,
            "uses_PF_Beam": False,
            "requires_outer_train_fit": False,
            "fit_wells_hash": None,
            "transform_version": transform_version,
            "unit": unit,
        }

    feature_definition = {
        "feature_version": SOURCE_FEATURE_VERSION,
        "feature_count": len(feature_columns),
        "target_excluded_from_features": "target_delta",
        "features": feature_definition_rows,
        "quality_field_definitions": {
            "finite_rate": "finite rows / all evaluation rows",
            "unique_count": "count of distinct finite values",
            "well_constant_rate": "fraction of wells with <=1 distinct value, treating NaN as a value",
            "correlation_with_existing_feature": "maximum absolute Pearson correlation with any other fixed model feature",
        },
    }
    return feature_definition, feature_lineage


def build_feature_quality_table(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    feature_lineage: dict[str, dict],
) -> pd.DataFrame:
    """计算固定字段顺序的一特征一行质量长表。"""

    if "well_id" not in feature_table.columns:
        raise ValueError("feature quality 需要 well_id")
    if set(feature_columns) != set(feature_lineage):
        raise ValueError("feature quality 的 lineage 与特征列表不一致")

    numeric_features = feature_table.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    finite_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    correlation_matrix = finite_features.corr(method="pearson").abs()
    quality_rows: list[dict] = []
    for feature_name in feature_columns:
        raw_values = numeric_features[feature_name]
        values = raw_values.to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        if len(finite_values) == 0:
            mean = std = p01 = p50 = p99 = float("nan")
        else:
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values, ddof=0))
            p01, p50, p99 = [
                float(value)
                for value in np.quantile(finite_values, [0.01, 0.50, 0.99])
            ]

        per_well_unique = (
            pd.DataFrame(
                {
                    "well_id": feature_table["well_id"].astype("string"),
                    "value": finite_features[feature_name],
                }
            )
            .groupby("well_id", observed=True)["value"]
            .nunique(dropna=False)
        )
        other_correlations = correlation_matrix.loc[feature_name].drop(
            labels=[feature_name]
        )
        finite_correlations = other_correlations[np.isfinite(other_correlations)]
        max_correlation = (
            float(finite_correlations.max())
            if len(finite_correlations) > 0
            else float("nan")
        )
        lineage = feature_lineage[feature_name]
        quality_rows.append(
            {
                "feature": feature_name,
                "dtype": str(feature_table[feature_name].dtype),
                "unit": str(lineage["unit"]),
                "source": " + ".join(str(item) for item in lineage["raw_source"]),
                "finite_rate": float(np.mean(finite_mask)),
                "unique_count": int(pd.Series(finite_values).nunique(dropna=True)),
                "mean": mean,
                "std": std,
                "p01": p01,
                "p50": p50,
                "p99": p99,
                "well_constant_rate": float(np.mean(per_well_unique <= 1)),
                "correlation_with_existing_feature": max_correlation,
            }
        )

    return pd.DataFrame(quality_rows, columns=list(FEATURE_QUALITY_COLUMNS))


def build_paired_bootstrap_artifacts(
    per_well_df: pd.DataFrame,
    *,
    candidate_id: str,
    baseline_id: str,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """返回井级配对 micro-RMSE bootstrap 的真实逐次明细和摘要。"""

    required_columns = {"rows", "prediction_sse", "baseline_sse"}
    if not required_columns.issubset(per_well_df.columns):
        raise ValueError("bootstrap 明细需要 rows、prediction_sse、baseline_sse")
    rows = per_well_df["rows"].to_numpy(dtype=np.int64)
    candidate_sse = per_well_df["prediction_sse"].to_numpy(dtype=np.float64)
    baseline_sse = per_well_df["baseline_sse"].to_numpy(dtype=np.float64)
    well_count = int(len(per_well_df))
    random_generator = np.random.default_rng(int(seed))
    replicate_rows: list[dict] = []
    for replicate_id in range(int(n_resamples)):
        sampled_indices = random_generator.integers(0, well_count, size=well_count)
        sampled_rows = int(rows[sampled_indices].sum())
        sampled_candidate_sse = float(candidate_sse[sampled_indices].sum())
        sampled_baseline_sse = float(baseline_sse[sampled_indices].sum())
        candidate_rmse = float(np.sqrt(sampled_candidate_sse / max(sampled_rows, 1)))
        baseline_rmse = float(np.sqrt(sampled_baseline_sse / max(sampled_rows, 1)))
        replicate_rows.append(
            {
                "replicate": replicate_id,
                "seed": int(seed),
                "candidate_id": str(candidate_id),
                "baseline_id": str(baseline_id),
                "sampled_wells": well_count,
                "unique_sampled_wells": int(np.unique(sampled_indices).size),
                "sampled_rows": sampled_rows,
                "candidate_sse": sampled_candidate_sse,
                "baseline_sse": sampled_baseline_sse,
                "candidate_rmse": candidate_rmse,
                "baseline_rmse": baseline_rmse,
                "delta_rmse": candidate_rmse - baseline_rmse,
            }
        )

    replicates = pd.DataFrame(replicate_rows)
    deltas = replicates["delta_rmse"].to_numpy(dtype=np.float64)
    summary = {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_better": float(np.mean(deltas < 0.0)),
    }
    return replicates, summary


def select_training_fingerprint(
    *,
    reporting_fingerprint: str,
    existing_manifest: dict | None,
    new_fold_runtimes: dict[int, dict],
    expected_contract: dict,
) -> tuple[str, str]:
    """区分报告代码指纹与可安全复用的历史训练指纹。"""

    if existing_manifest is None:
        return reporting_fingerprint, "new_training_uses_reporting_fingerprint"
    for field_name, expected_value in expected_contract.items():
        if existing_manifest.get(field_name) != expected_value:
            raise ValueError(f"existing manifest {field_name} 与当前合同不一致")

    stored_training_fingerprint = str(
        existing_manifest.get(
            "training_fingerprint",
            existing_manifest.get("audit_fingerprint", ""),
        )
    )
    if len(stored_training_fingerprint) != 64:
        raise ValueError("existing manifest 缺少有效 training fingerprint")
    if stored_training_fingerprint == reporting_fingerprint:
        return stored_training_fingerprint, "current_training_fingerprint"

    if set(new_fold_runtimes) != set(NEW_FOLDS):
        raise ValueError("报告代码变化时必须已有完整 folds 2–4 才能沿用训练指纹")
    for fold_id in NEW_FOLDS:
        runtime = new_fold_runtimes[fold_id]
        if int(runtime.get("fold", -1)) != fold_id:
            raise ValueError(f"existing fold {fold_id} runtime fold 不一致")
        if runtime.get("fingerprint") != stored_training_fingerprint:
            raise ValueError(f"existing fold {fold_id} training fingerprint 不一致")
    return stored_training_fingerprint, "legacy_completed_folds_reuse"


def preserve_or_write_prediction_table(
    predictions: pd.DataFrame,
    prediction_path: Path,
) -> dict:
    """已有预测逐位相同则保留原字节；仅在文件不存在时首次写入。"""

    if prediction_path.is_file():
        sha256_before = file_sha256(prediction_path)
        existing_predictions = pd.read_parquet(prediction_path)
        try:
            pd.testing.assert_frame_equal(
                existing_predictions.reset_index(drop=True),
                predictions.reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
                check_like=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"已有预测与本次纯复用汇总逐位不一致，禁止覆盖：{prediction_path}"
            ) from error
        sha256_after = file_sha256(prediction_path)
        if sha256_after != sha256_before:
            raise ValueError(f"只读验证期间预测文件 SHA 发生变化：{prediction_path}")
        return {
            "path": str(prediction_path.resolve()),
            "action": "preserved_existing_identical",
            "sha256_before": sha256_before,
            "sha256_after": sha256_after,
            "rows": int(len(existing_predictions)),
        }

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    written_hash = file_sha256(prediction_path)
    return {
        "path": str(prediction_path.resolve()),
        "action": "created_new_prediction_file",
        "sha256_before": None,
        "sha256_after": written_hash,
        "rows": int(len(predictions)),
    }


def build_diagnostic_reporting_artifacts(
    comparison_metrics: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """构造 B00 逐折、全体切片和 diagnostic-only 负对照说明。"""

    if comparison_metrics.get("baseline_id") != "B00_simple_lgbm_v1":
        raise ValueError("诊断报告只允许固定 B00 baseline")
    per_fold_rows: list[dict] = []
    for fold_summary in comparison_metrics.get("folds", []):
        per_fold_rows.append(
            {
                "fold": int(fold_summary["fold"]),
                "rows": int(fold_summary["rows"]),
                "wells": int(fold_summary["wells"]),
                "rf01a_micro_rmse": float(fold_summary["micro_rmse"]),
                "b00_micro_rmse": float(fold_summary["baseline_micro_rmse"]),
                "rf01a_minus_b00_micro_rmse": float(
                    fold_summary.get(
                        "rf01a_minus_b00_micro_rmse",
                        fold_summary["micro_rmse_delta_vs_baseline"],
                    )
                ),
                "macro_well_rmse": float(fold_summary["macro_well_rmse"]),
                "median_well_rmse": float(fold_summary["median_well_rmse"]),
                "p90_well_rmse": float(fold_summary["p90_well_rmse"]),
                "worst_well_rmse": float(fold_summary["worst_well_rmse"]),
                "well_win_rate_vs_B00": float(fold_summary["well_win_rate"]),
            }
        )
    per_fold = pd.DataFrame(per_fold_rows).sort_values("fold").reset_index(drop=True)

    overall = comparison_metrics["overall"]
    slice_metrics = pd.DataFrame(
        [
            {
                "slice": "all_natural_hidden_rows",
                "status": "full_population_only",
                "applicability": "no_pre_registered_additional_slices",
                "reason": (
                    "RF01 stability audit is diagnostic-only and registered no extra slice; "
                    "this row reports the complete fixed evaluation population."
                ),
                "rows": int(overall["rows"]),
                "wells": int(overall["wells"]),
                "rf01a_micro_rmse": float(overall["micro_rmse"]),
                "b00_micro_rmse": float(overall["baseline_micro_rmse"]),
                "rf01a_minus_b00_micro_rmse": float(
                    overall["micro_rmse_delta_vs_baseline"]
                ),
                "macro_well_rmse": float(overall["macro_well_rmse"]),
                "median_well_rmse": float(overall["median_well_rmse"]),
                "p90_well_rmse": float(overall["p90_well_rmse"]),
                "worst_well_rmse": float(overall["worst_well_rmse"]),
                "well_win_rate_vs_B00": float(overall["well_win_rate"]),
            }
        ]
    )
    negative_control = {
        "status": "not_applicable",
        "experiment_type": "diagnostic_only",
        "extra_randomized_control_run": False,
        "reason": (
            "This audit adds no feature group and only completes fixed RF01a folds 2-4; "
            "no additional shuffled or randomized negative control was pre-registered."
        ),
        "source_experiment": SOURCE_EXPERIMENT_ID,
        "original_rf01_decision": "not_promoted_and_unchanged",
        "must_not_be_used_for_promotion": True,
    }
    return per_fold, slice_metrics, negative_control


def build_leakage_tests_document(
    *,
    eval_row_hash: str,
    feature_lineage: dict[str, dict],
    cache_manifest: dict,
    prediction_file_records: dict[str, dict],
    negative_control: dict,
) -> dict:
    """按固定八个检查名记录本诊断实际执行或明确未重跑的证据。"""

    raw_source_text = " ".join(
        str(source)
        for lineage in feature_lineage.values()
        for source in lineage.get("raw_source", [])
    ).lower()
    surface_absent = "surface" not in raw_source_text and "contact" not in raw_source_text
    pf_beam_absent = all(
        lineage.get("uses_PF_Beam") is False
        for lineage in feature_lineage.values()
    )
    fold_intersections = {
        int(item["outer_fold"]): int(item["train_validation_intersection_count"])
        for item in cache_manifest.get("fold_artifacts", [])
    }
    outer_fold_passed = bool(fold_intersections) and all(
        intersection_count == 0
        for intersection_count in fold_intersections.values()
    )
    manifest_row_hash = str(cache_manifest.get("full_prediction_row_hash", ""))
    row_hash_passed = manifest_row_hash == str(eval_row_hash)
    prediction_hashes_unchanged = bool(prediction_file_records) and all(
        record.get("sha256_before") is not None
        and record.get("sha256_before") == record.get("sha256_after")
        for record in prediction_file_records.values()
    )

    checks = {
        "hidden TVT deletion invariance": {
            "status": "not_rerun_diagnostic_reuse",
            "reason": (
                "This diagnostic directly reuses the frozen RF01a feature cache and does "
                "not rebuild features after deleting hidden TVT. No new feature is introduced."
            ),
        },
        "hidden TVT mutation invariance": {
            "status": "not_rerun_diagnostic_reuse",
            "reason": (
                "This diagnostic directly reuses the frozen RF01a feature cache and does "
                "not rebuild features after mutating hidden TVT. No result is fabricated."
            ),
        },
        "surface deletion invariance": {
            "status": "passed_static_lineage" if surface_absent else "failed",
            "evidence": {"surface_or_contact_source_present": not surface_absent},
        },
        "outer-fold source exclusion": {
            "status": "passed" if outer_fold_passed else "failed",
            "evidence": {"train_validation_intersection_count_by_fold": fold_intersections},
        },
        "PF cache fold match": {
            "status": "not_applicable" if pf_beam_absent else "failed",
            "reason": "All 17 fixed features declare uses_PF_Beam=false.",
        },
        "row hash match": {
            "status": "passed" if row_hash_passed else "failed",
            "evidence": {
                "eval_row_hash": str(eval_row_hash),
                "full_prediction_row_hash": manifest_row_hash,
            },
        },
        "fixed-seed reproducibility": {
            "status": (
                "passed_prediction_sha_unchanged"
                if prediction_hashes_unchanged
                else "failed"
            ),
            "evidence": prediction_file_records,
            "note": "No model was retrained; existing prediction files were validated and preserved.",
        },
        "negative-control status": {
            "status": str(negative_control.get("status", "missing")),
            "reason": str(negative_control.get("reason", "")),
        },
    }
    if tuple(checks) != LEAKAGE_TEST_NAMES:
        raise ValueError("leakage check 名称或顺序与固定合同不一致")
    return {
        "experiment_id": AUDIT_EXPERIMENT_ID,
        "experiment_type": "diagnostic_only",
        "checks": checks,
    }


def _compute_identifier_list_hash(values: list[str]) -> str:
    """对排序去重后的标识符列表计算确定性 SHA-256。"""

    normalized_values = sorted(set(str(value) for value in values))
    encoded_values = json.dumps(
        normalized_values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_values).hexdigest()


def build_fold_cache_manifest(
    *,
    fold_id: int,
    artifact_source: str,
    registry: pd.DataFrame,
    prediction: pd.DataFrame,
    runtime: dict,
    fold_dir: Path,
) -> dict:
    """构造单折实际井集合、评价行与四类文件 hash 清单。"""

    train_wells = sorted(
        registry.loc[registry["fold"].astype(int) != int(fold_id), "well_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    validation_wells = sorted(
        registry.loc[registry["fold"].astype(int) == int(fold_id), "well_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    intersection_count = len(set(train_wells) & set(validation_wells))
    prediction_wells = set(prediction["well_id"].astype(str).unique())
    if prediction_wells != set(validation_wells):
        raise ValueError(f"fold {fold_id} prediction wells 与 registry 验证井不一致")
    if int(runtime.get("fold", -1)) != int(fold_id):
        raise ValueError(f"fold {fold_id} runtime fold 不一致")

    file_names = (
        "predictions.parquet",
        "runtime.json",
        "feature_importance.csv",
        "model.txt",
    )
    file_hashes: dict[str, str] = {}
    file_paths: dict[str, str] = {}
    for file_name in file_names:
        file_path = fold_dir / file_name
        _require_file(file_path, f"fold {fold_id} cache manifest {file_name}")
        file_hashes[file_name] = file_sha256(file_path)
        file_paths[file_name] = str(file_path.resolve())

    return {
        "outer_fold": int(fold_id),
        "artifact_source": str(artifact_source),
        "fingerprint": str(runtime.get("fingerprint", "")),
        "fit_wells_count": len(train_wells),
        "fit_wells_hash": _compute_identifier_list_hash(train_wells),
        "validation_wells_count": len(validation_wells),
        "validation_wells_hash": _compute_identifier_list_hash(validation_wells),
        "train_validation_intersection_count": intersection_count,
        "prediction_rows": int(len(prediction)),
        "prediction_wells": int(prediction["well_id"].nunique()),
        "prediction_row_hash": compute_row_key_hash(prediction),
        "file_paths": file_paths,
        "file_sha256": file_hashes,
    }


def classify_fold_pattern(fold_deltas: list[float]) -> str:
    """按五个 RF01a-minus-B00 折差值返回预注册的稳定性模式。"""

    # 五个位置依次代表 fold 0 到 fold 4，数量不对时不能猜测折含义。
    if len(fold_deltas) != 5:
        raise ValueError("折差值必须按 fold 0–4 提供五个数")

    # 转成普通浮点数，后续只判断相对零的方向，不修改原始差值。
    normalized_deltas = [float(value) for value in fold_deltas]

    # NaN 或 Inf 没有可解释方向，必须在写结论前停止。
    if not all(math.isfinite(value) for value in normalized_deltas):
        raise ValueError("折差值含 NaN 或 Inf")

    # 负数表示 RF01a 优于 B00；只有 fold 1 为正时，它是唯一反向折。
    fold1_is_only_reverse = normalized_deltas[1] > 0.0 and all(
        normalized_deltas[index] < 0.0 for index in (0, 2, 3, 4)
    )
    if fold1_is_only_reverse:
        return "fold1是唯一反向折"

    # 只有 fold 0 为负时，它是五折中唯一出现改善的正向折。
    fold0_is_only_positive = normalized_deltas[0] < 0.0 and all(
        normalized_deltas[index] > 0.0 for index in (1, 2, 3, 4)
    )
    if fold0_is_only_positive:
        return "fold0是唯一正向折"

    # 包含零或任何其他正负组合时，都只能报告其余折表现混合。
    return "其余折表现混合"


def compute_row_key_hash(frame: pd.DataFrame) -> str:
    """按固定行键排序并返回与输入行顺序无关的 SHA-256。"""

    # 先检查 schema；缺任一键时不能用剩余列生成貌似有效的指纹。
    missing_columns = set(ROW_KEY_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"行键表缺少列：{sorted(missing_columns)}")

    # 空集合和重复键都不是合法评价行合同，不能生成可被误用的哈希。
    if frame.empty:
        raise ValueError("行键表为空")
    if frame.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise ValueError("行键表含重复行键")

    # 只复制三列，确保非键特征或预测值不会影响评价行集合指纹。
    normalized_keys = frame.loc[:, list(ROW_KEY_COLUMNS)].copy()
    if normalized_keys.isna().any().any():
        raise ValueError("行键含缺失值")

    # well_id 统一为字符串；类别 dtype 与普通字符串应得到同一个指纹。
    normalized_keys["well_id"] = normalized_keys["well_id"].astype("string")

    # 分隔符只用于内部规范化文本，井号含这些字符时拒绝而不是产生歧义。
    forbidden_separator_pattern = "[\\x1e\\x1f\\r\\n]"
    if normalized_keys["well_id"].str.contains(
        forbidden_separator_pattern,
        regex=True,
    ).any():
        raise ValueError("well_id 含行键哈希保留分隔符")

    # fold 与 row_index 必须是有限整数，禁止 1.5 被静默截成 1。
    for numeric_column in ("fold", "row_index"):
        numeric_values = pd.to_numeric(normalized_keys[numeric_column], errors="raise")
        numeric_array = numeric_values.to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_array).all():
            raise ValueError(f"行键列 {numeric_column} 含 NaN 或 Inf")
        if not np.equal(numeric_array, np.floor(numeric_array)).all():
            raise ValueError(f"行键列 {numeric_column} 含非整数")
        normalized_keys[numeric_column] = numeric_array.astype(np.int64)

    # 稳定排序让相同行集合在任意输入顺序下生成完全相同的字节序列。
    normalized_keys = normalized_keys.sort_values(
        list(ROW_KEY_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)

    # 版本头把当前编码合同与未来可能的新算法隔离开。
    digest = hashlib.sha256(b"rogii-row-key-v1\n")

    # 分块编码避免一次构造 378 万行的巨大 Python 字符串。
    chunk_size = 100_000
    for start_index in range(0, len(normalized_keys), chunk_size):
        key_chunk = normalized_keys.iloc[start_index : start_index + chunk_size]
        serialized_rows = key_chunk["well_id"].str.cat(
            [
                key_chunk["fold"].astype(str),
                key_chunk["row_index"].astype(str),
            ],
            sep="\x1f",
        )
        serialized_chunk = "\x1e".join(serialized_rows.tolist()) + "\x1e"
        digest.update(serialized_chunk.encode("utf-8"))

    # 十六进制形式固定为 64 字符，便于写入 JSON 清单和人工比对。
    return digest.hexdigest()


def validate_config_contract(
    audit_config: dict,
    source_config: dict,
    model_config: dict,
    *,
    actual_fold_hash: str,
    actual_model_config_hash: str,
) -> list[str]:
    """验证审计、RF01a、模型和 fold 配置均未偏离预登记合同。"""

    # 审计身份与诊断属性必须固定，避免把脚本复用成新的候选实验。
    if audit_config.get("experiment_id") != AUDIT_EXPERIMENT_ID:
        raise ValueError("审计 experiment_id 与预登记不一致")
    if audit_config.get("experiment_type") != "diagnostic_only":
        raise ValueError("审计必须保持 diagnostic_only")
    if audit_config.get("source_experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("审计 source RF01a 实验不一致")
    if audit_config.get("original_rf01_decision") != "not_promoted_and_unchanged":
        raise ValueError("原 RF01 不晋级决定未锁定")

    # 两个布尔开关只能显式为 false；缺字段也不能被当成默认允许。
    if audit_config.get("allow_reselection") is not False:
        raise ValueError("审计禁止重新选择 RF01 版本")
    if audit_config.get("allow_parameter_change") is not False:
        raise ValueError("审计禁止调参或改变模型参数")
    if tuple(audit_config.get("pre_registered_selection_reasons", [])) != (
        PRE_REGISTERED_SELECTION_REASONS
    ):
        raise ValueError("审计四条预注册选择理由不一致")

    # 复用折和新折必须与公开常量逐项相同且互斥。
    if tuple(audit_config.get("reused_folds", [])) != REUSED_FOLDS:
        raise ValueError("reused_folds 必须固定为 folds 0–1")
    if tuple(audit_config.get("new_folds", [])) != NEW_FOLDS:
        raise ValueError("new_folds 必须固定为 folds 2–4")
    if not set(REUSED_FOLDS).isdisjoint(NEW_FOLDS):
        raise ValueError("复用折与新折发生交集")

    # 全量井数和评价行数是 B00/RF01a 的固定合同，不接受小范围替代运行。
    if int(audit_config.get("expected_wells", -1)) != EXPECTED_WELLS:
        raise ValueError("审计 expected_wells 不是固定 773 井")
    if int(audit_config.get("expected_rows", -1)) != EXPECTED_ROWS:
        raise ValueError("审计 expected_rows 不是固定 3,783,989 行")
    if int(source_config.get("expected_wells", -1)) != EXPECTED_WELLS:
        raise ValueError("RF01a 源配置井数与审计不一致")
    if int(source_config.get("expected_rows", -1)) != EXPECTED_ROWS:
        raise ValueError("RF01a 源配置行数与审计不一致")

    # 路径逐项锁死，特别是 source cache 不得改成其他 RF01 特征缓存。
    expected_audit_paths = {
        "source_config": EXPECTED_SOURCE_CONFIG,
        "source_artifact_dir": EXPECTED_SOURCE_ARTIFACT_DIR,
        "source_feature_cache": EXPECTED_SOURCE_FEATURE_CACHE,
        "model_config": EXPECTED_MODEL_CONFIG,
        "fold_registry": EXPECTED_FOLD_REGISTRY,
        "b00_prediction_file": EXPECTED_B00_PREDICTIONS,
    }
    for config_key, expected_path in expected_audit_paths.items():
        if audit_config.get(config_key) != expected_path:
            raise ValueError(f"审计路径 {config_key} 与预登记不一致")

    # 源配置身份、版本和 17 列顺序必须全部等于实验卡。
    if source_config.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("RF01a 源 experiment_id 不一致")
    if source_config.get("feature_version") != SOURCE_FEATURE_VERSION:
        raise ValueError("RF01a source feature_version 不一致")
    source_feature_columns = source_config.get("feature_columns")
    if source_feature_columns != list(RF01A_FEATURE_COLUMNS):
        raise ValueError("RF01a 固定 17 列及顺序不一致")
    if source_config.get("allow_nan_features") != list(RF01A_ALLOWED_NAN_FEATURES):
        raise ValueError("RF01a 允许 NaN 列与固定五个 slope 不一致")
    if source_config.get("target") != "target_delta":
        raise ValueError("RF01a 预测目标不是固定 target_delta")
    if source_config.get("final_prediction") != (
        "last_visible_tvt_plus_predicted_delta"
    ):
        raise ValueError("RF01a 绝对 TVT 还原方式不一致")

    # 审计和源配置必须指向同一模型文件，且模型文件内容与冻结版本完全一致。
    if source_config.get("model_config") != audit_config.get("model_config"):
        raise ValueError("RF01a 与审计模型配置路径不一致")
    if source_config.get("model_config") != EXPECTED_MODEL_CONFIG:
        raise ValueError("RF01a 模型配置不是冻结版本")
    if actual_model_config_hash.lower() != EXPECTED_MODEL_CONFIG_HASH:
        raise ValueError("冻结模型配置内容 hash 不一致，禁止调参")
    if model_config.get("config_version") != "lgbm_feature_baseline_v1":
        raise ValueError("冻结模型 config_version 不一致")
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("冻结模型 family 不一致")
    if model_config.get("training_policy", {}).get("early_stopping") is not False:
        raise ValueError("冻结模型不得启用 early stopping")

    # 关键训练参数再做可读检查；完整 JSON 仍由上面的内容 hash 锁定。
    model_params = model_config.get("params", {})
    if int(model_params.get("n_estimators", -1)) != 1734:
        raise ValueError("冻结模型树数不是 1,734")
    for seed_name in (
        "random_state",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
    ):
        if int(model_params.get(seed_name, -1)) != 29:
            raise ValueError(f"冻结模型 {seed_name} 不是 seed 29")

    # fold 路径和内容 hash 必须同时与源配置、审计配置及硬编码合同一致。
    if source_config.get("fold_registry") != audit_config.get("fold_registry"):
        raise ValueError("RF01a 与审计 fold 注册表路径不一致")
    fold_hashes = {
        str(audit_config.get("fold_registry_sha256", "")).lower(),
        str(source_config.get("fold_registry_sha256", "")).lower(),
        str(actual_fold_hash).lower(),
        EXPECTED_FOLD_REGISTRY_HASH,
    }
    if len(fold_hashes) != 1:
        raise ValueError("固定 fold hash 不一致")

    # 返回普通 list，直接作为 train_fold 的模型列顺序。
    return list(RF01A_FEATURE_COLUMNS)


def validate_evaluation_alignment(
    feature_table: pd.DataFrame,
    b00_predictions: pd.DataFrame,
    *,
    expected_rows: int,
    expected_wells: int,
) -> str:
    """验证 RF01a 缓存与 B00 的评价行键、fold 和 target 完全一致。"""

    # 两侧 schema 分开检查，错误信息明确指出缺少的输入来源。
    feature_required = set(ROW_KEY_COLUMNS) | {"target_tvt"}
    b00_required = feature_required | {"pred_tvt"}
    missing_feature_columns = feature_required - set(feature_table.columns)
    missing_b00_columns = b00_required - set(b00_predictions.columns)
    if missing_feature_columns:
        raise ValueError(f"RF01a 特征缓存缺少列：{sorted(missing_feature_columns)}")
    if missing_b00_columns:
        raise ValueError(f"B00 预测缺少列：{sorted(missing_b00_columns)}")

    # 行数与井数在哈希前检查，便于快速定位覆盖合同异常。
    if len(feature_table) != int(expected_rows):
        raise ValueError("RF01a 特征缓存评价行数不一致")
    if len(b00_predictions) != int(expected_rows):
        raise ValueError("B00 预测评价行数不一致")
    if int(feature_table["well_id"].nunique()) != int(expected_wells):
        raise ValueError("RF01a 特征缓存评价井数不一致")
    if int(b00_predictions["well_id"].nunique()) != int(expected_wells):
        raise ValueError("B00 预测评价井数不一致")

    # 三列联合键必须唯一；fold 被纳入键，但后面仍逐位核对。
    if feature_table.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise ValueError("RF01a 特征缓存含重复行键")
    if b00_predictions.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise ValueError("B00 预测含重复行键")

    # 审计只接受固定五折，不能在缺折时仍计算一个部分行集 hash。
    expected_fold_set = set(REUSED_FOLDS + NEW_FOLDS)
    feature_fold_set = set(feature_table["fold"].astype(int).unique())
    b00_fold_set = set(b00_predictions["fold"].astype(int).unique())
    if feature_fold_set != expected_fold_set or b00_fold_set != expected_fold_set:
        raise ValueError("RF01a 或 B00 不是固定 folds 0–4")

    # 目标和 B00 预测必须有限；任何 NaN/Inf 都不能进入对齐或评分。
    feature_targets = feature_table["target_tvt"].to_numpy(dtype=np.float64)
    b00_targets = b00_predictions["target_tvt"].to_numpy(dtype=np.float64)
    b00_values = b00_predictions["pred_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_targets).all():
        raise ValueError("RF01a target_tvt 含 NaN 或 Inf")
    if not np.isfinite(b00_targets).all():
        raise ValueError("B00 target_tvt 含 NaN 或 Inf")
    if not np.isfinite(b00_values).all():
        raise ValueError("B00 pred_tvt 含 NaN 或 Inf")

    # 分别从两侧独立计算行键 SHA-256，禁止用一侧排序结果代替另一侧指纹。
    feature_row_hash = compute_row_key_hash(feature_table)
    b00_row_hash = compute_row_key_hash(b00_predictions)
    if feature_row_hash != b00_row_hash:
        raise ValueError("RF01a 特征缓存与 B00 行键 hash 不一致")

    # 按三列固定排序后再做逐位键比较，避免仅依赖哈希碰撞假设。
    alignment_columns = list(ROW_KEY_COLUMNS) + ["target_tvt"]
    feature_aligned = feature_table.loc[:, alignment_columns].sort_values(
        list(ROW_KEY_COLUMNS)
    ).reset_index(drop=True)
    b00_aligned = b00_predictions.loc[:, alignment_columns].sort_values(
        list(ROW_KEY_COLUMNS)
    ).reset_index(drop=True)
    for key_column in ROW_KEY_COLUMNS:
        feature_key_values = feature_aligned[key_column].astype(str).to_numpy()
        b00_key_values = b00_aligned[key_column].astype(str).to_numpy()
        if not np.array_equal(feature_key_values, b00_key_values):
            raise ValueError(f"RF01a 与 B00 行键列 {key_column} 逐位不一致")

    # target_tvt 使用精确逐位比较，不以容差掩盖评价标签来源漂移。
    aligned_feature_targets = feature_aligned["target_tvt"].to_numpy(
        dtype=np.float64
    )
    aligned_b00_targets = b00_aligned["target_tvt"].to_numpy(dtype=np.float64)
    if not np.array_equal(aligned_feature_targets, aligned_b00_targets):
        raise ValueError("RF01a 与 B00 target_tvt 逐位不一致")

    # 该值来自真实固定行键，只在两侧完全一致后才允许写入运行清单。
    return feature_row_hash


def validate_well_fold_assignments(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    """逐井验证特征缓存的 fold 与固定注册表完全一致。"""

    # 两侧至少需要井号和 fold；缺列时不能只比较逐折行数。
    required_columns = {"well_id", "fold"}
    missing_feature_columns = required_columns - set(feature_table.columns)
    missing_registry_columns = required_columns - set(registry.columns)
    if missing_feature_columns:
        raise ValueError(f"特征缓存缺少逐井 fold 列：{sorted(missing_feature_columns)}")
    if missing_registry_columns:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing_registry_columns)}")

    # 一口井在特征缓存中只能有一个 fold，否则它会同时出现在不同验证折。
    feature_fold_counts = feature_table.groupby("well_id", observed=True)["fold"].nunique()
    if int(feature_fold_counts.max()) != 1:
        raise ValueError("特征缓存同一口井出现在多个 fold")
    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")

    # 各取一井一行并按字符串井号对齐，防止类别 dtype 影响比较。
    feature_well_folds = (
        feature_table.loc[:, ["well_id", "fold"]]
        .assign(well_id=lambda frame: frame["well_id"].astype("string"))
        .drop_duplicates("well_id")
        .sort_values("well_id")
        .reset_index(drop=True)
    )
    registry_well_folds = (
        registry.loc[:, ["well_id", "fold"]]
        .assign(well_id=lambda frame: frame["well_id"].astype("string"))
        .sort_values("well_id")
        .reset_index(drop=True)
    )

    # 井集合不同或任一折号不同都属于全局合同异常。
    if not np.array_equal(
        feature_well_folds["well_id"].to_numpy(),
        registry_well_folds["well_id"].to_numpy(),
    ):
        raise ValueError("特征缓存与注册表逐井 fold 的井集合不一致")
    if not np.array_equal(
        feature_well_folds["fold"].to_numpy(dtype=np.int64),
        registry_well_folds["fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("特征缓存与注册表逐井 fold 分配不一致")


def validate_reused_fold_runtime_contract(
    cache_metadata: dict,
    fold_runtimes: dict[int, dict],
) -> str:
    """验证 RF01a folds 0–1 runtime 与源 feature cache 指纹一致。"""

    # cache metadata 是源训练血缘的锚点，缺少指纹时不能复用既有折。
    source_fingerprint = str(cache_metadata.get("fingerprint", ""))
    if len(source_fingerprint) != 64:
        raise ValueError("RF01a feature cache metadata 缺少有效 fingerprint")

    # 只接受恰好两份源 runtime，禁止把其他 RF01 折混入。
    if set(fold_runtimes) != set(REUSED_FOLDS):
        raise ValueError("RF01a 源 runtime 必须恰好包含 folds 0–1")

    for fold_id in REUSED_FOLDS:
        runtime = fold_runtimes[fold_id]
        if int(runtime.get("fold", -1)) != fold_id:
            raise ValueError(f"RF01a fold {fold_id} runtime 的 fold 值不一致")
        if runtime.get("fingerprint") != source_fingerprint:
            raise ValueError(
                f"RF01a fold {fold_id} runtime fingerprint 与 cache metadata 不一致"
            )
        if int(runtime.get("features", -1)) != len(RF01A_FEATURE_COLUMNS):
            raise ValueError(f"RF01a fold {fold_id} runtime 特征数不是 17")
        if int(runtime.get("trees", -1)) != 1734:
            raise ValueError(f"RF01a fold {fold_id} runtime 树数不是 1,734")

    # 返回唯一源指纹，供审计指纹和运行清单显式记录。
    return source_fingerprint


def build_fixed_fold_directories(
    source_artifact_dir: Path,
    audit_artifact_dir: Path,
) -> dict[int, Path]:
    """返回 folds 0–1 源目录与 folds 2–4 审计目录的固定映射。"""

    # 显式逐折构造，避免 range(5) 意外把旧折指向审计目录。
    fold_directories = {
        0: source_artifact_dir / "fold_0",
        1: source_artifact_dir / "fold_1",
        2: audit_artifact_dir / "fold_2",
        3: audit_artifact_dir / "fold_3",
        4: audit_artifact_dir / "fold_4",
    }
    return fold_directories


def _require_file(path: Path, description: str) -> None:
    """要求合同文件存在且非空，不创建或修补输入。"""

    if not path.is_file():
        raise FileNotFoundError(f"缺少{description}：{path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{description}为空文件：{path}")


def read_source_runtime_contract(
    cache_metadata_path: Path,
    source_artifact_dir: Path,
) -> tuple[dict, dict[int, dict], str]:
    """从历史 cache metadata 与 folds 0–1 runtime 取得源指纹。"""

    # 历史源指纹由当时实际写入的三份清单共同证明，不用当前过宽 runner 倒推。
    _require_file(cache_metadata_path, "RF01a feature cache metadata")
    cache_metadata = read_json(cache_metadata_path)

    # 只读取小型 runtime；predictions/importance/model 会在完整折 bundle 中另验。
    fold_runtimes: dict[int, dict] = {}
    for fold_id in REUSED_FOLDS:
        runtime_path = source_artifact_dir / f"fold_{fold_id}" / "runtime.json"
        _require_file(runtime_path, f"RF01a fold {fold_id} runtime")
        fold_runtimes[fold_id] = read_json(runtime_path)

    # 统一函数要求 cache 与两个 runtime 的 fingerprint、折号、树数和特征数一致。
    source_fingerprint = validate_reused_fold_runtime_contract(
        cache_metadata,
        fold_runtimes,
    )
    return cache_metadata, fold_runtimes, source_fingerprint


def _validate_fold_artifact_bundle(
    fold_id: int,
    fold_dir: Path,
    expected_rows: pd.DataFrame,
    *,
    expected_fingerprint: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """读取并验证一折 prediction/runtime/importance/model 四类产物。"""

    # 每折固定需要四类文件；任何一项缺失都不能视为可复用完成折。
    prediction_path = fold_dir / "predictions.parquet"
    runtime_path = fold_dir / "runtime.json"
    importance_path = fold_dir / "feature_importance.csv"
    model_path = fold_dir / "model.txt"
    _require_file(prediction_path, f"fold {fold_id} predictions")
    _require_file(runtime_path, f"fold {fold_id} runtime")
    _require_file(importance_path, f"fold {fold_id} feature importance")
    _require_file(model_path, f"fold {fold_id} model")

    # 文件存在后才读取；本函数只读，不会调用 LightGBM 或写回原目录。
    prediction = pd.read_parquet(prediction_path)
    runtime = read_json(runtime_path)
    importance = pd.read_csv(importance_path)

    # runtime 必须属于请求折、相同指纹、17 列和固定 1,734 棵树。
    if int(runtime.get("fold", -1)) != int(fold_id):
        raise ValueError(f"fold {fold_id} runtime 的 fold 值不一致")
    if runtime.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"fold {fold_id} runtime fingerprint 不一致")
    if int(runtime.get("features", -1)) != len(feature_columns):
        raise ValueError(f"fold {fold_id} runtime 特征数不一致")
    if int(runtime.get("trees", -1)) != 1734:
        raise ValueError(f"fold {fold_id} runtime 树数不是 1,734")
    try:
        runtime_seconds = float(runtime["seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"fold {fold_id} runtime seconds 缺失或无效") from error
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0.0:
        raise ValueError(f"fold {fold_id} runtime seconds 必须是非负有限数")

    # 预测 schema、fold、覆盖和唯一性必须与 feature cache 的该折切片一致。
    prediction_required = set(ROW_KEY_COLUMNS) | {
        "target_tvt",
        "carry_tvt",
        "pred_tvt",
    }
    missing_prediction_columns = prediction_required - set(prediction.columns)
    if missing_prediction_columns:
        raise ValueError(
            f"fold {fold_id} predictions 缺少列：{sorted(missing_prediction_columns)}"
        )
    if set(prediction["fold"].astype(int).unique()) != {int(fold_id)}:
        raise ValueError(f"fold {fold_id} predictions 的 fold 值不一致")
    if len(prediction) != len(expected_rows):
        raise ValueError(f"fold {fold_id} predictions 行数不一致")
    expected_wells = int(expected_rows["well_id"].nunique())
    if int(prediction["well_id"].nunique()) != expected_wells:
        raise ValueError(f"fold {fold_id} predictions 井数不一致")
    if prediction.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise ValueError(f"fold {fold_id} predictions 含重复行键")
    scoring_values = prediction[["target_tvt", "carry_tvt", "pred_tvt"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(scoring_values).all():
        raise ValueError(
            f"fold {fold_id} predictions 的 target/carry/pred 含 NaN 或 Inf"
        )

    # runtime 中的验证覆盖也必须逐项等于实际文件。
    if int(runtime.get("validation_rows", -1)) != len(prediction):
        raise ValueError(f"fold {fold_id} runtime validation_rows 不一致")
    if int(runtime.get("validation_wells", -1)) != expected_wells:
        raise ValueError(f"fold {fold_id} runtime validation_wells 不一致")

    # 评价行键和 target 精确对齐，防止正确折号下仍混入错误行集合。
    if compute_row_key_hash(prediction) != compute_row_key_hash(expected_rows):
        raise ValueError(f"fold {fold_id} predictions 与 feature cache 行键不一致")
    scoring_alignment_columns = list(ROW_KEY_COLUMNS) + [
        "target_tvt",
        "carry_tvt",
    ]
    aligned_prediction = prediction.loc[:, scoring_alignment_columns].sort_values(
        list(ROW_KEY_COLUMNS)
    )
    aligned_expected = expected_rows.loc[:, scoring_alignment_columns].sort_values(
        list(ROW_KEY_COLUMNS)
    )
    for scoring_column in ("target_tvt", "carry_tvt"):
        prediction_values = aligned_prediction[scoring_column].to_numpy(
            dtype=np.float64
        )
        expected_values = aligned_expected[scoring_column].to_numpy(dtype=np.float64)
        if not np.array_equal(prediction_values, expected_values):
            raise ValueError(
                f"fold {fold_id} predictions {scoring_column} 不一致"
            )

    # importance 必须恰好一列一行，且 gain/split 都是有限数。
    importance_required = {"feature", "gain", "split"}
    missing_importance_columns = importance_required - set(importance.columns)
    if missing_importance_columns:
        raise ValueError(
            f"fold {fold_id} importance 缺少列：{sorted(missing_importance_columns)}"
        )
    if importance["feature"].duplicated().any():
        raise ValueError(f"fold {fold_id} importance 含重复特征")
    if set(importance["feature"]) != set(feature_columns):
        raise ValueError(f"fold {fold_id} importance 与固定 17 列不一致")
    if not np.isfinite(
        importance[["gain", "split"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(f"fold {fold_id} importance 含 NaN 或 Inf")

    # 返回内存对象供预检清单或最终五折汇总复用。
    return prediction, runtime, importance


def validate_existing_new_fold_reuse(
    feature_table: pd.DataFrame,
    audit_artifact_dir: Path,
    *,
    audit_fingerprint: str,
    feature_columns: list[str],
) -> list[int]:
    """在训练前完整验证会被通用 runner 复用的已有新折。"""

    # train_fold 只有在 prediction/model/runtime 三者都存在时才考虑复用。
    reusable_folds: list[int] = []
    for fold_id in NEW_FOLDS:
        fold_dir = audit_artifact_dir / f"fold_{fold_id}"
        prediction_path = fold_dir / "predictions.parquet"
        model_path = fold_dir / "model.txt"
        runtime_path = fold_dir / "runtime.json"
        generic_reuse_files_exist = all(
            path.is_file() for path in (prediction_path, model_path, runtime_path)
        )
        if not generic_reuse_files_exist:
            continue

        # runtime 无法解析时 generic runner 也会失败；这里提前到任何训练之前。
        runtime = read_json(runtime_path)
        if runtime.get("fingerprint") != audit_fingerprint:
            # 指纹不同的旧文件不会被复用，train_fold 会以当前合同重新训练覆盖。
            continue

        # 匹配指纹意味着 generic runner 会直接返回，因此必须额外检查 importance、
        # fold 值、评价行、target、有限预测和 model，不能等其他折训练后再发现损坏。
        expected_fold_rows = feature_table.loc[feature_table["fold"] == fold_id]
        _validate_fold_artifact_bundle(
            fold_id,
            fold_dir,
            expected_fold_rows,
            expected_fingerprint=audit_fingerprint,
            feature_columns=feature_columns,
        )
        reusable_folds.append(fold_id)

    # 返回列表仅用于日志；顺序始终跟随 NEW_FOLDS=(2,3,4)。
    return reusable_folds


def _build_reporting_fingerprint(
    audit_config: dict,
    source_config: dict,
    model_config: dict,
    *,
    registry_hash: str,
    source_feature_cache: Path,
    source_fingerprint: str,
) -> str:
    """生成报告/编排代码指纹；它不自动覆盖已完成折的训练指纹。"""

    # 文件大小和纳秒修改时间按计划进入指纹，识别被替换的源 cache。
    feature_cache_stat = source_feature_cache.stat()
    fingerprint_payload = {
        "audit_config": audit_config,
        "source_config": source_config,
        "model_config": model_config,
        "fold_registry_sha256": registry_hash,
        "source_feature_cache": {
            "path": str(source_feature_cache.resolve()),
            "size": int(feature_cache_stat.st_size),
            "mtime_ns": int(feature_cache_stat.st_mtime_ns),
        },
        "source_fingerprint": source_fingerprint,
        "audit_script_sha256": file_sha256(Path(__file__).resolve()),
        "generic_runner_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"
        ),
        "metrics_sha256": file_sha256(CLEAN_ROOT / "src" / "metrics.py"),
        "source_cache_metadata_sha256": file_sha256(
            source_feature_cache.with_name("feature_cache.meta.json")
        ),
    }
    encoded_payload = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def build_run_manifest(
    audit_config: dict,
    *,
    eval_row_hash: str,
    audit_fingerprint: str,
    source_fingerprint: str,
    source_feature_cache: Path,
    source_artifact_dir: Path,
    artifact_dir: Path,
    reporting_fingerprint: str | None = None,
    training_fingerprint_policy: str = "new_training_uses_reporting_fingerprint",
) -> dict:
    """构造训练前写入 config.json 的审计运行清单。"""

    # 三个哈希都必须是实际计算出的 SHA-256 形式，缺失时禁止生成清单。
    hash_values = {
        "eval_row_hash": eval_row_hash,
        "audit_fingerprint": audit_fingerprint,
        "source_fingerprint": source_fingerprint,
    }
    for hash_name, hash_value in hash_values.items():
        if len(str(hash_value)) != 64:
            raise ValueError(f"运行清单 {hash_name} 不是 64 字符 SHA-256")

    # 从原审计配置复制，保留所有预登记字段，再增加运行时计算血缘。
    manifest = dict(audit_config)
    manifest.update(
        {
            "eval_row_hash": str(eval_row_hash),
            "audit_fingerprint": str(audit_fingerprint),
            "audit_fingerprint_semantics": (
                "backward-compatible alias of training_fingerprint"
            ),
            "training_fingerprint": str(audit_fingerprint),
            "training_fingerprint_policy": str(training_fingerprint_policy),
            "reporting_code_fingerprint": str(
                reporting_fingerprint or audit_fingerprint
            ),
            "source_fingerprint": str(source_fingerprint),
            "source_feature_cache": str(source_feature_cache.resolve()),
            "source_artifact_dir": str(source_artifact_dir.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "reused_folds": list(REUSED_FOLDS),
            "new_folds": list(NEW_FOLDS),
            "row_key_columns": list(ROW_KEY_COLUMNS),
            "row_hash_algorithm": "sha256:rogii-row-key-v1",
            "source_fingerprint_policy": (
                "feature_cache.meta fingerprint must equal source folds 0-1 runtime; "
                "do not recompute with the later expanded generic runner source set"
            ),
        }
    )
    return manifest


def preflight_audit(
    audit_config_path: Path,
    *,
    clean_root: Path = CLEAN_ROOT,
) -> AuditInputs:
    """只读加载并验证全部输入；成功前不创建审计 artifact。"""

    # 规范化根目录和配置路径，所有相对路径均以 rogii_clean 为基准。
    resolved_clean_root = clean_root.resolve()
    resolved_audit_config_path = audit_config_path.resolve()
    _require_file(resolved_audit_config_path, "审计配置")
    audit_config = read_json(resolved_audit_config_path)

    # 根据预登记路径定位源配置、模型、fold、cache 和 B00；不提供替代回退。
    source_config_path = resolved_clean_root / str(audit_config["source_config"])
    model_config_path = resolved_clean_root / str(audit_config["model_config"])
    registry_path = resolved_clean_root / str(audit_config["fold_registry"])
    source_feature_cache = resolved_clean_root / str(
        audit_config["source_feature_cache"]
    )
    source_cache_metadata_path = source_feature_cache.with_name(
        "feature_cache.meta.json"
    )
    b00_prediction_path = resolved_clean_root / str(
        audit_config["b00_prediction_file"]
    )
    source_artifact_dir = resolved_clean_root / str(
        audit_config["source_artifact_dir"]
    )
    audit_artifact_dir = (
        resolved_clean_root / "artifacts" / str(audit_config["experiment_id"])
    )

    # 所有输入文件在读取前必须实际存在；这里仍不创建 audit_artifact_dir。
    _require_file(source_config_path, "RF01a 源配置")
    _require_file(model_config_path, "冻结模型配置")
    _require_file(registry_path, "固定 fold 注册表")
    _require_file(source_feature_cache, "RF01a feature cache")
    _require_file(source_cache_metadata_path, "RF01a feature cache metadata")
    _require_file(b00_prediction_path, "B00 完整预测")

    # 先读小型配置并验证所有冻结字段，再承担全量 Parquet I/O。
    source_config = read_json(source_config_path)
    model_config = read_json(model_config_path)
    registry_hash = file_sha256(registry_path)
    model_config_hash = file_sha256(model_config_path)
    feature_columns = validate_config_contract(
        audit_config,
        source_config,
        model_config,
        actual_fold_hash=registry_hash,
        actual_model_config_hash=model_config_hash,
    )
    model_params = dict(model_config["params"])

    # 历史源指纹以 cache metadata 与源 folds 0–1 runtime 三方一致为准。
    # 通用 runner 后来加入其他特征模块，不能用扩大后的源集合倒推旧指纹。
    cache_metadata, _, source_fingerprint = read_source_runtime_contract(
        source_cache_metadata_path,
        source_artifact_dir,
    )
    if int(cache_metadata.get("rows", -1)) != EXPECTED_ROWS:
        raise ValueError("RF01a feature cache metadata 行数不一致")
    if int(cache_metadata.get("wells", -1)) != EXPECTED_WELLS:
        raise ValueError("RF01a feature cache metadata 井数不一致")

    # 注册表按项目统一函数检查 773 井、隐藏行总数和 pad 不跨折。
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=EXPECTED_WELLS,
        expected_rows=EXPECTED_ROWS,
    )

    # 直接读取固定 feature_cache；本脚本没有任何 RF01 特征生成入口。
    feature_table = pd.read_parquet(source_feature_cache)
    validate_feature_table(
        feature_table,
        registry,
        EXPECTED_ROWS,
        feature_columns,
        list(RF01A_ALLOWED_NAN_FEATURES),
    )
    if int(feature_table["well_id"].nunique()) != EXPECTED_WELLS:
        raise ValueError("RF01a feature cache 实际井数不一致")

    # 缓存中模型列出现顺序必须等于固定 17 列，且不能夹入其他 u_ 特征。
    feature_column_set = set(feature_columns)
    actual_feature_order = [
        column for column in feature_table.columns if column in feature_column_set
    ]
    if actual_feature_order != feature_columns:
        raise ValueError("RF01a feature cache 固定 17 列顺序不一致")
    unexpected_u_columns = [
        column
        for column in feature_table.columns
        if column.startswith("u_") and column not in RF01A_FEATURE_COLUMNS
    ]
    if unexpected_u_columns:
        raise ValueError(
            f"RF01a feature cache 含其他 RF01 特征：{unexpected_u_columns}"
        )
    validate_well_fold_assignments(feature_table, registry)

    # 源 folds 0–1 的四类产物在任何新折训练前全部读取和验证。
    reused_fold_predictions: dict[int, pd.DataFrame] = {}
    reused_fold_runtimes: dict[int, dict] = {}
    reused_fold_importance: dict[int, pd.DataFrame] = {}
    for fold_id in REUSED_FOLDS:
        expected_fold_rows = feature_table.loc[feature_table["fold"] == fold_id]
        prediction, runtime, importance = _validate_fold_artifact_bundle(
            fold_id,
            source_artifact_dir / f"fold_{fold_id}",
            expected_fold_rows,
            expected_fingerprint=source_fingerprint,
            feature_columns=feature_columns,
        )
        reused_fold_predictions[fold_id] = prediction
        reused_fold_runtimes[fold_id] = runtime
        reused_fold_importance[fold_id] = importance
    validate_reused_fold_runtime_contract(cache_metadata, reused_fold_runtimes)

    # B00 必须覆盖同一 773 井、3,783,989 行，并逐位匹配 key/fold/target。
    b00_predictions = pd.read_parquet(b00_prediction_path)
    eval_row_hash = validate_evaluation_alignment(
        feature_table,
        b00_predictions,
        expected_rows=EXPECTED_ROWS,
        expected_wells=EXPECTED_WELLS,
    )

    # 最后计算审计 fingerprint；它将用于新 folds 的中断恢复。
    reporting_fingerprint = _build_reporting_fingerprint(
        audit_config,
        source_config,
        model_config,
        registry_hash=registry_hash,
        source_feature_cache=source_feature_cache,
        source_fingerprint=source_fingerprint,
    )

    # 已完成折可沿用历史训练指纹，但只允许在清单合同与 folds 2–4 全部一致时。
    existing_manifest_path = audit_artifact_dir / "config.json"
    existing_manifest = (
        read_json(existing_manifest_path) if existing_manifest_path.is_file() else None
    )
    existing_new_fold_runtimes: dict[int, dict] = {}
    for fold_id in NEW_FOLDS:
        runtime_path = audit_artifact_dir / f"fold_{fold_id}" / "runtime.json"
        if runtime_path.is_file():
            existing_new_fold_runtimes[fold_id] = read_json(runtime_path)
    expected_existing_contract = {
        "experiment_id": AUDIT_EXPERIMENT_ID,
        "eval_row_hash": eval_row_hash,
        "source_fingerprint": source_fingerprint,
        "fold_registry_sha256": EXPECTED_FOLD_REGISTRY_HASH,
        "expected_rows": EXPECTED_ROWS,
        "expected_wells": EXPECTED_WELLS,
        "model_config": EXPECTED_MODEL_CONFIG,
        "reused_folds": list(REUSED_FOLDS),
        "new_folds": list(NEW_FOLDS),
    }
    audit_fingerprint, training_fingerprint_policy = select_training_fingerprint(
        reporting_fingerprint=reporting_fingerprint,
        existing_manifest=existing_manifest,
        new_fold_runtimes=existing_new_fold_runtimes,
        expected_contract=expected_existing_contract,
    )

    # 若审计目录已有匹配指纹的新折，必须在第一次 train_fold 前完整验收其缓存。
    validate_existing_new_fold_reuse(
        feature_table,
        audit_artifact_dir,
        audit_fingerprint=audit_fingerprint,
        feature_columns=feature_columns,
    )

    # 返回时所有检查已经完成，且 audit_artifact_dir 仍未被创建。
    return AuditInputs(
        audit_config=audit_config,
        source_config=source_config,
        model_config=model_config,
        model_params=model_params,
        registry=registry,
        feature_table=feature_table,
        b00_predictions=b00_predictions,
        reused_fold_predictions=reused_fold_predictions,
        reused_fold_runtimes=reused_fold_runtimes,
        reused_fold_importance=reused_fold_importance,
        feature_columns=feature_columns,
        cache_metadata=cache_metadata,
        source_fingerprint=source_fingerprint,
        audit_fingerprint=audit_fingerprint,
        reporting_fingerprint=reporting_fingerprint,
        training_fingerprint_policy=training_fingerprint_policy,
        eval_row_hash=eval_row_hash,
        clean_root=resolved_clean_root,
        source_artifact_dir=source_artifact_dir,
        artifact_dir=audit_artifact_dir,
    )


def merge_fixed_fold_predictions(
    reused_fold_predictions: dict[int, pd.DataFrame],
    new_fold_predictions: dict[int, pd.DataFrame],
    feature_table: pd.DataFrame,
    *,
    expected_rows: int,
    expected_wells: int,
) -> pd.DataFrame:
    """合并源 folds 0–1 与审计 folds 2–4，并核对完整评价行。"""

    # 两组来源的键必须精确，不能缺折、补折或让新目录提供 folds 0–1。
    if set(reused_fold_predictions) != set(REUSED_FOLDS):
        raise ValueError("源预测必须恰好来自 folds 0–1")
    if set(new_fold_predictions) != set(NEW_FOLDS):
        raise ValueError("审计预测必须恰好来自 folds 2–4")

    # 每折先验证自身 fold 值和评分 schema，再进入纵向拼接。
    required_columns = set(ROW_KEY_COLUMNS) | {
        "target_tvt",
        "carry_tvt",
        "pred_tvt",
    }
    ordered_fold_predictions: list[pd.DataFrame] = []
    for fold_id in REUSED_FOLDS + NEW_FOLDS:
        source_mapping = (
            reused_fold_predictions if fold_id in REUSED_FOLDS else new_fold_predictions
        )
        fold_prediction = source_mapping[fold_id]
        missing_columns = required_columns - set(fold_prediction.columns)
        if missing_columns:
            raise ValueError(
                f"fold {fold_id} 预测缺少列：{sorted(missing_columns)}"
            )
        actual_folds = set(fold_prediction["fold"].astype(int).unique())
        if actual_folds != {fold_id}:
            raise ValueError(f"fold {fold_id} 预测文件中的 fold 值不一致")
        ordered_fold_predictions.append(fold_prediction.copy())

    # 合并后按三列键排序，使保存顺序与 B00 对齐规则一致。
    predictions = pd.concat(ordered_fold_predictions, ignore_index=True)
    predictions = predictions.sort_values(list(ROW_KEY_COLUMNS)).reset_index(drop=True)

    # 覆盖、唯一性和有限预测是完整五折保存前的硬门槛。
    if len(predictions) != int(expected_rows):
        raise ValueError("完整五折预测行数不一致")
    if int(predictions["well_id"].nunique()) != int(expected_wells):
        raise ValueError("完整五折预测井数不一致")
    if predictions.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise ValueError("完整五折预测含重复行键")
    if not np.isfinite(predictions["pred_tvt"].to_numpy(dtype=np.float64)).all():
        raise ValueError("完整五折 pred_tvt 含 NaN 或 Inf")

    # 完整预测的行集合必须等于预检通过的 RF01a feature cache。
    prediction_row_hash = compute_row_key_hash(predictions)
    feature_row_hash = compute_row_key_hash(feature_table)
    if prediction_row_hash != feature_row_hash:
        raise ValueError("完整五折预测与 RF01a feature cache 行键不一致")

    # 同一排序下 target_tvt 必须精确一致，不能只依赖行键哈希。
    aligned_features = feature_table.loc[
        :, list(ROW_KEY_COLUMNS) + ["target_tvt"]
    ].sort_values(list(ROW_KEY_COLUMNS)).reset_index(drop=True)
    prediction_targets = predictions["target_tvt"].to_numpy(dtype=np.float64)
    feature_targets = aligned_features["target_tvt"].to_numpy(dtype=np.float64)
    if not np.array_equal(prediction_targets, feature_targets):
        raise ValueError("完整五折预测与 RF01a feature cache target_tvt 不一致")

    # 返回可直接写为审计根 predictions.parquet 的固定顺序表。
    return predictions


def build_b00_comparison(
    candidate_predictions: pd.DataFrame,
    b00_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """构造显式以 B00 ``pred_tvt`` 为 baseline 的配对比较。"""

    # 两侧先按固定三列键排序；该函数可用于小表单测，也可用于完整五折。
    candidate_aligned = candidate_predictions.sort_values(
        list(ROW_KEY_COLUMNS)
    ).reset_index(drop=True)
    b00_aligned = b00_predictions.sort_values(list(ROW_KEY_COLUMNS)).reset_index(
        drop=True
    )
    if len(candidate_aligned) != len(b00_aligned):
        raise ValueError("RF01a 与 B00 比较行数不一致")
    if compute_row_key_hash(candidate_aligned) != compute_row_key_hash(b00_aligned):
        raise ValueError("RF01a 与 B00 比较行键不一致")

    # 逐位目标必须完全相同，B00 预测本身也必须有限。
    candidate_targets = candidate_aligned["target_tvt"].to_numpy(dtype=np.float64)
    b00_targets = b00_aligned["target_tvt"].to_numpy(dtype=np.float64)
    if not np.array_equal(candidate_targets, b00_targets):
        raise ValueError("RF01a 与 B00 比较 target_tvt 不一致")
    b00_values = b00_aligned["pred_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(b00_values).all():
        raise ValueError("B00 比较预测含 NaN 或 Inf")

    # 保留候选 pred_tvt，并把 B00 预测显式命名为 b00_pred_tvt。
    comparison_predictions = candidate_aligned.copy()
    comparison_predictions["b00_pred_tvt"] = b00_values

    # 关键处显式传 baseline_column；禁止使用 metrics 默认的 carry_tvt。
    comparison_per_well = build_per_well_metrics(
        comparison_predictions,
        prediction_column="pred_tvt",
        baseline_column="b00_pred_tvt",
    )
    overall = summarize_per_well_metrics(comparison_per_well)
    fold_summaries = summarize_by_fold(comparison_per_well)
    bootstrap = paired_well_bootstrap(
        comparison_per_well,
        n_resamples=2000,
        seed=42,
    )

    # 每折差值定义为 RF01a_RMSE - B00_RMSE，负数表示 RF01a 改善。
    for fold_summary in fold_summaries:
        fold_summary["rf01a_minus_b00_micro_rmse"] = float(
            fold_summary["micro_rmse"] - fold_summary["baseline_micro_rmse"]
        )

    comparison_metrics = {
        "baseline_id": "B00_simple_lgbm_v1",
        "delta_definition": "RF01a_RMSE - B00_RMSE",
        "overall": overall,
        "folds": fold_summaries,
        "paired_well_bootstrap": bootstrap,
    }
    return comparison_predictions, comparison_per_well, comparison_metrics


def aggregate_feature_importance(
    fold_importance: dict[int, pd.DataFrame],
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    """验证并平均固定五折的重要性表。"""

    # 必须恰好有 folds 0–4；调用方负责按固定源目录读取每一折。
    expected_folds = set(REUSED_FOLDS + NEW_FOLDS)
    if set(fold_importance) != expected_folds:
        raise ValueError("feature importance 必须恰好覆盖 folds 0–4")

    all_importance_tables: list[pd.DataFrame] = []
    for fold_id in REUSED_FOLDS + NEW_FOLDS:
        importance = fold_importance[fold_id].copy()
        required_columns = {"feature", "gain", "split"}
        missing_columns = required_columns - set(importance.columns)
        if missing_columns:
            raise ValueError(
                f"fold {fold_id} importance 缺少列：{sorted(missing_columns)}"
            )
        if importance["feature"].duplicated().any():
            raise ValueError(f"fold {fold_id} importance 含重复特征")
        if set(importance["feature"]) != set(feature_columns):
            raise ValueError(f"fold {fold_id} importance 特征列表与固定列不一致")
        numeric_values = importance[["gain", "split"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"fold {fold_id} importance 含 NaN 或 Inf")
        importance["fold"] = fold_id
        all_importance_tables.append(importance)

    # 五折等权平均 gain/split，与通用 runner 的完整五折汇总口径一致。
    all_importance = pd.concat(all_importance_tables, ignore_index=True)
    mean_importance = (
        all_importance.groupby("feature", as_index=False)[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    return mean_importance


def build_conclusion_text(fold_deltas: list[float]) -> str:
    """返回首行锁定原决定、次行仅含固定折模式的中文结论。"""

    # 分类函数会先拒绝缺折或非有限差值，避免写出无法解释的结论。
    fold_pattern = classify_fold_pattern(fold_deltas)

    # 只写两条固定陈述；审计结果不能添加任何追认晋级措辞。
    conclusion_lines = [
        "原 RF01 仍不晋级，本审计不改变原决定。",
        fold_pattern,
    ]
    return "\n".join(conclusion_lines) + "\n"


def run_new_folds(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    model_params: dict,
    artifact_dir: Path,
    fingerprint: str,
    feature_columns: list[str],
    new_folds: list[int] | tuple[int, ...],
    train_callable=train_fold,
) -> list[dict]:
    """仅按固定顺序把 folds 2–4 交给通用 ``train_fold``。"""

    # 配置必须逐项等于预注册集合，不能通过调用参数扩大审计范围。
    normalized_new_folds = tuple(int(fold_id) for fold_id in new_folds)
    if normalized_new_folds != NEW_FOLDS:
        raise ValueError(
            f"new_folds 必须固定为 {NEW_FOLDS}，实际为 {normalized_new_folds}"
        )

    # 每项摘要直接来自通用 runner；它会按相同 fingerprint 复用已完成折。
    fold_summaries: list[dict] = []
    for fold_id in normalized_new_folds:
        fold_summary = train_callable(
            feature_table,
            registry,
            fold_id,
            model_params,
            artifact_dir,
            fingerprint,
            feature_columns,
        )

        # 返回折号不一致意味着训练函数或缓存归属错误，不能继续拼接结果。
        if int(fold_summary.get("fold", -1)) != fold_id:
            raise ValueError(f"train_fold 返回的 fold 与请求的 fold {fold_id} 不一致")
        fold_summaries.append(fold_summary)

    # 列表顺序固定为 2、3、4，方便运行清单稳定记录。
    return fold_summaries


def _build_candidate_metrics(
    predictions: pd.DataFrame,
    prediction_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """使用统一指标生成 RF01a 相对 carry 的常规完整五折摘要。"""

    # 默认 baseline_column=carry_tvt 只用于候选自身常规指标，不用于 B00 比较。
    per_well = build_per_well_metrics(predictions)
    metrics = {
        "prediction_file": str(prediction_path.resolve()),
        "overall": summarize_per_well_metrics(per_well),
        "folds": summarize_by_fold(per_well),
        "paired_well_bootstrap": paired_well_bootstrap(
            per_well,
            n_resamples=2000,
            seed=42,
        ),
    }
    return per_well, metrics


def _manifest_from_inputs(audit_inputs: AuditInputs) -> dict:
    """从已预检对象构造可重复写入的 config.json 清单。"""

    source_feature_cache = audit_inputs.clean_root / str(
        audit_inputs.audit_config["source_feature_cache"]
    )
    return build_run_manifest(
        audit_inputs.audit_config,
        eval_row_hash=audit_inputs.eval_row_hash,
        audit_fingerprint=audit_inputs.audit_fingerprint,
        source_fingerprint=audit_inputs.source_fingerprint,
        source_feature_cache=source_feature_cache,
        source_artifact_dir=audit_inputs.source_artifact_dir,
        artifact_dir=audit_inputs.artifact_dir,
        reporting_fingerprint=audit_inputs.reporting_fingerprint,
        training_fingerprint_policy=audit_inputs.training_fingerprint_policy,
    )


def _reject_audit_reused_fold_directories(artifact_dir: Path) -> None:
    """拒绝审计目录中出现 fold_0 或 fold_1，防止错误来源被误解。"""

    unexpected_directories = [
        artifact_dir / f"fold_{fold_id}"
        for fold_id in REUSED_FOLDS
        if (artifact_dir / f"fold_{fold_id}").exists()
    ]
    if unexpected_directories:
        raise ValueError(
            "审计目录禁止包含 fold_0/fold_1："
            + ", ".join(str(path) for path in unexpected_directories)
        )


def finalize_audit(audit_inputs: AuditInputs) -> dict:
    """验证固定五折产物，保存完整指标、B00 比较和不可改判结论。"""

    # 汇总前再次确认审计根目录没有旧折，源 0–1 只能来自 RF01a 原目录。
    _reject_audit_reused_fold_directories(audit_inputs.artifact_dir)
    fold_directories = build_fixed_fold_directories(
        audit_inputs.source_artifact_dir,
        audit_inputs.artifact_dir,
    )

    # 五折全部重新读取四类产物；新折复用也必须通过 model/importance/fold 检查。
    all_fold_predictions: dict[int, pd.DataFrame] = {}
    all_fold_runtimes: dict[int, dict] = {}
    all_fold_importance: dict[int, pd.DataFrame] = {}
    for fold_id in REUSED_FOLDS + NEW_FOLDS:
        expected_fold_rows = audit_inputs.feature_table.loc[
            audit_inputs.feature_table["fold"] == fold_id
        ]
        expected_fingerprint = (
            audit_inputs.source_fingerprint
            if fold_id in REUSED_FOLDS
            else audit_inputs.audit_fingerprint
        )
        prediction, runtime, importance = _validate_fold_artifact_bundle(
            fold_id,
            fold_directories[fold_id],
            expected_fold_rows,
            expected_fingerprint=expected_fingerprint,
            feature_columns=audit_inputs.feature_columns,
        )
        all_fold_predictions[fold_id] = prediction
        all_fold_runtimes[fold_id] = runtime
        all_fold_importance[fold_id] = importance

    # 分开传入源折和新折，函数内部会拒绝任何来源集合漂移。
    reused_predictions = {
        fold_id: all_fold_predictions[fold_id] for fold_id in REUSED_FOLDS
    }
    new_predictions = {
        fold_id: all_fold_predictions[fold_id] for fold_id in NEW_FOLDS
    }
    predictions = merge_fixed_fold_predictions(
        reused_predictions,
        new_predictions,
        audit_inputs.feature_table,
        expected_rows=EXPECTED_ROWS,
        expected_wells=EXPECTED_WELLS,
    )

    # 纯补档不得重写已有预测；逐位一致后保留原文件和原 SHA。
    artifact_dir = audit_inputs.artifact_dir
    prediction_path = artifact_dir / "predictions.parquet"
    prediction_file_records: dict[str, dict] = {}
    prediction_file_records["predictions.parquet"] = (
        preserve_or_write_prediction_table(predictions, prediction_path)
    )

    # 使用项目统一指标生成相对 carry 的原有常规摘要，保持来源不变。
    per_well, metrics = _build_candidate_metrics(predictions, prediction_path)

    # B00 比较函数显式以 b00_pred_tvt 为 baseline，并生成逐折差值。
    comparison_predictions, comparison_per_well, comparison_metrics = (
        build_b00_comparison(predictions, audit_inputs.b00_predictions)
    )
    fold_delta_by_id = {
        int(fold_summary["fold"]): float(
            fold_summary["rf01a_minus_b00_micro_rmse"]
        )
        for fold_summary in comparison_metrics["folds"]
    }
    if set(fold_delta_by_id) != set(REUSED_FOLDS + NEW_FOLDS):
        raise ValueError("B00 比较逐折指标没有完整覆盖 folds 0–4")
    fold_deltas = [fold_delta_by_id[fold_id] for fold_id in range(5)]
    comparison_metrics["fold_pattern"] = classify_fold_pattern(fold_deltas)

    # 固定 artifact 的 bootstrap 明细对应正式 B00 baseline，不混入 carry 对照。
    bootstrap_replicates, bootstrap_summary = build_paired_bootstrap_artifacts(
        comparison_per_well,
        candidate_id=SOURCE_EXPERIMENT_ID,
        baseline_id="B00_simple_lgbm_v1",
        n_resamples=2000,
        seed=42,
    )
    if bootstrap_summary != comparison_metrics["paired_well_bootstrap"]:
        raise ValueError("bootstrap 明细重算摘要与 comparison_vs_B00 metrics 不一致")

    # 五折 importance 必须固定汇总源 0–1 与新 2–4，不从审计旧折目录读取。
    mean_importance = aggregate_feature_importance(
        all_fold_importance,
        feature_columns=audit_inputs.feature_columns,
    )

    # 定义、lineage 和质量表全部来自固定 17 列与实际 feature cache。
    feature_definition, feature_lineage = build_feature_contract_documents(
        audit_inputs.feature_columns
    )
    feature_quality = build_feature_quality_table(
        audit_inputs.feature_table,
        audit_inputs.feature_columns,
        feature_lineage,
    )

    # 逐折、全体切片和负对照说明都使用真实 B00 配对指标。
    per_fold, slice_metrics, negative_control_metrics = (
        build_diagnostic_reporting_artifacts(comparison_metrics)
    )

    # comparison 预测同样只验证并保留既有字节，不因补档而重写。
    comparison_dir = artifact_dir / "comparison_vs_B00"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_prediction_path = comparison_dir / "predictions.parquet"
    prediction_file_records["comparison_vs_B00/predictions.parquet"] = (
        preserve_or_write_prediction_table(
            comparison_predictions,
            comparison_prediction_path,
        )
    )

    # folds 2–4 已由 train_fold 以相同 training fingerprint 纯复用，记录当前 SHA。
    for fold_id in NEW_FOLDS:
        fold_prediction_path = fold_directories[fold_id] / "predictions.parquet"
        fold_prediction_hash = file_sha256(fold_prediction_path)
        prediction_file_records[f"fold_{fold_id}/predictions.parquet"] = {
            "path": str(fold_prediction_path.resolve()),
            "action": "reused_train_fold_matching_fingerprint",
            "sha256_before": fold_prediction_hash,
            "sha256_after": fold_prediction_hash,
            "rows": int(len(all_fold_predictions[fold_id])),
        }

    # 每折 cache manifest 保存实际 fit/validation 井 hash、行 hash 和四类文件 hash。
    fold_cache_records: list[dict] = []
    for fold_id in REUSED_FOLDS + NEW_FOLDS:
        fold_cache_records.append(
            build_fold_cache_manifest(
                fold_id=fold_id,
                artifact_source=(
                    "reused_RF01a" if fold_id in REUSED_FOLDS else "audit_new_fold"
                ),
                registry=audit_inputs.registry,
                prediction=all_fold_predictions[fold_id],
                runtime=all_fold_runtimes[fold_id],
                fold_dir=fold_directories[fold_id],
            )
        )

    source_feature_cache = audit_inputs.clean_root / str(
        audit_inputs.audit_config["source_feature_cache"]
    )
    source_cache_metadata_path = source_feature_cache.with_name(
        "feature_cache.meta.json"
    )
    registry_path = audit_inputs.clean_root / str(
        audit_inputs.audit_config["fold_registry"]
    )
    source_config_path = audit_inputs.clean_root / str(
        audit_inputs.audit_config["source_config"]
    )
    model_config_path = audit_inputs.clean_root / str(
        audit_inputs.audit_config["model_config"]
    )
    b00_prediction_path = audit_inputs.clean_root / str(
        audit_inputs.audit_config["b00_prediction_file"]
    )
    source_cache_stat = source_feature_cache.stat()
    feature_list_hash = hashlib.sha256(
        json.dumps(
            audit_inputs.feature_columns,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_manifest = {
        "manifest_version": "RF01_stability_audit_cache_manifest_v1",
        "experiment_id": AUDIT_EXPERIMENT_ID,
        "experiment_type": "diagnostic_only",
        "training_fingerprint": audit_inputs.audit_fingerprint,
        "training_fingerprint_policy": audit_inputs.training_fingerprint_policy,
        "reporting_code_fingerprint": audit_inputs.reporting_fingerprint,
        "source_fingerprint": audit_inputs.source_fingerprint,
        "eval_row_hash": audit_inputs.eval_row_hash,
        "full_prediction_row_hash": compute_row_key_hash(predictions),
        "feature_list_hash": feature_list_hash,
        "source_feature_cache": {
            "path": str(source_feature_cache.resolve()),
            "sha256": file_sha256(source_feature_cache),
            "metadata_path": str(source_cache_metadata_path.resolve()),
            "metadata_sha256": file_sha256(source_cache_metadata_path),
            "created_at": audit_inputs.cache_metadata.get("created_at"),
            "fingerprint": audit_inputs.cache_metadata.get("fingerprint"),
            "size_bytes": int(source_cache_stat.st_size),
            "mtime_ns": int(source_cache_stat.st_mtime_ns),
            "rows": int(audit_inputs.cache_metadata["rows"]),
            "wells": int(audit_inputs.cache_metadata["wells"]),
        },
        "fold_registry": {
            "path": str(registry_path.resolve()),
            "sha256": file_sha256(registry_path),
        },
        "source_config": {
            "path": str(source_config_path.resolve()),
            "sha256": file_sha256(source_config_path),
        },
        "model_config": {
            "path": str(model_config_path.resolve()),
            "sha256": file_sha256(model_config_path),
        },
        "b00_predictions": {
            "path": str(b00_prediction_path.resolve()),
            "sha256": file_sha256(b00_prediction_path),
            "row_hash": compute_row_key_hash(audit_inputs.b00_predictions),
        },
        "code_files": {
            "audit_script_sha256": file_sha256(Path(__file__).resolve()),
            "generic_runner_sha256": file_sha256(
                CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"
            ),
            "metrics_sha256": file_sha256(CLEAN_ROOT / "src" / "metrics.py"),
            "lgbm_features_sha256": file_sha256(
                CLEAN_ROOT / "src" / "lgbm_features.py"
            ),
            "rf01a_features_sha256": file_sha256(
                CLEAN_ROOT / "src" / "rf01a_features.py"
            ),
        },
        "fold_artifacts": fold_cache_records,
        "prediction_files": prediction_file_records,
        "PF_Beam_cache": {
            "status": "not_applicable",
            "reason": "All fixed RF01a features declare uses_PF_Beam=false.",
        },
    }
    if cache_manifest["full_prediction_row_hash"] != audit_inputs.eval_row_hash:
        raise ValueError("cache manifest 完整预测 row hash 与 eval_row_hash 不一致")

    leakage_tests = build_leakage_tests_document(
        eval_row_hash=audit_inputs.eval_row_hash,
        feature_lineage=feature_lineage,
        cache_manifest=cache_manifest,
        prediction_file_records=prediction_file_records,
        negative_control=negative_control_metrics,
    )

    # 运行清单同时写入 config.json 和 runtime.json，保存真实 eval row hash。
    run_manifest = _manifest_from_inputs(audit_inputs)
    fold_runtime_records: list[dict] = []
    for fold_id in REUSED_FOLDS + NEW_FOLDS:
        runtime_record = dict(all_fold_runtimes[fold_id])
        runtime_record["artifact_source"] = (
            "reused_RF01a" if fold_id in REUSED_FOLDS else "audit_new_fold"
        )
        fold_runtime_records.append(runtime_record)
    runtime_summary = {
        "fingerprint": audit_inputs.audit_fingerprint,
        "fingerprint_semantics": "backward-compatible training fingerprint",
        "training_fingerprint": audit_inputs.audit_fingerprint,
        "training_fingerprint_policy": audit_inputs.training_fingerprint_policy,
        "reporting_code_fingerprint": audit_inputs.reporting_fingerprint,
        "source_fingerprint": audit_inputs.source_fingerprint,
        "eval_row_hash": audit_inputs.eval_row_hash,
        "folds": fold_runtime_records,
        "reused_folds": list(REUSED_FOLDS),
        "new_folds": list(NEW_FOLDS),
        "new_fold_seconds": float(
            sum(float(all_fold_runtimes[fold_id]["seconds"]) for fold_id in NEW_FOLDS)
        ),
        "reused_source_fold_seconds": float(
            sum(
                float(all_fold_runtimes[fold_id]["seconds"])
                for fold_id in REUSED_FOLDS
            )
        ),
        "artifact_contract_version": "AGENTS_6.3_feature_roadmap_v2_section_14",
        "bootstrap_replicates": {
            "baseline_id": "B00_simple_lgbm_v1",
            "candidate_id": SOURCE_EXPERIMENT_ID,
            "seed": 42,
            "n_resamples": 2000,
            "path": str((artifact_dir / "bootstrap_replicates.parquet").resolve()),
        },
        "prediction_file_records": prediction_file_records,
    }
    run_manifest["artifact_contract_version"] = (
        "AGENTS_6.3_feature_roadmap_v2_section_14"
    )
    run_manifest["required_artifacts"] = list(FINAL_OUTPUT_RELATIVE_PATHS)

    # 根目录输出与通用 runner 同口径；comparison 子目录单独保存 B00 配对结果。
    write_json(artifact_dir / "metrics.json", metrics)
    per_well.to_csv(artifact_dir / "per_well.csv", index=False)
    write_json(
        artifact_dir / "feature_list.json",
        {
            "feature_version": SOURCE_FEATURE_VERSION,
            "feature_count": len(audit_inputs.feature_columns),
            "features": audit_inputs.feature_columns,
        },
    )
    write_json(artifact_dir / "parameter_list.json", audit_inputs.model_params)
    write_json(artifact_dir / "feature_definition.json", feature_definition)
    write_json(artifact_dir / "feature_lineage.json", feature_lineage)
    feature_quality.to_csv(artifact_dir / "feature_quality.csv", index=False)
    write_json(artifact_dir / "cache_manifest.json", cache_manifest)
    write_json(artifact_dir / "leakage_tests.json", leakage_tests)
    write_json(
        artifact_dir / "negative_control_metrics.json",
        negative_control_metrics,
    )
    per_fold.to_csv(artifact_dir / "per_fold.csv", index=False)
    slice_metrics.to_csv(artifact_dir / "slice_metrics.csv", index=False)
    bootstrap_replicates.to_parquet(
        artifact_dir / "bootstrap_replicates.parquet",
        index=False,
        compression="zstd",
    )
    write_json(artifact_dir / "runtime.json", runtime_summary)
    write_json(artifact_dir / "config.json", run_manifest)
    mean_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    # comparison prediction 已在上方逐位验证并保留原 SHA；这里只写指标表。
    comparison_per_well.to_csv(comparison_dir / "per_well.csv", index=False)
    comparison_metrics["prediction_file"] = str(
        comparison_prediction_path.resolve()
    )
    write_json(comparison_dir / "metrics.json", comparison_metrics)

    # conclusion.md 只含固定首行和一个预注册折模式，不写追认晋级解释。
    conclusion_text = build_conclusion_text(fold_deltas)
    (artifact_dir / "conclusion.md").write_text(conclusion_text, encoding="utf-8")

    # 最后逐项检查固定输出，缺一项都不能报告汇总成功。
    for relative_path in FINAL_OUTPUT_RELATIVE_PATHS:
        _require_file(artifact_dir / relative_path, "审计最终输出")
    _reject_audit_reused_fold_directories(artifact_dir)

    print(
        "RF01 稳定性审计完整五折："
        + json.dumps(metrics["overall"], ensure_ascii=False, indent=2),
        flush=True,
    )
    print(
        f"B00 折模式：{comparison_metrics['fold_pattern']}",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    """解析唯一审计配置路径；折集合不开放命令行覆盖。"""

    parser = argparse.ArgumentParser(
        description="复用 RF01a folds 0–1，只补 folds 2–4 的稳定性审计"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "rf01_stability_audit_v1.json",
    )
    return parser.parse_args()


def _print_long_run_contract(config_path: Path) -> None:
    """在任何全量读取或训练前打印长任务范围和恢复方式。"""

    print("将运行什么：固定 RF01a 稳定性审计，只补 folds 2–4。", flush=True)
    print("使用多少口井和哪些 folds：773 井；复用 0–1，新运行 2–4。", flush=True)
    print("当前阶段：训练前合同预检，随后连续运行三个新折。", flush=True)
    print("主要耗时：全量 cache/B00 对齐与三次固定 LightGBM 训练。", flush=True)
    print(
        "生成文件：fold_2/3/4 与完整五折指标、B00 比较和结论。",
        flush=True,
    )
    print("停止后如何继续：相同 fingerprint 会复用已完成的新折。", flush=True)
    print(f"审计配置：{config_path.resolve()}", flush=True)


def main() -> None:
    """执行只读预检、固定新折训练和独立五折汇总。"""

    args = parse_args()
    _print_long_run_contract(args.config)

    # 所有配置、cache、source folds 和 B00 对齐均在任何训练调用前完成。
    audit_inputs = preflight_audit(args.config, clean_root=CLEAN_ROOT)
    _reject_audit_reused_fold_directories(audit_inputs.artifact_dir)

    # 按缓存规则打印创建时间、源配置、当前配置和完全匹配状态。
    print(
        "缓存文件："
        f"{audit_inputs.clean_root / audit_inputs.audit_config['source_feature_cache']}",
        flush=True,
    )
    print(
        f"缓存创建时间：{audit_inputs.cache_metadata.get('created_at')}",
        flush=True,
    )
    print(f"缓存配置：{audit_inputs.source_fingerprint}", flush=True)
    print(f"当前配置：{audit_inputs.source_fingerprint}", flush=True)
    print("是否完全匹配：True", flush=True)
    print(f"固定 eval_row_hash：{audit_inputs.eval_row_hash}", flush=True)

    # 预检全部通过后才创建审计根目录，并在训练前保存真实 row hash 清单。
    audit_inputs.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        audit_inputs.artifact_dir / "config.json",
        _manifest_from_inputs(audit_inputs),
    )

    # 通用 train_fold 仅收到配置中固定的 [2,3,4]，同 fingerprint 支持恢复。
    run_new_folds(
        feature_table=audit_inputs.feature_table,
        registry=audit_inputs.registry,
        model_params=audit_inputs.model_params,
        artifact_dir=audit_inputs.artifact_dir,
        fingerprint=audit_inputs.audit_fingerprint,
        feature_columns=audit_inputs.feature_columns,
        new_folds=audit_inputs.audit_config["new_folds"],
    )

    # 自建汇总按固定来源读取五折；不调用通用 finalize_complete_cv。
    finalize_audit(audit_inputs)


if __name__ == "__main__":
    main()
