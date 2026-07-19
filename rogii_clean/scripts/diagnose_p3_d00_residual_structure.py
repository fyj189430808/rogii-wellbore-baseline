"""运行 P3-D00：审计 P2-P02 折外残差的低维平滑表示上限。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_d00_residual_structure import audit_well_residuals  # noqa: E402


EXPERIMENT_ID = "P3_D00_residual_structure_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_d00_residual_structure_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

# 左边是报告中的短名称，右边是单井审计模块的固定键。
REPRESENTATION_FIT_KEYS = {
    "constant_unanchored": "polynomial_degree_0_unanchored",
    "linear_unanchored": "polynomial_degree_1_unanchored",
    "linear_anchored": "polynomial_degree_1_anchored",
    "quadratic_unanchored": "polynomial_degree_2_unanchored",
    "quadratic_anchored": "polynomial_degree_2_anchored",
    "cubic_unanchored": "polynomial_degree_3_unanchored",
    "cubic_anchored": "polynomial_degree_3_anchored",
    "control4_unanchored": "control_points_4_unanchored",
    "control4_anchored": "control_points_4_anchored",
    "control8_unanchored": "control_points_8_unanchored",
    "control8_anchored": "control_points_8_anchored",
}
REPRESENTATIONS = ["baseline", *REPRESENTATION_FIT_KEYS]
PREDICTION_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "md",
    "target_tvt",
    "pred_tvt",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析冻结配置和独立输出目录。"""

    parser = argparse.ArgumentParser(description="运行 P3-D00 残差结构审计")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    """读取一个必须为 JSON 对象的文件。"""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以稳定格式写 JSON，并拒绝 NaN。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    """分块计算大 Parquet 的 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shadow_well_ids(shadow_path: Path) -> set[str]:
    """兼容“只列 shadow 井”和“全井带 is_shadow”两种登记表。"""

    shadow = pd.read_csv(shadow_path)
    if "well_id" not in shadow.columns:
        raise ValueError("shadow 文件缺少 well_id")
    if "is_shadow" not in shadow.columns:
        return set(shadow["well_id"].astype(str))
    flags = shadow["is_shadow"]
    if flags.dtype == bool:
        mask = flags.to_numpy()
    else:
        mask = flags.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
    return set(shadow.loc[mask, "well_id"].astype(str))


def load_development_predictions(
    prediction_path: Path,
    shadow_path: Path,
) -> pd.DataFrame:
    """在 Arrow 扫描阶段排除 shadow 井，再返回含目标的开发行。"""

    prediction_path = Path(prediction_path)
    shadow_ids = _shadow_well_ids(Path(shadow_path))
    dataset = arrow_dataset.dataset(prediction_path, format="parquet")
    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(dataset.schema.names))
    if missing_columns:
        raise ValueError(f"P2-P02 OOF 缺列：{missing_columns}")

    row_filter = None
    if shadow_ids:
        row_filter = ~arrow_dataset.field("well_id").isin(sorted(shadow_ids))
    table = dataset.to_table(columns=PREDICTION_COLUMNS, filter=row_filter)
    development = table.to_pandas()
    development["well_id"] = development["well_id"].astype(str)

    overlap = set(development["well_id"].unique()).intersection(shadow_ids)
    if overlap:
        raise RuntimeError(f"开发表仍含 shadow 井：{sorted(overlap)[:5]}")
    if development.duplicated(["well_id", "row_index"]).any():
        raise ValueError("P2-P02 OOF 含重复自然隐藏行键")
    numeric_columns = ["fold", "row_index", "md", "target_tvt", "pred_tvt"]
    numeric_values = development[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("开发 OOF 含 NaN 或无穷值")
    fold_counts = development.groupby("well_id")["fold"].nunique()
    if len(fold_counts) and int(fold_counts.max()) != 1:
        raise ValueError("同一口开发井出现多个 fold")
    return development.sort_values(
        ["well_id", "row_index"],
        kind="stable",
    ).reset_index(drop=True)


def _one_well_row(well_frame: pd.DataFrame) -> dict[str, Any]:
    """把单井嵌套审计结果压成一行，便于汇总但不保存逐行 oracle。"""

    audit = audit_well_residuals(well_frame)
    fold_values = well_frame["fold"].astype(int).unique()
    if len(fold_values) != 1:
        raise ValueError("单井审计收到多个 fold")
    row: dict[str, Any] = {
        "well_id": str(audit["well_id"]),
        "fold": int(fold_values[0]),
        "rows": int(audit["rows"]),
        "baseline_sse": float(audit["baseline_sse"]),
        "baseline_rmse": float(audit["baseline_rmse"]),
        "baseline_coefficients": "{}",
    }
    fits = audit["fits"]
    for report_name, fit_key in REPRESENTATION_FIT_KEYS.items():
        fit = fits[fit_key]
        row[f"{report_name}_sse"] = float(fit["sse"])
        row[f"{report_name}_rmse"] = float(fit["rmse"])
        row[f"{report_name}_coefficients"] = json.dumps(
            fit["coefficients"],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    return row


def _aggregate_subset(per_well: pd.DataFrame) -> dict[str, dict[str, float]]:
    """用总 SSE/总行数计算 pooled 指标，同时保留井等权指标。"""

    total_rows = int(per_well["rows"].sum())
    if total_rows <= 0:
        raise ValueError("没有可汇总的开发评价行")
    baseline_sse = float(per_well["baseline_sse"].sum())
    result: dict[str, dict[str, float]] = {}
    for representation in REPRESENTATIONS:
        sse = float(per_well[f"{representation}_sse"].sum())
        rmse_values = per_well[f"{representation}_rmse"].to_numpy(dtype=float)
        remaining_fraction = sse / baseline_sse if baseline_sse > 0.0 else 0.0
        result[representation] = {
            "pooled_rmse": float(np.sqrt(sse / total_rows)),
            "macro_well_rmse": float(np.mean(rmse_values)),
            "median_well_rmse": float(np.median(rmse_values)),
            "p90_well_rmse": float(np.quantile(rmse_values, 0.90)),
            "worst_well_rmse": float(np.max(rmse_values)),
            "sse": sse,
            "remaining_sse_fraction": float(remaining_fraction),
            "explained_sse_fraction": float(1.0 - remaining_fraction),
        }
    return result


def _decision(overall: dict[str, dict[str, float]]) -> dict[str, Any]:
    """按三阶段路线的预登记阈值给出下一条研究重点。"""

    linear = overall["linear_unanchored"]["pooled_rmse"]
    quadratic = overall["quadratic_unanchored"]["pooled_rmse"]
    cubic = overall["cubic_unanchored"]["pooled_rmse"]
    control8 = overall["control8_unanchored"]["pooled_rmse"]
    low_dimensional_supported = linear < 7.0 and min(quadratic, cubic) < 6.0
    branch_selection_supported = control8 > 8.0
    if low_dimensional_supported:
        priority = "strict_oof_low_dimensional_residual_coefficients"
    elif branch_selection_supported:
        priority = "pf_path_mode_and_branch_selection"
    else:
        priority = "mixed_residual_and_path_evidence"
    return {
        "linear_unanchored_rmse": float(linear),
        "quadratic_unanchored_rmse": float(quadratic),
        "cubic_unanchored_rmse": float(cubic),
        "control8_unanchored_high_frequency_rmse": float(control8),
        "low_dimensional_residual_route_supported": bool(low_dimensional_supported),
        "path_branch_selection_route_supported": bool(branch_selection_supported),
        "priority": priority,
        "next_registered_diagnostic": "P3_D01_pf_observation_weight_audit_v1",
    }


def run_audit(
    predictions: pd.DataFrame,
    source_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """逐井拟合全部固定表示，并返回逐井、逐折和总体摘要。"""

    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"开发预测缺列：{missing_columns}")
    if predictions.duplicated(["well_id", "row_index"]).any():
        raise ValueError("开发预测含重复自然隐藏行键")
    if len(predictions) == 0:
        raise ValueError("开发预测为空")

    well_rows = []
    for _well_id, well_frame in predictions.groupby("well_id", sort=True):
        ordered = well_frame.sort_values("row_index", kind="stable")
        well_rows.append(_one_well_row(ordered))
    per_well = pd.DataFrame(well_rows).sort_values("well_id").reset_index(drop=True)

    fold_rows: list[dict[str, Any]] = []
    for fold_id, fold_wells in per_well.groupby("fold", sort=True):
        fold_metrics = _aggregate_subset(fold_wells)
        for representation, metrics in fold_metrics.items():
            fold_rows.append(
                {
                    "fold": int(fold_id),
                    "representation": representation,
                    "wells": int(len(fold_wells)),
                    "rows": int(fold_wells["rows"].sum()),
                    **metrics,
                }
            )
    per_fold = pd.DataFrame(fold_rows)
    development_overall = _aggregate_subset(per_well)
    full_micro = float(source_metrics["overall"]["micro_rmse"])
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "lineage_full_773_micro_rmse": full_micro,
        "development_wells": int(len(per_well)),
        "development_rows": int(per_well["rows"].sum()),
        "development_folds": sorted(int(value) for value in per_well["fold"].unique()),
        "development_overall": development_overall,
        "high_frequency_residual_definition": "control8_unanchored",
        "decision": _decision(development_overall),
    }
    return per_well, per_fold, summary


def _sse_table(summary: dict[str, Any]) -> pd.DataFrame:
    """把总体嵌套指标整理成便于阅读的一行一表示表。"""

    rows = []
    for representation, metrics in summary["development_overall"].items():
        rows.append({"representation": representation, **metrics})
    return pd.DataFrame(rows)


def _conclusion_text(summary: dict[str, Any]) -> str:
    """按照事实、推断和证据边界写简短中文结论。"""

    metrics = summary["development_overall"]
    decision = summary["decision"]
    return f"""# P3-D00 结论

## 事实

- P2-P02 全 773 井血缘分数：`{summary['lineage_full_773_micro_rmse']:.10f}`。
- 本次严格排除 shadow 后使用 `{summary['development_wells']}` 口开发井、`{summary['development_rows']:,}` 个自然隐藏行。
- 开发井 P2-P02 基线：`{metrics['baseline']['pooled_rmse']:.6f}`。
- 常数 oracle：`{metrics['constant_unanchored']['pooled_rmse']:.6f}`。
- 一次 oracle：`{metrics['linear_unanchored']['pooled_rmse']:.6f}`。
- 二次 oracle：`{metrics['quadratic_unanchored']['pooled_rmse']:.6f}`。
- 三次 oracle：`{metrics['cubic_unanchored']['pooled_rmse']:.6f}`。
- 4 控制点 oracle：`{metrics['control4_unanchored']['pooled_rmse']:.6f}`。
- 8 控制点 oracle，也就是本实验定义的高频剩余 RMSE：`{metrics['control8_unanchored']['pooled_rmse']:.6f}`。

## 推断

预登记决策：`{decision['priority']}`。

## 仍未验证

本实验只证明残差的表示上限，没有证明测试期合法信息能够预测这些 oracle 系数。

## 当前只能否定

若某种基函数上限不足，只能否定该固定低维表示，不能否定 PF 分段权重、多峰或其他连续残差路线。

## 下一步

先运行只读 `P3-D01`，审计真实/插值 GR、似然尺度和有效路径数；不直接把本实验 oracle 系数加入模型。
"""


def validate_config(config: dict[str, Any]) -> None:
    """阻止 D00 静默改变基线、折、shadow 数或表示集合。"""

    expected = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "P3B00_group5_p2p02_v1",
        "source_experiment_id": "P2_P02_multiscale_pf_paths_v1",
        "fold_version": "balanced_well_5fold_v1",
        "expected_total_wells": 773,
        "expected_total_rows": 3_783_989,
        "expected_shadow_wells": 116,
        "expected_development_wells": 657,
        "formal_feature_output": False,
        "model_training": False,
        "shadow_target_access": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"P3-D00 {key} 必须固定为 {value!r}")
    if config.get("representations") != REPRESENTATIONS:
        raise ValueError("P3-D00 表示列表与代码冻结顺序不一致")


def main(argv: list[str] | None = None) -> None:
    """读取冻结来源，运行一次开发集 oracle 审计并保存独立产物。"""

    args = parse_args(argv)
    started = time.time()
    config = read_json(args.config.resolve())
    validate_config(config)
    prediction_path = (CLEAN_ROOT / str(config["source_predictions"])).resolve()
    source_metrics_path = (CLEAN_ROOT / str(config["source_metrics"])).resolve()
    shadow_path = (CLEAN_ROOT / str(config["shadow_holdout"])).resolve()
    output_dir = args.output_dir.resolve()

    print("P3-D00：读取时排除 shadow，再拟合开发井残差", flush=True)
    predictions = load_development_predictions(prediction_path, shadow_path)
    expected_development_wells = int(config["expected_development_wells"])
    actual_development_wells = int(predictions["well_id"].nunique())
    if actual_development_wells != expected_development_wells:
        raise ValueError(
            f"开发井数量应为 {expected_development_wells}，实际 {actual_development_wells}"
        )
    source_metrics = read_json(source_metrics_path)
    per_well, per_fold, summary = run_audit(predictions, source_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary.json", summary)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)
    _sse_table(summary).to_csv(output_dir / "sse_decomposition.csv", index=False)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "seconds": float(time.time() - started),
        "development_wells": actual_development_wells,
        "development_rows": int(len(predictions)),
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": file_sha256(prediction_path),
        "source_metrics": str(source_metrics_path),
        "source_metrics_sha256": file_sha256(source_metrics_path),
        "shadow_holdout": str(shadow_path),
        "shadow_holdout_sha256": file_sha256(shadow_path),
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "conclusion.md").write_text(
        _conclusion_text(summary),
        encoding="utf-8",
    )
    print(
        "P3-D00 完成："
        f"开发井={summary['development_wells']}，"
        f"基线={summary['development_overall']['baseline']['pooled_rmse']:.6f}，"
        f"三次={summary['development_overall']['cubic_unanchored']['pooled_rmse']:.6f}，"
        f"8控制点={summary['development_overall']['control8_unanchored']['pooled_rmse']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
