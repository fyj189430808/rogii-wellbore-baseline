"""运行 UP01：预测路径 U 的二/三次稳健投影及固定比例融合。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_up01_robust_u_projection import build_projection_candidates  # noqa: E402


EXPERIMENT_ID = "P3_UP01_robust_u_projection_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up01_robust_u_projection_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
LEGAL_P2_COLUMNS = ["well_id", "fold", "row_index", "md", "pred_tvt"]
TARGET_COLUMNS = ["well_id", "fold", "row_index", "target_tvt"]
RAW_HORIZONTAL_COLUMNS = ["MD", "Z", "TVT_input"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P3-UP01 预测路径稳健投影")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def shadow_well_ids(path: Path) -> set[str]:
    registry = pd.read_csv(path, usecols=lambda name: name in {"well_id", "is_shadow"})
    if "well_id" not in registry.columns:
        raise ValueError("影子集登记表缺少 well_id")
    if "is_shadow" not in registry.columns:
        return set(registry["well_id"].astype(str))
    flags = registry["is_shadow"]
    if flags.dtype == bool:
        mask = flags.to_numpy()
    else:
        mask = flags.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
    return set(registry.loc[mask, "well_id"].astype(str))


def load_legal_p2_rows(
    prediction_path: Path,
    folds: list[int],
    shadow_ids: set[str],
) -> pd.DataFrame:
    """Arrow 层只读取合法列；此函数的列清单中物理不存在 target_tvt。"""

    dataset = arrow_dataset.dataset(prediction_path, format="parquet")
    missing = sorted(set(LEGAL_P2_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"P2-P02 预测缺少合法列：{missing}")
    row_filter = arrow_dataset.field("fold").isin([int(value) for value in folds])
    if shadow_ids:
        row_filter = row_filter & ~arrow_dataset.field("well_id").isin(sorted(shadow_ids))
    frame = dataset.to_table(columns=LEGAL_P2_COLUMNS, filter=row_filter).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if set(frame["fold"].astype(int).unique()) != set(folds):
        raise ValueError("合法路径没有覆盖预登记 folds")
    if set(frame["well_id"].unique()).intersection(shadow_ids):
        raise RuntimeError("合法路径混入影子井")
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("P2-P02 合法路径存在重复行键")
    numeric = frame[["fold", "row_index", "md", "pred_tvt"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("P2-P02 合法路径含 NaN 或无穷值")
    return frame


def attach_z_without_target(legal_rows: pd.DataFrame, raw_train_dir: Path) -> pd.DataFrame:
    """逐井只读取 MD/Z/TVT_input，并用原文件行号对齐隐藏行。"""

    z_values = np.empty(len(legal_rows), dtype=np.float64)
    for well_number, (well_id, well_frame) in enumerate(legal_rows.groupby("well_id", sort=False), 1):
        horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=RAW_HORIZONTAL_COLUMNS)
        row_indices = well_frame["row_index"].to_numpy(dtype=np.int64)
        if int(row_indices.min()) < 0 or int(row_indices.max()) >= len(horizontal):
            raise ValueError(f"{well_id} 的 row_index 超出原始水平井范围")
        selected = horizontal.iloc[row_indices]
        if selected["TVT_input"].notna().any():
            raise ValueError(f"{well_id} 的评价行不是自然隐藏区")
        source_md = selected["MD"].to_numpy(dtype=np.float64)
        prediction_md = well_frame["md"].to_numpy(dtype=np.float64)
        if not np.allclose(source_md, prediction_md, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"{well_id} 的 MD 与 P2-P02 行键不一致")
        z_values[well_frame.index.to_numpy(dtype=np.int64)] = selected["Z"].to_numpy(dtype=np.float64)
        if well_number % 50 == 0:
            print(f"合法候选生成：已读取 {well_number} 口井的 MD/Z", flush=True)
    if not np.isfinite(z_values).all():
        raise ValueError("合法 Z 含 NaN 或无穷值")
    result = legal_rows.copy()
    result["z"] = z_values
    return result


def generate_all_legal_candidates(
    legal_rows: pd.DataFrame,
    degrees: tuple[int, ...],
    blends: tuple[float, ...],
) -> pd.DataFrame:
    """在读入任何隐藏目标之前，一次性生成全部六条预登记候选。"""

    candidate_names = [
        f"degree{degree}_blend{int(round(100.0 * blend)):02d}"
        for degree in degrees
        for blend in blends
    ]
    candidate_arrays = {
        candidate_name: np.empty(len(legal_rows), dtype=np.float64)
        for candidate_name in candidate_names
    }
    for well_number, (well_id, well_frame) in enumerate(legal_rows.groupby("well_id", sort=False), 1):
        row_positions = well_frame.index.to_numpy(dtype=np.int64)
        md = well_frame["md"].to_numpy(dtype=np.float64)
        z = well_frame["z"].to_numpy(dtype=np.float64)
        base_tvt = well_frame["pred_tvt"].to_numpy(dtype=np.float64)
        base_u = base_tvt + z
        well_candidates = build_projection_candidates(
            md,
            base_u,
            degrees=degrees,
            blend_fractions=blends,
        )
        if list(well_candidates) != candidate_names:
            raise RuntimeError(f"{well_id} 的候选顺序偏离预登记合同")
        for candidate_name, candidate_u in well_candidates.items():
            candidate_arrays[candidate_name][row_positions] = candidate_u - z
        if well_number % 50 == 0:
            print(f"合法候选生成：已投影 {well_number} 口井", flush=True)

    output = legal_rows[["well_id", "fold", "row_index", "md", "pred_tvt"]].copy()
    output = output.rename(columns={"pred_tvt": "p2_pred_tvt"})
    for candidate_name in candidate_names:
        column_name = f"{candidate_name}_pred_tvt"
        output[column_name] = candidate_arrays[candidate_name]
        if not np.isfinite(output[column_name].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"候选 {candidate_name} 含 NaN 或无穷值")
    return output


def load_targets_after_legal_generation(
    prediction_path: Path,
    folds: list[int],
    shadow_ids: set[str],
) -> pd.DataFrame:
    """唯一允许读取 target_tvt 的函数，只能在合法候选落盘后调用。"""

    dataset = arrow_dataset.dataset(prediction_path, format="parquet")
    row_filter = arrow_dataset.field("fold").isin([int(value) for value in folds])
    if shadow_ids:
        row_filter = row_filter & ~arrow_dataset.field("well_id").isin(sorted(shadow_ids))
    targets = dataset.to_table(columns=TARGET_COLUMNS, filter=row_filter).to_pandas()
    targets["well_id"] = targets["well_id"].astype(str)
    if set(targets["well_id"].unique()).intersection(shadow_ids):
        raise RuntimeError("评分目标混入影子井")
    if targets.duplicated(["well_id", "row_index"]).any():
        raise ValueError("评分目标存在重复行键")
    return targets


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def score_candidates(
    legal_candidates: pd.DataFrame,
    targets: pd.DataFrame,
    candidate_names: list[str],
    minimum_mean_improvement: float,
    maximum_fold_degradation: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scored = legal_candidates.merge(
        targets,
        on=["well_id", "fold", "row_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(legal_candidates) or len(scored) != len(targets):
        raise ValueError("合法候选与评分目标行键不完全一致")

    target = scored["target_tvt"].to_numpy(dtype=np.float64)
    baseline = scored["p2_pred_tvt"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(target, baseline)
    method_columns = {"baseline": "p2_pred_tvt"}
    method_columns.update(
        {candidate_name: f"{candidate_name}_pred_tvt" for candidate_name in candidate_names}
    )

    per_well_rows: list[dict[str, Any]] = []
    for (well_id, fold), well_frame in scored.groupby(["well_id", "fold"], sort=True):
        well_target = well_frame["target_tvt"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "well_id": str(well_id),
            "fold": int(fold),
            "rows": int(len(well_frame)),
        }
        well_baseline_rmse = rmse(well_target, well_frame["p2_pred_tvt"].to_numpy(dtype=np.float64))
        row["baseline_rmse"] = well_baseline_rmse
        for candidate_name in candidate_names:
            candidate_rmse = rmse(
                well_target,
                well_frame[f"{candidate_name}_pred_tvt"].to_numpy(dtype=np.float64),
            )
            row[f"{candidate_name}_rmse"] = candidate_rmse
            row[f"{candidate_name}_improvement_ft"] = well_baseline_rmse - candidate_rmse
        per_well_rows.append(row)
    per_well = pd.DataFrame(per_well_rows)

    per_fold_rows: list[dict[str, Any]] = []
    for fold, fold_frame in scored.groupby("fold", sort=True):
        fold_target = fold_frame["target_tvt"].to_numpy(dtype=np.float64)
        fold_baseline_rmse = rmse(
            fold_target,
            fold_frame["p2_pred_tvt"].to_numpy(dtype=np.float64),
        )
        for method_name, column_name in method_columns.items():
            method_rmse = rmse(fold_target, fold_frame[column_name].to_numpy(dtype=np.float64))
            per_fold_rows.append(
                {
                    "fold": int(fold),
                    "method": method_name,
                    "rows": int(len(fold_frame)),
                    "baseline_rmse": fold_baseline_rmse,
                    "rmse": method_rmse,
                    "improvement_ft": fold_baseline_rmse - method_rmse,
                    "degradation_ft": method_rmse - fold_baseline_rmse,
                }
            )
    per_fold = pd.DataFrame(per_fold_rows)

    candidates_metrics: dict[str, Any] = {}
    gate_passed_candidates: list[str] = []
    for candidate_name in candidate_names:
        candidate_prediction = scored[f"{candidate_name}_pred_tvt"].to_numpy(dtype=np.float64)
        candidate_rmse = rmse(target, candidate_prediction)
        candidate_fold_rows = per_fold.loc[per_fold["method"] == candidate_name]
        fold_improvements = candidate_fold_rows["improvement_ft"].to_numpy(dtype=np.float64)
        fold_degradations = candidate_fold_rows["degradation_ft"].to_numpy(dtype=np.float64)
        mean_fold_improvement = float(np.mean(fold_improvements))
        worst_fold_degradation = float(np.max(fold_degradations))
        passed = (
            mean_fold_improvement >= minimum_mean_improvement
            and worst_fold_degradation <= maximum_fold_degradation
        )
        if passed:
            gate_passed_candidates.append(candidate_name)
        candidates_metrics[candidate_name] = {
            "pooled_rmse": candidate_rmse,
            "pooled_improvement_ft": baseline_rmse - candidate_rmse,
            "arithmetic_mean_fold_improvement_ft": mean_fold_improvement,
            "maximum_fold_degradation_ft": worst_fold_degradation,
            "fold_improvements_ft": [float(value) for value in fold_improvements],
            "gate_passed": bool(passed),
        }

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "model_training": False,
        "candidate_generation_hidden_target_read": False,
        "shadow_target_read": False,
        "folds": sorted(int(value) for value in scored["fold"].unique()),
        "wells": int(scored["well_id"].nunique()),
        "rows": int(len(scored)),
        "baseline_pooled_rmse": baseline_rmse,
        "candidates": candidates_metrics,
        "gate": {
            "minimum_arithmetic_mean_fold_improvement_ft": minimum_mean_improvement,
            "maximum_any_fold_degradation_ft": maximum_fold_degradation,
            "passed_candidates": gate_passed_candidates,
            "continue_other_folds": bool(gate_passed_candidates),
        },
    }
    return metrics, per_fold, per_well


def conclusion_text(metrics: dict[str, Any]) -> str:
    lines = [
        "# P3-UP01 结论",
        "",
        "数据直接证明的事实：",
        "",
        f"- 只评分开发集 folds {metrics['folds']}，共 {metrics['wells']} 口井、{metrics['rows']} 行。",
        f"- 同行 P2-P02 路径自身基线 RMSE 为 {metrics['baseline_pooled_rmse']:.6f} ft。",
        "- 六个预登记候选全部报告如下：",
        "",
        "| 候选 | pooled RMSE | pooled 改善 | 两折平均改善 | 最差折恶化 | 过门槛 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate_name, values in metrics["candidates"].items():
        lines.append(
            f"| {candidate_name} | {values['pooled_rmse']:.6f} | "
            f"{values['pooled_improvement_ft']:+.6f} | "
            f"{values['arithmetic_mean_fold_improvement_ft']:+.6f} | "
            f"{values['maximum_fold_degradation_ft']:+.6f} | "
            f"{'是' if values['gate_passed'] else '否'} |"
        )
    passed = metrics["gate"]["passed_candidates"]
    lines.extend(
        [
            "",
            "基于事实的合理推断：",
            "",
            (
                f"- 有候选通过 folds 1–2 门槛：{passed}，才允许继续其他折。"
                if passed
                else "- 没有候选达到平均改善 0.15 ft 且同时满足单折最大恶化 0.15 ft，按合同立即停止。"
            ),
            "",
            "仍然没有验证的猜测：",
            "",
            "- 更高阶投影、按井自适应融合或其他窗口是否有效，本实验没有检验。",
            "",
            "当前实验只能否定的具体实现：",
            "",
            "- 只能否定固定 Huber-IRLS、二/三次整段投影和 25%/50%/75% 融合这一组实现。",
            "",
            "下一步最便宜的验证：",
            "",
            "- 若无候选过门槛，停止 UP01 并回到主路线；若过门槛，再固定通过候选运行剩余开发折。",
            "",
            "泄漏边界：六条候选及 legal_candidates.parquet 在读取 target_tvt 前已全部生成并落盘；影子 116 口井在 Arrow 扫描阶段排除。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("配置实验编号不匹配")
    if config.get("model_training") is not False:
        raise ValueError("UP01 禁止训练模型")
    if config.get("touch_pfs_legal_cache") is not False:
        raise ValueError("UP01 禁止触碰 PFS legal_cache")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics_path = output_dir / "metrics.json"
    if final_metrics_path.exists() and not args.force:
        print(f"实验已完成：{final_metrics_path}")
        return

    # 先复制冻结配置；之后任何候选次数或比例都不允许根据真值变化。
    shutil.copyfile(args.config, output_dir / "config.json")
    start_time = time.perf_counter()
    prediction_path = resolve_path(str(config["source_predictions"]))
    shadow_path = resolve_path(str(config["shadow_registry"]))
    raw_train_dir = resolve_path(str(config["raw_train_dir"]))
    folds = [int(value) for value in config["development_folds"]]
    degrees = tuple(int(value) for value in config["polynomial_degrees"])
    blends = tuple(float(value) for value in config["blend_fractions"])
    expected_candidate_names = [
        f"degree{degree}_blend{int(round(100.0 * blend)):02d}"
        for degree in degrees
        for blend in blends
    ]

    shadow_ids = shadow_well_ids(shadow_path)
    print("阶段 A：只读合法列并生成全部候选，不读取 target_tvt。", flush=True)
    legal_p2 = load_legal_p2_rows(prediction_path, folds, shadow_ids)
    legal_with_z = attach_z_without_target(legal_p2, raw_train_dir)
    legal_candidates = generate_all_legal_candidates(legal_with_z, degrees, blends)
    legal_path = output_dir / "legal_candidates.parquet"
    temporary_legal_path = output_dir / "legal_candidates.parquet.tmp"
    legal_candidates.to_parquet(temporary_legal_path, index=False)
    temporary_legal_path.replace(legal_path)
    legal_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_generation_complete": True,
        "hidden_target_read": False,
        "shadow_wells_excluded": len(shadow_ids),
        "shadow_overlap_wells": 0,
        "folds": folds,
        "wells": int(legal_candidates["well_id"].nunique()),
        "rows": int(len(legal_candidates)),
        "legal_p2_columns_read": LEGAL_P2_COLUMNS,
        "raw_horizontal_columns_read": RAW_HORIZONTAL_COLUMNS,
        "candidate_names": expected_candidate_names,
        "legal_candidates_sha256": file_sha256(legal_path),
        "source_predictions_sha256": file_sha256(prediction_path),
        "config_sha256": file_sha256(output_dir / "config.json"),
    }
    write_json(output_dir / "legal_generation_manifest.json", legal_manifest)
    print(
        f"阶段 A 完成：{legal_manifest['wells']} 口井、{legal_manifest['rows']} 行；现在才允许读取评分真值。",
        flush=True,
    )

    print("阶段 B：读取同一开发行 target_tvt，仅用于最终评分。", flush=True)
    targets = load_targets_after_legal_generation(prediction_path, folds, shadow_ids)
    gate = config["gate"]
    metrics, per_fold, per_well = score_candidates(
        legal_candidates,
        targets,
        expected_candidate_names,
        float(gate["minimum_arithmetic_mean_fold_improvement_ft"]),
        float(gate["maximum_any_fold_degradation_ft"]),
    )
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    write_json(final_metrics_path, metrics)
    (output_dir / "conclusion.md").write_text(conclusion_text(metrics), encoding="utf-8")
    write_json(
        output_dir / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "model_training": False,
            "pfs_legal_cache_touched": False,
            "hidden_target_read_stage": "after_legal_candidate_parquet_and_manifest",
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
