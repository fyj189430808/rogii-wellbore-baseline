"""运行 UP03：UP01 路径叠加已锁定的固定 PFS 修正。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pfs02_conservative_postblend import (
    development_registry,
    file_sha256,
    load_base_legal_rows,
    load_pfs_legal_rows,
    load_targets,
    read_json,
    resolve_path,
    shadow_ids,
    write_json,
)
from src.p3_up03_up01_plus_pfs_correction import build_up03_candidate, evaluate_confirmation_gate


EXPERIMENT_ID = "P3_UP03_up01_plus_pfs_correction_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up03_up01_plus_pfs_correction_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
UP01_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "md",
    "p2_pred_tvt",
    "degree2_blend50_pred_tvt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def load_up01_legal(path: Path, folds: list[int]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    frame = dataset.to_table(
        columns=UP01_COLUMNS,
        filter=ds.field("fold").isin(folds),
    ).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("UP01 合法路径存在重复行键")
    return frame


def generate_legal_candidate(
    config: dict[str, Any],
    registry: pd.DataFrame,
    folds: list[int],
    output: Path,
    suffix: str,
) -> tuple[pd.DataFrame, Path]:
    excluded = shadow_ids(resolve_path(config["shadow_registry"]))
    up01 = load_up01_legal(resolve_path(config["source_up01_legal_candidates"]), folds)
    if set(up01["well_id"]).intersection(excluded):
        raise RuntimeError("UP01 合法路径混入影子井")
    p3b00 = load_base_legal_rows(resolve_path(config["source_predictions"]), folds, excluded)
    pfs = load_pfs_legal_rows(
        registry,
        folds,
        resolve_path(config["pfs_legal_cache_dir"]),
        resolve_path(config["pfs_runtime_dir"]),
        str(config["pfs_fingerprint"]),
    )
    keys = ["well_id", "fold", "row_index"]
    merged = up01.merge(
        p3b00[keys + ["p3b00_pred_tvt"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    ).merge(pfs, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(up01) or len(merged) != len(p3b00) or len(merged) != len(pfs):
        raise ValueError("UP01、P3B00 与 PFS 行键不完全一致")
    if not np.allclose(
        merged["p2_pred_tvt"].to_numpy(dtype=np.float64),
        merged["p3b00_pred_tvt"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("UP01 内保存的 P3B00 路径与冻结 OOF 不一致")
    pfs_absolute = (
        merged["last_visible_tvt"].to_numpy(dtype=np.float64)
        + merged["pfs_lag1000_delta"].to_numpy(dtype=np.float64)
    )
    merged["pfs_lag1000_abs_tvt"] = pfs_absolute
    merged["candidate_pred_tvt"] = build_up03_candidate(
        merged["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64),
        merged["p3b00_pred_tvt"].to_numpy(dtype=np.float64),
        pfs_absolute,
        correction_fraction=float(config["pfs_correction_fraction"]),
    )
    legal_columns = keys + [
        "md",
        "p3b00_pred_tvt",
        "degree2_blend50_pred_tvt",
        "pfs_lag1000_abs_tvt",
        "candidate_pred_tvt",
    ]
    legal = merged[legal_columns].copy()
    path = output / f"legal_candidates_{suffix}.parquet"
    temporary = path.with_suffix(path.suffix + ".tmp")
    legal.to_parquet(temporary, index=False)
    temporary.replace(path)
    write_json(
        output / f"legal_manifest_{suffix}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "folds": folds,
            "wells": int(legal["well_id"].nunique()),
            "rows": int(len(legal)),
            "candidate_formula": "UP01_degree2_blend50 + 0.25 * (PFS_lag1000_abs - P3B00)",
            "candidate_generation_complete": True,
            "hidden_target_read": False,
            "shadow_overlap_wells": int(legal["well_id"].isin(excluded).sum()),
            "pfs_runtime_validated_per_well": True,
            "legal_candidates_sha256": file_sha256(path),
        },
    )
    return legal, path


def score(legal: pd.DataFrame, targets: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    keys = ["well_id", "fold", "row_index"]
    scored = legal.merge(targets, on=keys, how="inner", validate="one_to_one")
    if len(scored) != len(legal) or len(scored) != len(targets):
        raise ValueError("合法候选与真值行键不完全一致")
    target = scored["target_tvt"].to_numpy(dtype=np.float64)
    baseline = scored["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64)
    candidate = scored["candidate_pred_tvt"].to_numpy(dtype=np.float64)
    baseline_pooled = rmse(target, baseline)
    candidate_pooled = rmse(target, candidate)
    per_fold_rows: list[dict[str, Any]] = []
    for fold, frame in scored.groupby("fold", sort=True):
        fold_target = frame["target_tvt"].to_numpy(dtype=np.float64)
        baseline_value = rmse(fold_target, frame["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64))
        candidate_value = rmse(fold_target, frame["candidate_pred_tvt"].to_numpy(dtype=np.float64))
        per_fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(frame)),
                "up01_rmse": baseline_value,
                "candidate_rmse": candidate_value,
                "improvement_ft": baseline_value - candidate_value,
                "degradation_ft": candidate_value - baseline_value,
            }
        )
    per_fold = pd.DataFrame(per_fold_rows)
    per_well_rows: list[dict[str, Any]] = []
    for (well_id, fold), frame in scored.groupby(["well_id", "fold"], sort=True):
        well_target = frame["target_tvt"].to_numpy(dtype=np.float64)
        baseline_value = rmse(well_target, frame["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64))
        candidate_value = rmse(well_target, frame["candidate_pred_tvt"].to_numpy(dtype=np.float64))
        per_well_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "rows": int(len(frame)),
                "up01_rmse": baseline_value,
                "candidate_rmse": candidate_value,
                "improvement_ft": baseline_value - candidate_value,
            }
        )
    per_well = pd.DataFrame(per_well_rows)
    metrics = {
        "folds": sorted(int(v) for v in scored["fold"].unique()),
        "wells": int(scored["well_id"].nunique()),
        "rows": int(len(scored)),
        "up01_pooled_rmse": baseline_pooled,
        "candidate_pooled_rmse": candidate_pooled,
        "pooled_improvement_ft": baseline_pooled - candidate_pooled,
        "fold_improvements_ft": [float(v) for v in per_fold["improvement_ft"]],
    }
    return metrics, per_fold, per_well


def conclusion(metrics: dict[str, Any]) -> str:
    gate = metrics["confirmation_gate"]
    confirmation_metrics = metrics.get("independent_confirmation_metrics", metrics)
    lines = [
        "# P3-UP03 结论",
        "",
        "数据直接证明的事实：",
        "",
        f"- 首先只评价未参与两条组成路径选择的 folds {metrics['confirmation_folds']}，影子集保持关闭。",
        f"- 确认折 UP01 RMSE 为 {confirmation_metrics['up01_pooled_rmse']:.6f} ft，UP03 为 {confirmation_metrics['candidate_pooled_rmse']:.6f} ft，合并改善 {confirmation_metrics['pooled_improvement_ft']:+.6f} ft。",
        f"- 两个确认折改善为 {confirmation_metrics['fold_improvements_ft']}；算术平均 {gate['arithmetic_mean_fold_improvement_ft']:+.6f} ft；确认门槛{'通过' if gate['passed'] else '未通过'}。",
        "",
        "基于事实的合理推断：",
        "",
        "- 只有确认折通过预登记门槛，才能把 UP01 与固定 PFS 修正的互补性视为值得继续汇总的证据。",
        "",
        "仍然没有验证的猜测：",
        "",
        "- 其他修正强度、其他滞后距离或自适应融合未在本实验中搜索。",
        "",
        "当前实验只能否定的具体实现：",
        "",
        "- 若失败，只否定 degree2_blend50 + 0.25×(lag1000−P3B00) 这一固定组合。",
        "",
        "下一步最便宜的验证：",
        "",
        "- 未通过则停止 UP03；通过才汇总 folds 0–2，并保持 folds 3–4 的候选和门槛不变。",
        "",
        "选择血缘：UP01 degree2_blend50 由 folds 1–2 选定；PFS lag1000、0.25 修正由 PFS02 folds 0–1 锁定；因此 folds 3–4 是唯一没有参与任何组成选择的确认折。",
    ]
    if "independent_confirmation_metrics" in metrics:
        lines.insert(
            8,
            f"- 完整开发集 UP01 RMSE 为 {metrics['up01_pooled_rmse']:.6f} ft，UP03 为 {metrics['candidate_pooled_rmse']:.6f} ft，合并改善 {metrics['pooled_improvement_ft']:+.6f} ft。",
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("model_training") is not False:
        raise ValueError("实验配置不匹配或意外启用了训练")
    output = args.output_dir.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() and not args.force:
        print(f"已有结果：{output / 'metrics.json'}")
        return
    shutil.copyfile(args.config.resolve(), output / "config.json")
    start = time.perf_counter()
    up01_path = resolve_path(config["source_up01_legal_candidates"])
    if file_sha256(up01_path) != config["source_up01_sha256"]:
        raise RuntimeError("UP01 合法候选哈希不匹配")
    up01_manifest = read_json(resolve_path(config["source_up01_manifest"]))
    if up01_manifest.get("hidden_target_read") is not False or up01_manifest.get("shadow_overlap_wells") != 0:
        raise RuntimeError("UP01 合法候选血缘不合格")
    prediction_path = resolve_path(config["source_predictions"])
    if file_sha256(prediction_path) != config["source_predictions_sha256"]:
        raise RuntimeError("P3B00 OOF 哈希不匹配")
    shadow_path = resolve_path(config["shadow_registry"])
    excluded = shadow_ids(shadow_path)
    registry = development_registry(resolve_path(config["fold_registry"]), shadow_path)

    confirmation_folds = [int(v) for v in config["independent_confirmation_folds"]]
    print("阶段 A：只生成 folds 3–4 固定候选，不读取 target_tvt。", flush=True)
    legal_confirmation, confirmation_path = generate_legal_candidate(
        config, registry, confirmation_folds, output, "folds34"
    )
    print("阶段 B：确认候选已落盘，现在才读取 folds 3–4 开发真值。", flush=True)
    targets_confirmation = load_targets(prediction_path, confirmation_folds, excluded)
    metrics, per_fold, per_well = score(legal_confirmation, targets_confirmation)
    gate_config = config["confirmation_gate"]
    gate = evaluate_confirmation_gate(
        metrics["fold_improvements_ft"],
        float(gate_config["minimum_arithmetic_mean_fold_improvement_ft"]),
        float(gate_config["maximum_any_fold_degradation_ft"]),
    )
    metrics.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "confirmation_folds": confirmation_folds,
            "confirmation_gate": gate,
            "model_training": False,
            "shadow_target_read": False,
            "candidate_generation_hidden_target_read": False,
            "selection_lineage": config["selection_lineage"],
        }
    )
    per_fold.to_csv(output / "per_fold_folds34.csv", index=False)
    per_well.to_csv(output / "per_well_folds34.csv", index=False)
    write_json(output / "metrics_folds34.json", metrics)
    if not gate["passed"]:
        per_fold.to_csv(output / "per_fold.csv", index=False)
        per_well.to_csv(output / "per_well.csv", index=False)
        write_json(output / "metrics.json", metrics)
        (output / "conclusion.md").write_text(conclusion(metrics), encoding="utf-8")
        write_json(
            output / "runtime.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "elapsed_seconds": float(time.perf_counter() - start),
                "stopped_after_independent_confirmation": True,
                "confirmation_legal_candidates": str(confirmation_path),
                "shadow_target_read": False,
            },
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return

    summary_folds = [int(v) for v in config["summary_folds_after_pass"]]
    print("确认折通过；阶段 C 先生成 folds 0–2 合法候选，不读取其真值。", flush=True)
    legal_summary, summary_path = generate_legal_candidate(config, registry, summary_folds, output, "folds012")
    print("阶段 D：folds 0–2 候选已落盘，现在才读取其开发真值并汇总。", flush=True)
    targets_summary = load_targets(prediction_path, summary_folds, excluded)
    full_legal = pd.concat([legal_summary, legal_confirmation], ignore_index=True)
    full_targets = pd.concat([targets_summary, targets_confirmation], ignore_index=True)
    full_metrics, full_per_fold, full_per_well = score(full_legal, full_targets)
    full_metrics.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "confirmation_folds": confirmation_folds,
            "confirmation_gate": gate,
            "independent_confirmation_metrics": metrics,
            "model_training": False,
            "shadow_target_read": False,
            "candidate_generation_hidden_target_read": False,
            "selection_lineage": config["selection_lineage"],
        }
    )
    full_per_fold.to_csv(output / "per_fold.csv", index=False)
    full_per_well.to_csv(output / "per_well.csv", index=False)
    write_json(output / "metrics.json", full_metrics)
    (output / "conclusion.md").write_text(conclusion(full_metrics), encoding="utf-8")
    write_json(
        output / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start),
            "stopped_after_independent_confirmation": False,
            "confirmation_legal_candidates": str(confirmation_path),
            "summary_legal_candidates": str(summary_path),
            "shadow_target_read": False,
        },
    )
    print(json.dumps(full_metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
