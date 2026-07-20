"""执行 P3-UP04：固定 1:1 融合 UP01 全局路径与 UP02 局部路径。"""

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


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_up01_robust_u_projection import (  # noqa: E402
    file_sha256,
    load_targets_after_legal_generation,
    read_json,
    resolve_path,
    shadow_well_ids,
    write_json,
)
from src.p3_up04_equal_blend import equal_blend_paths  # noqa: E402


EXPERIMENT_ID = "P3_UP04_global_local_equal_blend_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up04_global_local_equal_blend_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
KEY_COLUMNS = ["well_id", "fold", "row_index"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P3-UP04 固定全局/局部路径等权融合")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def read_locked_candidate(
    metrics_path: Path,
    selected_field: str,
    selection_provenance_path: Path | None = None,
) -> tuple[str | None, list[int]]:
    """读取锁定候选；若主指标没保存该字段，则使用当时独立落盘的选择记录。"""

    metrics = read_json(metrics_path)
    selected = metrics.get(selected_field)
    if selected is not None:
        return str(selected), []
    if selection_provenance_path is None:
        return None, []
    provenance = read_json(selection_provenance_path)
    provenance_selected = provenance.get("selected_candidate")
    selection_folds = [int(value) for value in provenance.get("selection_folds", [])]
    return (
        str(provenance_selected) if provenance_selected is not None else None,
        selection_folds,
    )


def validate_source_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """确认两条源路径正是先前各自锁定的候选，且合法缓存不含影子目标。"""

    global_dir = resolve_path(str(Path(config["global_source"]).parent))
    local_dir = resolve_path(str(Path(config["local_source"]).parent))
    global_selection_path = resolve_path(str(config["global_selection_provenance"]))
    global_selected, recorded_global_folds = read_locked_candidate(
        global_dir / "metrics.json",
        selected_field="selected_by_folds12",
        selection_provenance_path=global_selection_path,
    )
    local_selected, _ = read_locked_candidate(
        local_dir / "metrics.json",
        selected_field="selected_by_folds01",
    )
    global_manifest = read_json(global_dir / "legal_generation_manifest.json")
    local_manifest = read_json(local_dir / "legal_generation_manifest.json")

    if global_selected != "degree2_blend50":
        raise ValueError("UP01 锁定候选不是 degree2_blend50")
    if recorded_global_folds != [1, 2]:
        raise ValueError("UP01 独立选择记录不是 folds [1, 2]")
    if local_selected != "smooth1000_blend75":
        raise ValueError("UP02 锁定候选不是 smooth1000_blend75")
    for source_name, manifest in (("UP01", global_manifest), ("UP02", local_manifest)):
        if manifest.get("hidden_target_read") is not False:
            raise ValueError(f"{source_name} 合法缓存生成阶段读取过隐藏目标")
        if int(manifest.get("shadow_overlap_wells", -1)) != 0:
            raise ValueError(f"{source_name} 合法缓存混入影子井")
        if int(manifest.get("wells", -1)) != 657:
            raise ValueError(f"{source_name} 合法缓存井数不是 657")
    return {
        "global_selected_by_folds": list(config["global_selection_folds"]),
        "global_selected_candidate": global_selected,
        "global_selection_provenance": str(global_selection_path),
        "local_selected_by_folds": list(config["local_selection_folds"]),
        "local_selected_candidate": local_selected,
        "global_manifest_hidden_target_read": global_manifest["hidden_target_read"],
        "local_manifest_hidden_target_read": local_manifest["hidden_target_read"],
    }


def build_legal_candidate(config: dict[str, Any]) -> pd.DataFrame:
    """只读取两条合法路径和行键，逐行生成固定算术平均。"""

    global_path = resolve_path(str(config["global_source"]))
    local_path = resolve_path(str(config["local_source"]))
    global_column = str(config["global_source_column"])
    local_column = str(config["local_source_column"])
    global_rows = pd.read_parquet(global_path, columns=KEY_COLUMNS + [global_column])
    local_rows = pd.read_parquet(local_path, columns=KEY_COLUMNS + [local_column])
    global_rows["well_id"] = global_rows["well_id"].astype(str)
    local_rows["well_id"] = local_rows["well_id"].astype(str)
    global_rows = global_rows.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    local_rows = local_rows.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)

    if len(global_rows) != len(local_rows):
        raise ValueError("UP01 与 UP02 合法路径行数不一致")
    for key in KEY_COLUMNS:
        if not np.array_equal(global_rows[key].to_numpy(), local_rows[key].to_numpy()):
            raise ValueError(f"UP01 与 UP02 的 {key} 行键不一致")

    global_prediction = global_rows[global_column].to_numpy(dtype=np.float64)
    local_prediction = local_rows[local_column].to_numpy(dtype=np.float64)
    blended_prediction = equal_blend_paths(global_prediction, local_prediction)
    output = global_rows[KEY_COLUMNS].copy()
    output["up01_degree2_blend50_pred_tvt"] = global_prediction
    output["up02_smooth1000_blend75_pred_tvt"] = local_prediction
    output["equal_blend_pred_tvt"] = blended_prediction
    return output


def score_paths(legal: pd.DataFrame, targets: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scored = legal.merge(targets, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(scored) != len(legal) or len(scored) != len(targets):
        raise ValueError("合法候选与目标行键未完全对齐")

    baseline_column = "up01_degree2_blend50_pred_tvt"
    local_column = "up02_smooth1000_blend75_pred_tvt"
    candidate_column = "equal_blend_pred_tvt"
    target = scored["target_tvt"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(target, scored[baseline_column].to_numpy(dtype=np.float64))
    local_rmse = rmse(target, scored[local_column].to_numpy(dtype=np.float64))
    candidate_rmse = rmse(target, scored[candidate_column].to_numpy(dtype=np.float64))

    per_fold_rows: list[dict[str, Any]] = []
    for fold, fold_rows in scored.groupby("fold", sort=True):
        fold_target = fold_rows["target_tvt"].to_numpy(dtype=np.float64)
        fold_baseline = rmse(fold_target, fold_rows[baseline_column].to_numpy(dtype=np.float64))
        for method, column in (
            ("up01_degree2_blend50", baseline_column),
            ("up02_smooth1000_blend75", local_column),
            ("equal_blend", candidate_column),
        ):
            method_rmse = rmse(fold_target, fold_rows[column].to_numpy(dtype=np.float64))
            per_fold_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "rows": int(len(fold_rows)),
                    "baseline_rmse": fold_baseline,
                    "rmse": method_rmse,
                    "improvement_vs_up01_ft": fold_baseline - method_rmse,
                    "degradation_vs_up01_ft": method_rmse - fold_baseline,
                }
            )
    per_fold = pd.DataFrame(per_fold_rows)

    per_well_rows: list[dict[str, Any]] = []
    for (well_id, fold), well_rows in scored.groupby(["well_id", "fold"], sort=True):
        well_target = well_rows["target_tvt"].to_numpy(dtype=np.float64)
        well_baseline = rmse(well_target, well_rows[baseline_column].to_numpy(dtype=np.float64))
        well_local = rmse(well_target, well_rows[local_column].to_numpy(dtype=np.float64))
        well_candidate = rmse(well_target, well_rows[candidate_column].to_numpy(dtype=np.float64))
        per_well_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "rows": int(len(well_rows)),
                "up01_rmse": well_baseline,
                "up02_rmse": well_local,
                "equal_blend_rmse": well_candidate,
                "improvement_vs_up01_ft": well_baseline - well_candidate,
            }
        )
    per_well = pd.DataFrame(per_well_rows)

    candidate_fold_rows = per_fold.loc[per_fold["method"] == "equal_blend"].sort_values("fold")
    fold_improvements = candidate_fold_rows["improvement_vs_up01_ft"].to_numpy(dtype=np.float64)
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "folds": sorted(int(value) for value in scored["fold"].unique()),
        "wells": int(scored["well_id"].nunique()),
        "rows": int(len(scored)),
        "up01_baseline_rmse": baseline_rmse,
        "up02_local_rmse": local_rmse,
        "equal_blend_rmse": candidate_rmse,
        "pooled_improvement_vs_up01_ft": baseline_rmse - candidate_rmse,
        "arithmetic_mean_fold_improvement_vs_up01_ft": float(np.mean(fold_improvements)),
        "fold_improvements_vs_up01_ft": [float(value) for value in fold_improvements],
        "maximum_fold_degradation_vs_up01_ft": float(
            candidate_fold_rows["degradation_vs_up01_ft"].max()
        ),
        "model_training": False,
        "shadow_target_read": False,
    }
    return metrics, per_fold, per_well


def bootstrap_ci(per_well: pd.DataFrame, draws: int = 10000, seed: int = 29) -> list[float]:
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    baseline_sse = rows * per_well["up01_rmse"].to_numpy(dtype=np.float64) ** 2
    candidate_sse = rows * per_well["equal_blend_rmse"].to_numpy(dtype=np.float64) ** 2
    generator = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    well_count = len(per_well)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        indices = generator.integers(0, well_count, size=(end - start, well_count))
        sampled_rows = rows[indices].sum(axis=1)
        baseline_rmse = np.sqrt(baseline_sse[indices].sum(axis=1) / sampled_rows)
        candidate_rmse = np.sqrt(candidate_sse[indices].sum(axis=1) / sampled_rows)
        differences[start:end] = candidate_rmse - baseline_rmse
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def add_full_diagnostics(metrics: dict[str, Any], per_well: pd.DataFrame) -> None:
    baseline = per_well["up01_rmse"].to_numpy(dtype=np.float64)
    candidate = per_well["equal_blend_rmse"].to_numpy(dtype=np.float64)
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    gain = rows * (baseline**2 - candidate**2)
    total_gain = float(gain.sum())
    top_count = max(1, int(np.ceil(0.05 * len(per_well))))
    metrics["full_development_diagnostics"] = {
        "improved_folds": int(sum(value > 0.0 for value in metrics["fold_improvements_vs_up01_ft"])),
        "paired_well_bootstrap_candidate_minus_up01_ci95_ft": bootstrap_ci(per_well),
        "well_win_rate": float(np.mean(candidate < baseline)),
        "p90_well_rmse_degradation_ft": float(np.quantile(candidate, 0.90) - np.quantile(baseline, 0.90)),
        "strongest_5pct_gain_share": float(np.sort(gain)[::-1][:top_count].sum() / total_gain)
        if total_gain > 0.0
        else None,
    }


def conclusion_text(metrics: dict[str, Any]) -> str:
    confirmation = metrics["confirmation"] if "confirmation" in metrics else metrics
    passed = bool(confirmation["gate"]["passed"])
    lines = [
        "# P3-UP04 全局/局部路径固定等权融合结论",
        "",
        "数据直接证明的事实：",
        "",
        f"- 候选在读取真值前已覆盖 {metrics['legal_wells']} 口开发井、{metrics['legal_rows']} 行，影子重叠为 0。",
        "- 唯一候选严格固定为 0.5×UP01 degree2_blend50 + 0.5×UP02 smooth1000_blend75。",
        f"- 完全未参与两条源路径选择的 folds 3-4：UP01 RMSE {confirmation['up01_baseline_rmse']:.6f}，等权融合 RMSE {confirmation['equal_blend_rmse']:.6f}，合并改善 {confirmation['pooled_improvement_vs_up01_ft']:+.6f} ft。",
        f"- folds 3-4 算术平均改善 {confirmation['arithmetic_mean_fold_improvement_vs_up01_ft']:+.6f} ft；逐折为 {confirmation['fold_improvements_vs_up01_ft']}。",
        f"- 预登记确认门槛：{'通过' if passed else '未通过'}。",
        "",
        "基于事实的合理推断：",
        "",
    ]
    if passed and "confirmation" in metrics:
        lines.append(
            f"- 门槛通过后汇总五折：UP01 {metrics['up01_baseline_rmse']:.6f} → 等权融合 {metrics['equal_blend_rmse']:.6f}，改善 {metrics['pooled_improvement_vs_up01_ft']:+.6f} ft。"
        )
    else:
        lines.append("- 固定等权融合没有达到继续汇总其他折的预登记证据要求。")
    lines.extend(
        [
            "",
            "仍然没有验证的猜测：",
            "",
            "- 其他融合权重是否更优；本实验禁止搜索权重，因此不能回答。",
            "",
            "当前实验只能否定的具体实现：",
            "",
            "- 若失败，只能否定这两条锁定路径的固定 1:1 融合，不能否定全局与局部路径互补。",
            "",
            "下一步最便宜的验证：",
            "",
            "- 若确认通过，再判断是否把固定等权路径作为冻结 LightGBM 的单一新增路径特征。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("配置实验编号不匹配")
    if config.get("blend") != {"global_weight": 0.5, "local_weight": 0.5}:
        raise ValueError("UP04 权重必须固定为 0.5/0.5")
    if config.get("model_training") is not False or config.get("shadow_target_read") is not False:
        raise ValueError("UP04 禁止训练模型或读取影子目标")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists() and not args.force:
        print(f"实验已经完成：{output_dir / 'metrics.json'}")
        return
    shutil.copyfile(args.config, output_dir / "config.json")
    start = time.perf_counter()

    provenance = validate_source_provenance(config)
    print("阶段 A：只读取两条合法源路径，生成 657 口井固定等权候选；不读取 target_tvt。", flush=True)
    legal = build_legal_candidate(config)
    shadow_ids = shadow_well_ids(resolve_path(str(config["shadow_registry"])))
    if set(legal["well_id"].astype(str)).intersection(shadow_ids):
        raise RuntimeError("UP04 合法候选混入影子井")
    if legal["well_id"].nunique() != 657:
        raise ValueError("UP04 合法候选井数不是 657")

    legal_path = output_dir / "legal_candidates.parquet"
    temporary_path = output_dir / "legal_candidates.parquet.tmp"
    legal.to_parquet(temporary_path, index=False)
    temporary_path.replace(legal_path)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_generation_complete": True,
        "hidden_target_read": False,
        "shadow_target_read": False,
        "shadow_overlap_wells": 0,
        "wells": int(legal["well_id"].nunique()),
        "rows": int(len(legal)),
        "candidate_formula": "0.5 * UP01_degree2_blend50 + 0.5 * UP02_smooth1000_blend75",
        "columns": list(legal.columns),
        "provenance": provenance,
        "legal_candidates_sha256": file_sha256(legal_path),
        "global_source_sha256": file_sha256(resolve_path(str(config["global_source"]))),
        "local_source_sha256": file_sha256(resolve_path(str(config["local_source"]))),
        "config_sha256": file_sha256(output_dir / "config.json"),
    }
    write_json(output_dir / "legal_generation_manifest.json", manifest)
    print(f"阶段 A 完成：{manifest['wells']} 口、{manifest['rows']} 行已落盘；现在只读 folds 3-4 真值。", flush=True)

    prediction_path = resolve_path(str(config["source_predictions"]))
    confirmation_folds = [int(value) for value in config["confirmation_folds"]]
    if confirmation_folds != [3, 4]:
        raise ValueError("确认折必须固定为 [3, 4]")
    confirmation_legal = legal.loc[legal["fold"].isin(confirmation_folds)].copy()
    confirmation_targets = load_targets_after_legal_generation(
        prediction_path,
        confirmation_folds,
        shadow_ids,
    )
    confirmation_metrics, confirmation_per_fold, confirmation_per_well = score_paths(
        confirmation_legal,
        confirmation_targets,
    )
    gate = config["gate"]
    passed = (
        confirmation_metrics["arithmetic_mean_fold_improvement_vs_up01_ft"]
        >= float(gate["minimum_arithmetic_mean_fold_improvement_ft"])
        and confirmation_metrics["maximum_fold_degradation_vs_up01_ft"]
        <= float(gate["maximum_any_fold_degradation_ft"])
    )
    confirmation_metrics["gate"] = {
        "minimum_arithmetic_mean_fold_improvement_ft": float(
            gate["minimum_arithmetic_mean_fold_improvement_ft"]
        ),
        "maximum_any_fold_degradation_ft": float(gate["maximum_any_fold_degradation_ft"]),
        "passed": bool(passed),
    }
    write_json(output_dir / "metrics_folds34_confirmation.json", confirmation_metrics)
    confirmation_per_fold.to_csv(output_dir / "per_fold_folds34_confirmation.csv", index=False)
    confirmation_per_well.to_csv(output_dir / "per_well_folds34_confirmation.csv", index=False)
    print(
        f"folds 3-4：平均改善={confirmation_metrics['arithmetic_mean_fold_improvement_vs_up01_ft']:+.6f}，门槛通过={passed}",
        flush=True,
    )

    if not passed:
        confirmation_metrics["legal_wells"] = manifest["wells"]
        confirmation_metrics["legal_rows"] = manifest["rows"]
        confirmation_metrics["other_fold_targets_read"] = False
        write_json(output_dir / "metrics.json", confirmation_metrics)
        confirmation_per_fold.to_csv(output_dir / "per_fold.csv", index=False)
        confirmation_per_well.to_csv(output_dir / "per_well.csv", index=False)
        (output_dir / "conclusion.md").write_text(
            conclusion_text(confirmation_metrics), encoding="utf-8"
        )
    else:
        print("确认门槛通过：现在才读取 folds 0-2 目标并汇总完整开发五折。", flush=True)
        remaining_folds = [int(value) for value in config["remaining_folds_after_gate"]]
        if remaining_folds != [0, 1, 2]:
            raise ValueError("门槛后剩余折必须固定为 [0, 1, 2]")
        remaining_targets = load_targets_after_legal_generation(
            prediction_path,
            remaining_folds,
            shadow_ids,
        )
        full_targets = pd.concat([confirmation_targets, remaining_targets], ignore_index=True)
        full_metrics, full_per_fold, full_per_well = score_paths(legal, full_targets)
        add_full_diagnostics(full_metrics, full_per_well)
        full_metrics["confirmation"] = confirmation_metrics
        full_metrics["legal_wells"] = manifest["wells"]
        full_metrics["legal_rows"] = manifest["rows"]
        full_metrics["other_fold_targets_read_after_gate"] = True
        write_json(output_dir / "metrics.json", full_metrics)
        full_per_fold.to_csv(output_dir / "per_fold.csv", index=False)
        full_per_well.to_csv(output_dir / "per_well.csv", index=False)
        (output_dir / "conclusion.md").write_text(conclusion_text(full_metrics), encoding="utf-8")

    write_json(
        output_dir / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start),
            "model_training": False,
            "shadow_target_read": False,
            "truth_access_order": "all legal candidates -> folds 3-4 gate -> folds 0-2 only if passed",
        },
    )
    print(f"结果目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
