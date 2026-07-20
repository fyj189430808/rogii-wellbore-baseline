"""UP01 通过 folds 1–2 后，按预登记继续剩余开发折并汇总五折。"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = CLEAN_ROOT / "artifacts" / "P3_UP01_robust_u_projection_v1"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_up01_robust_u_projection_v1.json"

import sys

sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_up01_robust_u_projection import (  # noqa: E402
    EXPERIMENT_ID,
    attach_z_without_target,
    conclusion_text,
    file_sha256,
    generate_all_legal_candidates,
    load_legal_p2_rows,
    load_targets_after_legal_generation,
    read_json,
    resolve_path,
    score_candidates,
    shadow_well_ids,
    write_json,
)


def bootstrap_metric_difference(
    per_well: pd.DataFrame,
    candidate_name: str,
    *,
    draws: int = 10000,
    seed: int = 29,
) -> tuple[float, float]:
    """按井重采样，每次将整口井的 SSE 与行数一起带入 pooled RMSE。"""

    rows = per_well["rows"].to_numpy(dtype=np.float64)
    baseline_sse = rows * per_well["baseline_rmse"].to_numpy(dtype=np.float64) ** 2
    candidate_sse = rows * per_well[f"{candidate_name}_rmse"].to_numpy(dtype=np.float64) ** 2
    random_generator = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    well_count = len(per_well)
    for draw_start in range(0, draws, 250):
        draw_end = min(draw_start + 250, draws)
        indices = random_generator.integers(0, well_count, size=(draw_end - draw_start, well_count))
        sampled_rows = rows[indices].sum(axis=1)
        baseline_rmse = np.sqrt(baseline_sse[indices].sum(axis=1) / sampled_rows)
        candidate_rmse = np.sqrt(candidate_sse[indices].sum(axis=1) / sampled_rows)
        differences[draw_start:draw_end] = candidate_rmse - baseline_rmse
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def add_full_development_gate(metrics: dict[str, Any], per_fold: pd.DataFrame, per_well: pd.DataFrame) -> None:
    """补充项目统一的开发集五折门槛；所有六候选仍逐个报告。"""

    passed_candidates: list[str] = []
    for candidate_name, values in metrics["candidates"].items():
        candidate_fold = per_fold.loc[per_fold["method"] == candidate_name].sort_values("fold")
        fold_improvements = candidate_fold["improvement_ft"].to_numpy(dtype=np.float64)
        improved_folds = int(np.sum(fold_improvements > 0.0))
        folds_2_to_4 = candidate_fold.loc[candidate_fold["fold"].isin([2, 3, 4]), "improvement_ft"]
        improved_folds_2_to_4 = int(np.sum(folds_2_to_4.to_numpy(dtype=np.float64) > 0.0))
        worst_fold_degradation = float(max(0.0, -float(np.min(fold_improvements))))
        candidate_rmse_values = per_well[f"{candidate_name}_rmse"].to_numpy(dtype=np.float64)
        baseline_rmse_values = per_well["baseline_rmse"].to_numpy(dtype=np.float64)
        win_rate = float(np.mean(candidate_rmse_values < baseline_rmse_values))
        p90_degradation = float(
            np.quantile(candidate_rmse_values, 0.90) - np.quantile(baseline_rmse_values, 0.90)
        )
        ci_low, ci_high = bootstrap_metric_difference(per_well, candidate_name)

        rows = per_well["rows"].to_numpy(dtype=np.float64)
        sse_gain = rows * (baseline_rmse_values**2 - candidate_rmse_values**2)
        total_gain = float(np.sum(sse_gain))
        strongest_count = max(1, int(np.ceil(0.05 * len(per_well))))
        strongest_gain = float(np.sort(sse_gain)[::-1][:strongest_count].sum())
        strongest_share = strongest_gain / total_gain if total_gain > 0.0 else float("inf")

        passed = (
            values["pooled_improvement_ft"] >= 0.10
            and improved_folds >= 4
            and improved_folds_2_to_4 >= 2
            and worst_fold_degradation <= 0.25
            and ci_high < 0.0
            and win_rate >= 0.55
            and p90_degradation <= 0.20
            and strongest_share <= 0.60
        )
        values["full_development_gate"] = {
            "improved_folds": improved_folds,
            "improved_folds_2_to_4": improved_folds_2_to_4,
            "worst_fold_degradation_ft": worst_fold_degradation,
            "paired_well_bootstrap_candidate_minus_baseline_ci95_ft": [ci_low, ci_high],
            "well_win_rate": win_rate,
            "p90_well_rmse_degradation_ft": p90_degradation,
            "strongest_5pct_gain_share": strongest_share,
            "passed": bool(passed),
        }
        if passed:
            passed_candidates.append(candidate_name)
    metrics["full_development_gate"] = {
        "passed_candidates": passed_candidates,
        "minimum_pooled_improvement_ft": 0.10,
        "minimum_improved_folds": 4,
        "minimum_improved_folds_2_to_4": 2,
        "maximum_fold_degradation_ft": 0.25,
        "maximum_bootstrap_ci_upper_ft": 0.0,
        "minimum_well_win_rate": 0.55,
        "maximum_p90_degradation_ft": 0.20,
        "maximum_strongest_5pct_gain_share": 0.60,
    }


def full_conclusion(metrics: dict[str, Any]) -> str:
    text = conclusion_text(metrics)
    extra = [
        "",
        "## 开发集完整五折补充判定",
        "",
        f"- 通过统一完整五折门槛的候选：{metrics['full_development_gate']['passed_candidates']}。",
        "- 这里仍是排除 116 口影子井后的 657 口开发集结果，不是 773 口正式主 CV。",
        "- 未读取影子井目标，也没有训练新模型。",
        "",
    ]
    return text + "\n".join(extra)


def main() -> None:
    start = time.perf_counter()
    config = read_json(CONFIG_PATH)
    screening_metrics = read_json(OUTPUT_DIR / "metrics.json")
    if not screening_metrics["gate"]["continue_other_folds"]:
        raise RuntimeError("folds 1–2 未过门槛，不允许继续剩余折")
    screening_folds = [int(value) for value in config["development_folds"]]
    continuation_folds = [int(value) for value in config["gate"]["continuation_folds_if_passed"]]
    all_folds = sorted(screening_folds + continuation_folds)
    if all_folds != [0, 1, 2, 3, 4] or set(screening_folds).intersection(continuation_folds):
        raise ValueError("预登记延续折必须与筛查折共同构成无重复五折")

    # 永久保留当时的 folds 1–2 筛查产物，证明延续决定发生在门槛通过之后。
    preserve_pairs = {
        "metrics.json": "metrics_folds12_screen.json",
        "per_fold.csv": "per_fold_folds12_screen.csv",
        "per_well.csv": "per_well_folds12_screen.csv",
        "conclusion.md": "conclusion_folds12_screen.md",
        "legal_candidates.parquet": "legal_candidates_folds12.parquet",
        "legal_generation_manifest.json": "legal_generation_manifest_folds12.json",
    }
    for source_name, destination_name in preserve_pairs.items():
        shutil.copyfile(OUTPUT_DIR / source_name, OUTPUT_DIR / destination_name)

    prediction_path = resolve_path(str(config["source_predictions"]))
    shadow_path = resolve_path(str(config["shadow_registry"]))
    raw_train_dir = resolve_path(str(config["raw_train_dir"]))
    shadow_ids = shadow_well_ids(shadow_path)
    degrees = tuple(int(value) for value in config["polynomial_degrees"])
    blends = tuple(float(value) for value in config["blend_fractions"])
    candidate_names = [
        f"degree{degree}_blend{int(round(100.0 * blend)):02d}"
        for degree in degrees
        for blend in blends
    ]

    print(f"延续阶段 A：生成剩余 folds {continuation_folds} 的合法候选，不读取目标。", flush=True)
    legal_p2 = load_legal_p2_rows(prediction_path, continuation_folds, shadow_ids)
    legal_with_z = attach_z_without_target(legal_p2, raw_train_dir)
    continuation_candidates = generate_all_legal_candidates(legal_with_z, degrees, blends)
    continuation_path = OUTPUT_DIR / "legal_candidates_continuation.parquet"
    continuation_candidates.to_parquet(continuation_path, index=False)
    write_json(
        OUTPUT_DIR / "legal_generation_manifest_continuation.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_generation_complete": True,
            "hidden_target_read": False,
            "shadow_overlap_wells": 0,
            "folds": continuation_folds,
            "wells": int(continuation_candidates["well_id"].nunique()),
            "rows": int(len(continuation_candidates)),
            "candidate_names": candidate_names,
            "legal_candidates_sha256": file_sha256(continuation_path),
        },
    )

    screening_candidates = pd.read_parquet(OUTPUT_DIR / "legal_candidates_folds12.parquet")
    full_candidates = pd.concat([screening_candidates, continuation_candidates], ignore_index=True)
    full_candidates = full_candidates.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if full_candidates.duplicated(["well_id", "row_index"]).any():
        raise ValueError("五折合法候选出现重复行键")
    temporary_full_path = OUTPUT_DIR / "legal_candidates.parquet.tmp"
    full_candidates.to_parquet(temporary_full_path, index=False)
    temporary_full_path.replace(OUTPUT_DIR / "legal_candidates.parquet")
    write_json(
        OUTPUT_DIR / "legal_generation_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_generation_complete": True,
            "hidden_target_read": False,
            "shadow_overlap_wells": 0,
            "folds": all_folds,
            "wells": int(full_candidates["well_id"].nunique()),
            "rows": int(len(full_candidates)),
            "candidate_names": candidate_names,
            "legal_candidates_sha256": file_sha256(OUTPUT_DIR / "legal_candidates.parquet"),
            "screening_gate_passed_before_continuation": True,
        },
    )
    print("延续阶段 A 完成：五折候选全部落盘；现在才读取五折开发目标。", flush=True)

    targets = load_targets_after_legal_generation(prediction_path, all_folds, shadow_ids)
    metrics, per_fold, per_well = score_candidates(
        full_candidates,
        targets,
        candidate_names,
        0.15,
        0.15,
    )
    metrics["screening_folds12"] = screening_metrics
    metrics["development_scope"] = "657 wells excluding frozen 116-well shadow holdout"
    add_full_development_gate(metrics, per_fold, per_well)
    per_fold.to_csv(OUTPUT_DIR / "per_fold.csv", index=False)
    per_well.to_csv(OUTPUT_DIR / "per_well.csv", index=False)
    write_json(OUTPUT_DIR / "metrics.json", metrics)
    (OUTPUT_DIR / "conclusion.md").write_text(full_conclusion(metrics), encoding="utf-8")
    runtime = read_json(OUTPUT_DIR / "runtime.json")
    runtime["full_development_continuation_seconds"] = float(time.perf_counter() - start)
    runtime["full_development_complete"] = True
    write_json(OUTPUT_DIR / "runtime.json", runtime)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
