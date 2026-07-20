"""执行 P3-UP02：固定局部平滑 U 路径，不训练模型。"""

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

from scripts.continue_p3_up01_full_development import add_full_development_gate  # noqa: E402
from scripts.run_p3_up01_robust_u_projection import (  # noqa: E402
    file_sha256,
    load_legal_p2_rows,
    load_targets_after_legal_generation,
    read_json,
    resolve_path,
    score_candidates,
    shadow_well_ids,
    write_json,
)
from src.p3_up02_local_u_smoothing import build_local_smoothing_candidates  # noqa: E402


EXPERIMENT_ID = "P3_UP02_local_u_smoothing_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up02_local_u_smoothing_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
RAW_HORIZONTAL_COLUMNS = ["MD", "Z", "TVT_input"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P3-UP02 固定局部 U 平滑实验")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def candidate_names(windows_ft: tuple[float, ...], blends: tuple[float, ...]) -> list[str]:
    return [
        f"smooth{int(round(window_ft))}_blend{int(round(100.0 * blend)):02d}"
        for window_ft in windows_ft
        for blend in blends
    ]


def generate_legal_candidates(
    legal_p2: pd.DataFrame,
    raw_train_dir: Path,
    windows_ft: tuple[float, ...],
    blends: tuple[float, ...],
    grid_step_ft: float,
    sigma_fraction: float,
) -> pd.DataFrame:
    """逐井拼接可见真实 U 与隐藏预测 U，然后只返回隐藏区候选 TVT。"""

    names = candidate_names(windows_ft, blends)
    arrays = {name: np.empty(len(legal_p2), dtype=np.float32) for name in names}

    for well_number, (well_id, well_rows) in enumerate(
        legal_p2.groupby("well_id", sort=False),
        start=1,
    ):
        horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=RAW_HORIZONTAL_COLUMNS)
        full_md = horizontal["MD"].to_numpy(dtype=np.float64)
        full_z = horizontal["Z"].to_numpy(dtype=np.float64)
        visible_tvt = horizontal["TVT_input"].to_numpy(dtype=np.float64)
        hidden_indices = well_rows["row_index"].to_numpy(dtype=np.int64)
        expected_hidden_indices = np.flatnonzero(np.isnan(visible_tvt))

        if not np.array_equal(hidden_indices, expected_hidden_indices):
            raise ValueError(f"{well_id}: P2 隐藏行与 TVT_input 缺失行不完全一致")
        if not np.allclose(
            full_md[hidden_indices],
            well_rows["md"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(f"{well_id}: P2 的 MD 与原始井文件不一致")

        full_u = visible_tvt + full_z
        hidden_prediction_u = (
            well_rows["pred_tvt"].to_numpy(dtype=np.float64) + full_z[hidden_indices]
        )
        full_u[hidden_indices] = hidden_prediction_u
        if not np.isfinite(full_md).all() or not np.isfinite(full_z).all():
            raise ValueError(f"{well_id}: MD/Z 含缺失或无穷值")
        if not np.isfinite(full_u).all():
            raise ValueError(f"{well_id}: 可见 U 与隐藏预测 U 未覆盖整口井")

        well_candidates = build_local_smoothing_candidates(
            full_md,
            full_u,
            smoothing_windows_ft=windows_ft,
            blend_fractions=blends,
            grid_step_ft=grid_step_ft,
            sigma_fraction_of_window=sigma_fraction,
        )
        if list(well_candidates) != names:
            raise RuntimeError(f"{well_id}: 候选顺序偏离预登记合同")

        output_positions = well_rows.index.to_numpy(dtype=np.int64)
        for name in names:
            candidate_tvt = well_candidates[name][hidden_indices] - full_z[hidden_indices]
            arrays[name][output_positions] = candidate_tvt.astype(np.float32)

        if well_number % 50 == 0 or well_number == legal_p2["well_id"].nunique():
            print(f"合法候选生成：{well_number} / {legal_p2['well_id'].nunique()} 口井", flush=True)

    output = legal_p2[["well_id", "fold", "row_index", "md", "pred_tvt"]].copy()
    output = output.rename(columns={"pred_tvt": "p2_pred_tvt"})
    for name in names:
        output[f"{name}_pred_tvt"] = arrays[name]
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name}: 合法候选含 NaN 或无穷值")
    return output


def apply_screen_gate(
    metrics: dict[str, Any],
    per_fold: pd.DataFrame,
    minimum_combined_improvement: float,
    maximum_fold_degradation: float,
) -> str | None:
    """使用预登记的 folds 0-1 合并 micro 门槛，并锁定唯一候选。"""

    passed: list[str] = []
    for name, values in metrics["candidates"].items():
        fold_rows = per_fold.loc[per_fold["method"] == name]
        worst_degradation = float(fold_rows["degradation_ft"].max())
        combined_improvement = float(values["pooled_improvement_ft"])
        gate_passed = (
            combined_improvement >= minimum_combined_improvement
            and worst_degradation <= maximum_fold_degradation
        )
        values["maximum_fold_degradation_ft"] = worst_degradation
        values["gate_passed"] = bool(gate_passed)
        if gate_passed:
            passed.append(name)

    selected = None
    if passed:
        selected = min(passed, key=lambda name: metrics["candidates"][name]["pooled_rmse"])
    metrics["gate"] = {
        "screening_folds": [0, 1],
        "minimum_combined_micro_improvement_ft": minimum_combined_improvement,
        "maximum_any_fold_degradation_ft": maximum_fold_degradation,
        "passed_candidates": passed,
        "selected_candidate": selected,
        "continue_confirmation_folds": selected is not None,
    }
    return selected


def make_conclusion(metrics: dict[str, Any]) -> str:
    screening = metrics["screening"] if "screening" in metrics else metrics
    lines = [
        "# P3-UP02 固定局部 U 平滑结论",
        "",
        "数据直接证明的事实：",
        "",
        f"- 合法候选在读取隐藏目标前已覆盖 {metrics['legal_wells']} 口开发井、{metrics['legal_rows']} 行。",
        "- 影子集 116 口井已在合法候选扫描阶段排除，未读取其目标。",
        f"- folds 0-1 的 P2-P02 合并基线 RMSE 为 {screening['baseline_pooled_rmse']:.6f} ft。",
        "- 九个预登记候选的 folds 0-1 结果如下：",
        "",
        "| 候选 | RMSE | 合并改善 | 最差折恶化 | 过筛 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, values in screening["candidates"].items():
        lines.append(
            f"| {name} | {values['pooled_rmse']:.6f} | "
            f"{values['pooled_improvement_ft']:+.6f} | "
            f"{values['maximum_fold_degradation_ft']:+.6f} | "
            f"{'是' if values['gate_passed'] else '否'} |"
        )

    selected = screening["gate"]["selected_candidate"]
    lines.extend(["", "基于事实的合理推断：", ""])
    if selected is None:
        lines.append("- 没有候选通过预登记门槛，因此不读取 folds 2-4 目标，也不晋级该实现。")
    else:
        selected_values = metrics["candidates"][selected]
        lines.append(f"- folds 0-1 锁定候选为 `{selected}`，选择依据未使用 folds 2-4。")
        lines.append(
            f"- 该候选五折开发集路径 RMSE 为 {selected_values['pooled_rmse']:.6f} ft，"
            f"相对 P2-P02 改善 {selected_values['pooled_improvement_ft']:+.6f} ft。"
        )

    lines.extend(
        [
            "",
            "仍然没有验证的猜测：",
            "",
            "- 局部平滑路径作为 LightGBM 新特征后是否还能提供条件增益，本实验没有训练模型。",
            "",
            "当前实验只能否定的具体实现：",
            "",
            "- 若失败，只能否定固定 250/500/1000 ft 高斯尺度与 25/50/75% 回混这一组实现。",
            "",
            "下一步最便宜的验证：",
            "",
            "- 若路径通过门槛，只把锁定路径作为一个新特征加入冻结 LightGBM；否则返回下一条预登记路径实验。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("配置实验编号不匹配")
    if config.get("model_training") is not False:
        raise ValueError("UP02 禁止训练模型")
    if config.get("shadow_target_read") is not False:
        raise ValueError("UP02 禁止读取影子目标")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics_path = output_dir / "metrics.json"
    if final_metrics_path.exists() and not args.force:
        print(f"实验已经完成：{final_metrics_path}")
        return
    shutil.copyfile(args.config, output_dir / "config.json")

    start_time = time.perf_counter()
    prediction_path = resolve_path(str(config["source_predictions"]))
    shadow_path = resolve_path(str(config["shadow_registry"]))
    raw_train_dir = resolve_path(str(config["raw_train_dir"]))
    all_folds = [int(value) for value in config["development_folds"]]
    screen_folds = [int(value) for value in config["screening_folds"]]
    confirmation_folds = [int(value) for value in config["confirmation_folds"]]
    if sorted(all_folds) != [0, 1, 2, 3, 4]:
        raise ValueError("开发折必须固定为 0-4")
    if screen_folds != [0, 1] or confirmation_folds != [2, 3, 4]:
        raise ValueError("筛查折和确认折偏离预登记合同")

    windows_ft = tuple(float(value) for value in config["smoothing_windows_ft"])
    blends = tuple(float(value) for value in config["blend_fractions"])
    if windows_ft != (250.0, 500.0, 1000.0) or blends != (0.25, 0.5, 0.75):
        raise ValueError("窗口或融合比例偏离预登记合同")
    names = candidate_names(windows_ft, blends)
    shadow_ids = shadow_well_ids(shadow_path)

    print("阶段 A：生成全部 657 口开发井的九条合法候选；此时不读取 target_tvt。", flush=True)
    legal_p2 = load_legal_p2_rows(prediction_path, all_folds, shadow_ids)
    legal_candidates = generate_legal_candidates(
        legal_p2,
        raw_train_dir,
        windows_ft,
        blends,
        float(config["grid_step_ft"]),
        float(config["sigma_fraction_of_window"]),
    )
    legal_path = output_dir / "legal_candidates.parquet"
    temporary_legal_path = output_dir / "legal_candidates.parquet.tmp"
    legal_candidates.to_parquet(temporary_legal_path, index=False)
    temporary_legal_path.replace(legal_path)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_generation_complete": True,
        "hidden_target_read": False,
        "shadow_target_read": False,
        "shadow_wells_excluded": len(shadow_ids),
        "shadow_overlap_wells": 0,
        "folds": all_folds,
        "wells": int(legal_candidates["well_id"].nunique()),
        "rows": int(len(legal_candidates)),
        "candidate_names": names,
        "source_columns_read": ["well_id", "fold", "row_index", "md", "pred_tvt"],
        "raw_horizontal_columns_read": RAW_HORIZONTAL_COLUMNS,
        "visible_prefix_used_as_smoothing_context": True,
        "legal_candidates_sha256": file_sha256(legal_path),
        "source_predictions_sha256": file_sha256(prediction_path),
        "config_sha256": file_sha256(output_dir / "config.json"),
    }
    write_json(output_dir / "legal_generation_manifest.json", manifest)
    print(
        f"阶段 A 完成：{manifest['wells']} 口井、{manifest['rows']} 行全部落盘；现在才读取开发目标。",
        flush=True,
    )

    targets = load_targets_after_legal_generation(prediction_path, all_folds, shadow_ids)
    screen_candidates = legal_candidates.loc[legal_candidates["fold"].isin(screen_folds)].copy()
    screen_targets = targets.loc[targets["fold"].isin(screen_folds)].copy()
    screen_metrics, screen_per_fold, screen_per_well = score_candidates(
        screen_candidates,
        screen_targets,
        names,
        0.0,
        999.0,
    )
    screen_metrics["experiment_id"] = EXPERIMENT_ID
    gate = config["gate"]
    selected = apply_screen_gate(
        screen_metrics,
        screen_per_fold,
        float(gate["minimum_combined_fold01_improvement_ft"]),
        float(gate["maximum_any_fold_degradation_ft"]),
    )
    write_json(output_dir / "metrics_folds01_screen.json", screen_metrics)
    screen_per_fold.to_csv(output_dir / "per_fold_folds01_screen.csv", index=False)
    screen_per_well.to_csv(output_dir / "per_well_folds01_screen.csv", index=False)
    print(
        f"folds 0-1 门槛：通过候选={screen_metrics['gate']['passed_candidates']}，锁定={selected}",
        flush=True,
    )

    if selected is None:
        screen_metrics["legal_wells"] = manifest["wells"]
        screen_metrics["legal_rows"] = manifest["rows"]
        screen_metrics["confirmation_target_read"] = False
        write_json(final_metrics_path, screen_metrics)
        screen_per_fold.to_csv(output_dir / "per_fold.csv", index=False)
        screen_per_well.to_csv(output_dir / "per_well.csv", index=False)
        (output_dir / "conclusion.md").write_text(make_conclusion(screen_metrics), encoding="utf-8")
    else:
        print("阶段 B：候选已锁定，现在确认 folds 2-4 并汇总完整五折。", flush=True)
        full_metrics, full_per_fold, full_per_well = score_candidates(
            legal_candidates,
            targets,
            names,
            0.0,
            999.0,
        )
        full_metrics["experiment_id"] = EXPERIMENT_ID
        add_full_development_gate(full_metrics, full_per_fold, full_per_well)
        full_metrics["screening"] = screen_metrics
        full_metrics["selected_by_folds01"] = selected
        full_metrics["legal_wells"] = manifest["wells"]
        full_metrics["legal_rows"] = manifest["rows"]
        full_metrics["confirmation_target_read"] = True
        full_metrics["shadow_target_read"] = False
        write_json(final_metrics_path, full_metrics)
        full_per_fold.to_csv(output_dir / "per_fold.csv", index=False)
        full_per_well.to_csv(output_dir / "per_well.csv", index=False)
        (output_dir / "conclusion.md").write_text(make_conclusion(full_metrics), encoding="utf-8")

    write_json(
        output_dir / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "model_training": False,
            "hidden_target_read_stage": "after all 657-well legal candidates and manifest",
            "shadow_target_read": False,
        },
    )
    print(f"结果目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
