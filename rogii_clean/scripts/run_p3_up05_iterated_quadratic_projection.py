"""执行 P3-UP05：对 UP03 绝对预测路径再做一次固定二次稳健投影。"""

from __future__ import annotations

import argparse
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
    attach_z_without_target,
    file_sha256,
    load_targets_after_legal_generation,
    read_json,
    resolve_path,
    shadow_well_ids,
    write_json,
)
from src.p3_up01_robust_u_projection import (  # noqa: E402
    DEFAULT_HUBER_K,
    DEFAULT_IRLS_ITERATIONS,
    DEFAULT_SCALE_FLOOR,
)
from src.p3_up05_iterated_quadratic_projection import iterated_quadratic_tvt  # noqa: E402


EXPERIMENT_ID = "P3_UP05_iterated_quadratic_projection_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up05_iterated_quadratic_projection_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
KEY_COLUMNS = ["well_id", "fold", "row_index"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P3-UP05 迭代二次稳健投影")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def validate_fixed_projection(config: dict[str, Any]) -> None:
    projection = config["projection"]
    expected = {
        "coordinate": "U=UP03_pred_tvt+Z",
        "degree": 2,
        "huber_k": DEFAULT_HUBER_K,
        "irls_iterations": DEFAULT_IRLS_ITERATIONS,
        "mad_consistency_factor": 1.4826,
        "scale_floor": DEFAULT_SCALE_FLOOR,
        "blend_fraction": 0.5,
    }
    if projection != expected:
        raise ValueError("UP05 投影参数偏离 UP01 degree2/blend50 固定合同")


def load_up03_legal_rows(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """只读取 UP03 合法绝对路径，不读取任何目标。"""

    source_paths = [resolve_path(str(value)) for value in config["source_up03_legal_paths"]]
    manifest_paths = [resolve_path(str(value)) for value in config["source_up03_manifests"]]
    if len(source_paths) != 2 or len(manifest_paths) != 2:
        raise ValueError("UP03 必须由固定 folds012 与 folds34 两份缓存组成")

    manifests: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    source_column = str(config["source_up03_column"])
    for source_path, manifest_path in zip(source_paths, manifest_paths, strict=True):
        manifest = read_json(manifest_path)
        if manifest.get("hidden_target_read") is not False:
            raise ValueError(f"UP03 合法缓存曾读取隐藏目标：{manifest_path}")
        if int(manifest.get("shadow_overlap_wells", -1)) != 0:
            raise ValueError(f"UP03 合法缓存混入影子井：{manifest_path}")
        if file_sha256(source_path) != str(manifest["legal_candidates_sha256"]):
            raise ValueError(f"UP03 合法缓存哈希不匹配：{source_path}")
        manifests.append(manifest)
        frame = pd.read_parquet(
            source_path,
            columns=["well_id", "fold", "row_index", "md", source_column],
        )
        frame = frame.rename(columns={source_column: "pred_tvt"})
        frames.append(frame)

    legal = pd.concat(frames, ignore_index=True)
    legal["well_id"] = legal["well_id"].astype(str)
    legal = legal.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if legal.duplicated(KEY_COLUMNS).any():
        raise ValueError("UP03 两份合法缓存存在重复行键")
    if sorted(int(value) for value in legal["fold"].unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("UP03 合法缓存没有覆盖五折")
    if legal["well_id"].nunique() != 657:
        raise ValueError("UP03 合法缓存井数不是 657")
    return legal, manifests


def generate_candidate(legal_with_z: pd.DataFrame) -> pd.DataFrame:
    candidate = np.empty(len(legal_with_z), dtype=np.float64)
    well_count = int(legal_with_z["well_id"].nunique())
    for well_number, (well_id, well_rows) in enumerate(
        legal_with_z.groupby("well_id", sort=False),
        start=1,
    ):
        positions = well_rows.index.to_numpy(dtype=np.int64)
        candidate[positions] = iterated_quadratic_tvt(
            well_rows["md"].to_numpy(dtype=np.float64),
            well_rows["z"].to_numpy(dtype=np.float64),
            well_rows["pred_tvt"].to_numpy(dtype=np.float64),
        )
        if well_number % 100 == 0 or well_number == well_count:
            print(f"合法候选生成：{well_number} / {well_count} 口井", flush=True)
    if not np.isfinite(candidate).all():
        raise ValueError("UP05 候选含 NaN 或无穷值")

    output = legal_with_z[["well_id", "fold", "row_index", "md", "pred_tvt"]].copy()
    output = output.rename(columns={"pred_tvt": "up03_pred_tvt"})
    output["candidate_pred_tvt"] = candidate
    return output


def score_candidate(legal: pd.DataFrame, targets: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scored = legal.merge(targets, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(scored) != len(legal) or len(scored) != len(targets):
        raise ValueError("UP05 合法候选与目标行键未完全对齐")
    target = scored["target_tvt"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(target, scored["up03_pred_tvt"].to_numpy(dtype=np.float64))
    candidate_rmse = rmse(target, scored["candidate_pred_tvt"].to_numpy(dtype=np.float64))

    per_fold_rows: list[dict[str, Any]] = []
    for fold, fold_rows in scored.groupby("fold", sort=True):
        fold_target = fold_rows["target_tvt"].to_numpy(dtype=np.float64)
        fold_baseline = rmse(fold_target, fold_rows["up03_pred_tvt"].to_numpy(dtype=np.float64))
        fold_candidate = rmse(fold_target, fold_rows["candidate_pred_tvt"].to_numpy(dtype=np.float64))
        per_fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(fold_rows)),
                "up03_rmse": fold_baseline,
                "candidate_rmse": fold_candidate,
                "improvement_ft": fold_baseline - fold_candidate,
                "degradation_ft": fold_candidate - fold_baseline,
            }
        )
    per_fold = pd.DataFrame(per_fold_rows)

    per_well_rows: list[dict[str, Any]] = []
    for (well_id, fold), well_rows in scored.groupby(["well_id", "fold"], sort=True):
        well_target = well_rows["target_tvt"].to_numpy(dtype=np.float64)
        well_baseline = rmse(well_target, well_rows["up03_pred_tvt"].to_numpy(dtype=np.float64))
        well_candidate = rmse(well_target, well_rows["candidate_pred_tvt"].to_numpy(dtype=np.float64))
        per_well_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "rows": int(len(well_rows)),
                "up03_rmse": well_baseline,
                "candidate_rmse": well_candidate,
                "improvement_ft": well_baseline - well_candidate,
            }
        )
    per_well = pd.DataFrame(per_well_rows)
    improvements = per_fold["improvement_ft"].to_numpy(dtype=np.float64)
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "folds": [int(value) for value in per_fold["fold"]],
        "wells": int(scored["well_id"].nunique()),
        "rows": int(len(scored)),
        "up03_baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "pooled_improvement_ft": baseline_rmse - candidate_rmse,
        "arithmetic_mean_fold_improvement_ft": float(np.mean(improvements)),
        "fold_improvements_ft": [float(value) for value in improvements],
        "maximum_fold_degradation_ft": float(per_fold["degradation_ft"].max()),
        "model_training": False,
        "shadow_target_read": False,
        "development_redevelopment": True,
        "independent_confirmation_claim": False,
    }
    return metrics, per_fold, per_well


def bootstrap_ci(per_well: pd.DataFrame, draws: int = 10000, seed: int = 29) -> list[float]:
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    base_sse = rows * per_well["up03_rmse"].to_numpy(dtype=np.float64) ** 2
    candidate_sse = rows * per_well["candidate_rmse"].to_numpy(dtype=np.float64) ** 2
    generator = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    count = len(per_well)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        index = generator.integers(0, count, size=(end - start, count))
        sampled_rows = rows[index].sum(axis=1)
        base_rmse = np.sqrt(base_sse[index].sum(axis=1) / sampled_rows)
        cand_rmse = np.sqrt(candidate_sse[index].sum(axis=1) / sampled_rows)
        differences[start:end] = cand_rmse - base_rmse
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def add_full_diagnostics(metrics: dict[str, Any], per_well: pd.DataFrame) -> None:
    base = per_well["up03_rmse"].to_numpy(dtype=np.float64)
    candidate = per_well["candidate_rmse"].to_numpy(dtype=np.float64)
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    gain = rows * (base**2 - candidate**2)
    total_gain = float(gain.sum())
    top_count = max(1, int(np.ceil(0.05 * len(per_well))))
    metrics["full_development_diagnostics"] = {
        "improved_folds": int(sum(value > 0.0 for value in metrics["fold_improvements_ft"])),
        "paired_well_bootstrap_candidate_minus_up03_ci95_ft": bootstrap_ci(per_well),
        "well_win_rate": float(np.mean(candidate < base)),
        "p90_well_rmse_degradation_ft": float(np.quantile(candidate, 0.90) - np.quantile(base, 0.90)),
        "strongest_5pct_gain_share": float(np.sort(gain)[::-1][:top_count].sum() / total_gain)
        if total_gain > 0.0
        else None,
    }


def conclusion_text(metrics: dict[str, Any]) -> str:
    screen = metrics["screening"] if "screening" in metrics else metrics
    passed = bool(screen["gate"]["passed"])
    lines = [
        "# P3-UP05 迭代二次投影结论",
        "",
        "数据直接证明的事实：",
        "",
        f"- 候选在读取开发目标前已覆盖 {metrics['legal_wells']} 口、{metrics['legal_rows']} 行，影子重叠为 0。",
        "- 唯一公式为：先对 UP03 的 U 使用 UP01 原始 degree2 Huber-IRLS 投影，再与 UP03 固定 1:1 回混。",
        f"- folds 0-1：UP03 {screen['up03_baseline_rmse']:.6f} → 候选 {screen['candidate_rmse']:.6f}，合并改善 {screen['pooled_improvement_ft']:+.6f} ft。",
        f"- folds 0-1 逐折改善为 {screen['fold_improvements_ft']}，预登记门槛{'通过' if passed else '未通过'}。",
        "",
        "基于事实的合理推断：",
        "",
    ]
    if passed and "screening" in metrics:
        lines.append(
            f"- 门槛通过后汇总开发五折：UP03 {metrics['up03_baseline_rmse']:.6f} → 候选 {metrics['candidate_rmse']:.6f}，改善 {metrics['pooled_improvement_ft']:+.6f} ft。"
        )
    else:
        lines.append("- 该固定二次迭代投影没有达到继续读取 folds 2-4 的门槛。")
    lines.extend(
        [
            "",
            "仍然没有验证的猜测：",
            "",
            "- 其他次数、IRLS 参数或回混比例是否更好；本实验严格禁止搜索。",
            "",
            "当前实验只能否定的具体实现：",
            "",
            "- 若失败，只能否定 UP03 上再次应用同一个 degree2/blend50 变换。",
            "",
            "下一步最便宜的验证：",
            "",
            "- 返回路线图中下一条固定路径变换，不对本实验事后调参。",
            "",
            "重要限制：",
            "",
            "- 所有开发折均参与过此前路线选择或确认；本结果属于开发集再开发，不能称为独立确认。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("配置实验编号不匹配")
    if config.get("model_training") is not False or config.get("shadow_target_read") is not False:
        raise ValueError("UP05 禁止训练模型或读取影子目标")
    validate_fixed_projection(config)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists() and not args.force:
        print(f"实验已经完成：{output_dir / 'metrics.json'}")
        return
    shutil.copyfile(args.config, output_dir / "config.json")
    start = time.perf_counter()

    print("阶段 A：读取 UP03 合法绝对路径并生成全部 657 口固定候选；不读取 target_tvt。", flush=True)
    up03_legal, source_manifests = load_up03_legal_rows(config)
    shadow_ids = shadow_well_ids(resolve_path(str(config["shadow_registry"])))
    if set(up03_legal["well_id"]).intersection(shadow_ids):
        raise RuntimeError("UP03 源路径混入影子井")
    legal_with_z = attach_z_without_target(
        up03_legal,
        resolve_path(str(config["raw_train_dir"])),
    )
    legal = generate_candidate(legal_with_z)
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
        "candidate_formula": "0.5 * UP03 + 0.5 * (degree2_HuberIRLS(UP03+Z) - Z)",
        "source_manifest_folds": [manifest_value["folds"] for manifest_value in source_manifests],
        "legal_candidates_sha256": file_sha256(legal_path),
        "config_sha256": file_sha256(output_dir / "config.json"),
        "development_redevelopment": True,
        "independent_confirmation_claim": False,
    }
    write_json(output_dir / "legal_generation_manifest.json", manifest)
    print(f"阶段 A 完成：{manifest['wells']} 口、{manifest['rows']} 行已落盘；现在只读 folds 0-1 目标。", flush=True)

    prediction_path = resolve_path(str(config["source_predictions"]))
    screen_folds = [int(value) for value in config["screening_folds"]]
    if screen_folds != [0, 1]:
        raise ValueError("UP05 筛查折必须固定为 [0, 1]")
    screen_legal = legal.loc[legal["fold"].isin(screen_folds)].copy()
    screen_targets = load_targets_after_legal_generation(prediction_path, screen_folds, shadow_ids)
    screen_metrics, screen_per_fold, screen_per_well = score_candidate(screen_legal, screen_targets)
    gate = config["gate"]
    passed = (
        screen_metrics["pooled_improvement_ft"]
        >= float(gate["minimum_combined_fold01_improvement_ft"])
        and screen_metrics["maximum_fold_degradation_ft"]
        <= float(gate["maximum_any_fold_degradation_ft"])
    )
    screen_metrics["gate"] = {
        "minimum_combined_fold01_improvement_ft": float(
            gate["minimum_combined_fold01_improvement_ft"]
        ),
        "maximum_any_fold_degradation_ft": float(gate["maximum_any_fold_degradation_ft"]),
        "passed": bool(passed),
    }
    write_json(output_dir / "metrics_folds01_screen.json", screen_metrics)
    screen_per_fold.to_csv(output_dir / "per_fold_folds01_screen.csv", index=False)
    screen_per_well.to_csv(output_dir / "per_well_folds01_screen.csv", index=False)
    print(f"folds 0-1：合并改善={screen_metrics['pooled_improvement_ft']:+.6f}，门槛通过={passed}", flush=True)

    if not passed:
        screen_metrics["legal_wells"] = manifest["wells"]
        screen_metrics["legal_rows"] = manifest["rows"]
        screen_metrics["folds2_to_4_target_read"] = False
        write_json(output_dir / "metrics.json", screen_metrics)
        screen_per_fold.to_csv(output_dir / "per_fold.csv", index=False)
        screen_per_well.to_csv(output_dir / "per_well.csv", index=False)
        (output_dir / "conclusion.md").write_text(conclusion_text(screen_metrics), encoding="utf-8")
    else:
        print("筛查门槛通过：现在才读取 folds 2-4 目标并汇总开发五折。", flush=True)
        confirmation_folds = [int(value) for value in config["confirmation_folds_after_gate"]]
        if confirmation_folds != [2, 3, 4]:
            raise ValueError("UP05 门槛后确认折必须固定为 [2, 3, 4]")
        confirmation_targets = load_targets_after_legal_generation(
            prediction_path,
            confirmation_folds,
            shadow_ids,
        )
        all_targets = pd.concat([screen_targets, confirmation_targets], ignore_index=True)
        full_metrics, full_per_fold, full_per_well = score_candidate(legal, all_targets)
        add_full_diagnostics(full_metrics, full_per_well)
        full_metrics["screening"] = screen_metrics
        full_metrics["legal_wells"] = manifest["wells"]
        full_metrics["legal_rows"] = manifest["rows"]
        full_metrics["folds2_to_4_target_read_after_gate"] = True
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
            "development_redevelopment": True,
            "truth_access_order": "all legal candidates -> folds 0-1 gate -> folds 2-4 only if passed",
        },
    )
    print(f"结果目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
