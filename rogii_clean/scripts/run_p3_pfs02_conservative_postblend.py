"""运行 P3-PFS02：固定滞后 PF 路径与 P3B00 OOF 的保守后融合。"""

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
import pyarrow.dataset as ds


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_pfs02_conservative_postblend import build_postblend_candidates, score_candidates


EXPERIMENT_ID = "P3_PFS02_conservative_postblend_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pfs02_conservative_postblend_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
BASE_LEGAL_COLUMNS = ["well_id", "fold", "row_index", "md", "pred_tvt"]
TARGET_COLUMNS = ["well_id", "fold", "row_index", "target_tvt"]
PFS_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    "pfs_lag250_delta",
    "pfs_lag500_delta",
    "pfs_lag1000_delta",
    "_cache_fingerprint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return (CLEAN_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shadow_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, usecols=["well_id", "is_shadow"])
    return set(frame.loc[frame["is_shadow"].astype(bool), "well_id"].astype(str))


def development_registry(fold_path: Path, shadow_path: Path) -> pd.DataFrame:
    folds = pd.read_csv(fold_path, usecols=["well_id", "fold", "hidden_rows"])
    folds["well_id"] = folds["well_id"].astype(str)
    excluded = shadow_ids(shadow_path)
    result = folds.loc[~folds["well_id"].isin(excluded)].copy()
    if len(result) != 657 or result["well_id"].nunique() != 657:
        raise ValueError("开发集不等于冻结的 657 口井")
    return result


def load_base_legal_rows(path: Path, folds: list[int], excluded_ids: set[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    condition = ds.field("fold").isin(folds) & ~ds.field("well_id").isin(sorted(excluded_ids))
    frame = dataset.to_table(columns=BASE_LEGAL_COLUMNS, filter=condition).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.rename(columns={"pred_tvt": "p3b00_pred_tvt"})
    frame = frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("P3B00 OOF 存在重复行键")
    return frame


def load_pfs_legal_rows(
    registry: pd.DataFrame,
    folds: list[int],
    cache_dir: Path,
    runtime_dir: Path,
    expected_fingerprint: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    selected = registry.loc[registry["fold"].isin(folds)].sort_values("well_id")
    for index, row in enumerate(selected.itertuples(index=False), 1):
        well_id = str(row.well_id)
        cache_path = cache_dir / f"{well_id}.parquet"
        runtime_path = runtime_dir / f"{well_id}.json"
        if not cache_path.exists() or not runtime_path.exists():
            raise FileNotFoundError(f"PFS 合法缓存或运行记录缺失：{well_id}")
        runtime = read_json(runtime_path)
        if runtime.get("hidden_tvt_read") is not False:
            raise RuntimeError(f"PFS 缓存血缘不合法：{well_id} 读取过隐藏 TVT")
        if runtime.get("experiment_fingerprint") != expected_fingerprint:
            raise RuntimeError(f"PFS 指纹不匹配：{well_id}")
        if int(runtime.get("fold", -1)) != int(row.fold):
            raise RuntimeError(f"PFS fold 不匹配：{well_id}")
        if int(runtime.get("hidden_rows", -1)) != int(row.hidden_rows):
            raise RuntimeError(f"PFS 行数不匹配：{well_id}")
        if runtime.get("cache_sha256") != file_sha256(cache_path):
            raise RuntimeError(f"PFS 缓存哈希不匹配：{well_id}")
        part = pd.read_parquet(cache_path, columns=PFS_COLUMNS)
        parts.append(part)
        if index % 100 == 0:
            print(f"合法缓存检查：{index}/{len(selected)} 口井", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame["well_id"] = frame["well_id"].astype(str)
    if set(frame["_cache_fingerprint"].unique()) != {expected_fingerprint}:
        raise RuntimeError("PFS parquet 内部指纹不匹配")
    frame = frame.drop(columns=["_cache_fingerprint"])
    frame = frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    return frame


def generate_legal_stage(
    config: dict[str, Any],
    registry: pd.DataFrame,
    folds: list[int],
    output_dir: Path,
    suffix: str,
) -> tuple[pd.DataFrame, list[str], Path]:
    excluded = shadow_ids(resolve_path(config["shadow_registry"]))
    base = load_base_legal_rows(resolve_path(config["source_predictions"]), folds, excluded)
    pfs = load_pfs_legal_rows(
        registry,
        folds,
        resolve_path(config["pfs_legal_cache_dir"]),
        resolve_path(config["pfs_runtime_dir"]),
        str(config["pfs_fingerprint"]),
    )
    keys = ["well_id", "fold", "row_index"]
    merged = base.merge(pfs, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(pfs):
        raise ValueError("P3B00 与 PFS 合法路径行键不一致")
    legal, names = build_postblend_candidates(
        merged,
        config["lag_distances_ft"],
        config["blend_fractions"],
    )
    path = output_dir / f"legal_candidates_{suffix}.parquet"
    temporary = path.with_suffix(path.suffix + ".tmp")
    legal.to_parquet(temporary, index=False)
    temporary.replace(path)
    write_json(
        output_dir / f"legal_manifest_{suffix}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "folds": folds,
            "wells": int(legal["well_id"].nunique()),
            "rows": int(len(legal)),
            "candidate_names": names,
            "candidate_generation_complete": True,
            "hidden_target_read": False,
            "shadow_overlap_wells": int(legal["well_id"].isin(excluded).sum()),
            "pfs_runtime_validated_per_well": True,
            "legal_candidates_sha256": file_sha256(path),
        },
    )
    return legal, names, path


def load_targets(path: Path, folds: list[int], excluded_ids: set[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    condition = ds.field("fold").isin(folds) & ~ds.field("well_id").isin(sorted(excluded_ids))
    frame = dataset.to_table(columns=TARGET_COLUMNS, filter=condition).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    if set(frame["well_id"]).intersection(excluded_ids):
        raise RuntimeError("评分阶段混入影子井")
    return frame


def bootstrap_ci(per_well: pd.DataFrame, candidate: str, draws: int = 10000) -> list[float]:
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    baseline_sse = rows * per_well["baseline_rmse"].to_numpy(dtype=np.float64) ** 2
    candidate_sse = rows * per_well[f"{candidate}_rmse"].to_numpy(dtype=np.float64) ** 2
    rng = np.random.default_rng(29)
    differences = np.empty(draws, dtype=np.float64)
    count = len(per_well)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        indices = rng.integers(0, count, size=(end - start, count))
        sampled_rows = rows[indices].sum(axis=1)
        differences[start:end] = np.sqrt(candidate_sse[indices].sum(axis=1) / sampled_rows) - np.sqrt(
            baseline_sse[indices].sum(axis=1) / sampled_rows
        )
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def add_full_gate(metrics: dict[str, Any], per_fold: pd.DataFrame, per_well: pd.DataFrame, locked: str) -> None:
    values = metrics["candidates"][locked]
    fold_rows = per_fold.loc[per_fold["method"] == locked].sort_values("fold")
    fold_improvements = fold_rows["improvement_ft"].to_numpy(dtype=np.float64)
    baseline = per_well["baseline_rmse"].to_numpy(dtype=np.float64)
    candidate = per_well[f"{locked}_rmse"].to_numpy(dtype=np.float64)
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    sse_gain = rows * (baseline**2 - candidate**2)
    total_gain = float(sse_gain.sum())
    strongest = max(1, int(np.ceil(0.05 * len(per_well))))
    share = float(np.sort(sse_gain)[::-1][:strongest].sum() / total_gain) if total_gain > 0 else float("inf")
    ci = bootstrap_ci(per_well, locked)
    details = {
        "improved_folds": int((fold_improvements > 0).sum()),
        "improved_folds_2_to_4": int((fold_rows.loc[fold_rows["fold"].isin([2, 3, 4]), "improvement_ft"] > 0).sum()),
        "worst_fold_degradation_ft": float(max(0.0, -fold_improvements.min())),
        "paired_well_bootstrap_candidate_minus_baseline_ci95_ft": ci,
        "well_win_rate": float(np.mean(candidate < baseline)),
        "p90_well_rmse_degradation_ft": float(np.quantile(candidate, 0.90) - np.quantile(baseline, 0.90)),
        "strongest_5pct_gain_share": share,
    }
    contract = {
        "minimum_pooled_improvement_ft": 0.10,
        "minimum_improved_folds": 4,
        "minimum_improved_folds_2_to_4": 2,
        "maximum_single_fold_degradation_ft": 0.25,
        "maximum_bootstrap_ci_upper_ft": 0.0,
        "minimum_well_win_rate": 0.55,
        "maximum_p90_degradation_ft": 0.20,
        "maximum_top5_percent_positive_gain_share": 0.60,
    }
    details["passed"] = bool(
        values["pooled_improvement_ft"] >= 0.10
        and details["improved_folds"] >= 4
        and details["improved_folds_2_to_4"] >= 2
        and details["worst_fold_degradation_ft"] <= 0.25
        and ci[1] < 0.0
        and details["well_win_rate"] >= 0.55
        and details["p90_well_rmse_degradation_ft"] <= 0.20
        and share <= 0.60
    )
    metrics["full_development_gate"] = {"contract": contract, "locked_candidate_result": details}


def conclusion(metrics: dict[str, Any]) -> str:
    selected = metrics["locked_screening_candidate"]
    selected_values = metrics["candidates"][selected]
    lines = [
        "# P3-PFS02 保守后融合结论",
        "",
        "数据直接证明的事实：",
        "",
        f"- 影子 116 口井保持关闭；本实验只使用 {metrics['wells']} 口开发井、folds {metrics['folds']}。",
        "- 九条候选在读取对应开发真值前已经生成并落盘；没有训练新模型。",
        f"- folds 0–1 锁定候选为 `{selected}`，锁定后未根据 folds 2–4 改选。",
        f"- 当前报告范围内 P3B00 RMSE 为 {metrics['baseline_pooled_rmse']:.6f} ft；锁定候选为 {selected_values['pooled_rmse']:.6f} ft，改善 {selected_values['pooled_improvement_ft']:+.6f} ft。",
        "",
        "| 候选 | RMSE | 改善 | 最差单折恶化 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in metrics["candidates"].items():
        lines.append(f"| {name} | {values['pooled_rmse']:.6f} | {values['pooled_improvement_ft']:+.6f} | {values['maximum_fold_degradation_ft']:+.6f} |")
    lines.extend(
        [
            "",
            "基于事实的合理推断：",
            "",
            "- 固定比例后融合是否有价值，只能按上面的严格门槛判断；不能把单条 PFS 路径的 oracle 或路径自身分数当成正式 CV 提升。",
            "",
            "仍然没有验证的猜测：",
            "",
            "- 按井或按位置自适应融合是否更好，本实验没有测试。",
            "",
            "当前实验只能否定的具体实现：",
            "",
            "- 若未过门槛，只否定 lag={250,500,1000} 与 alpha={0.10,0.25,0.50} 的固定比例后融合。",
            "",
            "下一步最便宜的验证：",
            "",
            "- 未过门槛则停止 PFS02；过门槛则按已经锁定的候选检查 folds 2–4 与完整开发集门槛。",
            "",
            "血缘说明：PFS01 的全开发集 path_diagnostics 在本实验前已经存在并被看过，因此本实验不是完全独立的全新确认；但本次九候选网格、folds 0–1 选择规则和门槛在读本次目标前固定。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("model_training") is not False:
        raise ValueError("实验配置不匹配或意外启用了模型训练")
    output = args.output_dir.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() and not args.force:
        print(f"实验已有结果：{output / 'metrics.json'}")
        return
    shutil.copyfile(args.config.resolve(), output / "config.json")
    start = time.perf_counter()
    prediction_path = resolve_path(config["source_predictions"])
    if file_sha256(prediction_path) != config["source_predictions_sha256"]:
        raise RuntimeError("P3B00 predictions 哈希不匹配")
    shadow_path = resolve_path(config["shadow_registry"])
    excluded = shadow_ids(shadow_path)
    registry = development_registry(resolve_path(config["fold_registry"]), shadow_path)

    screening_folds = [int(v) for v in config["screening_folds"]]
    print("阶段 A：生成 folds 0–1 的九条合法候选，不读取 target_tvt。", flush=True)
    legal_screen, candidate_names, screen_path = generate_legal_stage(
        config, registry, screening_folds, output, "folds01"
    )
    print("阶段 B：合法候选已落盘，现在才读取 folds 0–1 的开发真值。", flush=True)
    targets_screen = load_targets(prediction_path, screening_folds, excluded)
    gate = config["screening_gate"]
    screen_metrics, screen_per_fold, screen_per_well = score_candidates(
        legal_screen,
        targets_screen,
        candidate_names,
        float(gate["minimum_pooled_improvement_ft"]),
        float(gate["maximum_any_fold_degradation_ft"]),
    )
    locked = str(screen_metrics["selected_candidate"])
    screen_metrics.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "model_training": False,
            "shadow_target_read": False,
            "candidate_generation_hidden_target_read": False,
            "locked_screening_candidate": locked,
            "screening_only": not bool(screen_metrics["gate"]["passed"]),
            "known_prior_inspection": "PFS01 full-development path_diagnostics existed and had already been inspected before this experiment",
        }
    )
    screen_per_fold.to_csv(output / "per_fold_folds01.csv", index=False)
    screen_per_well.to_csv(output / "per_well_folds01.csv", index=False)
    write_json(output / "metrics_folds01.json", screen_metrics)

    if not screen_metrics["gate"]["passed"]:
        screen_per_fold.to_csv(output / "per_fold.csv", index=False)
        screen_per_well.to_csv(output / "per_well.csv", index=False)
        write_json(output / "metrics.json", screen_metrics)
        (output / "conclusion.md").write_text(conclusion(screen_metrics), encoding="utf-8")
        write_json(
            output / "runtime.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "elapsed_seconds": float(time.perf_counter() - start),
                "stopped_after_screening": True,
                "model_training": False,
                "shadow_target_read": False,
                "screening_legal_candidates": str(screen_path),
            },
        )
        print(json.dumps(screen_metrics, ensure_ascii=False, indent=2), flush=True)
        return

    confirmation_folds = [int(v) for v in config["confirmation_folds"]]
    print(f"筛查通过并锁定 {locked}；阶段 C 先生成 folds 2–4 合法候选，仍不读其真值。", flush=True)
    legal_confirmation, confirmation_names, confirmation_path = generate_legal_stage(
        config, registry, confirmation_folds, output, "folds234"
    )
    if confirmation_names != candidate_names:
        raise RuntimeError("确认折候选顺序漂移")
    print("阶段 D：folds 2–4 合法候选已落盘，现在才读取其开发真值。", flush=True)
    targets_confirmation = load_targets(prediction_path, confirmation_folds, excluded)
    full_legal = pd.concat([legal_screen, legal_confirmation], ignore_index=True)
    full_targets = pd.concat([targets_screen, targets_confirmation], ignore_index=True)
    full_metrics, full_per_fold, full_per_well = score_candidates(
        full_legal,
        full_targets,
        candidate_names,
        float(gate["minimum_pooled_improvement_ft"]),
        float(gate["maximum_any_fold_degradation_ft"]),
    )
    descriptive_full_best = full_metrics["selected_candidate"]
    full_metrics.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "model_training": False,
            "shadow_target_read": False,
            "candidate_generation_hidden_target_read": False,
            "locked_screening_candidate": locked,
            "selected_candidate": locked,
            "selection_rule": "locked on folds 0-1; folds 2-4 never used to change the candidate",
            "descriptive_full5_best_candidate_not_used_for_selection": descriptive_full_best,
            "screening_folds01": screen_metrics,
            "known_prior_inspection": "PFS01 full-development path_diagnostics existed and had already been inspected before this experiment",
        }
    )
    add_full_gate(full_metrics, full_per_fold, full_per_well, locked)
    full_per_fold.to_csv(output / "per_fold.csv", index=False)
    full_per_well.to_csv(output / "per_well.csv", index=False)
    write_json(output / "metrics.json", full_metrics)
    (output / "conclusion.md").write_text(conclusion(full_metrics), encoding="utf-8")
    write_json(
        output / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start),
            "stopped_after_screening": False,
            "model_training": False,
            "shadow_target_read": False,
            "screening_legal_candidates": str(screen_path),
            "confirmation_legal_candidates": str(confirmation_path),
        },
    )
    print(json.dumps(full_metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
