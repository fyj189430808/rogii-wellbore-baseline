"""执行 P3-PFS03：按剩余 MD 选择最长可用固定滞后路径。"""

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
    load_pfs_legal_rows,
    read_json,
    resolve_path,
    shadow_ids,
    write_json,
)
from src.p3_pfs03_max_available_lag_postcorrection import (
    build_fixed_candidate,
    choose_dynamic_pfs_absolute,
)


EXPERIMENT_ID = "P3_PFS03_max_available_lag_postcorrection_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pfs03_max_available_lag_postcorrection_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
KEYS = ["well_id", "fold", "row_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def load_up03_legal(config: dict[str, Any], excluded: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source_value, manifest_value in zip(
        config["source_up03_legal_candidates"],
        config["source_up03_legal_manifests"],
        strict=True,
    ):
        source = resolve_path(source_value)
        manifest = read_json(resolve_path(manifest_value))
        if manifest.get("hidden_target_read") is not False or manifest.get("shadow_overlap_wells") != 0:
            raise RuntimeError(f"UP03 合法路径血缘不合格：{source}")
        if manifest.get("legal_candidates_sha256") != file_sha256(source):
            raise RuntimeError(f"UP03 合法路径哈希不匹配：{source}")
        part = pd.read_parquet(
            source,
            columns=KEYS
            + ["md", "p3b00_pred_tvt", "degree2_blend50_pred_tvt", "candidate_pred_tvt"],
        )
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["well_id"] = frame["well_id"].astype(str)
    if frame.duplicated(KEYS).any() or set(frame["well_id"]).intersection(excluded):
        raise RuntimeError("UP03 合法路径存在重复行键或混入影子井")
    if frame["well_id"].nunique() != 657 or set(frame["fold"].unique()) != {0, 1, 2, 3, 4}:
        raise RuntimeError("UP03 合法路径没有覆盖完整 657 口开发井五折")
    return frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)


def load_pf_scale8(registry: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for index, row in enumerate(registry.sort_values("well_id").itertuples(index=False), 1):
        well_id = str(row.well_id)
        part = pd.read_parquet(
            cache_dir / f"{well_id}.parquet",
            columns=["well_id", "row_index", "last_visible_tvt", "pf128_scale_8_delta"],
        )
        part["well_id"] = part["well_id"].astype(str)
        if len(part) != int(row.hidden_rows) or set(part["well_id"]) != {well_id}:
            raise RuntimeError(f"P2-P01 scale8 缓存行数或井号不匹配：{well_id}")
        parts.append(part)
        if index % 100 == 0:
            print(f"读取原 PF scale8：{index}/{len(registry)} 口井", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["well_id", "row_index"]).any():
        raise RuntimeError("P2-P01 scale8 缓存存在重复行键")
    return frame


def generate_all_legal_candidates(
    config: dict[str, Any], registry: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, Path]:
    excluded = shadow_ids(resolve_path(config["shadow_registry"]))
    up03 = load_up03_legal(config, excluded)
    pfs = load_pfs_legal_rows(
        registry,
        [0, 1, 2, 3, 4],
        resolve_path(config["source_pfs_legal_cache_dir"]),
        resolve_path(config["source_pfs_runtime_dir"]),
        str(config["pfs_fingerprint"]),
    )
    pf = load_pf_scale8(registry, resolve_path(config["source_p01_legal_cache_dir"]))
    merged = up03.merge(pfs, on=KEYS, how="inner", validate="one_to_one").merge(
        pf, on=["well_id", "row_index"], how="inner", validate="one_to_one", suffixes=("", "_pf")
    )
    if len(merged) != len(up03) or len(merged) != len(pfs) or len(merged) != len(pf):
        raise RuntimeError("UP03、PFS 和原 PF 的逐行键未完整对齐")
    if not np.allclose(
        merged["last_visible_tvt"].to_numpy(dtype=np.float64),
        merged["last_visible_tvt_pf"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-3,
    ):
        raise RuntimeError("PFS 与原 PF 的可见末值不一致")

    md_end = merged.groupby("well_id", sort=False)["md"].transform("max").to_numpy(dtype=np.float64)
    remaining_md = md_end - merged["md"].to_numpy(dtype=np.float64)
    last_visible = merged["last_visible_tvt"].to_numpy(dtype=np.float64)
    dynamic, selected_lag = choose_dynamic_pfs_absolute(
        remaining_md,
        merged["last_visible_tvt_pf"].to_numpy(dtype=np.float64)
        + merged["pf128_scale_8_delta"].to_numpy(dtype=np.float64),
        last_visible + merged["pfs_lag250_delta"].to_numpy(dtype=np.float64),
        last_visible + merged["pfs_lag500_delta"].to_numpy(dtype=np.float64),
        last_visible + merged["pfs_lag1000_delta"].to_numpy(dtype=np.float64),
    )
    candidate = build_fixed_candidate(
        merged["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64),
        merged["p3b00_pred_tvt"].to_numpy(dtype=np.float64),
        dynamic,
        correction_fraction=float(config["correction_fraction"]),
    )
    legal = merged[KEYS + ["md", "p3b00_pred_tvt", "degree2_blend50_pred_tvt", "candidate_pred_tvt"]].copy()
    legal = legal.rename(columns={"candidate_pred_tvt": "up03_pred_tvt"})
    legal["remaining_md"] = remaining_md
    legal["selected_lag_ft"] = selected_lag
    legal["dynamic_pfs_abs_tvt"] = dynamic
    legal["candidate_pred_tvt"] = candidate
    legal = legal.sort_values(KEYS, kind="stable").reset_index(drop=True)
    path = output_dir / "legal_candidates.parquet"
    temporary = path.with_suffix(path.suffix + ".tmp")
    legal.to_parquet(temporary, index=False)
    temporary.replace(path)
    counts = legal["selected_lag_ft"].value_counts().sort_index().to_dict()
    write_json(
        output_dir / "legal_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_generation_complete": True,
            "hidden_target_read": False,
            "shadow_target_read": False,
            "shadow_overlap_wells": 0,
            "wells": int(legal["well_id"].nunique()),
            "rows": int(len(legal)),
            "folds": sorted(int(value) for value in legal["fold"].unique()),
            "selected_lag_row_counts": {str(key): int(value) for key, value in counts.items()},
            "candidate_formula": config["candidate_formula"],
            "development_redevelopment": True,
            "legal_candidates_sha256": file_sha256(path),
        },
    )
    return legal, path


def load_targets(path: Path, folds: list[int], excluded: set[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    condition = ds.field("fold").isin(folds) & ~ds.field("well_id").isin(sorted(excluded))
    frame = dataset.to_table(columns=KEYS + ["target_tvt"], filter=condition).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    if set(frame["well_id"]).intersection(excluded):
        raise RuntimeError("评分阶段混入影子井")
    return frame


def score(legal: pd.DataFrame, targets: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scored = legal.merge(targets, on=KEYS, how="inner", validate="one_to_one")
    if len(scored) != len(legal) or len(scored) != len(targets):
        raise RuntimeError("合法候选与目标行键未完整对齐")
    target = scored["target_tvt"].to_numpy(dtype=np.float64)
    baseline = scored["up03_pred_tvt"].to_numpy(dtype=np.float64)
    candidate = scored["candidate_pred_tvt"].to_numpy(dtype=np.float64)
    per_fold_rows: list[dict[str, Any]] = []
    for fold, frame in scored.groupby("fold", sort=True):
        fold_target = frame["target_tvt"].to_numpy(dtype=np.float64)
        base_value = rmse(fold_target, frame["up03_pred_tvt"].to_numpy(dtype=np.float64))
        candidate_value = rmse(fold_target, frame["candidate_pred_tvt"].to_numpy(dtype=np.float64))
        per_fold_rows.append({
            "fold": int(fold), "rows": int(len(frame)), "up03_rmse": base_value,
            "candidate_rmse": candidate_value, "improvement_ft": base_value - candidate_value,
            "degradation_ft": candidate_value - base_value,
        })
    per_fold = pd.DataFrame(per_fold_rows)
    per_well_rows: list[dict[str, Any]] = []
    for (well_id, fold), frame in scored.groupby(["well_id", "fold"], sort=True):
        well_target = frame["target_tvt"].to_numpy(dtype=np.float64)
        base_value = rmse(well_target, frame["up03_pred_tvt"].to_numpy(dtype=np.float64))
        candidate_value = rmse(well_target, frame["candidate_pred_tvt"].to_numpy(dtype=np.float64))
        per_well_rows.append({
            "well_id": str(well_id), "fold": int(fold), "rows": int(len(frame)),
            "up03_rmse": base_value, "candidate_rmse": candidate_value,
            "improvement_ft": base_value - candidate_value,
        })
    per_well = pd.DataFrame(per_well_rows)
    base_value = rmse(target, baseline)
    candidate_value = rmse(target, candidate)
    return {
        "folds": sorted(int(value) for value in scored["fold"].unique()),
        "wells": int(scored["well_id"].nunique()), "rows": int(len(scored)),
        "up03_pooled_rmse": base_value, "candidate_pooled_rmse": candidate_value,
        "pooled_improvement_ft": base_value - candidate_value,
        "fold_improvements_ft": [float(value) for value in per_fold["improvement_ft"]],
    }, per_fold, per_well


def make_conclusion(metrics: dict[str, Any]) -> str:
    gate = metrics["screen_gate"]
    return f"""# P3-PFS03 结论

数据直接证明的事实：

- 全部 657 口开发井候选在读取任何隐藏真值前已经落盘；影子集保持关闭。
- folds 0–1：UP03 为 {metrics['up03_pooled_rmse']:.6f} ft，PFS03 为 {metrics['candidate_pooled_rmse']:.6f} ft，改善 {metrics['pooled_improvement_ft']:+.6f} ft。
- 分折改善为 {metrics['fold_improvements_ft']}；筛查门槛{'通过' if gate['passed'] else '未通过'}。

基于事实的合理推断：

- 本结果属于多次开发后的 redevelopment 证据，不是独立确认。

仍然没有验证的猜测：

- 其他尾段过渡规则、其他修正强度是否更好，本实验没有搜索。

当前实验只能否定的具体实现：

- 若失败，只否定按剩余 MD 硬切换最长可用 lag，再固定乘 0.25 的实现。

下一步最便宜的验证：

- 过门才读取 folds 2–4；不过门立即停止，不打开影子集。
"""


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("model_training") is not False:
        raise ValueError("实验配置不匹配或意外启用了模型训练")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists() and not args.force:
        print(f"实验已有结果：{output_dir / 'metrics.json'}")
        return
    shutil.copyfile(args.config.resolve(), output_dir / "config.json")
    start = time.perf_counter()
    shadow_path = resolve_path(config["shadow_registry"])
    excluded = shadow_ids(shadow_path)
    registry = development_registry(resolve_path(config["fold_registry"]), shadow_path)

    print("阶段 A：生成全部 657 口动态滞后候选；此阶段不读取 target_tvt。", flush=True)
    legal, legal_path = generate_all_legal_candidates(config, registry, output_dir)
    print(f"阶段 A 完成：{legal['well_id'].nunique()} 口井、{len(legal)} 行已落盘。", flush=True)

    screen_folds = [int(value) for value in config["screen_folds"]]
    print("阶段 B：合法候选已全部落盘，现在只读取 folds 0–1 开发真值。", flush=True)
    legal_screen = legal.loc[legal["fold"].isin(screen_folds)].copy()
    target_screen = load_targets(resolve_path(config["source_predictions"]), screen_folds, excluded)
    metrics, per_fold, per_well = score(legal_screen, target_screen)
    gate_config = config["screen_gate"]
    worst_degradation = float(max(0.0, per_fold["degradation_ft"].max()))
    gate = {
        "minimum_pooled_improvement_ft": float(gate_config["minimum_pooled_improvement_ft"]),
        "maximum_any_fold_degradation_ft": float(gate_config["maximum_any_fold_degradation_ft"]),
        "maximum_fold_degradation_ft": worst_degradation,
        "passed": bool(
            metrics["pooled_improvement_ft"] >= float(gate_config["minimum_pooled_improvement_ft"])
            and worst_degradation <= float(gate_config["maximum_any_fold_degradation_ft"])
        ),
    }
    metrics.update({
        "experiment_id": EXPERIMENT_ID, "screen_gate": gate, "model_training": False,
        "candidate_generation_hidden_target_read": False, "shadow_target_read": False,
        "development_redevelopment": True,
    })
    per_fold.to_csv(output_dir / "per_fold_folds01.csv", index=False)
    per_well.to_csv(output_dir / "per_well_folds01.csv", index=False)
    write_json(output_dir / "metrics_folds01.json", metrics)

    if gate["passed"]:
        print("folds 0–1 过门；现在才读取 folds 2–4 真值并汇总五折。", flush=True)
        other_folds = [int(value) for value in config["remaining_folds_after_pass"]]
        other_targets = load_targets(resolve_path(config["source_predictions"]), other_folds, excluded)
        full_targets = pd.concat([target_screen, other_targets], ignore_index=True)
        full_metrics, full_per_fold, full_per_well = score(legal, full_targets)
        full_metrics.update({
            "experiment_id": EXPERIMENT_ID, "screen_metrics": metrics, "screen_gate": gate,
            "model_training": False, "candidate_generation_hidden_target_read": False,
            "shadow_target_read": False, "development_redevelopment": True,
        })
        metrics, per_fold, per_well = full_metrics, full_per_fold, full_per_well
    else:
        print("folds 0–1 未过门；停止，不读取 folds 2–4 真值。", flush=True)

    per_fold.to_csv(output_dir / "per_fold.csv", index=False)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "conclusion.md").write_text(make_conclusion(metrics), encoding="utf-8")
    write_json(output_dir / "runtime.json", {
        "experiment_id": EXPERIMENT_ID,
        "elapsed_seconds": float(time.perf_counter() - start),
        "legal_candidates": str(legal_path),
        "stopped_after_folds01": bool(not gate["passed"]),
        "model_training": False,
        "shadow_target_read": False,
        "development_redevelopment": True,
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

