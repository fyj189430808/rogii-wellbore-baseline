"""运行 P3-N01b：将 N01a 邻井残差修正固定缩放为 0.5。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_n01a_neighbor_residual_audit import (  # noqa: E402
    _cache_fingerprint as n01a_cache_fingerprint,
    _development_registry,
    _generate_fold as generate_n01a_fold,
    _read_json,
    _tail_fingerprints,
    _workspace_path,
)
from src.p3_n01b_half_strength_neighbor_residual import (  # noqa: E402
    FOLD0_SELECTION_GRID,
    FROZEN_MULTIPLIER,
    apply_neighbor_multiplier,
    pooled_multiplier_grid,
    select_grid_multiplier,
)


EXPERIMENT_ID = "P3_N01b_half_strength_neighbor_residual_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_n01b_half_strength_neighbor_residual_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
N01A_CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_n01a_pf_scale8_neighbor_residual_audit_v1.json"
LEGAL_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "baseline_scale8_tvt",
    "n01a_full_neighbor_tvt",
    "n01b_half_neighbor_tvt",
    "support_well_count",
    "support_total_weight",
    "eta",
]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _n01a_source_path(
    well_id: str,
    fold: int,
    existing_fold01_dir: Path,
    generated_dir: Path,
) -> Path:
    if int(fold) in {0, 1}:
        return existing_fold01_dir / f"{well_id}.parquet"
    return generated_dir / "legal_cache" / f"{well_id}.parquet"


def _validate_n01a_source(frame: pd.DataFrame, well_id: str, fold: int, expected_rows: int) -> None:
    required = {
        "well_id",
        "fold",
        "row_index",
        "baseline_scale8_tvt",
        "neighbor_corrected_tvt",
        "support_well_count",
        "support_total_weight",
        "eta",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{well_id} N01a source 缺列：{missing}")
    if len(frame) != int(expected_rows):
        raise ValueError(f"{well_id} N01a source 行数错误")
    if not frame["well_id"].astype(str).eq(well_id).all():
        raise ValueError(f"{well_id} N01a source 混入其他井")
    if not frame["fold"].astype(int).eq(int(fold)).all():
        raise ValueError(f"{well_id} N01a source fold 错误")
    if frame["row_index"].duplicated().any():
        raise ValueError(f"{well_id} N01a source row_index 重复")


def _build_half_cache(
    row: Any,
    source_path: Path,
    output_path: Path,
) -> None:
    well_id = str(row.well_id)
    fold = int(row.fold)
    source = pd.read_parquet(source_path)
    _validate_n01a_source(source, well_id, fold, int(row.hidden_rows))
    baseline = source["baseline_scale8_tvt"].to_numpy(dtype=np.float64)
    full = source["neighbor_corrected_tvt"].to_numpy(dtype=np.float64)
    half = apply_neighbor_multiplier(baseline, full, FROZEN_MULTIPLIER)
    output = pd.DataFrame(
        {
            "well_id": well_id,
            "fold": fold,
            "row_index": source["row_index"].to_numpy(dtype=np.int64),
            "baseline_scale8_tvt": baseline,
            "n01a_full_neighbor_tvt": full,
            "n01b_half_neighbor_tvt": half,
            "support_well_count": source["support_well_count"].to_numpy(dtype=np.int16),
            "support_total_weight": source["support_total_weight"].to_numpy(dtype=np.float64),
            "eta": source["eta"].to_numpy(dtype=np.float64),
        },
        columns=LEGAL_COLUMNS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False, compression="zstd")


def _half_cache_valid(path: Path, row: Any) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
        return bool(
            list(frame.columns) == LEGAL_COLUMNS
            and len(frame) == int(row.hidden_rows)
            and frame["well_id"].astype(str).eq(str(row.well_id)).all()
            and frame["fold"].astype(int).eq(int(row.fold)).all()
        )
    except Exception:
        return False


def _materialize_half_caches(
    targets: pd.DataFrame,
    existing_fold01_dir: Path,
    generated_dir: Path,
    output_dir: Path,
) -> None:
    legal_dir = output_dir / "legal_cache"
    legal_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(targets.itertuples(index=False), start=1):
        output_path = legal_dir / f"{row.well_id}.parquet"
        if not _half_cache_valid(output_path, row):
            source_path = _n01a_source_path(
                str(row.well_id),
                int(row.fold),
                existing_fold01_dir,
                generated_dir,
            )
            _build_half_cache(row, source_path, output_path)
        if number % 100 == 0 or number == len(targets):
            print(f"  N01b half cache {number}/{len(targets)}", flush=True)


def _read_target(raw_dir: Path, well_id: str, expected_row_index: np.ndarray) -> np.ndarray:
    horizontal = pd.read_csv(
        raw_dir / f"{well_id}__horizontal_well.csv",
        usecols=["TVT", "TVT_input"],
    )
    hidden = horizontal.loc[horizontal["TVT_input"].isna(), "TVT"]
    if not np.array_equal(hidden.index.to_numpy(dtype=np.int64), expected_row_index):
        raise ValueError(f"{well_id} 目标行与 N01b 缓存不一致")
    target = hidden.to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError(f"{well_id} 目标含非有限值")
    return target


def _score(
    targets: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    per_well_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold0_target_chunks: list[np.ndarray] = []
    fold0_baseline_chunks: list[np.ndarray] = []
    fold0_full_chunks: list[np.ndarray] = []
    for row in targets.itertuples(index=False):
        well_id = str(row.well_id)
        legal = pd.read_parquet(output_dir / "legal_cache" / f"{well_id}.parquet")
        row_index = legal["row_index"].to_numpy(dtype=np.int64)
        target = _read_target(raw_dir, well_id, row_index)
        baseline = legal["baseline_scale8_tvt"].to_numpy(dtype=np.float64)
        full = legal["n01a_full_neighbor_tvt"].to_numpy(dtype=np.float64)
        half = legal["n01b_half_neighbor_tvt"].to_numpy(dtype=np.float64)
        baseline_error = target - baseline
        half_error = target - half
        baseline_sse = float(baseline_error @ baseline_error)
        half_sse = float(half_error @ half_error)
        per_well_rows.append(
            {
                "well_id": well_id,
                "fold": int(row.fold),
                "rows": int(len(target)),
                "support_well_count": int(legal["support_well_count"].iloc[0]),
                "baseline_sse": baseline_sse,
                "half_sse": half_sse,
                "baseline_rmse": float(np.sqrt(baseline_sse / len(target))),
                "half_rmse": float(np.sqrt(half_sse / len(target))),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "fold": int(row.fold),
                    "row_index": row_index,
                    "target_tvt": target,
                    "baseline_scale8_tvt": baseline,
                    "n01b_half_neighbor_tvt": half,
                }
            )
        )
        if int(row.fold) == 0:
            fold0_target_chunks.append(target)
            fold0_baseline_chunks.append(baseline)
            fold0_full_chunks.append(full)
    per_well = pd.DataFrame(per_well_rows).sort_values(["fold", "well_id"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_rows: list[dict[str, Any]] = []
    for fold, subset in per_well.groupby("fold", sort=True):
        rows = int(subset["rows"].sum())
        baseline_rmse = float(np.sqrt(subset["baseline_sse"].sum() / rows))
        half_rmse = float(np.sqrt(subset["half_sse"].sum() / rows))
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(len(subset)),
                "rows": rows,
                "baseline_rmse": baseline_rmse,
                "half_rmse": half_rmse,
                "improvement_ft": baseline_rmse - half_rmse,
            }
        )
    per_fold = pd.DataFrame(fold_rows)
    grid_rows = pooled_multiplier_grid(
        fold0_target_chunks,
        fold0_baseline_chunks,
        fold0_full_chunks,
        FOLD0_SELECTION_GRID,
    )
    grid = pd.DataFrame(grid_rows)
    selected = select_grid_multiplier(grid_rows)
    if selected != FROZEN_MULTIPLIER:
        raise RuntimeError(f"fold 0 网格最优不是冻结的 0.5，而是 {selected}")
    total_rows = int(per_well["rows"].sum())
    baseline_rmse = float(np.sqrt(per_well["baseline_sse"].sum() / total_rows))
    half_rmse = float(np.sqrt(per_well["half_sse"].sum() / total_rows))
    improvement = baseline_rmse - half_rmse
    improved_folds = int((per_fold["improvement_ft"] > 0.0).sum())
    is_full = set(per_fold["fold"].astype(int)) == {0, 1, 2, 3, 4}
    gates = {
        "minimum_pooled_improvement": bool(improvement >= 0.10) if is_full else None,
        "minimum_four_improved_folds": bool(improved_folds >= 4) if is_full else None,
    }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "wells": int(len(per_well)),
        "rows": total_rows,
        "shadow_overlap_wells": 0,
        "frozen_multiplier": FROZEN_MULTIPLIER,
        "fold0_grid": list(FOLD0_SELECTION_GRID),
        "fold0_selected_multiplier": selected,
        "baseline_scale8_rmse": baseline_rmse,
        "half_strength_rmse": half_rmse,
        "improvement_ft": improvement,
        "improved_folds": improved_folds,
        "full5_gates": gates,
        "full5_passed": bool(all(value is True for value in gates.values())) if is_full else None,
        "lgbm_training_run": False,
    }
    return per_well, per_fold, grid, predictions, summary


def _write_predictions(path: Path, predictions: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(predictions, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd")


def _conclusion(summary: dict[str, Any], per_fold: pd.DataFrame) -> str:
    fold_text = "、".join(
        f"fold {int(row.fold)} {float(row.improvement_ft):+.6f} ft"
        for row in per_fold.itertuples(index=False)
    )
    if summary["full5_passed"] is None:
        decision = "当前只完成 folds 0～1 确认，尚未进行全五折晋级判断。"
    elif summary["full5_passed"]:
        decision = "通过全五折直接路径门槛；下一步只准备固定单模 LightGBM 接入，不在本实验内训练。"
    else:
        decision = "未通过全五折直接路径门槛；按规则停止，不接入 LightGBM。"
    return (
        f"数据直接证明的事实：固定 multiplier=0.5 后，scale8 路径 RMSE 从 "
        f"{summary['baseline_scale8_rmse']:.6f} 降至 {summary['half_strength_rmse']:.6f} ft，"
        f"改善 {summary['improvement_ft']:.6f} ft；{fold_text}。\n\n"
        f"基于事实的合理推断：{decision}\n\n"
        "仍然没有验证的猜测：固定单模 LightGBM 能否利用半强度路径以及局部可靠性特征进一步改善 OOF。\n\n"
        "当前实验只能否定的具体实现：若失败，只否定统一 0.5 强度，不否定严格 OOF 门控或逐位置权重。\n\n"
        "下一步最便宜的验证：若全五折通过，准备但不自动运行与冻结 P3B00 完全相同参数的单模 LightGBM 接入。\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P3-N01b 半强度邻井残差路径")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["fold01", "all"], default="fold01")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    config = _read_json(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("实验 ID 与 N01b runner 不一致")
    if float(config["frozen_multiplier"]) != FROZEN_MULTIPLIER:
        raise ValueError("N01b multiplier 必须锁死为 0.5")
    if tuple(float(value) for value in config["fold0_selection_grid"]) != FOLD0_SELECTION_GRID:
        raise ValueError("fold 0 预设网格发生变化")
    development, shadow = _development_registry(config)
    if set(development["well_id"].astype(str)).intersection(shadow):
        raise RuntimeError("开发井含 shadow")
    targets = development if args.mode == "all" else development.loc[development["fold"].astype(int).isin([0, 1])].copy()
    if args.mode == "fold01" and len(targets) != int(config["fold01_wells"]):
        raise ValueError("folds 0-1 井数错误")
    raw_dir = _workspace_path(str(config["raw_train_dir"]))
    pf_cache_dir = _workspace_path(str(config["pf_legal_cache_dir"]))
    existing_fold01_dir = _workspace_path(str(config["n01a_existing_cache_dir"]))
    generated_dir = args.output_dir / "generated_n01a_source"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "config.json", config)

    if args.mode == "all":
        n01a_config = _read_json(N01A_CONFIG_PATH)
        fingerprint = n01a_cache_fingerprint(N01A_CONFIG_PATH, n01a_config)
        tail_fingerprints = _tail_fingerprints(
            development["well_id"].astype(str).tolist(),
            raw_dir,
            float(n01a_config["typewell_tail_span_ft"]),
        )
        for outer_fold in [2, 3, 4]:
            fold_targets = development.loc[development["fold"].astype(int) == outer_fold].copy()
            generate_n01a_fold(
                outer_fold=outer_fold,
                target_rows=fold_targets,
                development=development,
                raw_dir=raw_dir,
                pf_cache_dir=pf_cache_dir,
                tail_fingerprints=tail_fingerprints,
                output_dir=generated_dir,
                fingerprint=fingerprint,
                shuffle_seed=int(n01a_config["shuffle_seed"]),
            )

    _materialize_half_caches(
        targets,
        existing_fold01_dir,
        generated_dir,
        args.output_dir,
    )
    per_well, per_fold, grid, predictions, summary = _score(
        targets,
        raw_dir,
        args.output_dir,
        args.mode,
    )
    suffix = args.mode
    per_well.to_csv(args.output_dir / f"per_well_{suffix}.csv", index=False, lineterminator="\n")
    per_fold.to_csv(args.output_dir / f"per_fold_{suffix}.csv", index=False, lineterminator="\n")
    grid.to_csv(args.output_dir / "fold0_multiplier_grid.csv", index=False, lineterminator="\n")
    _write_predictions(args.output_dir / f"predictions_{suffix}.parquet", predictions)
    _write_json(args.output_dir / f"metrics_{suffix}.json", summary)
    _write_json(
        args.output_dir / f"runtime_{suffix}.json",
        {
            "mode": args.mode,
            "elapsed_seconds": time.perf_counter() - started,
            "wells": int(len(targets)),
            "resume": "rerun same command; N01a source and N01b half caches are per-well reusable",
        },
    )
    (args.output_dir / f"conclusion_{suffix}.md").write_text(
        _conclusion(summary, per_fold),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
