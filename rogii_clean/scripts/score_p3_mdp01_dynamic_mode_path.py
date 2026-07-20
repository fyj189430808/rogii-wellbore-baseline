"""在 P3-MDP01 合法缓存全部完成后，独立读取隐藏目标并评分固定路径。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_mdp01_dynamic_mode_path as legal_runner  # noqa: E402


EXPERIMENT_ID = legal_runner.EXPERIMENT_ID
DEFAULT_CONFIG_PATH = legal_runner.DEFAULT_CONFIG_PATH
DEFAULT_ARTIFACT_DIR = legal_runner.DEFAULT_ARTIFACT_DIR
TARGET_COLUMNS = ["well_id", "fold", "row_index", "target_tvt", "pred_tvt"]
METHOD_TO_DELTA = {
    "mdp_dynamic": "mdp_dynamic_delta",
    "mdp_safe_10": "mdp_safe_10_delta",
    "mdp_safe_25": "mdp_safe_25_delta",
    "mdp_gr_shift_dynamic": "mdp_gr_shift_dynamic_delta",
    "mdp_gr_shift_safe_10": "mdp_gr_shift_safe_10_delta",
    "mdp_gr_shift_safe_25": "mdp_gr_shift_safe_25_delta",
    "mdp_cost_permutation_dynamic": "mdp_cost_permutation_dynamic_delta",
    "mdp_cost_permutation_safe_10": "mdp_cost_permutation_safe_10_delta",
    "mdp_cost_permutation_safe_25": "mdp_cost_permutation_safe_25_delta",
}
NORMAL_METHODS = ("mdp_dynamic", "mdp_safe_10", "mdp_safe_25")
CONTROL_PAIRS = {
    "dynamic": (
        "mdp_dynamic",
        "mdp_gr_shift_dynamic",
        "mdp_cost_permutation_dynamic",
    ),
    "safe_10": (
        "mdp_safe_10",
        "mdp_gr_shift_safe_10",
        "mdp_cost_permutation_safe_10",
    ),
    "safe_25": (
        "mdp_safe_25",
        "mdp_gr_shift_safe_25",
        "mdp_cost_permutation_safe_25",
    ),
}


def load_target_rows(predictions_path: Path, selected_wells: list[str]) -> pd.DataFrame:
    """评分阶段才调用；Arrow 先过滤开发井，再读取目标和冻结 P2 预测。"""

    dataset = ds.dataset(predictions_path, format="parquet")
    table = dataset.to_table(
        columns=TARGET_COLUMNS,
        filter=ds.field("well_id").isin(list(map(str, selected_wells))),
    )
    frame = table.to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    return frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(
        drop=True
    )


def validate_legal_generation_complete(
    registry: pd.DataFrame, artifact_dir: Path, run_mode: str
) -> tuple[Path, str, dict[str, Path]]:
    """逐井验证完整性；该函数不打开任何含隐藏目标的文件。"""

    run_dir = legal_runner.resolve_run_dir(artifact_dir, run_mode)
    summary_path = run_dir / f"runtime_{run_mode}.json"
    uses_all_superset = False
    if not summary_path.is_file() and run_mode == "folds12":
        all_summary_path = run_dir / "runtime_all.json"
        if all_summary_path.is_file():
            summary_path = all_summary_path
            uses_all_superset = True
    if not summary_path.is_file():
        raise ValueError("legal generation summary is missing")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("legal generation summary is invalid") from error
    if summary.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("legal generation experiment_id mismatch")
    summary_mode_matches = summary.get("run_mode") == run_mode
    summary_is_allowed_superset = uses_all_superset and summary.get("run_mode") == "all"
    if not (summary_mode_matches or summary_is_allowed_superset):
        raise ValueError("legal generation mode mismatch")
    if summary.get("completed") is not True:
        raise ValueError("legal generation is not complete")
    if summary.get("hidden_tvt_read") is not False:
        raise ValueError("legal generation read hidden target")
    expected_wells = int(len(registry))
    expected_rows = int(registry["hidden_rows"].sum())
    selected_wells = int(summary.get("selected_wells", -1))
    selected_rows = int(summary.get("selected_rows", -1))
    completed_wells = int(summary.get("completed_wells", -1))
    completed_rows = int(summary.get("completed_rows", -1))
    if uses_all_superset:
        if selected_wells < expected_wells or completed_wells < expected_wells:
            raise ValueError("legal generation well superset is incomplete")
        if selected_rows < expected_rows or completed_rows < expected_rows:
            raise ValueError("legal generation row superset is incomplete")
    else:
        if selected_wells != expected_wells:
            raise ValueError("legal generation well count mismatch")
        if selected_rows != expected_rows:
            raise ValueError("legal generation row count mismatch")
        if completed_wells != expected_wells:
            raise ValueError("legal generation completed-well count mismatch")
        if completed_rows != expected_rows:
            raise ValueError("legal generation completed-row count mismatch")
    if int(summary.get("shadow_overlap", -1)) != 0:
        raise ValueError("legal generation contains shadow wells")
    if summary.get("errors") not in ([], None):
        raise ValueError("legal generation contains errors")
    fingerprint = str(summary.get("experiment_fingerprint", ""))
    if len(fingerprint) != 64:
        raise ValueError("legal generation fingerprint is invalid")

    cache_paths: dict[str, Path] = {}
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        fold = int(row.fold)
        hidden_rows = int(row.hidden_rows)
        cache_path = run_dir / "legal_cache" / f"{well_id}.parquet"
        runtime_path = run_dir / "legal_runtime" / f"{well_id}.json"
        if not cache_path.is_file() or not runtime_path.is_file():
            raise ValueError(f"legal generation cache is missing for {well_id}")
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"legal generation runtime is invalid for {well_id}") from error
        if runtime.get("experiment_fingerprint") != fingerprint:
            raise ValueError(f"legal generation fingerprint mismatch for {well_id}")
        if runtime.get("well_id") != well_id or int(runtime.get("fold", -1)) != fold:
            raise ValueError(f"legal generation natural key mismatch for {well_id}")
        if int(runtime.get("hidden_rows", -1)) != hidden_rows:
            raise ValueError(f"legal generation rows mismatch for {well_id}")
        if runtime.get("hidden_tvt_read") is not False:
            raise ValueError(f"legal generation target boundary failed for {well_id}")
        if runtime.get("cache_sha256") != legal_runner.file_sha256(cache_path):
            raise ValueError(f"legal generation cache hash mismatch for {well_id}")
        frame = pd.read_parquet(cache_path)
        if frame.columns.tolist() != legal_runner.LEGAL_CACHE_COLUMNS:
            raise ValueError(f"legal generation schema mismatch for {well_id}")
        if len(frame) != hidden_rows:
            raise ValueError(f"legal generation cache rows mismatch for {well_id}")
        if not frame["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"legal generation cache well mismatch for {well_id}")
        if not pd.to_numeric(frame["fold"], errors="raise").eq(fold).all():
            raise ValueError(f"legal generation cache fold mismatch for {well_id}")
        if frame["row_index"].duplicated().any():
            raise ValueError(f"legal generation duplicate row_index for {well_id}")
        if frame["_cache_fingerprint"].astype(str).unique().tolist() != [fingerprint]:
            raise ValueError(f"legal generation cache fingerprint mismatch for {well_id}")
        cache_paths[well_id] = cache_path
    return run_dir, fingerprint, cache_paths


def _aligned_target_and_legal(
    target: pd.DataFrame,
    legal: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    hidden_rows: int,
    fingerprint: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(target) != hidden_rows or len(legal) != hidden_rows:
        raise ValueError(f"{well_id}: score row count mismatch")
    for frame, name in ((target, "target"), (legal, "legal")):
        if not frame["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"{well_id}: {name} well_id mismatch")
        if not pd.to_numeric(frame["fold"], errors="raise").eq(fold).all():
            raise ValueError(f"{well_id}: {name} fold mismatch")
    target_sorted = target.sort_values("row_index", kind="stable").reset_index(drop=True)
    legal_sorted = legal.sort_values("row_index", kind="stable").reset_index(drop=True)
    target_keys = target_sorted["row_index"].to_numpy(dtype=np.int64)
    legal_keys = legal_sorted["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(target_keys, legal_keys):
        raise ValueError(f"{well_id}: target/legal row_index mismatch")
    if legal_sorted["_cache_fingerprint"].astype(str).unique().tolist() != [fingerprint]:
        raise ValueError(f"{well_id}: legal fingerprint mismatch during scoring")
    return target_sorted, legal_sorted


def score_one_well(
    target: pd.DataFrame,
    legal: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    hidden_rows: int,
    fingerprint: str,
) -> dict[str, Any]:
    """分别评价每条预先冻结路径，不做逐井 oracle 挑选。"""

    target_sorted, legal_sorted = _aligned_target_and_legal(
        target,
        legal,
        well_id=well_id,
        fold=fold,
        hidden_rows=hidden_rows,
        fingerprint=fingerprint,
    )
    truth = target_sorted["target_tvt"].to_numpy(dtype=np.float64)
    p2_path = target_sorted["pred_tvt"].to_numpy(dtype=np.float64)
    last_visible = legal_sorted["last_visible_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(np.concatenate([truth, p2_path, last_visible])).all():
        raise ValueError(f"{well_id}: non-finite score arrays")
    paths: dict[str, np.ndarray] = {"p2": p2_path}
    for method, delta_column in METHOD_TO_DELTA.items():
        paths[method] = last_visible + legal_sorted[delta_column].to_numpy(
            dtype=np.float64
        )
    record: dict[str, Any] = {
        "well_id": well_id,
        "fold": fold,
        "rows": hidden_rows,
    }
    for method, path in paths.items():
        if not np.isfinite(path).all():
            raise ValueError(f"{well_id}: {method} contains non-finite values")
        sse = float(np.square(path - truth).sum(dtype=np.float64))
        record[f"{method}_sse"] = sse
        record[f"{method}_rmse"] = float(math.sqrt(sse / hidden_rows))
    return record


def summarize_method(
    per_well: pd.DataFrame, *, method: str, baseline_method: str = "p2"
) -> tuple[dict[str, Any], pd.DataFrame]:
    """报告 pooled micro、井级分布、胜井率及逐折差值。"""

    rows = int(per_well["rows"].sum())
    baseline_sse = float(per_well[f"{baseline_method}_sse"].sum())
    method_sse = float(per_well[f"{method}_sse"].sum())
    baseline_rmse = float(math.sqrt(baseline_sse / rows))
    rmse = float(math.sqrt(method_sse / rows))
    method_well_rmse = per_well[f"{method}_rmse"].to_numpy(dtype=np.float64)
    baseline_well_rmse = per_well[f"{baseline_method}_rmse"].to_numpy(
        dtype=np.float64
    )
    metrics = {
        "method": method,
        "wells": int(len(per_well)),
        "rows": rows,
        "baseline_rmse": baseline_rmse,
        "rmse": rmse,
        "improvement_ft": float(baseline_rmse - rmse),
        "macro_well_rmse": float(np.mean(method_well_rmse)),
        "median_well_rmse": float(np.median(method_well_rmse)),
        "p90_well_rmse": float(np.quantile(method_well_rmse, 0.90)),
        "worst_well_rmse": float(np.max(method_well_rmse)),
        "win_rate": float(np.mean(method_well_rmse < baseline_well_rmse)),
    }
    fold_records: list[dict[str, Any]] = []
    for fold, group in per_well.groupby("fold", sort=True):
        fold_rows = int(group["rows"].sum())
        fold_baseline = float(
            math.sqrt(float(group[f"{baseline_method}_sse"].sum()) / fold_rows)
        )
        fold_rmse = float(math.sqrt(float(group[f"{method}_sse"].sum()) / fold_rows))
        fold_records.append(
            {
                "method": method,
                "fold": int(fold),
                "wells": int(len(group)),
                "rows": fold_rows,
                "baseline_rmse": fold_baseline,
                "rmse": fold_rmse,
                "improvement_ft": float(fold_baseline - fold_rmse),
            }
        )
    return metrics, pd.DataFrame(fold_records)


def paired_well_bootstrap(
    per_well: pd.DataFrame,
    *,
    method: str,
    repeats: int = 2000,
    seed: int = 20260720,
) -> dict[str, float] | None:
    """按井成簇重采样，井被抽中时其全部评价行一起进入 micro RMSE。"""

    if len(per_well) < 10:
        return None
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    baseline_sse = per_well["p2_sse"].to_numpy(dtype=np.float64)
    method_sse = per_well[f"{method}_sse"].to_numpy(dtype=np.float64)
    random = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = random.integers(0, len(per_well), size=len(per_well))
        sampled_rows = float(rows[sampled].sum())
        baseline_rmse = math.sqrt(float(baseline_sse[sampled].sum()) / sampled_rows)
        method_rmse = math.sqrt(float(method_sse[sampled].sum()) / sampled_rows)
        differences[repeat] = method_rmse - baseline_rmse
    return {
        "difference_mean": float(np.mean(differences)),
        "difference_ci_low": float(np.quantile(differences, 0.025)),
        "difference_ci_high": float(np.quantile(differences, 0.975)),
    }


def screen_gate(per_fold: pd.DataFrame) -> dict[str, Any]:
    """三阶段新合同：folds 1、2 平均至少改善 0.15，单折不坏过 0.15。"""

    values = {
        int(row.fold): float(row.improvement_ft)
        for row in per_fold.itertuples(index=False)
    }
    if set(values) != {1, 2}:
        return {"applicable": False, "passed": False, "reason": "requires folds 1 and 2"}
    average = float(np.mean(list(values.values())))
    worst = float(min(values.values()))
    passed = average >= 0.15 and worst >= -0.15
    return {
        "applicable": True,
        "passed": bool(passed),
        "average_fold_improvement_ft": average,
        "worst_fold_improvement_ft": worst,
        "required_average_improvement_ft": 0.15,
        "maximum_allowed_fold_degradation_ft": 0.15,
    }


def score_experiment(
    *,
    registry: pd.DataFrame,
    artifact_dir: Path,
    p2_predictions_path: Path,
    run_mode: str,
) -> dict[str, Any]:
    """先完成全井合法性门，再读取目标并保存物理分离的评分产物。"""

    started = time.perf_counter()
    run_dir, fingerprint, cache_paths = validate_legal_generation_complete(
        registry, artifact_dir, run_mode
    )
    selected_ids = registry["well_id"].astype(str).tolist()
    target_all = load_target_rows(p2_predictions_path, selected_ids)
    target_groups = {
        str(well_id): group.copy()
        for well_id, group in target_all.groupby("well_id", sort=False)
    }
    if set(target_groups) != set(selected_ids):
        raise ValueError("score targets do not cover exactly the selected wells")
    records: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        legal = pd.read_parquet(cache_paths[well_id])
        records.append(
            score_one_well(
                target_groups[well_id],
                legal,
                well_id=well_id,
                fold=int(row.fold),
                hidden_rows=int(row.hidden_rows),
                fingerprint=fingerprint,
            )
        )
    per_well = pd.DataFrame(records).sort_values("well_id", kind="stable")
    method_metrics: dict[str, Any] = {}
    fold_frames: list[pd.DataFrame] = []
    for method in ("p2", *METHOD_TO_DELTA):
        metrics, per_fold = summarize_method(
            per_well, method=method, baseline_method="p2"
        )
        if method in NORMAL_METHODS:
            metrics["paired_well_bootstrap"] = paired_well_bootstrap(
                per_well, method=method
            )
            metrics["folds12_gate"] = screen_gate(per_fold)
        method_metrics[method] = metrics
        fold_frames.append(per_fold)
    per_fold_all = pd.concat(fold_frames, ignore_index=True)
    controls: dict[str, Any] = {}
    for label, (normal, gr_shift, cost_permutation) in CONTROL_PAIRS.items():
        controls[label] = {
            "normal_rmse": method_metrics[normal]["rmse"],
            "gr_shift_rmse": method_metrics[gr_shift]["rmse"],
            "cost_permutation_rmse": method_metrics[cost_permutation]["rmse"],
            "gr_shift_minus_normal_ft": float(
                method_metrics[gr_shift]["rmse"] - method_metrics[normal]["rmse"]
            ),
            "cost_permutation_minus_normal_ft": float(
                method_metrics[cost_permutation]["rmse"]
                - method_metrics[normal]["rmse"]
            ),
        }
    output_dir = run_dir / "scoring" / run_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    per_fold_all.to_csv(output_dir / "per_fold.csv", index=False)
    metrics_payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_mode": run_mode,
        "experiment_fingerprint": fingerprint,
        "wells": int(len(per_well)),
        "rows": int(per_well["rows"].sum()),
        "methods": method_metrics,
        "negative_controls": controls,
        "target_read_after_legal_completion": True,
        "no_oracle_path_selection": True,
    }
    legal_runner.write_json_atomic(output_dir / "metrics.json", metrics_payload)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "run_mode": run_mode,
        "legal_generation_validated": True,
        "hidden_tvt_read": True,
        "hidden_tvt_read_stage": "independent_scorer_only",
        "wells": int(len(per_well)),
        "rows": int(per_well["rows"].sum()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    legal_runner.write_json_atomic(output_dir / "runtime.json", runtime)
    return metrics_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=legal_runner.RUN_MODES, default="smoke3")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_args(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong MDP01 experiment_id")
    folds_path = (CLEAN_ROOT / str(config["fold_registry"])).resolve()
    shadow_path = (CLEAN_ROOT / str(config["shadow_registry"])).resolve()
    predictions_path = (CLEAN_ROOT / str(config["p2_predictions"])).resolve()
    development, _ = legal_runner.load_development_registry(folds_path, shadow_path)
    registry = legal_runner.select_registry(development, arguments.mode)
    print(
        f"P3-MDP01 score {arguments.mode}: validating {len(registry)} legal caches first",
        flush=True,
    )
    metrics = score_experiment(
        registry=registry,
        artifact_dir=arguments.artifact_dir.resolve(),
        p2_predictions_path=predictions_path,
        run_mode=arguments.mode,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
