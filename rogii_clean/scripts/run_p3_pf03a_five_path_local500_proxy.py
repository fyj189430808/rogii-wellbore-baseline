"""运行 P3-PF03a：不重跑 PF，用五条冻结路径筛查 500 ft 局部 GR 加权。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p3_d01_pf_observation_weights import (  # noqa: E402
    file_sha256,
    read_legal_inputs,
    stable_json_hash,
    write_csv_atomic,
    write_json_atomic,
)
from src.p3_pf03a_five_path_local500_proxy import (  # noqa: E402
    CANDIDATE_PATH_COLUMNS,
    OUTPUT_PATH_COLUMN,
    build_local500_proxy_path,
)


BASE_EXPERIMENT_ID = "P3_PF03a_five_path_local500_proxy_v1"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_pf03a_five_path_local500_proxy_v1.json"
WINDOW_LENGTHS_FT = (250.0, 500.0, 1000.0)
OLD_PATH_NAMES = {
    "mean": "pf128_mean_delta",
    "scale3": "pf128_scale_3_delta",
    "scale5": "pf128_scale_5_delta",
    "scale8": "pf128_scale_8_delta",
    "scale12": "pf128_scale_12_delta",
}


def resolve_clean_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (CLEAN_ROOT / path).resolve()


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_config(segment_length_ft: float = 500.0) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": BASE_EXPERIMENT_ID,
        "experiment_role": "zero_pf_rerun_direction_screen_not_formal_pf03",
        "candidate_paths": list(CANDIDATE_PATH_COLUMNS),
        "segment_length_ft": 500.0,
        "minimum_observed_gr_rows_per_segment": 50,
        "target_ess": 4.0,
        "fallback_path": "pf128_mean_delta",
        "development_wells": 657,
        "development_hidden_rows": 3_211_872,
        "shadow_target_access": False,
        "model_training": False,
        "formal_pf03": False,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"PF03a 配置 {name} 未保持冻结值 {value}")
    selected_length = float(segment_length_ft)
    if selected_length not in WINDOW_LENGTHS_FT:
        raise ValueError(f"PF03a 只允许预注册窗口 {WINDOW_LENGTHS_FT}")
    variant = config["window_variants"][str(int(selected_length))]
    selected = dict(config)
    selected["experiment_id"] = str(variant["experiment_id"])
    selected["artifact_dir"] = str(variant["artifact_dir"])
    selected["segment_length_ft"] = selected_length
    return selected


def choose_window_using_fold0(
    per_fold_comparison: pd.DataFrame,
    minimum_improvement_ft: float,
) -> dict[str, Any]:
    """只看 fold 0 选择窗口；fold 1 即使更好也不能改变选择。"""
    required = {"segment_length_ft", "fold", "local500_rmse", "improvement_ft"}
    if not required.issubset(per_fold_comparison.columns):
        raise ValueError("三窗口逐折表缺少选择所需字段")
    fold0 = per_fold_comparison.loc[per_fold_comparison["fold"].eq(0)].copy()
    if sorted(fold0["segment_length_ft"].astype(float).tolist()) != list(WINDOW_LENGTHS_FT):
        raise ValueError("fold 0 必须正好包含 250/500/1000 三个窗口")
    eligible = fold0.loc[fold0["improvement_ft"] >= float(minimum_improvement_ft)].copy()
    if eligible.empty:
        return {
            "selected_segment_length_ft": None,
            "selection_used_fold1": False,
            "fold0_minimum_improvement_ft": float(minimum_improvement_ft),
            "selection_reason": "no_window_passed_fold0_threshold",
        }
    # 先按 fold 0 的 local 路径 RMSE；完全并列时固定优先 500、250、1000。
    tie_priority = {500.0: 0, 250.0: 1, 1000.0: 2}
    eligible["tie_priority"] = eligible["segment_length_ft"].map(tie_priority)
    selected = eligible.sort_values(
        ["local500_rmse", "tie_priority"], kind="mergesort"
    ).iloc[0]
    return {
        "selected_segment_length_ft": float(selected["segment_length_ft"]),
        "selection_used_fold1": False,
        "fold0_minimum_improvement_ft": float(minimum_improvement_ft),
        "selected_fold0_local500_rmse": float(selected["local500_rmse"]),
        "selected_fold0_improvement_ft": float(selected["improvement_ft"]),
        "selection_reason": "lowest_fold0_rmse_among_fold0_threshold_passers",
    }


def load_development_table(config: dict[str, Any]) -> pd.DataFrame:
    """D01 表只读取四个无标签字段；误差、RMSE 和 oracle 列不进入内存。"""
    development = pd.read_csv(
        resolve_clean_path(config["source_d01_per_well"]),
        usecols=["well_id", "fold", "hidden_rows", "gr_sigma"],
        dtype={"well_id": str},
    )
    development["well_id"] = development["well_id"].str.zfill(8)
    development = development.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if development["well_id"].duplicated().any():
        raise ValueError("开发井表存在重复井号")
    if len(development) != int(config["development_wells"]):
        raise ValueError("开发井数不等于冻结的 657")
    if int(development["hidden_rows"].sum()) != int(config["development_hidden_rows"]):
        raise ValueError("开发井隐藏行数不一致")
    return development


def select_mode(development: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "smoke":
        return development.iloc[[0]].copy()
    if mode == "fold01":
        return development.loc[development["fold"].isin([0, 1])].copy()
    if mode == "all":
        return development.copy()
    raise ValueError(f"不支持 mode={mode}")


def cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def load_cache_hit(task: dict[str, Any]) -> dict[str, Any] | None:
    cache_path, runtime_path = cache_paths(Path(task["artifact_dir"]), str(task["well_id"]))
    if not cache_path.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("experiment_fingerprint") != task["fingerprint"]:
            return None
        cached = pd.read_parquet(
            cache_path,
            columns=["well_id", "row_index", OUTPUT_PATH_COLUMN, "_cache_fingerprint"],
        )
        if len(cached) != int(task["hidden_rows"]):
            return None
        if cached["well_id"].astype(str).unique().tolist() != [str(task["well_id"])]:
            return None
        if cached["_cache_fingerprint"].astype(str).unique().tolist() != [task["fingerprint"]]:
            return None
        if not np.isfinite(cached[OUTPUT_PATH_COLUMN].to_numpy(dtype=np.float64)).all():
            return None
        runtime["cache_hit"] = True
        return runtime
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """只读取合法当前井数据和旧五路径；函数内从不读取隐藏 TVT。"""
    cached = load_cache_hit(task)
    if cached is not None:
        return cached
    started = time.perf_counter()
    well_id = str(task["well_id"])
    horizontal, typewell = read_legal_inputs(Path(task["raw_train_dir"]), well_id)
    if int(horizontal["TVT_input"].isna().sum()) != int(task["hidden_rows"]):
        raise ValueError(f"井 {well_id} 隐藏行数不一致")

    source_path = Path(task["source_cache_dir"]) / f"{well_id}.parquet"
    source = pd.read_parquet(
        source_path,
        columns=[
            "well_id",
            "row_index",
            "last_visible_tvt",
            *CANDIDATE_PATH_COLUMNS,
            "_cache_fingerprint",
        ],
    )
    source_fingerprints = source["_cache_fingerprint"].astype(str).unique().tolist()
    if source_fingerprints != [task["source_pf_artifact_fingerprint"]]:
        raise ValueError(f"井 {well_id} 冻结 PF 路径指纹不一致")
    if len(source) != int(task["hidden_rows"]):
        raise ValueError(f"井 {well_id} 冻结 PF 路径行数不一致")

    proxy, report = build_local500_proxy_path(
        frozen_paths=source,
        horizontal_well=horizontal,
        typewell=typewell,
        gr_sigma=float(task["gr_sigma"]),
        segment_length_ft=float(task["segment_length_ft"]),
        minimum_observed_rows=int(task["minimum_observed_gr_rows_per_segment"]),
        target_ess=float(task["target_ess"]),
        squared_residual_cap=float(task["squared_gr_residual_cap"]),
    )
    output = proxy.copy()
    output.insert(0, "fold", int(task["fold"]))
    output.insert(0, "well_id", well_id)
    output["_cache_fingerprint"] = str(task["fingerprint"])
    cache_path, runtime_path = cache_paths(Path(task["artifact_dir"]), well_id)
    write_parquet_atomic(cache_path, output)
    runtime = {
        "experiment_id": str(task["experiment_id"]),
        "experiment_fingerprint": task["fingerprint"],
        "well_id": well_id,
        "fold": int(task["fold"]),
        "hidden_rows": int(task["hidden_rows"]),
        "hidden_tvt_read": False,
        "formal_pf03": False,
        "old_gr_sigma": float(task["gr_sigma"]),
        "source_cache_sha256": file_sha256(source_path),
        "report": report,
        "cache_hit": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(runtime_path, runtime)
    return runtime


def generate_selected_paths(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_one_well, task): str(task["well_id"]) for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            well_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                    state = "缓存" if result.get("cache_hit") else "新算"
                    report = result["report"]
                    print(
                        f"P3-PF03a {completed}/{len(tasks)}：{well_id}（{state}，"
                        f"有效段 {report['scored_segments']}/{report['segments']}）",
                        flush=True,
                    )
            except Exception as error:  # noqa: BLE001
                errors.append(f"{well_id}: {error!r}")
    if errors:
        raise RuntimeError(f"PF03a 有 {len(errors)} 口井失败：{errors[:3]}")
    return sorted(results, key=lambda item: item["well_id"])


def read_targets(prediction_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """本函数只在全部合法路径生成结束后调用。"""
    selected_ids = selected["well_id"].astype(str).tolist()
    targets = pd.read_parquet(
        prediction_path,
        columns=["well_id", "fold", "row_index", "target_tvt"],
        filters=[("well_id", "in", selected_ids)],
    )
    targets["well_id"] = targets["well_id"].astype(str).str.zfill(8)
    targets = targets.loc[targets["well_id"].isin(set(selected_ids))].copy()
    if len(targets) != int(selected["hidden_rows"].sum()):
        raise ValueError("目标行数不等于所选开发井隐藏行数")
    return targets


def score_paths(
    selected: pd.DataFrame,
    artifact_dir: Path,
    source_cache_dir: Path,
    prediction_path: Path,
    experiment_id: str,
    segment_length_ft: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """比较 local500 与同源五条旧路径；不训练模型。"""
    targets = read_targets(prediction_path, selected)
    path_columns = {**OLD_PATH_NAMES, "local500": OUTPUT_PATH_COLUMN}
    total_sse = {name: 0.0 for name in path_columns}
    total_count = 0
    fold_sse: dict[tuple[int, str], float] = {}
    fold_count: dict[int, int] = {}
    per_well_rows: list[dict[str, Any]] = []
    for registry_row in selected.itertuples(index=False):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        local_path, _ = cache_paths(artifact_dir, well_id)
        local = pd.read_parquet(local_path, columns=["row_index", "last_visible_tvt", OUTPUT_PATH_COLUMN])
        old = pd.read_parquet(
            source_cache_dir / f"{well_id}.parquet",
            columns=["row_index", "last_visible_tvt", *OLD_PATH_NAMES.values()],
        )
        truth = targets.loc[targets["well_id"].eq(well_id), ["row_index", "target_tvt"]]
        local = local.sort_values("row_index", kind="mergesort")
        old = old.sort_values("row_index", kind="mergesort")
        truth = truth.sort_values("row_index", kind="mergesort")
        target_index = truth["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(target_index, local["row_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"井 {well_id} local500 row_index 不一致")
        if not np.array_equal(target_index, old["row_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"井 {well_id} 旧路径 row_index 不一致")

        target_tvt = truth["target_tvt"].to_numpy(dtype=np.float64)
        row_result: dict[str, Any] = {"well_id": well_id, "fold": fold, "hidden_rows": len(target_tvt)}
        fold_count[fold] = fold_count.get(fold, 0) + len(target_tvt)
        total_count += len(target_tvt)
        for path_name, path_column in path_columns.items():
            frame = local if path_name == "local500" else old
            prediction = (
                frame["last_visible_tvt"].to_numpy(dtype=np.float64)
                + frame[path_column].to_numpy(dtype=np.float64)
            )
            squared_error = np.square(prediction - target_tvt)
            sse = float(squared_error.sum())
            total_sse[path_name] += sse
            fold_sse[(fold, path_name)] = fold_sse.get((fold, path_name), 0.0) + sse
            row_result[f"{path_name}_rmse"] = float(math.sqrt(sse / len(target_tvt)))
        per_well_rows.append(row_result)

    pooled_rmse = {
        name: float(math.sqrt(total_sse[name] / total_count)) for name in path_columns
    }
    best_old = min(OLD_PATH_NAMES, key=lambda name: pooled_rmse[name])
    per_fold_rows: list[dict[str, Any]] = []
    fold_improvements: list[float] = []
    for fold in sorted(selected["fold"].unique().tolist()):
        fold_old_rmse = {
            name: math.sqrt(fold_sse[(int(fold), name)] / fold_count[int(fold)])
            for name in OLD_PATH_NAMES
        }
        best_old_within_fold = min(fold_old_rmse, key=fold_old_rmse.get)
        old_rmse = fold_old_rmse[best_old_within_fold]
        local_rmse = math.sqrt(fold_sse[(int(fold), "local500")] / fold_count[int(fold)])
        improvement = float(old_rmse - local_rmse)
        fold_improvements.append(improvement)
        per_fold_rows.append(
            {
                "fold": int(fold),
                "segment_length_ft": float(segment_length_ft),
                "hidden_rows": int(fold_count[int(fold)]),
                "best_old_fixed_path": best_old_within_fold,
                "best_old_fixed_rmse": float(old_rmse),
                "local500_rmse": float(local_rmse),
                "improvement_ft": improvement,
            }
        )
    combined_improvement = float(pooled_rmse[best_old] - pooled_rmse["local500"])
    is_fold01 = set(selected["fold"].unique().tolist()) == {0, 1}
    metrics = {
        "experiment_id": str(experiment_id),
        "experiment_role": "zero_pf_rerun_direction_screen_not_formal_pf03",
        "segment_length_ft": float(segment_length_ft),
        "wells": int(len(selected)),
        "hidden_rows": int(total_count),
        "path_rmse": pooled_rmse,
        "best_old_fixed_path": best_old,
        "improvement_vs_best_old_fixed_ft": combined_improvement,
        "fold01_gate_passed": bool(
            is_fold01 and combined_improvement >= 0.20 and min(fold_improvements) >= -0.10
        ),
        "failure_scope": "only_five_aggregated_paths_local500_mixing_not_formal_128_seed_pf03",
    }
    return (
        metrics,
        pd.DataFrame(per_fold_rows),
        pd.DataFrame(per_well_rows).sort_values("well_id", kind="mergesort"),
    )


def compare_window_results() -> dict[str, Any]:
    """汇总三个预注册窗口，并严格用 fold 0 完成窗口选择。"""
    raw_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pooled_rows: list[dict[str, Any]] = []
    per_fold_frames: list[pd.DataFrame] = []
    for segment_length_ft in WINDOW_LENGTHS_FT:
        config = load_config(segment_length_ft)
        artifact_dir = resolve_clean_path(config["artifact_dir"])
        metrics_path = artifact_dir / "metrics_fold01.json"
        per_fold_path = artifact_dir / "per_fold_fold01.csv"
        if not metrics_path.is_file() or not per_fold_path.is_file():
            raise FileNotFoundError(f"窗口 {segment_length_ft:g} 缺少 folds01 结果")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_fold = pd.read_csv(per_fold_path)
        per_fold["segment_length_ft"] = float(segment_length_ft)
        per_fold_frames.append(per_fold)
        pooled_rows.append(
            {
                "segment_length_ft": float(segment_length_ft),
                "experiment_id": str(config["experiment_id"]),
                "wells": int(metrics["wells"]),
                "hidden_rows": int(metrics["hidden_rows"]),
                "best_old_fixed_path": str(metrics["best_old_fixed_path"]),
                "best_old_fixed_rmse": float(metrics["path_rmse"][metrics["best_old_fixed_path"]]),
                "local500_rmse": float(metrics["path_rmse"]["local500"]),
                "improvement_ft": float(metrics["improvement_vs_best_old_fixed_ft"]),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values("segment_length_ft", kind="mergesort")
    per_fold_comparison = pd.concat(per_fold_frames, ignore_index=True).sort_values(
        ["fold", "segment_length_ft"], kind="mergesort"
    )
    selection = choose_window_using_fold0(
        per_fold_comparison=per_fold_comparison,
        minimum_improvement_ft=float(raw_config["fold01_minimum_improvement_ft"]),
    )
    selected_length = selection["selected_segment_length_ft"]
    if selected_length is None:
        selection["fold1_confirmation_passed"] = False
        selection["combined_confirmation_passed"] = False
        selection["overall_window_stability_passed"] = False
    else:
        selected_fold1 = per_fold_comparison.loc[
            per_fold_comparison["segment_length_ft"].eq(selected_length)
            & per_fold_comparison["fold"].eq(1)
        ].iloc[0]
        selected_pooled = pooled.loc[pooled["segment_length_ft"].eq(selected_length)].iloc[0]
        selection["selected_fold1_improvement_ft"] = float(selected_fold1["improvement_ft"])
        selection["selected_combined_improvement_ft"] = float(selected_pooled["improvement_ft"])
        selection["fold1_confirmation_passed"] = bool(
            selected_fold1["improvement_ft"]
            >= -float(raw_config["fold01_maximum_single_fold_degradation_ft"])
        )
        selection["combined_confirmation_passed"] = bool(
            selected_pooled["improvement_ft"]
            >= float(raw_config["fold01_minimum_improvement_ft"])
        )
        selection["overall_window_stability_passed"] = bool(
            selection["fold1_confirmation_passed"]
            and selection["combined_confirmation_passed"]
        )

    comparison_dir = resolve_clean_path(raw_config["window_stability_artifact_dir"])
    write_csv_atomic(comparison_dir / "pooled_fold01.csv", pooled)
    write_csv_atomic(comparison_dir / "per_fold_fold01.csv", per_fold_comparison)
    summary = {
        "experiment_id": "P3_PF03a_window_stability_v1",
        "windows_ft": list(WINDOW_LENGTHS_FT),
        "selection_rule": "select_on_fold0_only_then_confirm_without_switching_on_fold1",
        "selection": selection,
        "failure_scope": "only_five_aggregated_paths_local_window_mixing_not_formal_128_seed_pf03",
    }
    write_json_atomic(comparison_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "fold01", "all"), default="smoke")
    parser.add_argument(
        "--segment-length-ft",
        type=float,
        choices=WINDOW_LENGTHS_FT,
        default=500.0,
    )
    parser.add_argument("--compare-windows", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    if args.compare_windows:
        print(json.dumps(compare_window_results(), ensure_ascii=False, indent=2), flush=True)
        return
    config = load_config(args.segment_length_ft)
    experiment_id = str(config["experiment_id"])
    development = load_development_table(config)
    selected = select_mode(development, args.mode)
    source_cache_dir = resolve_clean_path(config["source_pf_legal_cache_dir"])
    artifact_dir = resolve_clean_path(config["artifact_dir"])
    fingerprint = stable_json_hash(
        {
            "experiment_id": experiment_id,
            "candidate_paths": config["candidate_paths"],
            "segment_length_ft": config["segment_length_ft"],
            "minimum_observed_gr_rows_per_segment": config["minimum_observed_gr_rows_per_segment"],
            "target_ess": config["target_ess"],
            "squared_gr_residual_cap": config["squared_gr_residual_cap"],
            "fallback_path": config["fallback_path"],
            "source_pf_artifact_fingerprint": config["source_pf_artifact_fingerprint"],
            "core_sha256": file_sha256(CLEAN_ROOT / "src" / "p3_pf03a_five_path_local500_proxy.py"),
        }
    )
    print(
        f"P3-PF03a {config['segment_length_ft']:g}ft {args.mode}：{len(selected)} 口井，"
        f"{int(selected['hidden_rows'].sum())} 行；"
        "零 PF 重跑，只筛查五条聚合路径的局部混合。",
        flush=True,
    )
    tasks: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        tasks.append(
            {
                **row,
                "experiment_id": experiment_id,
                "raw_train_dir": str(resolve_clean_path(config["raw_train_dir"])),
                "source_cache_dir": str(source_cache_dir),
                "source_pf_artifact_fingerprint": config["source_pf_artifact_fingerprint"],
                "artifact_dir": str(artifact_dir),
                "segment_length_ft": config["segment_length_ft"],
                "minimum_observed_gr_rows_per_segment": config["minimum_observed_gr_rows_per_segment"],
                "target_ess": config["target_ess"],
                "squared_gr_residual_cap": config["squared_gr_residual_cap"],
                "fingerprint": fingerprint,
            }
        )
    runtimes = generate_selected_paths(tasks, workers=int(config["workers"]))
    # 到这里为止，所有新路径已经生成；现在才允许读取开发目标用于评分。
    metrics, per_fold, per_well = score_paths(
        selected=selected,
        artifact_dir=artifact_dir,
        source_cache_dir=source_cache_dir,
        prediction_path=resolve_clean_path(config["source_model_predictions"]),
        experiment_id=experiment_id,
        segment_length_ft=float(config["segment_length_ft"]),
    )
    metrics["mode"] = args.mode
    metrics["experiment_fingerprint"] = fingerprint
    metrics["cache_hits"] = int(sum(bool(item["cache_hit"]) for item in runtimes))
    metrics["elapsed_seconds"] = float(time.perf_counter() - started)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifact_dir / "config.json", config)
    write_json_atomic(artifact_dir / f"metrics_{args.mode}.json", metrics)
    write_csv_atomic(artifact_dir / f"per_fold_{args.mode}.csv", per_fold)
    write_csv_atomic(artifact_dir / f"per_well_{args.mode}.csv", per_well)
    write_json_atomic(
        artifact_dir / f"runtime_{args.mode}.json",
        {
            "experiment_id": experiment_id,
            "mode": args.mode,
            "wells": int(len(selected)),
            "hidden_rows": int(selected["hidden_rows"].sum()),
            "shadow_wells": 0,
            "hidden_target_read_after_generation": True,
            "formal_pf03": False,
            "cache_hits": metrics["cache_hits"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "experiment_fingerprint": fingerprint,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
