"""运行 P3-PF01a：只改可见前缀 GR sigma，生成路径并和旧 PF 路径评分。"""

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
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p3_d01_pf_observation_weights import (  # noqa: E402
    file_sha256,
    load_development_registry,
    read_legal_inputs,
    select_mode_registry,
    stable_json_hash,
    write_csv_atomic,
    write_json_atomic,
)
from src.p2_p01_multiseed_pf import FEATURE_COLUMNS  # noqa: E402
from src.p3_pf01a_observed_only_sigma import (  # noqa: E402
    build_observed_only_sigma_pf_features,
)


EXPERIMENT_ID = "P3_PF01a_observed_only_sigma_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pf01a_observed_only_sigma_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
PATH_NAMES = ("mean", "scale3", "scale5", "scale8", "scale12")
PATH_COLUMNS = {
    "mean": "pf128_mean_tvt",
    "scale3": "pf128_scale_3_delta",
    "scale5": "pf128_scale_5_delta",
    "scale8": "pf128_scale_8_delta",
    "scale12": "pf128_scale_12_delta",
}


def resolve_clean_path(value: str | Path) -> Path:
    """相对路径统一相对于 rogii_clean。"""

    path = Path(value)
    return path if path.is_absolute() else (CLEAN_ROOT / path).resolve()


def cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    """返回单井路径缓存和运行信息路径。"""

    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存单井路径，支持中断后重跑。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_cache_hit(task: dict[str, Any]) -> dict[str, Any] | None:
    """缓存指纹、哈希、井号和行数都正确时才复用。"""

    cache_path, runtime_path = cache_paths(Path(task["artifact_dir"]), task["well_id"])
    if not cache_path.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime["experiment_fingerprint"] != task["fingerprint"]:
            return None
        if runtime["cache_sha256"] != file_sha256(cache_path):
            return None
        frame = pd.read_parquet(cache_path, columns=["well_id", "row_index", "_cache_fingerprint"])
        if len(frame) != int(task["hidden_rows"]):
            return None
        if not frame["well_id"].astype(str).eq(str(task["well_id"])).all():
            return None
        if frame["_cache_fingerprint"].astype(str).unique().tolist() != [task["fingerprint"]]:
            return None
        result = dict(runtime)
        result["cache_hit"] = True
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """读取合法列，运行只改 sigma 的 PF，并保存单井路径。"""

    cached = load_cache_hit(task)
    if cached is not None:
        return cached
    started = time.perf_counter()
    well_id = str(task["well_id"])
    horizontal, typewell = read_legal_inputs(Path(task["raw_train_dir"]), well_id)
    if int(horizontal["TVT_input"].isna().sum()) != int(task["hidden_rows"]):
        raise ValueError(f"井 {well_id} 隐藏行数不匹配")
    features, quality = build_observed_only_sigma_pf_features(
        horizontal,
        typewell,
        task["particle_filter"],
    )
    if list(features.columns) != FEATURE_COLUMNS or len(features) != int(task["hidden_rows"]):
        raise ValueError(f"井 {well_id} PF 特征 schema 或行数不匹配")
    output = features.copy()
    output.insert(0, "fold", int(task["fold"]))
    output.insert(0, "well_id", well_id)
    output["_cache_fingerprint"] = str(task["fingerprint"])
    cache_path, runtime_path = cache_paths(Path(task["artifact_dir"]), well_id)
    write_parquet_atomic(cache_path, output)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": task["fingerprint"],
        "well_id": well_id,
        "fold": int(task["fold"]),
        "hidden_rows": int(task["hidden_rows"]),
        "hidden_tvt_read": False,
        "changed_factor": "visible_observed_gr_sigma_only",
        "quality": quality,
        "cache_sha256": file_sha256(cache_path),
        "cache_hit": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(runtime_path, runtime)
    return runtime


def run_generation(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """并行运行并逐井打印进度；任一井失败就停止评分。"""

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_well = {executor.submit(generate_one_well, task): task["well_id"] for task in tasks}
        for completed, future in enumerate(as_completed(future_to_well), start=1):
            well_id = str(future_to_well[future])
            try:
                runtime = future.result()
                results.append(runtime)
                state = "缓存" if runtime.get("cache_hit") else "新算"
                print(
                    f"P3-PF01a {completed}/{len(tasks)}：{well_id} "
                    f"({state}, sigma {runtime['quality']['legacy_gr_sigma']:.2f}→"
                    f"{runtime['quality']['observed_only_gr_sigma']:.2f}, "
                    f"{runtime.get('elapsed_seconds', 0.0):.1f}秒)",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - 保留井号后统一阻断。
                errors.append(f"{well_id}: {error!r}")
                print(f"P3-PF01a {well_id} 失败：{error}", flush=True)
    if errors:
        raise RuntimeError(f"P3-PF01a 有 {len(errors)} 口井失败：{errors[:3]}")
    return results


def absolute_path(frame: pd.DataFrame, path_name: str) -> np.ndarray:
    """把缓存中的 mean 绝对 TVT 或 scale delta 统一恢复成绝对 TVT。"""

    column = PATH_COLUMNS[path_name]
    values = frame[column].to_numpy(dtype=np.float64)
    if path_name == "mean":
        return values
    return frame["last_visible_tvt"].to_numpy(dtype=np.float64) + values


def read_selected_targets(prediction_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """路径全部生成后，只读取选中开发井的目标；影子井不会进入评分。"""

    selected_ids = selected["well_id"].astype(str).tolist()
    targets = pd.read_parquet(
        prediction_path,
        columns=["well_id", "fold", "row_index", "target_tvt"],
        filters=[("well_id", "in", selected_ids)],
    )
    targets["well_id"] = targets["well_id"].astype(str)
    targets = targets.loc[targets["well_id"].isin(set(selected_ids))].copy()
    expected_rows = int(selected["hidden_rows"].sum())
    if len(targets) != expected_rows:
        raise ValueError(f"目标行数 {len(targets)} 不等于开发井注册表 {expected_rows}")
    return targets


def score_paths(
    selected: pd.DataFrame,
    artifact_dir: Path,
    legacy_cache_dir: Path,
    prediction_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """按相同行键计算新旧 mean/scale 路径的 pooled、逐折和逐井 RMSE。"""

    targets = read_selected_targets(prediction_path, selected)
    sums: dict[tuple[str, str, int | str], float] = {}
    counts: dict[tuple[str, str, int | str], int] = {}
    per_well_rows: list[dict[str, Any]] = []
    for registry_row in selected.itertuples(index=False):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        new_cache, _ = cache_paths(artifact_dir, well_id)
        old_cache = legacy_cache_dir / f"{well_id}.parquet"
        columns = ["row_index", "last_visible_tvt", *PATH_COLUMNS.values()]
        new_frame = pd.read_parquet(new_cache, columns=list(dict.fromkeys(columns)))
        old_frame = pd.read_parquet(old_cache, columns=list(dict.fromkeys(columns)))
        target = targets.loc[targets["well_id"].eq(well_id), ["row_index", "target_tvt"]]
        target = target.sort_values("row_index", kind="mergesort").reset_index(drop=True)
        new_frame = new_frame.sort_values("row_index", kind="mergesort").reset_index(drop=True)
        old_frame = old_frame.sort_values("row_index", kind="mergesort").reset_index(drop=True)
        target_row_index = target["row_index"].to_numpy(dtype=np.int64)
        new_row_index = new_frame["row_index"].to_numpy(dtype=np.int64)
        old_row_index = old_frame["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(target_row_index, new_row_index) or not np.array_equal(
            target_row_index,
            old_row_index,
        ):
            raise ValueError(f"井 {well_id} 新旧路径与目标 row_index 不一致")
        truth = target["target_tvt"].to_numpy(dtype=np.float64)
        row_result: dict[str, Any] = {"well_id": well_id, "fold": fold, "hidden_rows": len(truth)}
        for path_name in PATH_NAMES:
            for version, frame in (("old", old_frame), ("new", new_frame)):
                squared_error = np.square(absolute_path(frame, path_name) - truth)
                rmse = float(math.sqrt(float(squared_error.mean())))
                row_result[f"{version}_{path_name}_rmse"] = rmse
                for group in (fold, "all"):
                    key = (version, path_name, group)
                    sums[key] = sums.get(key, 0.0) + float(squared_error.sum())
                    counts[key] = counts.get(key, 0) + len(squared_error)
            row_result[f"{path_name}_improvement"] = (
                row_result[f"old_{path_name}_rmse"] - row_result[f"new_{path_name}_rmse"]
            )
        per_well_rows.append(row_result)

    per_well = pd.DataFrame(per_well_rows).sort_values("well_id", kind="mergesort")
    per_fold_rows: list[dict[str, Any]] = []
    for fold in sorted(selected["fold"].unique().tolist()):
        for path_name in PATH_NAMES:
            old_rmse = math.sqrt(sums[("old", path_name, int(fold))] / counts[("old", path_name, int(fold))])
            new_rmse = math.sqrt(sums[("new", path_name, int(fold))] / counts[("new", path_name, int(fold))])
            per_fold_rows.append(
                {"fold": int(fold), "path": path_name, "old_rmse": old_rmse, "new_rmse": new_rmse,
                 "improvement_ft": old_rmse - new_rmse, "hidden_rows": counts[("new", path_name, int(fold))]}
            )
    per_fold = pd.DataFrame(per_fold_rows)

    pooled: dict[str, Any] = {}
    for path_name in PATH_NAMES:
        old_rmse = math.sqrt(sums[("old", path_name, "all")] / counts[("old", path_name, "all")])
        new_rmse = math.sqrt(sums[("new", path_name, "all")] / counts[("new", path_name, "all")])
        pooled[path_name] = {"old_rmse": old_rmse, "new_rmse": new_rmse, "improvement_ft": old_rmse - new_rmse}
    old_best = min(PATH_NAMES, key=lambda name: pooled[name]["old_rmse"])
    new_best = min(PATH_NAMES, key=lambda name: pooled[name]["new_rmse"])
    cross_improvement = pooled[old_best]["old_rmse"] - pooled[new_best]["new_rmse"]
    fold_cross = []
    for fold in sorted(selected["fold"].unique().tolist()):
        old_value = float(per_fold.query("fold == @fold and path == @old_best")["old_rmse"].iloc[0])
        new_value = float(per_fold.query("fold == @fold and path == @new_best")["new_rmse"].iloc[0])
        fold_cross.append({"fold": int(fold), "improvement_ft": old_value - new_value})
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(selected)),
        "hidden_rows": int(selected["hidden_rows"].sum()),
        "paths": pooled,
        "old_best_fixed_path": old_best,
        "new_best_fixed_path": new_best,
        "best_fixed_cross_improvement_ft": float(cross_improvement),
        "best_fixed_cross_improvement_by_fold": fold_cross,
        "fold01_gate_passed": bool(
            set(selected["fold"].unique()) == {0, 1}
            and cross_improvement >= 0.20
            and min(item["improvement_ft"] for item in fold_cross) >= -0.25
        ),
    }
    return metrics, per_fold, per_well


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """只开放 smoke、fold01、all 三个固定范围。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "fold01", "all"), default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """先生成全部合法路径，成功后再读取目标并评分。"""

    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fold_path = resolve_clean_path(config["fold_registry"])
    shadow_path = resolve_clean_path(config["shadow_registry"])
    source_pf_config_path = resolve_clean_path(config["source_pf_config"])
    source_pf_config = json.loads(source_pf_config_path.read_text(encoding="utf-8"))
    particle_filter = source_pf_config["particle_filter"]
    development = load_development_registry(fold_path, shadow_path, config)
    selected = select_mode_registry(development, args.mode)
    artifact_dir = args.artifact_dir.resolve()
    fingerprint = stable_json_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "particle_filter": particle_filter,
            "core_sha256": file_sha256(CLEAN_ROOT / "src" / "p3_pf01a_observed_only_sigma.py"),
        }
    )
    tasks = [
        {
            "well_id": str(row.well_id), "fold": int(row.fold), "hidden_rows": int(row.hidden_rows),
            "raw_train_dir": str(resolve_clean_path(config["raw_train_dir"])), "artifact_dir": str(artifact_dir),
            "fingerprint": fingerprint, "particle_filter": particle_filter,
        }
        for row in selected.itertuples(index=False)
    ]
    print(
        f"P3-PF01a {args.mode}：{len(tasks)}口开发井，{int(selected['hidden_rows'].sum()):,}隐藏行，"
        f"{config['workers']}线程；输出 {artifact_dir}",
        flush=True,
    )
    print("主要耗时是每井 128-seed PF；中断后原命令重跑会逐井续算。", flush=True)
    started = time.perf_counter()
    runtimes = run_generation(tasks, int(config["workers"]))
    legal_rows = []
    for runtime in runtimes:
        legal_rows.append(
            {
                "well_id": runtime["well_id"], "fold": runtime["fold"], "hidden_rows": runtime["hidden_rows"],
                "cache_hit": runtime["cache_hit"], "elapsed_seconds": runtime["elapsed_seconds"], **runtime["quality"],
            }
        )
    write_csv_atomic(artifact_dir / "legal" / f"per_well_{args.mode}.csv", pd.DataFrame(legal_rows))
    metrics, per_fold, per_well = score_paths(
        selected,
        artifact_dir,
        resolve_clean_path(config["source_pf_legal_cache_dir"]),
        resolve_clean_path(config["source_model_predictions"]),
    )
    metrics["mode"] = args.mode
    metrics["experiment_fingerprint"] = fingerprint
    metrics["elapsed_seconds"] = float(time.perf_counter() - started)
    write_json_atomic(artifact_dir / f"metrics_{args.mode}.json", metrics)
    write_csv_atomic(artifact_dir / f"per_fold_{args.mode}.csv", per_fold)
    write_csv_atomic(artifact_dir / "oracle" / f"per_well_{args.mode}.csv", per_well)
    write_json_atomic(
        artifact_dir / f"runtime_{args.mode}.json",
        {"experiment_id": EXPERIMENT_ID, "mode": args.mode, "wells": len(tasks), "fingerprint": fingerprint,
         "cache_hits": int(sum(bool(item["cache_hit"]) for item in runtimes)), "elapsed_seconds": metrics["elapsed_seconds"]},
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
