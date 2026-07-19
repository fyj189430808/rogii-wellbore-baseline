"""运行正式 P3-PF03：128 条冻结 seed 路径的 250/500/1000 ft 局部似然加权。"""

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
from src.p3_d01_pf_observation_audit import read_likelihood_cache  # noqa: E402
from src.p3_pf03_segmented_likelihood import (  # noqa: E402
    FORMAL_CENTER_STEP_FT,
    FORMAL_TARGET_ESS,
    FORMAL_WINDOW_SIZES_FT,
    LOCAL_PATH_COLUMNS,
    build_segmented_pf_features,
    write_pf03_shared_cache_atomic,
)


EXPERIMENT_ID = "P3_PF03_segmented_likelihood_v1"
SHARED_CACHE_ID = "P3_shared_pf_seed_paths_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pf03_segmented_likelihood_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
OLD_PATH_COLUMNS = {
    "scale3": "pf128_scale_3_delta",
    "scale5": "pf128_scale_5_delta",
    "scale8": "pf128_scale_8_delta",
    "scale12": "pf128_scale_12_delta",
}
NEW_PATH_COLUMNS = {
    "local250": "pf128_local250_delta",
    "local500": "pf128_local500_delta",
    "local1000": "pf128_local1000_delta",
}
EXPECTED_FOLD0_WELLS = 131
EXPECTED_FOLD0_HIDDEN_ROWS = 651_881


def resolve_clean_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (CLEAN_ROOT / path).resolve()


def path_cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_cache_hit(task: dict[str, Any]) -> dict[str, Any] | None:
    path_cache, runtime_path = path_cache_paths(
        Path(task["artifact_dir"]), str(task["well_id"])
    )
    shared_cache = Path(task["shared_cache_dir"]) / f"{task['well_id']}.npz"
    if not path_cache.is_file() or not runtime_path.is_file() or not shared_cache.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime["experiment_fingerprint"] != task["fingerprint"]:
            return None
        if runtime["shared_cache_fingerprint"] != task["shared_fingerprint"]:
            return None
        if runtime["path_cache_sha256"] != file_sha256(path_cache):
            return None
        if runtime["shared_cache_sha256"] != file_sha256(shared_cache):
            return None
        cached = pd.read_parquet(
            path_cache,
            columns=["well_id", "row_index", "_cache_fingerprint"],
        )
        if len(cached) != int(task["hidden_rows"]):
            return None
        if not cached["well_id"].astype(str).eq(str(task["well_id"])).all():
            return None
        if cached["_cache_fingerprint"].astype(str).unique().tolist() != [
            task["fingerprint"]
        ]:
            return None
        result = dict(runtime)
        result["cache_hit"] = True
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """一次冻结 PF 回放，同时写三条 PF03 路径和 MODE01 共享 seed 缓存。"""

    cached = load_cache_hit(task)
    if cached is not None:
        return cached
    started = time.perf_counter()
    well_id = str(task["well_id"])
    horizontal, typewell = read_legal_inputs(Path(task["raw_train_dir"]), well_id)
    if int(horizontal["TVT_input"].isna().sum()) != int(task["hidden_rows"]):
        raise ValueError(f"井 {well_id} 隐藏行数不匹配")
    d01_likelihoods = read_likelihood_cache(
        Path(task["d01_likelihood_cache_dir"]) / f"{well_id}.npz",
        expected_fingerprint=str(task["source_d01_fingerprint"]),
        expected_number_of_seeds=int(task["particle_filter"]["number_of_seeds"]),
    )
    features, quality, shared_arrays = build_segmented_pf_features(
        horizontal_well=horizontal,
        typewell=typewell,
        parameters=task["particle_filter"],
        d01_log_likelihoods=d01_likelihoods,
        window_sizes_ft=tuple(task["window_sizes_ft"]),
        target_ess=float(task["target_ess"]),
        center_step_ft=float(task["center_step_ft"]),
    )
    expected_columns = ["row_index", "last_visible_tvt", *LOCAL_PATH_COLUMNS]
    if list(features.columns) != expected_columns or len(features) != int(
        task["hidden_rows"]
    ):
        raise ValueError(f"井 {well_id} PF03 特征 schema 或行数不匹配")

    output = features.copy()
    output.insert(0, "fold", int(task["fold"]))
    output.insert(0, "well_id", well_id)
    output["_cache_fingerprint"] = str(task["fingerprint"])
    path_cache, runtime_path = path_cache_paths(
        Path(task["artifact_dir"]), well_id
    )
    shared_cache = Path(task["shared_cache_dir"]) / f"{well_id}.npz"
    write_pf03_shared_cache_atomic(
        shared_cache,
        shared_arrays,
        fingerprint=str(task["shared_fingerprint"]),
    )
    write_parquet_atomic(path_cache, output)

    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "shared_cache_id": SHARED_CACHE_ID,
        "experiment_fingerprint": task["fingerprint"],
        "shared_cache_fingerprint": task["shared_fingerprint"],
        "well_id": well_id,
        "fold": int(task["fold"]),
        "hidden_rows": int(task["hidden_rows"]),
        "hidden_tvt_read": False,
        "changed_factor": "local_seed_likelihood_weighting_only",
        "quality": quality,
        "path_cache_sha256": file_sha256(path_cache),
        "shared_cache_sha256": file_sha256(shared_cache),
        "cache_hit": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(runtime_path, runtime)
    return runtime


def run_generation(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_well = {
            executor.submit(generate_one_well, task): str(task["well_id"])
            for task in tasks
        }
        for completed, future in enumerate(as_completed(future_to_well), start=1):
            well_id = future_to_well[future]
            try:
                runtime = future.result()
                results.append(runtime)
                state = "缓存" if runtime.get("cache_hit") else "新算"
                quality = runtime["quality"]
                print(
                    f"P3-PF03 {completed}/{len(tasks)}：{well_id}（{state}，"
                    f"GR={quality['observed_gr_fraction']:.1%}，"
                    f"250ft无证据窗={quality['local250_no_observation_window_count']}，"
                    f"{runtime.get('elapsed_seconds', 0.0):.1f}秒）",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - 聚合井号后统一阻断评分。
                errors.append(f"{well_id}: {error!r}")
                print(f"P3-PF03 {well_id} 失败：{error}", flush=True)
    if errors:
        raise RuntimeError(f"P3-PF03 有 {len(errors)} 口井失败：{errors[:3]}")
    return results


def read_selected_targets(prediction_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = selected["well_id"].astype(str).tolist()
    targets = pd.read_parquet(
        prediction_path,
        columns=["well_id", "fold", "row_index", "target_tvt"],
        filters=[("well_id", "in", selected_ids)],
    )
    targets["well_id"] = targets["well_id"].astype(str)
    targets = targets.loc[targets["well_id"].isin(set(selected_ids))].copy()
    if len(targets) != int(selected["hidden_rows"].sum()):
        raise ValueError("目标行数不等于开发井注册表")
    return targets


def score_paths(
    selected: pd.DataFrame,
    artifact_dir: Path,
    old_cache_dir: Path,
    prediction_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """只在合法路径全部生成后，比较三条局部路径与四条冻结固定温度路径。"""

    targets = read_selected_targets(prediction_path, selected)
    all_columns = {**OLD_PATH_COLUMNS, **NEW_PATH_COLUMNS}
    sums: dict[tuple[str, int | str], float] = {}
    counts: dict[tuple[str, int | str], int] = {}
    per_well_rows: list[dict[str, Any]] = []
    for registry_row in selected.itertuples(index=False):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        new_cache, _ = path_cache_paths(artifact_dir, well_id)
        new_frame = pd.read_parquet(
            new_cache,
            columns=["row_index", "last_visible_tvt", *NEW_PATH_COLUMNS.values()],
        ).sort_values("row_index", kind="mergesort")
        old_frame = pd.read_parquet(
            old_cache_dir / f"{well_id}.parquet",
            columns=["row_index", "last_visible_tvt", *OLD_PATH_COLUMNS.values()],
        ).sort_values("row_index", kind="mergesort")
        target = targets.loc[
            targets["well_id"].eq(well_id), ["row_index", "target_tvt"]
        ].sort_values("row_index", kind="mergesort")
        target_index = target["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            target_index, new_frame["row_index"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"井 {well_id} 新路径 row_index 不一致")
        if not np.array_equal(
            target_index, old_frame["row_index"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"井 {well_id} 旧路径 row_index 不一致")
        truth = target["target_tvt"].to_numpy(dtype=np.float64)
        row_result: dict[str, Any] = {
            "well_id": well_id,
            "fold": fold,
            "hidden_rows": len(truth),
        }
        for path_name, column in all_columns.items():
            frame = old_frame if path_name in OLD_PATH_COLUMNS else new_frame
            prediction = (
                frame["last_visible_tvt"].to_numpy(dtype=np.float64)
                + frame[column].to_numpy(dtype=np.float64)
            )
            squared_error = np.square(prediction - truth)
            row_result[f"{path_name}_rmse"] = float(
                math.sqrt(float(squared_error.mean()))
            )
            for group in (fold, "all"):
                key = (path_name, group)
                sums[key] = sums.get(key, 0.0) + float(squared_error.sum())
                counts[key] = counts.get(key, 0) + len(squared_error)
        per_well_rows.append(row_result)

    per_well = pd.DataFrame(per_well_rows).sort_values("well_id", kind="mergesort")
    pooled = {
        path_name: float(
            math.sqrt(sums[(path_name, "all")] / counts[(path_name, "all")])
        )
        for path_name in all_columns
    }
    best_old = min(OLD_PATH_COLUMNS, key=lambda name: pooled[name])
    best_new = min(NEW_PATH_COLUMNS, key=lambda name: pooled[name])
    cross_improvement = float(pooled[best_old] - pooled[best_new])
    per_fold_rows: list[dict[str, Any]] = []
    fold_improvements: list[float] = []
    for fold in sorted(selected["fold"].unique().tolist()):
        old_rmse = math.sqrt(
            sums[(best_old, int(fold))] / counts[(best_old, int(fold))]
        )
        new_rmse = math.sqrt(
            sums[(best_new, int(fold))] / counts[(best_new, int(fold))]
        )
        improvement = float(old_rmse - new_rmse)
        fold_improvements.append(improvement)
        per_fold_rows.append(
            {
                "fold": int(fold),
                "hidden_rows": counts[(best_new, int(fold))],
                "best_old_fixed_path": best_old,
                "best_old_fixed_rmse": float(old_rmse),
                "best_local_path": best_new,
                "best_local_rmse": float(new_rmse),
                "improvement_ft": improvement,
            }
        )
    is_fold01 = set(selected["fold"].unique().tolist()) == {0, 1}
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(selected)),
        "hidden_rows": int(selected["hidden_rows"].sum()),
        "path_rmse": pooled,
        "best_old_fixed_path": best_old,
        "best_local_path": best_new,
        "best_path_cross_improvement_ft": cross_improvement,
        "fold01_gate_passed": bool(
            is_fold01
            and cross_improvement >= 0.20
            and min(fold_improvements) >= -0.10
        ),
    }
    return metrics, pd.DataFrame(per_fold_rows), per_well


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    run_choice = parser.add_mutually_exclusive_group()
    run_choice.add_argument("--mode", choices=("smoke", "fold01", "all"), default=None)
    run_choice.add_argument("--generation-only-fold", type=int, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args(argv)
    if args.mode is None and args.generation_only_fold is None:
        args.mode = "smoke"
    return args


def select_generation_only_registry(
    development: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    """只选择指定开发折；fold0 额外锁死 131 井和 651881 行合同。"""

    selected = development.loc[development["fold"].astype(int).eq(int(fold))].copy()
    if selected.empty or not selected["fold"].astype(int).eq(int(fold)).all():
        raise ValueError(f"development 中没有可生成的 fold {fold}")
    if int(fold) == 0 and (
        len(selected) != EXPECTED_FOLD0_WELLS
        or int(selected["hidden_rows"].sum()) != EXPECTED_FOLD0_HIDDEN_ROWS
    ):
        raise ValueError("generation-only fold0 必须精确覆盖 131 井/651881 行")
    return selected


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("配置实验编号错误")
    if tuple(config["window_sizes_ft"]) != FORMAL_WINDOW_SIZES_FT:
        raise ValueError("正式 PF03 窗口必须固定为 250/500/1000 ft")
    if float(config["target_ess"]) != FORMAL_TARGET_ESS:
        raise ValueError("正式 PF03 目标 ESS 必须固定为 16")
    if float(config["center_step_ft"]) != FORMAL_CENTER_STEP_FT:
        raise ValueError("正式 PF03 窗口中心步长必须固定为 100 ft")
    if config["model_training"] or config["shadow_target_access"]:
        raise ValueError("路径生成阶段不得训练模型或打开影子标签")
    source_pf_core = resolve_clean_path(config["source_pf_core"])
    if file_sha256(source_pf_core) != config["source_pf_core_sha256"]:
        raise ValueError("冻结 P2-P01 PF 核心哈希不匹配")
    source_pf_config_path = resolve_clean_path(config["source_pf_config"])
    source_pf_config = json.loads(source_pf_config_path.read_text(encoding="utf-8"))
    particle_filter = source_pf_config["particle_filter"]
    if int(particle_filter["number_of_seeds"]) != 128:
        raise ValueError("正式 PF03 必须使用冻结的 128 随机种子")

    development = load_development_registry(
        resolve_clean_path(config["fold_registry"]),
        resolve_clean_path(config["shadow_registry"]),
        config,
    )
    if args.generation_only_fold is None:
        selected = select_mode_registry(development, args.mode)
        run_label = str(args.mode)
    else:
        selected = select_generation_only_registry(
            development, int(args.generation_only_fold)
        )
        run_label = f"generation_only_fold{int(args.generation_only_fold)}"
    artifact_dir = args.artifact_dir.resolve()
    shared_cache_dir = resolve_clean_path(config["shared_seed_cache_dir"])
    shared_fingerprint = stable_json_hash(
        {
            "cache_id": SHARED_CACHE_ID,
            "particle_filter": particle_filter,
            "source_pf_core_sha256": config["source_pf_core_sha256"],
            "source_d01_fingerprint": config["source_d01_fingerprint"],
            "storage_schema": {
                "seed_delta": "float32[128,N]",
                "row_index": "int32[N]",
                "hidden_md": "float32[N]",
                "last_tvt": "float64[1]",
                "final_ll": "float64[128]",
                "seed_ids": "int32[128]",
            },
        }
    )
    fingerprint = stable_json_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "particle_filter": particle_filter,
            "shared_cache_fingerprint": shared_fingerprint,
            "formal_core_sha256": file_sha256(
                CLEAN_ROOT / "src" / "p3_pf03_segmented_likelihood.py"
            ),
        }
    )
    tasks = [
        {
            "well_id": str(row.well_id),
            "fold": int(row.fold),
            "hidden_rows": int(row.hidden_rows),
            "raw_train_dir": str(resolve_clean_path(config["raw_train_dir"])),
            "artifact_dir": str(artifact_dir),
            "shared_cache_dir": str(shared_cache_dir),
            "fingerprint": fingerprint,
            "shared_fingerprint": shared_fingerprint,
            "particle_filter": particle_filter,
            "window_sizes_ft": config["window_sizes_ft"],
            "target_ess": config["target_ess"],
            "center_step_ft": config["center_step_ft"],
            "d01_likelihood_cache_dir": str(
                resolve_clean_path(config["source_d01_likelihood_cache_dir"])
            ),
            "source_d01_fingerprint": config["source_d01_fingerprint"],
        }
        for row in selected.itertuples(index=False)
    ]
    print(
        f"P3-PF03 {run_label}：{len(tasks)}口开发井，"
        f"{int(selected['hidden_rows'].sum()):,}隐藏行，{config['workers']}线程；"
        f"路径输出 {artifact_dir}；共享seed缓存 {shared_cache_dir}",
        flush=True,
    )
    print("主要耗时是每井一次 128-seed 冻结 PF；逐井原子缓存，同命令可续跑。", flush=True)
    started = time.perf_counter()
    runtimes = run_generation(tasks, workers=int(config["workers"]))
    legal_rows = [
        {
            "well_id": runtime["well_id"],
            "fold": runtime["fold"],
            "hidden_rows": runtime["hidden_rows"],
            "cache_hit": runtime["cache_hit"],
            "elapsed_seconds": runtime["elapsed_seconds"],
            **runtime["quality"],
        }
        for runtime in runtimes
    ]
    write_csv_atomic(
        artifact_dir / "legal" / f"per_well_{run_label}.csv",
        pd.DataFrame(legal_rows),
    )
    if args.generation_only_fold is not None:
        cache_hits = int(sum(bool(runtime["cache_hit"]) for runtime in runtimes))
        generation_runtime = {
            "experiment_id": EXPERIMENT_ID,
            "generation_only_fold": int(args.generation_only_fold),
            "wells": int(len(tasks)),
            "rows": int(selected["hidden_rows"].sum()),
            "cache_hits": cache_hits,
            "new": int(len(runtimes) - cache_hits),
            "generated": int(len(runtimes)),
            "fingerprints": {
                "experiment": fingerprint,
                "shared_cache": shared_fingerprint,
            },
            "completed": bool(len(runtimes) == len(tasks)),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        write_json_atomic(
            artifact_dir / f"runtime_{run_label}.json",
            generation_runtime,
        )
        print(json.dumps(generation_runtime, ensure_ascii=False, indent=2), flush=True)
        return
    metrics, per_fold, per_well = score_paths(
        selected=selected,
        artifact_dir=artifact_dir,
        old_cache_dir=resolve_clean_path(config["source_pf_legal_cache_dir"]),
        prediction_path=resolve_clean_path(config["source_model_predictions"]),
    )
    metrics["mode"] = args.mode
    metrics["experiment_fingerprint"] = fingerprint
    metrics["shared_cache_fingerprint"] = shared_fingerprint
    metrics["elapsed_seconds"] = float(time.perf_counter() - started)
    metrics["cache_hits"] = int(sum(bool(runtime["cache_hit"]) for runtime in runtimes))
    write_json_atomic(artifact_dir / f"metrics_{args.mode}.json", metrics)
    write_csv_atomic(artifact_dir / f"per_fold_{args.mode}.csv", per_fold)
    write_csv_atomic(artifact_dir / "oracle" / f"per_well_{args.mode}.csv", per_well)
    write_json_atomic(
        artifact_dir / f"runtime_{args.mode}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "mode": args.mode,
            "wells": len(tasks),
            "hidden_rows": int(selected["hidden_rows"].sum()),
            "cache_hits": metrics["cache_hits"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "experiment_fingerprint": fingerprint,
            "shared_cache_fingerprint": shared_fingerprint,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
