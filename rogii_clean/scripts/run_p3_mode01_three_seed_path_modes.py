"""运行 P3-MODE01：从共享的 128 条 seed 路径提取三个固定路径模式。"""

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
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p3_d01_pf_observation_weights import (  # noqa: E402
    file_sha256,
    load_development_registry,
    stable_json_hash,
    write_csv_atomic,
    write_json_atomic,
)
from src.p3_mode01_three_seed_path_modes import (  # noqa: E402
    FORMAL_FEATURE_COLUMNS,
    FORMAL_LIKELIHOOD_SCALE,
    FORMAL_NUMBER_OF_MODES,
    FORMAL_NUMBER_OF_SEEDS,
    FORMAL_SAMPLE_POINTS,
    FORMAL_STABILITY_RANDOM_SEED,
    FORMAL_STABILITY_REPEATS,
    FORMAL_STABILITY_SAMPLE_SIZE,
    build_three_mode_features,
    compute_half_seed_stability,
    read_mode01_shared_cache,
    validate_shared_cache_arrays,
)


EXPERIMENT_ID = "P3_MODE01_three_seed_path_modes_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_mode01_three_seed_path_modes_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
OLD_PATH_COLUMNS = {
    "mean": "pf128_mean_delta",
    "scale3": "pf128_scale_3_delta",
    "scale5": "pf128_scale_5_delta",
    "scale8": "pf128_scale_8_delta",
    "scale12": "pf128_scale_12_delta",
}
MODE_PATH_COLUMNS = {
    "mode1": "pf_mode1_delta",
    "mode2": "pf_mode2_delta",
    "mode3": "pf_mode3_delta",
}
ORACLE_SUMMARY_FIELDS = {
    "per_well_best_old_oracle_rmse",
    "per_well_best_mode_oracle_rmse",
}


def resolve_clean_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (CLEAN_ROOT / path).resolve()


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def mode_cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def _read_shadow_ids(path: Path) -> set[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = ["well_id"]
    if "is_shadow" in header:
        columns.append("is_shadow")
    shadow = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    if "is_shadow" in shadow:
        values = shadow["is_shadow"]
        if pd.api.types.is_bool_dtype(values):
            mask = values.fillna(False).astype(bool)
        else:
            mask = values.astype(str).str.strip().str.lower().isin(
                {"1", "true", "yes", "y"}
            )
        shadow = shadow.loc[mask]
    return set(shadow["well_id"].astype(str))


def select_available_registry(
    development: pd.DataFrame,
    shared_cache_dir: Path,
    mode: str,
    smoke_well_ids: list[str],
) -> pd.DataFrame:
    """smoke 最多取三个已存在缓存；正式模式要求所需缓存全部齐全。"""

    if mode not in {"smoke", "fold01", "all"}:
        raise ValueError(f"MODE01 不支持运行模式 {mode}")
    ordered = development.copy()
    ordered["well_id"] = ordered["well_id"].astype(str)
    ordered = ordered.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    available_ids = {
        path.stem for path in Path(shared_cache_dir).glob("*.npz") if path.is_file()
    }

    if mode == "smoke":
        development_ids = set(ordered["well_id"])
        preferred = [
            str(well_id)
            for well_id in smoke_well_ids
            if str(well_id) in development_ids and str(well_id) in available_ids
        ]
        if len(preferred) < 3:
            fallback = [
                well_id
                for well_id in ordered["well_id"].tolist()
                if well_id in available_ids and well_id not in preferred
            ]
            preferred.extend(fallback[: 3 - len(preferred)])
        if not preferred:
            raise FileNotFoundError("MODE01 smoke 没有可复用的开发井共享缓存")
        return ordered.loc[ordered["well_id"].isin(preferred[:3])].reset_index(drop=True)

    selected = ordered.loc[ordered["fold"].isin([0, 1])].copy() if mode == "fold01" else ordered
    missing = sorted(set(selected["well_id"]) - available_ids)
    if missing:
        raise FileNotFoundError(
            f"MODE01 缺少 {len(missing)} 口共享 seed 缓存；先续跑 PF03。示例：{missing[:3]}"
        )
    return selected.reset_index(drop=True)


def build_mode_cache_frame(
    well_id: str,
    fold: int,
    arrays: dict[str, np.ndarray],
    experiment_fingerprint: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只从合法共享数组生成八项特征，并把稳定性留在诊断而非特征中。"""

    validate_shared_cache_arrays(arrays)
    features, diagnostics = build_three_mode_features(
        seed_delta=arrays["seed_delta"],
        hidden_md=arrays["hidden_md"],
        final_ll=arrays["final_ll"],
        seed_ids=arrays["seed_ids"],
    )
    stability = compute_half_seed_stability(
        seed_delta=arrays["seed_delta"],
        hidden_md=arrays["hidden_md"],
        final_ll=arrays["final_ll"],
        seed_ids=arrays["seed_ids"],
    )
    diagnostics.update(stability)

    output = features.copy()
    output.insert(
        0,
        "last_visible_tvt",
        np.full(len(output), float(arrays["last_tvt"][0]), dtype=np.float64),
    )
    output.insert(0, "row_index", arrays["row_index"].astype(np.int32, copy=False))
    output.insert(0, "fold", int(fold))
    output.insert(0, "well_id", str(well_id))
    output["_cache_fingerprint"] = str(experiment_fingerprint)
    expected_columns = [
        "well_id",
        "fold",
        "row_index",
        "last_visible_tvt",
        *FORMAL_FEATURE_COLUMNS,
        "_cache_fingerprint",
    ]
    if output.columns.tolist() != expected_columns:
        raise RuntimeError("MODE01 正式缓存出现未登记字段")
    return output, diagnostics


def _load_cache_hit(task: dict[str, Any]) -> dict[str, Any] | None:
    path_cache, runtime_path = mode_cache_paths(
        Path(task["artifact_dir"]), str(task["well_id"])
    )
    if not path_cache.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime["experiment_fingerprint"] != task["experiment_fingerprint"]:
            return None
        if runtime["shared_cache_fingerprint"] != task["shared_cache_fingerprint"]:
            return None
        if runtime["source_shared_cache_sha256"] != file_sha256(
            Path(task["shared_cache_path"])
        ):
            return None
        if runtime["path_cache_sha256"] != file_sha256(path_cache):
            return None
        cached = pd.read_parquet(
            path_cache,
            columns=["well_id", "row_index", "_cache_fingerprint"],
        )
        if len(cached) != int(task["hidden_rows"]):
            return None
        if cached["_cache_fingerprint"].astype(str).unique().tolist() != [
            task["experiment_fingerprint"]
        ]:
            return None
        output = dict(runtime)
        output["cache_hit"] = True
        return output
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    cached = _load_cache_hit(task)
    if cached is not None:
        return cached
    started = time.perf_counter()
    shared_cache_path = Path(task["shared_cache_path"])
    arrays = read_mode01_shared_cache(
        str(shared_cache_path),
        expected_fingerprint=str(task["shared_cache_fingerprint"]),
    )
    if arrays["seed_delta"].shape[1] != int(task["hidden_rows"]):
        raise ValueError(f"井 {task['well_id']} 共享缓存隐藏行数不一致")
    output, diagnostics = build_mode_cache_frame(
        well_id=str(task["well_id"]),
        fold=int(task["fold"]),
        arrays=arrays,
        experiment_fingerprint=str(task["experiment_fingerprint"]),
    )
    path_cache, runtime_path = mode_cache_paths(
        Path(task["artifact_dir"]), str(task["well_id"])
    )
    write_parquet_atomic(path_cache, output)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "well_id": str(task["well_id"]),
        "fold": int(task["fold"]),
        "hidden_rows": int(task["hidden_rows"]),
        "experiment_fingerprint": str(task["experiment_fingerprint"]),
        "shared_cache_fingerprint": str(task["shared_cache_fingerprint"]),
        "source_shared_cache_sha256": file_sha256(shared_cache_path),
        "path_cache_sha256": file_sha256(path_cache),
        "hidden_tvt_read": False,
        "formal_feature_count": len(FORMAL_FEATURE_COLUMNS),
        "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
        "diagnostics": diagnostics,
        "elapsed_seconds": float(time.perf_counter() - started),
        "cache_hit": False,
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
                diagnostic = runtime["diagnostics"]
                print(
                    f"P3-MODE01 {completed}/{len(tasks)}：{well_id}（{state}，"
                    f"mass={diagnostic['mode1_mass']:.3f}/{diagnostic['mode2_mass']:.3f}，"
                    f"稳定RMSE={diagnostic['stability_center_rmse_mean']:.3f}ft）",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - 汇总井号后阻断目标读取。
                errors.append(f"{well_id}: {error!r}")
                print(f"P3-MODE01 {well_id} 失败：{error}", flush=True)
    if errors:
        raise RuntimeError(f"P3-MODE01 有 {len(errors)} 口井失败：{errors[:3]}")
    return results


def read_selected_targets(
    prediction_path: Path,
    selected: pd.DataFrame,
    shadow_ids: set[str],
) -> pd.DataFrame:
    """在 Arrow 扫描阶段同时限定开发井并反选 shadow，之后才返回目标。"""

    selected_ids = selected["well_id"].astype(str).tolist()
    if set(selected_ids).intersection(shadow_ids):
        raise ValueError("MODE01 待评分井与 shadow 重叠")
    dataset = arrow_dataset.dataset(prediction_path, format="parquet")
    row_filter = arrow_dataset.field("well_id").isin(selected_ids)
    if shadow_ids:
        row_filter = row_filter & ~arrow_dataset.field("well_id").isin(
            sorted(shadow_ids)
        )
    table = dataset.to_table(
        columns=["well_id", "fold", "row_index", "target_tvt", "pred_tvt"],
        filter=row_filter,
    )
    targets = table.to_pandas()
    targets["well_id"] = targets["well_id"].astype(str)
    if set(targets["well_id"]).intersection(shadow_ids):
        raise RuntimeError("MODE01 目标表仍含 shadow 井")
    if len(targets) != int(selected["hidden_rows"].sum()):
        raise ValueError("MODE01 目标行数不等于所选开发井隐藏行数")
    return targets


def _rmse(squared_error: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(squared_error))))


def separate_oracle_metrics(
    combined_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """把事后逐井最佳指标从根指标中取走，保证 oracle 物理隔离。"""

    missing = sorted(ORACLE_SUMMARY_FIELDS - set(combined_metrics))
    if missing:
        raise ValueError(f"MODE01 聚合结果缺少 oracle 字段：{missing}")
    legal_metrics = {
        key: value
        for key, value in combined_metrics.items()
        if key not in ORACLE_SUMMARY_FIELDS
    }
    if any("oracle" in key.lower() for key in legal_metrics):
        raise RuntimeError("MODE01 根指标仍包含 oracle 字段")
    oracle_metrics = {
        "experiment_id": combined_metrics["experiment_id"],
        "wells": combined_metrics["wells"],
        "hidden_rows": combined_metrics["hidden_rows"],
        **{
            key: combined_metrics[key]
            for key in sorted(ORACLE_SUMMARY_FIELDS)
        },
    }
    return legal_metrics, oracle_metrics


def score_path_diagnostics(
    selected: pd.DataFrame,
    shadow_ids: set[str],
    artifact_dir: Path,
    old_path_cache_dir: Path,
    prediction_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """路径全部生成后才读开发目标；oracle 单独保存，绝不写回正式特征。"""

    targets = read_selected_targets(prediction_path, selected, shadow_ids)
    path_names = [*OLD_PATH_COLUMNS, *MODE_PATH_COLUMNS]
    squared_sums = {name: 0.0 for name in path_names}
    row_counts = {name: 0 for name in path_names}
    fold_sums: dict[tuple[int, str], float] = {}
    fold_counts: dict[tuple[int, str], int] = {}
    oracle_mode_squared_sum = 0.0
    oracle_old_squared_sum = 0.0
    oracle_rows = 0
    path_self_rows: list[dict[str, Any]] = []
    oracle_rows_output: list[dict[str, Any]] = []
    residual_x_parts: list[np.ndarray] = []
    residual_y_parts: list[np.ndarray] = []

    for registry_row in selected.itertuples(index=False):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        legal_path, runtime_path = mode_cache_paths(artifact_dir, well_id)
        mode_frame = pd.read_parquet(
            legal_path,
            columns=["row_index", "last_visible_tvt", *FORMAL_FEATURE_COLUMNS],
        ).sort_values("row_index", kind="mergesort")
        old_frame = pd.read_parquet(
            old_path_cache_dir / f"{well_id}.parquet",
            columns=["row_index", "last_visible_tvt", *OLD_PATH_COLUMNS.values()],
        ).sort_values("row_index", kind="mergesort")
        target = targets.loc[
            targets["well_id"].eq(well_id),
            ["row_index", "target_tvt", "pred_tvt"],
        ].sort_values("row_index", kind="mergesort")
        target_index = target["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            target_index, mode_frame["row_index"].to_numpy(dtype=np.int64)
        ) or not np.array_equal(
            target_index, old_frame["row_index"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"井 {well_id} 的路径和目标 row_index 不一致")
        truth = target["target_tvt"].to_numpy(dtype=np.float64)
        last_tvt = mode_frame["last_visible_tvt"].to_numpy(dtype=np.float64)
        predictions: dict[str, np.ndarray] = {}
        for path_name, column in OLD_PATH_COLUMNS.items():
            predictions[path_name] = (
                old_frame["last_visible_tvt"].to_numpy(dtype=np.float64)
                + old_frame[column].to_numpy(dtype=np.float64)
            )
        for path_name, column in MODE_PATH_COLUMNS.items():
            predictions[path_name] = (
                last_tvt + mode_frame[column].to_numpy(dtype=np.float64)
            )

        per_well_rmse: dict[str, float] = {}
        per_well_squared: dict[str, np.ndarray] = {}
        for path_name, prediction in predictions.items():
            squared_error = np.square(prediction - truth)
            per_well_squared[path_name] = squared_error
            per_well_rmse[path_name] = _rmse(squared_error)
            squared_sums[path_name] += float(np.sum(squared_error))
            row_counts[path_name] += len(squared_error)
            fold_sums[(fold, path_name)] = fold_sums.get((fold, path_name), 0.0) + float(
                np.sum(squared_error)
            )
            fold_counts[(fold, path_name)] = fold_counts.get((fold, path_name), 0) + len(
                squared_error
            )

        best_mode = min(MODE_PATH_COLUMNS, key=lambda name: per_well_rmse[name])
        best_old = min(OLD_PATH_COLUMNS, key=lambda name: per_well_rmse[name])
        oracle_mode_squared_sum += float(np.sum(per_well_squared[best_mode]))
        oracle_old_squared_sum += float(np.sum(per_well_squared[best_old]))
        oracle_rows += len(truth)
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        diagnostic = runtime["diagnostics"]
        scale8_delta = old_frame[OLD_PATH_COLUMNS["scale8"]].to_numpy(dtype=np.float64)
        path_self_rows.append(
            {
                "well_id": well_id,
                "fold": fold,
                "hidden_rows": len(truth),
                **diagnostic,
                "mode1_vs_scale8_path_rmse": _rmse(
                    np.square(
                        mode_frame["pf_mode1_delta"].to_numpy(dtype=np.float64)
                        - scale8_delta
                    )
                ),
                "mode2_vs_scale8_path_rmse": _rmse(
                    np.square(
                        mode_frame["pf_mode2_delta"].to_numpy(dtype=np.float64)
                        - scale8_delta
                    )
                ),
                "mode3_vs_scale8_path_rmse": _rmse(
                    np.square(
                        mode_frame["pf_mode3_delta"].to_numpy(dtype=np.float64)
                        - scale8_delta
                    )
                ),
            }
        )
        oracle_rows_output.append(
            {
                "well_id": well_id,
                "fold": fold,
                "hidden_rows": len(truth),
                **{f"{name}_rmse": value for name, value in per_well_rmse.items()},
                "best_old_path": best_old,
                "best_old_rmse": per_well_rmse[best_old],
                "best_mode_path": best_mode,
                "best_mode_rmse": per_well_rmse[best_mode],
                "oracle_mode_minus_old_rmse": (
                    per_well_rmse[best_mode] - per_well_rmse[best_old]
                ),
            }
        )
        residual_x_parts.append(
            mode_frame["pf_mode12_separation"].to_numpy(dtype=np.float64)
        )
        residual_y_parts.append(
            truth - target["pred_tvt"].to_numpy(dtype=np.float64)
        )

    pooled_rmse = {
        name: float(math.sqrt(squared_sums[name] / row_counts[name]))
        for name in path_names
    }
    best_pooled_mode = min(MODE_PATH_COLUMNS, key=lambda name: pooled_rmse[name])
    best_pooled_old = min(OLD_PATH_COLUMNS, key=lambda name: pooled_rmse[name])
    x = np.concatenate(residual_x_parts)
    y = np.concatenate(residual_y_parts)
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    metrics: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(selected)),
        "hidden_rows": int(selected["hidden_rows"].sum()),
        "path_rmse": pooled_rmse,
        "best_pooled_old_path": best_pooled_old,
        "best_pooled_old_rmse": pooled_rmse[best_pooled_old],
        "best_pooled_mode_path": best_pooled_mode,
        "best_pooled_mode_rmse": pooled_rmse[best_pooled_mode],
        "best_mode_improvement_vs_best_old_ft": (
            pooled_rmse[best_pooled_old] - pooled_rmse[best_pooled_mode]
        ),
        "per_well_best_old_oracle_rmse": float(
            math.sqrt(oracle_old_squared_sum / oracle_rows)
        ),
        "per_well_best_mode_oracle_rmse": float(
            math.sqrt(oracle_mode_squared_sum / oracle_rows)
        ),
        "mode12_separation_vs_p2p02_residual_pearson": correlation,
        "shadow_overlap_wells": 0,
        "shadow_target_access": False,
    }
    per_fold_rows: list[dict[str, Any]] = []
    for fold in sorted(selected["fold"].unique().tolist()):
        for path_name in path_names:
            per_fold_rows.append(
                {
                    "fold": int(fold),
                    "path": path_name,
                    "hidden_rows": fold_counts[(int(fold), path_name)],
                    "rmse": float(
                        math.sqrt(
                            fold_sums[(int(fold), path_name)]
                            / fold_counts[(int(fold), path_name)]
                        )
                    ),
                }
            )
    return (
        metrics,
        pd.DataFrame(per_fold_rows),
        pd.DataFrame(path_self_rows),
        pd.DataFrame(oracle_rows_output),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "fold01", "all"), default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def _validate_config(config: dict[str, Any]) -> None:
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("MODE01 配置实验编号错误")
    frozen_values = {
        "number_of_seeds": FORMAL_NUMBER_OF_SEEDS,
        "number_of_modes": FORMAL_NUMBER_OF_MODES,
        "sample_points": FORMAL_SAMPLE_POINTS,
        "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
        "stability_repeats": FORMAL_STABILITY_REPEATS,
        "stability_half_seed_count": FORMAL_STABILITY_SAMPLE_SIZE,
        "stability_random_seed": FORMAL_STABILITY_RANDOM_SEED,
        "source_shared_cache_format_version": 1,
    }
    for name, expected in frozen_values.items():
        if config[name] != expected:
            raise ValueError(f"MODE01 固定参数 {name} 被改变")
    if config["formal_feature_columns"] != FORMAL_FEATURE_COLUMNS:
        raise ValueError("MODE01 首批正式特征必须严格为登记的八项")
    if config["model_training"] or config["shadow_target_access"]:
        raise ValueError("MODE01 路径诊断不得训练模型或打开影子标签")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    shared_writer_path = resolve_clean_path(config["source_shared_writer"])
    if file_sha256(shared_writer_path) != config["source_shared_writer_sha256"]:
        raise ValueError("MODE01 共享缓存写入核心哈希不匹配")
    fold_path = resolve_clean_path(config["fold_registry"])
    shadow_path = resolve_clean_path(config["shadow_registry"])
    shared_cache_dir = resolve_clean_path(config["source_shared_cache_dir"])
    development = load_development_registry(fold_path, shadow_path, config)
    shadow_ids = _read_shadow_ids(shadow_path)
    selected = select_available_registry(
        development,
        shared_cache_dir=shared_cache_dir,
        mode=args.mode,
        smoke_well_ids=[str(value) for value in config["smoke_well_ids"]],
    )
    if set(selected["well_id"].astype(str)).intersection(shadow_ids):
        raise RuntimeError("MODE01 选择阶段仍含 shadow 井")

    artifact_dir = args.artifact_dir.resolve()
    experiment_fingerprint = stable_json_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "formal_core_sha256": file_sha256(
                CLEAN_ROOT / "src" / "p3_mode01_three_seed_path_modes.py"
            ),
            "runner_contract": "three_mode_paths_eight_features_oracle_separate_v1",
        }
    )
    tasks = [
        {
            "well_id": str(row.well_id),
            "fold": int(row.fold),
            "hidden_rows": int(row.hidden_rows),
            "shared_cache_path": str(shared_cache_dir / f"{row.well_id}.npz"),
            "shared_cache_fingerprint": config["source_shared_cache_fingerprint"],
            "artifact_dir": str(artifact_dir),
            "experiment_fingerprint": experiment_fingerprint,
        }
        for row in selected.itertuples(index=False)
    ]
    print(
        f"P3-MODE01 {args.mode}：{len(tasks)}口开发井，"
        f"{int(selected['hidden_rows'].sum()):,}隐藏行；输出 {artifact_dir}",
        flush=True,
    )
    print("主要耗时是每井 Ward K=3 与 8 次半样本稳定性；逐井缓存，可中断续跑。", flush=True)
    started = time.perf_counter()
    runtimes = run_generation(tasks, workers=min(int(config["workers"]), len(tasks)))

    path_self_legal = pd.DataFrame(
        [
            {
                "well_id": runtime["well_id"],
                "fold": runtime["fold"],
                "hidden_rows": runtime["hidden_rows"],
                "cache_hit": runtime["cache_hit"],
                "elapsed_seconds": runtime["elapsed_seconds"],
                **runtime["diagnostics"],
            }
            for runtime in runtimes
        ]
    ).sort_values("well_id", kind="mergesort")
    write_csv_atomic(artifact_dir / "legal" / f"per_well_{args.mode}.csv", path_self_legal)

    combined_metrics, per_fold, path_self, oracle = score_path_diagnostics(
        selected=selected,
        shadow_ids=shadow_ids,
        artifact_dir=artifact_dir,
        old_path_cache_dir=resolve_clean_path(config["source_pf_legal_cache_dir"]),
        prediction_path=resolve_clean_path(config["source_model_predictions"]),
    )
    metrics, oracle_metrics = separate_oracle_metrics(combined_metrics)
    metrics.update(
        {
            "mode": args.mode,
            "experiment_fingerprint": experiment_fingerprint,
            "source_shared_cache_fingerprint": config[
                "source_shared_cache_fingerprint"
            ],
            "formal_feature_count": len(FORMAL_FEATURE_COLUMNS),
            "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
            "cache_hits": int(sum(bool(runtime["cache_hit"]) for runtime in runtimes)),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    write_json_atomic(artifact_dir / f"metrics_{args.mode}.json", metrics)
    write_csv_atomic(artifact_dir / f"per_fold_{args.mode}.csv", per_fold)
    write_csv_atomic(artifact_dir / "path_self" / f"per_well_{args.mode}.csv", path_self)
    write_csv_atomic(artifact_dir / "oracle" / f"per_well_{args.mode}.csv", oracle)
    oracle_metrics.update(
        {
            "mode": args.mode,
            "experiment_fingerprint": experiment_fingerprint,
            "shadow_overlap_wells": 0,
            "shadow_target_access": False,
        }
    )
    write_json_atomic(
        artifact_dir / "oracle" / f"metrics_{args.mode}.json",
        oracle_metrics,
    )
    write_json_atomic(
        artifact_dir / f"runtime_{args.mode}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "mode": args.mode,
            "wells": len(tasks),
            "hidden_rows": int(selected["hidden_rows"].sum()),
            "elapsed_seconds": metrics["elapsed_seconds"],
            "cache_hits": metrics["cache_hits"],
            "experiment_fingerprint": experiment_fingerprint,
            "source_shared_cache_fingerprint": config[
                "source_shared_cache_fingerprint"
            ],
            "shadow_overlap_wells": 0,
            "shadow_target_access": False,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
