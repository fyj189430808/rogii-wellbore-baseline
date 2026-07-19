"""运行 P3-PF02a：按固定目标 ESS 从四条冻结 PF 路径中逐井选路。"""

from __future__ import annotations

import argparse
import hashlib
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

from src.p3_pf02a_target_ess import (  # noqa: E402
    SCALE_TO_PATH_COLUMN,
    build_target_ess_path,
)


EXPERIMENT_ID = "P3_PF02a_target_ess_v1"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_pf02a_target_ess_v1.json"
ESS_COLUMN_BY_SCALE = {
    3.0: "scale_3_effective_sample_size",
    5.0: "scale_5_effective_sample_size",
    8.0: "scale_8_effective_sample_size",
    12.0: "scale_12_effective_sample_size",
}


def resolve_clean_path(path_text: str) -> Path:
    """把配置中的相对路径统一解释为相对 rogii_clean。"""
    path = Path(path_text)
    return path if path.is_absolute() else CLEAN_ROOT / path


def file_sha256(path: Path) -> str:
    """计算输入文件指纹，防止静默换源。"""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """用稳定 JSON 计算实验指纹。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("配置实验编号错误")
    if config["candidate_scales"] != [3.0, 5.0, 8.0, 12.0]:
        raise ValueError("PF02a 只允许冻结的四个离散温度")
    if config["target_ess"] != 96.0 or config["number_of_seeds"] != 128:
        raise ValueError("PF02a 已冻结为 target ESS 96/128")
    if config["shadow_target_access"] or config["model_training"]:
        raise ValueError("PF02a 不允许打开影子标签或训练模型")
    return config


def load_development_table(config: dict[str, Any]) -> pd.DataFrame:
    """读取 D01 已冻结的 657 口开发井无标签统计。"""
    source_path = resolve_clean_path(config["source_d01_per_well"])
    required = ["well_id", "fold", "hidden_rows", *ESS_COLUMN_BY_SCALE.values()]
    development = pd.read_csv(source_path, usecols=required, dtype={"well_id": str})
    development["well_id"] = development["well_id"].astype(str).str.zfill(8)
    development = development.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if development["well_id"].duplicated().any():
        raise ValueError("D01 开发井表存在重复井")
    if len(development) != int(config["development_wells"]):
        raise ValueError("D01 开发井数量与配置不一致")
    if int(development["hidden_rows"].sum()) != int(config["development_hidden_rows"]):
        raise ValueError("D01 开发井隐藏行数与配置不一致")
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


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """复制一口井最接近目标 ESS 的冻结路径，并保存可恢复缓存。"""
    well_id = str(task["well_id"])
    artifact_dir = Path(task["artifact_dir"])
    output_path, runtime_path = cache_paths(artifact_dir, well_id)
    if output_path.is_file() and runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("experiment_fingerprint") == task["fingerprint"]:
            cached = pd.read_parquet(output_path, columns=["well_id", "_cache_fingerprint"])
            if (
                len(cached) == int(task["hidden_rows"])
                and cached["well_id"].astype(str).unique().tolist() == [well_id]
                and cached["_cache_fingerprint"].astype(str).unique().tolist()
                == [task["fingerprint"]]
            ):
                runtime["cache_hit"] = True
                return runtime

    start = time.perf_counter()
    old_path = Path(task["source_cache_dir"]) / f"{well_id}.parquet"
    frozen_columns = [
        "well_id",
        "row_index",
        "last_visible_tvt",
        *SCALE_TO_PATH_COLUMN.values(),
        "_cache_fingerprint",
    ]
    frozen_paths = pd.read_parquet(old_path, columns=frozen_columns)
    source_fingerprints = frozen_paths["_cache_fingerprint"].astype(str).unique().tolist()
    if source_fingerprints != [task["source_pf_artifact_fingerprint"]]:
        raise ValueError(f"井 {well_id} 的冻结 PF 缓存指纹不一致")
    if len(frozen_paths) != int(task["hidden_rows"]):
        raise ValueError(f"井 {well_id} 的冻结路径行数不一致")

    ess_by_scale = {
        scale: float(task[ESS_COLUMN_BY_SCALE[scale]]) for scale in ESS_COLUMN_BY_SCALE
    }
    output = build_target_ess_path(
        frozen_paths=frozen_paths,
        ess_by_scale=ess_by_scale,
        target_ess=float(task["target_ess"]),
    )
    output["_cache_fingerprint"] = str(task["fingerprint"])
    write_parquet_atomic(output_path, output)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": task["fingerprint"],
        "well_id": well_id,
        "hidden_rows": int(len(output)),
        "selected_scale": float(output["selected_scale"].iloc[0]),
        "selected_ess": float(output["selected_ess"].iloc[0]),
        "target_ess": float(task["target_ess"]),
        "elapsed_seconds": float(time.perf_counter() - start),
        "cache_hit": False,
    }
    write_json_atomic(runtime_path, runtime)
    return runtime


def generate_selected_paths(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_one_well, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                print(
                    f"P3-PF02a 路径缓存 {completed}/{len(tasks)}；"
                    f"最新井 {result['well_id']}，scale={result['selected_scale']:g}",
                    flush=True,
                )
    return sorted(results, key=lambda item: item["well_id"])


def read_targets(prediction_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = selected["well_id"].astype(str).tolist()
    targets = pd.read_parquet(
        prediction_path,
        columns=["well_id", "fold", "row_index", "target_tvt"],
        filters=[("well_id", "in", selected_ids)],
    )
    targets["well_id"] = targets["well_id"].astype(str).str.zfill(8)
    targets = targets.loc[targets["well_id"].isin(set(selected_ids))].copy()
    if len(targets) != int(selected["hidden_rows"].sum()):
        raise ValueError("目标行数与开发井注册表不一致")
    return targets


def score_paths(
    selected: pd.DataFrame,
    artifact_dir: Path,
    source_cache_dir: Path,
    prediction_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """用相同目标行比较目标 ESS 路径和四条冻结固定温度路径。"""
    targets = read_targets(prediction_path, selected)
    path_names = ["target_ess", "scale3", "scale5", "scale8", "scale12"]
    total_sse = {name: 0.0 for name in path_names}
    total_count = 0
    fold_sse: dict[tuple[int, str], float] = {}
    fold_count: dict[int, int] = {}
    per_well_rows: list[dict[str, Any]] = []

    for registry_row in selected.itertuples(index=False):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        new_path, _ = cache_paths(artifact_dir, well_id)
        selected_path = pd.read_parquet(new_path)
        frozen_path = pd.read_parquet(
            source_cache_dir / f"{well_id}.parquet",
            columns=["row_index", "last_visible_tvt", *SCALE_TO_PATH_COLUMN.values()],
        )
        truth = targets.loc[
            targets["well_id"].eq(well_id), ["row_index", "target_tvt"]
        ].sort_values("row_index", kind="mergesort")
        selected_path = selected_path.sort_values("row_index", kind="mergesort")
        frozen_path = frozen_path.sort_values("row_index", kind="mergesort")
        truth_index = truth["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(truth_index, selected_path["row_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"井 {well_id} 的目标 ESS 路径 row_index 不一致")
        if not np.array_equal(truth_index, frozen_path["row_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"井 {well_id} 的冻结路径 row_index 不一致")

        target_tvt = truth["target_tvt"].to_numpy(dtype=np.float64)
        last_visible_tvt = frozen_path["last_visible_tvt"].to_numpy(dtype=np.float64)
        predictions = {
            "target_ess": selected_path["last_visible_tvt"].to_numpy(dtype=np.float64)
            + selected_path["pf128_target_ess_delta"].to_numpy(dtype=np.float64),
        }
        for scale, column in SCALE_TO_PATH_COLUMN.items():
            predictions[f"scale{int(scale)}"] = (
                last_visible_tvt + frozen_path[column].to_numpy(dtype=np.float64)
            )

        row_result: dict[str, Any] = {
            "well_id": well_id,
            "fold": fold,
            "hidden_rows": len(target_tvt),
            "selected_scale": float(selected_path["selected_scale"].iloc[0]),
            "selected_ess": float(selected_path["selected_ess"].iloc[0]),
        }
        fold_count[fold] = fold_count.get(fold, 0) + len(target_tvt)
        total_count += len(target_tvt)
        for path_name, prediction in predictions.items():
            squared_error = np.square(prediction - target_tvt)
            sse = float(squared_error.sum())
            total_sse[path_name] += sse
            fold_sse[(fold, path_name)] = fold_sse.get((fold, path_name), 0.0) + sse
            row_result[f"{path_name}_rmse"] = float(math.sqrt(sse / len(target_tvt)))
        per_well_rows.append(row_result)

    pooled_rmse = {
        path_name: float(math.sqrt(total_sse[path_name] / total_count))
        for path_name in path_names
    }
    fixed_names = ["scale3", "scale5", "scale8", "scale12"]
    best_fixed_name = min(fixed_names, key=lambda name: pooled_rmse[name])
    per_fold_rows: list[dict[str, Any]] = []
    per_fold_improvement: list[float] = []
    for fold in sorted(selected["fold"].unique().tolist()):
        target_rmse = math.sqrt(fold_sse[(int(fold), "target_ess")] / fold_count[int(fold)])
        fixed_rmse = math.sqrt(fold_sse[(int(fold), best_fixed_name)] / fold_count[int(fold)])
        improvement = float(fixed_rmse - target_rmse)
        per_fold_improvement.append(improvement)
        per_fold_rows.append(
            {
                "fold": int(fold),
                "hidden_rows": int(fold_count[int(fold)]),
                "target_ess_rmse": float(target_rmse),
                "best_fixed_path": best_fixed_name,
                "best_fixed_path_rmse": float(fixed_rmse),
                "improvement_ft": improvement,
            }
        )

    improvement = float(pooled_rmse[best_fixed_name] - pooled_rmse["target_ess"])
    scale_counts = {
        str(int(scale)): int(sum(row["selected_scale"] == scale for row in per_well_rows))
        for scale in SCALE_TO_PATH_COLUMN
    }
    is_fold01 = set(selected["fold"].unique().tolist()) == {0, 1}
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(selected)),
        "hidden_rows": int(total_count),
        "target_ess": 96.0,
        "selected_scale_counts": scale_counts,
        "pooled_rmse": pooled_rmse,
        "best_fixed_path": best_fixed_name,
        "improvement_vs_best_fixed_ft": improvement,
        "fold01_gate_passed": bool(
            is_fold01 and improvement >= 0.20 and min(per_fold_improvement) >= -0.10
        ),
    }
    per_fold = pd.DataFrame(per_fold_rows)
    per_well = pd.DataFrame(per_well_rows).sort_values("well_id", kind="mergesort")
    return metrics, per_fold, per_well


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "fold01", "all"), default="smoke")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.perf_counter()
    config = load_config()
    development = load_development_table(config)
    selected = select_mode(development, args.mode)
    source_d01_path = resolve_clean_path(config["source_d01_per_well"])
    source_cache_dir = resolve_clean_path(config["source_pf_legal_cache_dir"])
    source_predictions = resolve_clean_path(config["source_model_predictions"])
    artifact_dir = resolve_clean_path(config["artifact_dir"])
    fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "target_ess": config["target_ess"],
            "candidate_scales": config["candidate_scales"],
            "source_pf_artifact_fingerprint": config["source_pf_artifact_fingerprint"],
            "source_d01_per_well_sha256": file_sha256(source_d01_path),
            "core_sha256": file_sha256(CLEAN_ROOT / "src" / "p3_pf02a_target_ess.py"),
        }
    )
    print(
        f"P3-PF02a {args.mode}：{len(selected)} 口井，"
        f"{int(selected['hidden_rows'].sum())} 行；只复制冻结路径，不重跑 PF。",
        flush=True,
    )
    tasks: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        tasks.append(
            {
                **row,
                "artifact_dir": str(artifact_dir),
                "source_cache_dir": str(source_cache_dir),
                "source_pf_artifact_fingerprint": config["source_pf_artifact_fingerprint"],
                "target_ess": config["target_ess"],
                "fingerprint": fingerprint,
            }
        )
    runtimes = generate_selected_paths(tasks, workers=int(config["workers"]))
    metrics, per_fold, per_well = score_paths(
        selected=selected,
        artifact_dir=artifact_dir,
        source_cache_dir=source_cache_dir,
        prediction_path=source_predictions,
    )
    metrics["mode"] = args.mode
    metrics["experiment_fingerprint"] = fingerprint
    metrics["elapsed_seconds"] = float(time.perf_counter() - start)
    metrics["cache_hits"] = int(sum(bool(item["cache_hit"]) for item in runtimes))
    write_json_atomic(artifact_dir / f"metrics_{args.mode}.json", metrics)
    write_csv_atomic(artifact_dir / f"per_fold_{args.mode}.csv", per_fold)
    write_csv_atomic(artifact_dir / f"per_well_{args.mode}.csv", per_well)
    write_json_atomic(
        artifact_dir / f"runtime_{args.mode}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "mode": args.mode,
            "wells": len(selected),
            "hidden_rows": int(selected["hidden_rows"].sum()),
            "cache_hits": metrics["cache_hits"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "experiment_fingerprint": fingerprint,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

