"""运行 P2-S01 严格 outer-fold EGFDU 局部平面路径诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_s01_outer_fold_surface import (  # noqa: E402
    LEGAL_TARGET_COLUMNS,
    build_legal_surface_path,
    build_source_representative,
    permute_source_surfaces,
)


EXPERIMENT_ID = "P2_S01_outer_fold_surface_path_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_s01_outer_fold_surface_path_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "all")


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大 parquet 一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """把配置或 lineage 变成顺序稳定的指纹。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_frame_hash(frame: pd.DataFrame) -> str:
    """为已排序的小表生成稳定指纹。"""

    hashed_rows = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed_rows.tobytes()).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """先写临时文件再替换，防止中断留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存 parquet，支持单井断点续跑。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    """确认运行配置仍是实验卡预先冻结的唯一版本。"""

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("P2-S01 experiment_id 不匹配")
    if config.get("surface_name") != "EGFDU":
        raise ValueError("P2-S01 v1 只允许 EGFDU")
    if int(config.get("nearest_source_wells", -1)) != 10:
        raise ValueError("P2-S01 v1 最近 source 井数必须固定为 10")
    if float(config.get("control_step_horizontal_ft", -1.0)) != 50.0:
        raise ValueError("P2-S01 v1 控制点间隔必须固定为 50 ft")
    if config.get("supported_modes") != list(SUPPORTED_MODES):
        raise ValueError("P2-S01 supported_modes 被修改")


def load_registry(config: dict[str, Any]) -> pd.DataFrame:
    """读取并验证冻结的按井五折表。"""

    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    if file_sha256(registry_path) != str(config["fold_registry_sha256"]):
        raise ValueError("P2-S01 fold 注册表 SHA-256 不一致")

    registry = pd.read_csv(registry_path, dtype={"well_id": str, "pad_id": str})
    required_columns = {"well_id", "fold", "hidden_rows"}
    if not required_columns.issubset(registry.columns):
        raise ValueError("P2-S01 fold 注册表缺少必要列")
    registry["well_id"] = registry["well_id"].astype(str)
    registry["fold"] = pd.to_numeric(registry["fold"], errors="raise").astype(int)
    if bool(registry["well_id"].duplicated().any()):
        raise ValueError("P2-S01 fold 注册表含重复 well_id")
    if sorted(registry["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("P2-S01 fold 注册表必须正好包含 0～4")
    if len(registry) != int(config["expected_wells"]):
        raise ValueError("P2-S01 fold 注册表井数不等于冻结值")
    if int(registry["hidden_rows"].sum()) != int(config["expected_hidden_rows"]):
        raise ValueError("P2-S01 fold 注册表隐藏行数不等于冻结值")
    return registry.sort_values("well_id", kind="stable").reset_index(drop=True)


def validate_external_inputs(config: dict[str, Any]) -> None:
    """验证地层面血缘审计和 P2B00 预测来自冻结产物。"""

    lineage_path = CLEAN_ROOT / str(config["surface_lineage_summary"])
    if file_sha256(lineage_path) != str(config["surface_lineage_summary_sha256"]):
        raise ValueError("P2-S01 surface lineage summary SHA-256 不一致")
    lineage_summary = json.loads(lineage_path.read_text(encoding="utf-8"))
    relationship_supported = lineage_summary.get("relationship_checks", {}).get(
        "derived_relationship_supported"
    )
    if not bool(lineage_summary.get("full_registry_processed")) or not bool(relationship_supported):
        raise ValueError("P2-S01 surface 血缘审计没有通过")

    baseline_path = CLEAN_ROOT / str(config["baseline_predictions"])
    if file_sha256(baseline_path) != str(config["baseline_predictions_sha256"]):
        raise ValueError("P2-S01 P2B00 predictions SHA-256 不一致")


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """把配置、核心代码和冻结输入一起放入实验指纹。"""

    fingerprint_parts = {
        "config": config,
        "core_sha256": file_sha256(CLEAN_ROOT / "src" / "p2_s01_outer_fold_surface.py"),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "fold_registry_sha256": config["fold_registry_sha256"],
        "baseline_predictions_sha256": config["baseline_predictions_sha256"],
        "surface_lineage_summary_sha256": config["surface_lineage_summary_sha256"],
    }
    return stable_json_hash(fingerprint_parts)


def horizontal_path(raw_train_dir: Path, well_id: str) -> Path:
    """返回一口训练水平井的固定文件名。"""

    path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    if not path.is_file():
        raise FileNotFoundError(f"找不到水平井文件：{path}")
    return path


def build_all_source_representatives(
    registry: pd.DataFrame,
    raw_train_dir: Path,
    surface_name: str,
    workers: int,
) -> pd.DataFrame:
    """读取 773 口井，每井只保留一个 EGFDU 中位代表点。"""

    well_ids = registry["well_id"].astype(str).tolist()

    def load_one(well_id: str) -> dict[str, float | str]:
        source_frame = pd.read_csv(
            horizontal_path(raw_train_dir, well_id),
            usecols=["X", "Y", str(surface_name)],
        )
        return build_source_representative(well_id, source_frame, str(surface_name))

    representatives: list[dict[str, float | str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_by_well = {executor.submit(load_one, well_id): well_id for well_id in well_ids}
        for completed_number, future in enumerate(as_completed(future_by_well), start=1):
            representatives.append(future.result())
            if completed_number % 100 == 0 or completed_number == len(well_ids):
                print(
                    f"source 代表点：{completed_number}/{len(well_ids)}",
                    flush=True,
                )

    result = pd.DataFrame(representatives)
    result = result.sort_values("well_id", kind="stable").reset_index(drop=True)
    if result["well_id"].tolist() != sorted(well_ids):
        raise ValueError("source 代表点没有完整覆盖 fold 注册表")
    return result


def load_or_build_source_representatives(
    registry: pd.DataFrame,
    raw_train_dir: Path,
    config: dict[str, Any],
    artifact_dir: Path,
    fingerprint: str,
) -> pd.DataFrame:
    """复用来源完全一致的 source 代表点缓存，否则重新构建。"""

    cache_path = artifact_dir / "source_representatives.parquet"
    metadata_path = artifact_dir / "source_representatives_meta.json"
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("experiment_fingerprint") == fingerprint:
            cached = pd.read_parquet(cache_path)
            if len(cached) == len(registry):
                print(f"命中 source 代表点缓存：{cache_path}", flush=True)
                return cached

    print("未命中 source 代表点缓存，开始读取 773 口 EGFDU。", flush=True)
    representatives = build_all_source_representatives(
        registry=registry,
        raw_train_dir=raw_train_dir,
        surface_name=str(config["surface_name"]),
        workers=int(config["workers"]),
    )
    write_parquet_atomic(cache_path, representatives)
    write_json_atomic(
        metadata_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "wells": int(len(representatives)),
            "surface_name": str(config["surface_name"]),
            "representatives_sha256": file_sha256(cache_path),
        },
    )
    return representatives


def build_fold_source_table(
    registry: pd.DataFrame,
    representatives: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    """严格排除完整 outer-valid fold，返回当前折唯一合法的 source 表。"""

    fold_lookup = registry.loc[:, ["well_id", "fold"]].copy()
    fold_lookup["well_id"] = fold_lookup["well_id"].astype(str)
    merged = representatives.merge(fold_lookup, on="well_id", how="left", validate="one_to_one")
    if bool(merged["fold"].isna().any()):
        raise ValueError("source 代表点中存在 fold 未知的井")
    sources = merged.loc[merged["fold"].astype(int).ne(int(outer_fold))].copy()
    sources = sources.drop(columns="fold").sort_values("well_id", kind="stable").reset_index(drop=True)
    excluded_ids = set(registry.loc[registry["fold"].eq(int(outer_fold)), "well_id"].astype(str))
    if not set(sources["well_id"].astype(str)).isdisjoint(excluded_ids):
        raise ValueError("surface source 表混入了 outer-valid 井")
    return sources


def build_own_surface_oracle_path(
    oracle_target_df: pd.DataFrame,
    surface_name: str,
) -> np.ndarray:
    """用目标井自身 surface 检查公式上限；该函数只在合法路径完成后调用。"""

    required_columns = ("Z", "TVT_input", str(surface_name))
    missing = [column for column in required_columns if column not in oracle_target_df.columns]
    if missing:
        raise ValueError(f"目标自身 surface oracle 缺少列：{missing}")
    oracle = oracle_target_df.loc[:, list(required_columns)].apply(pd.to_numeric, errors="coerce")
    visible_mask = oracle["TVT_input"].notna().to_numpy(dtype=bool)
    hidden_mask = ~visible_mask
    if not bool(visible_mask.any()) or not bool(hidden_mask.any()):
        raise ValueError("目标自身 surface oracle 需要可见前缀和隐藏后缀")

    visible_offset = (
        oracle.loc[visible_mask, "TVT_input"].to_numpy(dtype=np.float64)
        + oracle.loc[visible_mask, "Z"].to_numpy(dtype=np.float64)
        - oracle.loc[visible_mask, str(surface_name)].to_numpy(dtype=np.float64)
    )
    offset = float(np.median(visible_offset))
    prediction = (
        oracle.loc[hidden_mask, str(surface_name)].to_numpy(dtype=np.float64)
        + offset
        - oracle.loc[hidden_mask, "Z"].to_numpy(dtype=np.float64)
    )
    return prediction


def rmse_from_sse(sum_squared_error: float, row_count: int) -> float:
    """由 SSE 和行数计算 RMSE。"""

    return float(np.sqrt(float(sum_squared_error) / int(row_count)))


def build_summary(
    per_well_df: pd.DataFrame,
    config: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """汇总逐井 SSE，并严格应用实验卡四条晋级门槛。"""

    if per_well_df.empty:
        raise ValueError("P2-S01 没有逐井结果")
    total_rows = int(per_well_df["hidden_rows"].sum())

    sse_columns = {
        "surface": "sse_surface",
        "negative": "sse_negative",
        "own_surface": "sse_own_surface",
        "carry": "sse_carry",
        "p2b00": "sse_p2b00",
    }
    micro_rmse = {
        name: rmse_from_sse(float(per_well_df[column].sum()), total_rows)
        for name, column in sse_columns.items()
    }

    fold_rows: list[dict[str, float | int]] = []
    for fold_id, fold_group in per_well_df.groupby("fold", sort=True):
        fold_hidden_rows = int(fold_group["hidden_rows"].sum())
        fold_row: dict[str, float | int] = {
            "fold": int(fold_id),
            "wells": int(len(fold_group)),
            "hidden_rows": fold_hidden_rows,
        }
        for name, column in sse_columns.items():
            fold_row[f"{name}_rmse"] = rmse_from_sse(
                float(fold_group[column].sum()),
                fold_hidden_rows,
            )
        fold_row["improvement_vs_carry_ft"] = float(
            fold_row["carry_rmse"] - fold_row["surface_rmse"]
        )
        fold_rows.append(fold_row)

    folds_better_than_carry = int(
        sum(float(row["surface_rmse"]) < float(row["carry_rmse"]) for row in fold_rows)
    )
    improvement_vs_carry = float(micro_rmse["carry"] - micro_rmse["surface"])
    improvement_vs_negative = float(micro_rmse["negative"] - micro_rmse["surface"])
    conditions = config["success_conditions"]
    checks = {
        "own_surface_oracle_pass": bool(
            micro_rmse["own_surface"]
            <= float(conditions["maximum_own_surface_oracle_micro_rmse_ft"])
        ),
        "carry_improvement_pass": bool(
            improvement_vs_carry >= float(conditions["minimum_improvement_vs_carry_ft"])
        ),
        "fold_consistency_pass": bool(
            folds_better_than_carry >= int(conditions["minimum_folds_better_than_carry"])
        ),
        "negative_control_pass": bool(
            improvement_vs_negative
            >= float(conditions["minimum_improvement_vs_permuted_surface_ft"])
        ),
    }

    return {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(per_well_df)),
        "hidden_rows": total_rows,
        "micro_rmse": micro_rmse,
        "macro_well_rmse": {
            "surface": float(per_well_df["surface_rmse"].mean()),
            "negative": float(per_well_df["negative_rmse"].mean()),
            "own_surface": float(per_well_df["own_surface_rmse"].mean()),
            "carry": float(per_well_df["carry_rmse"].mean()),
            "p2b00": float(per_well_df["p2b00_rmse"].mean()),
        },
        "p90_well_rmse": {
            "surface": float(per_well_df["surface_rmse"].quantile(0.9)),
            "carry": float(per_well_df["carry_rmse"].quantile(0.9)),
            "p2b00": float(per_well_df["p2b00_rmse"].quantile(0.9)),
        },
        "well_win_rate_vs_carry": float(
            per_well_df["surface_rmse"].lt(per_well_df["carry_rmse"]).mean()
        ),
        "well_win_rate_vs_p2b00": float(
            per_well_df["surface_rmse"].lt(per_well_df["p2b00_rmse"]).mean()
        ),
        "improvement_vs_carry_ft": improvement_vs_carry,
        "improvement_vs_permuted_surface_ft": improvement_vs_negative,
        "folds_better_than_carry": folds_better_than_carry,
        "folds": fold_rows,
        "checks": checks,
        "surface_path_supported": bool(all(checks.values())),
        "wall_seconds": float(wall_seconds),
    }


def build_distance_slices(per_well_df: pd.DataFrame) -> pd.DataFrame:
    """按合法最近 source 距离把井分四组，检查空间支撑是否解释误差。"""

    sliced = per_well_df.copy()
    unique_distances = int(sliced["median_nearest_source_distance"].nunique())
    number_of_bins = min(4, unique_distances)
    if number_of_bins < 2:
        sliced["distance_slice"] = "all"
    else:
        sliced["distance_slice"] = pd.qcut(
            sliced["median_nearest_source_distance"],
            q=number_of_bins,
            labels=[f"Q{number}" for number in range(1, number_of_bins + 1)],
            duplicates="drop",
        ).astype(str)

    rows: list[dict[str, Any]] = []
    for slice_name, group in sliced.groupby("distance_slice", sort=True):
        hidden_rows = int(group["hidden_rows"].sum())
        rows.append(
            {
                "distance_slice": str(slice_name),
                "wells": int(len(group)),
                "hidden_rows": hidden_rows,
                "median_nearest_source_distance": float(
                    group["median_nearest_source_distance"].median()
                ),
                "surface_rmse": rmse_from_sse(float(group["sse_surface"].sum()), hidden_rows),
                "carry_rmse": rmse_from_sse(float(group["sse_carry"].sum()), hidden_rows),
                "p2b00_rmse": rmse_from_sse(float(group["sse_p2b00"].sum()), hidden_rows),
            }
        )
    return pd.DataFrame(rows)


def load_baseline_predictions(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """读取冻结 P2B00 OOF，并建立井到行位置的只读索引。"""

    baseline_path = CLEAN_ROOT / str(config["baseline_predictions"])
    baseline = pd.read_parquet(baseline_path)
    required = {"well_id", "fold", "row_index", "target_tvt", "carry_tvt", "pred_tvt"}
    if not required.issubset(baseline.columns):
        raise ValueError("P2B00 predictions 缺少必要列")
    baseline["well_id"] = baseline["well_id"].astype(str)
    if len(baseline) != int(config["expected_hidden_rows"]):
        raise ValueError("P2B00 predictions 行数不等于冻结评价行")
    positions = baseline.groupby("well_id", sort=False).indices
    return baseline, positions


def process_one_well(
    well_id: str,
    outer_fold: int,
    raw_train_dir: Path,
    real_source_table: pd.DataFrame,
    negative_source_table: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    config: dict[str, Any],
    artifact_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    """生成一口验证井的合法路径，再在隔离步骤中附加 oracle 指标。"""

    per_well_path = artifact_dir / "per_well" / f"{well_id}.parquet"
    runtime_path = artifact_dir / "per_well_runtime" / f"{well_id}.json"
    if per_well_path.is_file() and runtime_path.is_file():
        cached_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if cached_runtime.get("experiment_fingerprint") == fingerprint:
            return dict(cached_runtime["metrics"])

    started = time.perf_counter()
    raw_path = horizontal_path(raw_train_dir, well_id)

    # 第一次物理读取只取合法目标列；此时 TVT 和 EGFDU 没有进入内存中的 target 表。
    legal_target = pd.read_csv(raw_path, usecols=list(LEGAL_TARGET_COLUMNS))
    legal_path = build_legal_surface_path(
        target_horizontal_df=legal_target,
        source_table=real_source_table,
        nearest_source_wells=int(config["nearest_source_wells"]),
        control_step_horizontal_ft=float(config["control_step_horizontal_ft"]),
        distance_weight_epsilon=float(config["distance_weight_epsilon"]),
    )
    negative_path = build_legal_surface_path(
        target_horizontal_df=legal_target,
        source_table=negative_source_table,
        nearest_source_wells=int(config["nearest_source_wells"]),
        control_step_horizontal_ft=float(config["control_step_horizontal_ft"]),
        distance_weight_epsilon=float(config["distance_weight_epsilon"]),
    )

    if not np.array_equal(legal_path["row_index"], negative_path["row_index"]):
        raise ValueError(f"{well_id} 主路径与负对照行键不一致")

    hidden_rows = legal_path["row_index"].to_numpy(dtype=np.int64)
    expected_baseline = baseline_rows.sort_values("row_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(expected_baseline["row_index"].to_numpy(dtype=np.int64), hidden_rows):
        raise ValueError(f"{well_id} P2B00 与 surface 路径 row_index 不一致")
    if not expected_baseline["fold"].astype(int).eq(int(outer_fold)).all():
        raise ValueError(f"{well_id} P2B00 fold 与新折表不一致")

    # 合法路径已经完成并保存后，才第二次读取 oracle 所需的目标 surface 和 TVT。
    oracle_columns = pd.read_csv(raw_path, usecols=["TVT", str(config["surface_name"])])
    truth = oracle_columns.loc[hidden_rows, "TVT"].to_numpy(dtype=np.float64)
    own_surface_frame = pd.DataFrame(
        {
            "Z": legal_target["Z"],
            "TVT_input": legal_target["TVT_input"],
            str(config["surface_name"]): oracle_columns[str(config["surface_name"])],
        }
    )
    own_surface_prediction = build_own_surface_oracle_path(
        own_surface_frame,
        str(config["surface_name"]),
    )

    baseline_truth = expected_baseline["target_tvt"].to_numpy(dtype=np.float64)
    if not np.allclose(truth, baseline_truth, atol=1e-9, rtol=0.0):
        raise ValueError(f"{well_id} 原始 TVT 与冻结 P2B00 target 不一致")

    surface_prediction = legal_path["surface_tvt"].to_numpy(dtype=np.float64)
    negative_prediction = negative_path["surface_tvt"].to_numpy(dtype=np.float64)
    carry_prediction = expected_baseline["carry_tvt"].to_numpy(dtype=np.float64)
    p2_prediction = expected_baseline["pred_tvt"].to_numpy(dtype=np.float64)

    error_surface = surface_prediction - truth
    error_negative = negative_prediction - truth
    error_own_surface = own_surface_prediction - truth
    error_carry = carry_prediction - truth
    error_p2 = p2_prediction - truth

    if np.std(error_surface) > 0.0 and np.std(error_p2) > 0.0:
        residual_correlation = float(np.corrcoef(error_surface, error_p2)[0, 1])
    else:
        residual_correlation = float("nan")

    number_of_hidden_rows = int(len(truth))
    sse_surface = float(np.sum(np.square(error_surface)))
    sse_negative = float(np.sum(np.square(error_negative)))
    sse_own_surface = float(np.sum(np.square(error_own_surface)))
    sse_carry = float(np.sum(np.square(error_carry)))
    sse_p2 = float(np.sum(np.square(error_p2)))

    legal_output = legal_path.copy()
    legal_output.insert(0, "well_id", str(well_id))
    legal_output.insert(1, "fold", int(outer_fold))
    legal_output["negative_surface_tvt"] = negative_prediction
    write_parquet_atomic(per_well_path, legal_output)

    metrics = {
        "well_id": str(well_id),
        "fold": int(outer_fold),
        "hidden_rows": number_of_hidden_rows,
        "sse_surface": sse_surface,
        "sse_negative": sse_negative,
        "sse_own_surface": sse_own_surface,
        "sse_carry": sse_carry,
        "sse_p2b00": sse_p2,
        "surface_rmse": rmse_from_sse(sse_surface, number_of_hidden_rows),
        "negative_rmse": rmse_from_sse(sse_negative, number_of_hidden_rows),
        "own_surface_rmse": rmse_from_sse(sse_own_surface, number_of_hidden_rows),
        "carry_rmse": rmse_from_sse(sse_carry, number_of_hidden_rows),
        "p2b00_rmse": rmse_from_sse(sse_p2, number_of_hidden_rows),
        "surface_p2b00_residual_correlation": residual_correlation,
        "median_nearest_source_distance": float(
            legal_path["nearest_source_distance"].median()
        ),
        "median_kth_source_distance": float(legal_path["kth_source_distance"].median()),
        "median_local_plane_rmse": float(legal_path["local_plane_rmse"].median()),
        "prefix_calibration_rmse": float(legal_path["prefix_calibration_rmse"].iloc[0]),
        "fallback_fraction": float(legal_path["used_fallback"].mean()),
    }
    write_json_atomic(
        runtime_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "well_id": str(well_id),
            "fold": int(outer_fold),
            "real_source_wells": int(len(real_source_table)),
            "validation_fold_excluded": True,
            "elapsed_seconds": float(time.perf_counter() - started),
            "metrics": metrics,
        },
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke/all 和配置路径。"""

    parser = argparse.ArgumentParser(description="运行 P2-S01 严格外折地层面路径诊断")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """构建 source、逐井生成路径、汇总预注册证据并保存产物。"""

    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    validate_external_inputs(config)
    registry = load_registry(config)
    fingerprint = experiment_fingerprint(config)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifact_dir / "config.json", config)

    raw_train_dir = (CLEAN_ROOT / str(config["raw_train_dir"])).resolve()
    representatives = load_or_build_source_representatives(
        registry=registry,
        raw_train_dir=raw_train_dir,
        config=config,
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
    )

    real_source_by_fold: dict[int, pd.DataFrame] = {}
    negative_source_by_fold: dict[int, pd.DataFrame] = {}
    source_lineage: dict[str, Any] = {}
    for fold_id in range(5):
        real_sources = build_fold_source_table(registry, representatives, fold_id)
        negative_sources = permute_source_surfaces(
            real_sources,
            seed=int(config["negative_control_seed"]) + fold_id,
        )
        real_source_by_fold[fold_id] = real_sources
        negative_source_by_fold[fold_id] = negative_sources
        valid_ids = set(registry.loc[registry["fold"].eq(fold_id), "well_id"].astype(str))
        source_ids = set(real_sources["well_id"].astype(str))
        source_lineage[str(fold_id)] = {
            "outer_fold": fold_id,
            "source_wells": int(len(real_sources)),
            "source_table_hash": stable_frame_hash(real_sources),
            "negative_source_table_hash": stable_frame_hash(negative_sources),
            "outer_valid_intersection_count": int(len(valid_ids & source_ids)),
            "validation_fold_excluded": bool(valid_ids.isdisjoint(source_ids)),
        }
    if not all(item["validation_fold_excluded"] for item in source_lineage.values()):
        raise ValueError("P2-S01 source lineage 未能排除完整验证折")
    write_json_atomic(
        artifact_dir / "lineage.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "legal_target_columns": list(LEGAL_TARGET_COLUMNS),
            "target_tvt_or_surface_in_legal_path": False,
            "folds": source_lineage,
        },
    )

    baseline, baseline_positions = load_baseline_predictions(config)
    if args.mode == "smoke":
        selected_registry = registry.head(int(config["smoke_wells"])).copy()
    else:
        selected_registry = registry.copy()

    print(
        f"P2-S01 {args.mode}：{len(selected_registry)} 口井，"
        f"每折约 {len(real_source_by_fold[0])} 口合法 source，"
        f"控制点间隔 {config['control_step_horizontal_ft']} ft，输出 {artifact_dir}",
        flush=True,
    )
    started = time.perf_counter()
    metrics_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def submit_one(row: Any) -> dict[str, Any]:
        well_id = str(row.well_id)
        fold_id = int(row.fold)
        positions = baseline_positions.get(well_id)
        if positions is None:
            raise ValueError(f"P2B00 predictions 缺少井 {well_id}")
        baseline_rows = baseline.iloc[positions].copy()
        return process_one_well(
            well_id=well_id,
            outer_fold=fold_id,
            raw_train_dir=raw_train_dir,
            real_source_table=real_source_by_fold[fold_id],
            negative_source_table=negative_source_by_fold[fold_id],
            baseline_rows=baseline_rows,
            config=config,
            artifact_dir=artifact_dir,
            fingerprint=fingerprint,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(config["workers"]))) as executor:
        future_by_well = {
            executor.submit(submit_one, row): str(row.well_id)
            for row in selected_registry.itertuples(index=False)
        }
        for completed_number, future in enumerate(as_completed(future_by_well), start=1):
            well_id = future_by_well[future]
            try:
                metrics_rows.append(future.result())
            except Exception as error:  # noqa: BLE001 - 逐井记录后统一失败，便于续跑定位。
                errors.append({"well_id": well_id, "error": repr(error)})
            if completed_number % 10 == 0 or completed_number == len(selected_registry):
                print(
                    f"P2-S01 路径：{completed_number}/{len(selected_registry)}，失败 {len(errors)}",
                    flush=True,
                )

    wall_seconds = float(time.perf_counter() - started)
    if errors:
        write_json_atomic(
            artifact_dir / f"errors_{args.mode}.json",
            {"experiment_fingerprint": fingerprint, "errors": errors},
        )
        raise RuntimeError(f"P2-S01 有 {len(errors)} 口井失败，详见 errors_{args.mode}.json")

    per_well = pd.DataFrame(metrics_rows).sort_values("well_id", kind="stable").reset_index(drop=True)
    summary = build_summary(per_well, config, wall_seconds)
    summary["mode"] = str(args.mode)
    summary["experiment_fingerprint"] = fingerprint
    output_suffix = "smoke" if args.mode == "smoke" else "all"
    per_well.to_csv(artifact_dir / f"per_well_{output_suffix}.csv", index=False)
    build_distance_slices(per_well).to_csv(
        artifact_dir / f"distance_slices_{output_suffix}.csv",
        index=False,
    )
    write_json_atomic(artifact_dir / f"summary_{output_suffix}.json", summary)
    write_json_atomic(
        artifact_dir / f"runtime_{output_suffix}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "mode": str(args.mode),
            "wells": int(len(per_well)),
            "wall_seconds": wall_seconds,
            "workers": int(config["workers"]),
        },
    )

    if args.mode == "all":
        per_well.to_csv(artifact_dir / "per_well.csv", index=False)
        build_distance_slices(per_well).to_csv(artifact_dir / "distance_slices.csv", index=False)
        write_json_atomic(artifact_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
