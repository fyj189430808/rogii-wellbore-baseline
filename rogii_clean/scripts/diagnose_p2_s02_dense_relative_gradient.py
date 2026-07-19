"""运行 P2-S02 dense EGFDU 相对梯度路径诊断。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p2_s01_outer_fold_surface import (  # noqa: E402
    build_distance_slices,
    build_own_surface_oracle_path,
    file_sha256,
    horizontal_path,
    load_baseline_predictions,
    load_registry,
    rmse_from_sse,
    stable_frame_hash,
    stable_json_hash,
    validate_external_inputs,
    write_json_atomic,
    write_parquet_atomic,
)
from src.p2_s01_outer_fold_surface import LEGAL_TARGET_COLUMNS  # noqa: E402
from src.p2_s02_dense_relative_gradient import (  # noqa: E402
    DenseSurfaceGradientIndex,
    build_dense_source_profile,
    build_legal_relative_gradient_path,
    compute_outer_train_gradient_cap,
    reverse_source_profile_gradients,
)


EXPERIMENT_ID = "P2_S02_dense_relative_gradient_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_s02_dense_relative_gradient_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "all")


def validate_config(config: dict[str, Any]) -> None:
    """确认所有会改变路径的参数仍与实验卡一致。"""

    expected_values = {
        "experiment_id": EXPERIMENT_ID,
        "surface_name": "EGFDU",
        "source_control_step_horizontal_ft": 50.0,
        "target_control_step_horizontal_ft": 50.0,
        "nearest_distinct_source_wells": 10,
        "distance_weight_epsilon": 0.001,
        "gradient_cap_quantile": 0.99,
    }
    for key, expected_value in expected_values.items():
        if config.get(key) != expected_value:
            raise ValueError(f"P2-S02 配置 {key} 不等于冻结值 {expected_value}")
    if config.get("supported_modes") != list(SUPPORTED_MODES):
        raise ValueError("P2-S02 supported_modes 被修改")


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """组合配置、代码和冻结输入的完整指纹。"""

    return stable_json_hash(
        {
            "config": config,
            "core_sha256": file_sha256(
                CLEAN_ROOT / "src" / "p2_s02_dense_relative_gradient.py"
            ),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "fold_registry_sha256": config["fold_registry_sha256"],
            "baseline_predictions_sha256": config["baseline_predictions_sha256"],
            "surface_lineage_summary_sha256": config["surface_lineage_summary_sha256"],
        }
    )


def build_all_dense_source_points(
    registry: pd.DataFrame,
    raw_train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """读取全部训练井，按每井 50 ft 建立 dense EGFDU 点表。"""

    well_ids = registry["well_id"].astype(str).tolist()

    def load_one(well_id: str) -> pd.DataFrame:
        horizontal = pd.read_csv(
            horizontal_path(raw_train_dir, well_id),
            usecols=["X", "Y", str(config["surface_name"])],
        )
        return build_dense_source_profile(
            well_id=well_id,
            horizontal_df=horizontal,
            surface_name=str(config["surface_name"]),
            control_step_horizontal_ft=float(config["source_control_step_horizontal_ft"]),
        )

    profiles: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, int(config["workers"]))) as executor:
        future_by_well = {executor.submit(load_one, well_id): well_id for well_id in well_ids}
        for completed_number, future in enumerate(as_completed(future_by_well), start=1):
            profiles.append(future.result())
            if completed_number % 100 == 0 or completed_number == len(well_ids):
                print(f"dense source：{completed_number}/{len(well_ids)} 口井", flush=True)

    dense_points = pd.concat(profiles, ignore_index=True)
    dense_points = dense_points.sort_values(
        ["source_well_id", "source_distance_ft", "source_row_index"],
        kind="stable",
    ).reset_index(drop=True)
    if dense_points["source_well_id"].nunique() != len(registry):
        raise ValueError("dense source 没有覆盖全部 773 口井")
    return dense_points


def load_or_build_dense_source_points(
    registry: pd.DataFrame,
    raw_train_dir: Path,
    config: dict[str, Any],
    artifact_dir: Path,
    fingerprint: str,
) -> pd.DataFrame:
    """命中完整指纹时复用 dense source，否则重新构建。"""

    cache_path = artifact_dir / "dense_source_points.parquet"
    metadata_path = artifact_dir / "dense_source_points_meta.json"
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("experiment_fingerprint") == fingerprint:
            cached = pd.read_parquet(cache_path)
            if cached["source_well_id"].astype(str).nunique() == len(registry):
                print(f"命中 dense source 缓存：{cache_path}", flush=True)
                return cached

    print("未命中 dense source 缓存，开始读取 773 口井。", flush=True)
    dense_points = build_all_dense_source_points(registry, raw_train_dir, config)
    write_parquet_atomic(cache_path, dense_points)
    write_json_atomic(
        metadata_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "source_wells": int(dense_points["source_well_id"].nunique()),
            "source_points": int(len(dense_points)),
            "cache_sha256": file_sha256(cache_path),
        },
    )
    return dense_points


def build_fold_source_points(
    registry: pd.DataFrame,
    dense_points: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    """从 dense 总表中完整排除 outer-valid 井。"""

    outer_valid_ids = set(
        registry.loc[registry["fold"].astype(int).eq(int(outer_fold)), "well_id"].astype(str)
    )
    source_mask = ~dense_points["source_well_id"].astype(str).isin(outer_valid_ids)
    source_points = dense_points.loc[source_mask].reset_index(drop=True)
    actual_source_ids = set(source_points["source_well_id"].astype(str))
    if not actual_source_ids.isdisjoint(outer_valid_ids):
        raise ValueError("P2-S02 dense source 混入 outer-valid 井")
    expected_source_ids = set(
        registry.loc[registry["fold"].astype(int).ne(int(outer_fold)), "well_id"].astype(str)
    )
    if actual_source_ids != expected_source_ids:
        raise ValueError("P2-S02 dense source 井集合不等于另外四折")
    return source_points


def build_summary(
    per_well_df: pd.DataFrame,
    config: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """从逐井 SSE 计算 pooled 指标并应用四条固定门槛。"""

    if per_well_df.empty:
        raise ValueError("P2-S02 没有逐井指标")
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
        "reversed_gradient_control_pass": bool(
            improvement_vs_negative
            >= float(conditions["minimum_improvement_vs_reversed_gradient_ft"])
        ),
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(per_well_df)),
        "hidden_rows": total_rows,
        "micro_rmse": micro_rmse,
        "macro_well_rmse": {
            name: float(per_well_df[column.replace("sse_", "") + "_rmse"].mean())
            if name != "p2b00"
            else float(per_well_df["p2b00_rmse"].mean())
            for name, column in sse_columns.items()
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
        "improvement_vs_reversed_gradient_ft": improvement_vs_negative,
        "folds_better_than_carry": folds_better_than_carry,
        "folds": fold_rows,
        "checks": checks,
        "surface_path_supported": bool(all(checks.values())),
        "wall_seconds": float(wall_seconds),
    }


def process_one_well(
    well_id: str,
    outer_fold: int,
    raw_train_dir: Path,
    real_index: DenseSurfaceGradientIndex,
    negative_index: DenseSurfaceGradientIndex,
    gradient_cap: float,
    baseline_rows: pd.DataFrame,
    config: dict[str, Any],
    artifact_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    """先生成合法梯度路径，再隔离读取目标真值和自身 surface。"""

    path_output = artifact_dir / "per_well" / f"{well_id}.parquet"
    runtime_output = artifact_dir / "per_well_runtime" / f"{well_id}.json"
    if path_output.is_file() and runtime_output.is_file():
        cached = json.loads(runtime_output.read_text(encoding="utf-8"))
        if cached.get("experiment_fingerprint") == fingerprint:
            return dict(cached["metrics"])

    started = time.perf_counter()
    raw_path = horizontal_path(raw_train_dir, well_id)
    legal_target = pd.read_csv(raw_path, usecols=list(LEGAL_TARGET_COLUMNS))
    real_path = build_legal_relative_gradient_path(
        target_horizontal_df=legal_target,
        dense_surface_index=real_index,
        nearest_distinct_source_wells=int(config["nearest_distinct_source_wells"]),
        target_control_step_horizontal_ft=float(config["target_control_step_horizontal_ft"]),
        distance_weight_epsilon=float(config["distance_weight_epsilon"]),
        gradient_magnitude_cap=float(gradient_cap),
    )
    negative_path = build_legal_relative_gradient_path(
        target_horizontal_df=legal_target,
        dense_surface_index=negative_index,
        nearest_distinct_source_wells=int(config["nearest_distinct_source_wells"]),
        target_control_step_horizontal_ft=float(config["target_control_step_horizontal_ft"]),
        distance_weight_epsilon=float(config["distance_weight_epsilon"]),
        gradient_magnitude_cap=float(gradient_cap),
    )
    if not np.array_equal(real_path["row_index"], negative_path["row_index"]):
        raise ValueError(f"{well_id} 主路径与梯度反转负对照行键不一致")

    hidden_rows = real_path["row_index"].to_numpy(dtype=np.int64)
    expected_baseline = baseline_rows.sort_values("row_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(expected_baseline["row_index"].to_numpy(dtype=np.int64), hidden_rows):
        raise ValueError(f"{well_id} P2B00 与 P2-S02 行键不一致")
    if not expected_baseline["fold"].astype(int).eq(int(outer_fold)).all():
        raise ValueError(f"{well_id} P2B00 fold 与 P2-S02 不一致")

    oracle = pd.read_csv(raw_path, usecols=["TVT", str(config["surface_name"])])
    truth = oracle.loc[hidden_rows, "TVT"].to_numpy(dtype=np.float64)
    own_surface_frame = pd.DataFrame(
        {
            "Z": legal_target["Z"],
            "TVT_input": legal_target["TVT_input"],
            str(config["surface_name"]): oracle[str(config["surface_name"])],
        }
    )
    own_surface_prediction = build_own_surface_oracle_path(
        own_surface_frame,
        str(config["surface_name"]),
    )
    if not np.allclose(
        truth,
        expected_baseline["target_tvt"].to_numpy(dtype=np.float64),
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError(f"{well_id} 原始 TVT 与 P2B00 target 不一致")

    surface_prediction = real_path["surface_tvt"].to_numpy(dtype=np.float64)
    negative_prediction = negative_path["surface_tvt"].to_numpy(dtype=np.float64)
    carry_prediction = expected_baseline["carry_tvt"].to_numpy(dtype=np.float64)
    p2_prediction = expected_baseline["pred_tvt"].to_numpy(dtype=np.float64)
    errors = {
        "surface": surface_prediction - truth,
        "negative": negative_prediction - truth,
        "own_surface": own_surface_prediction - truth,
        "carry": carry_prediction - truth,
        "p2b00": p2_prediction - truth,
    }
    sse = {name: float(np.sum(np.square(values))) for name, values in errors.items()}
    number_of_hidden_rows = int(len(truth))
    if np.std(errors["surface"]) > 0.0 and np.std(errors["p2b00"]) > 0.0:
        residual_correlation = float(np.corrcoef(errors["surface"], errors["p2b00"])[0, 1])
    else:
        residual_correlation = float("nan")

    legal_output = real_path.copy()
    legal_output.insert(0, "well_id", str(well_id))
    legal_output.insert(1, "fold", int(outer_fold))
    legal_output["negative_surface_tvt"] = negative_prediction
    write_parquet_atomic(path_output, legal_output)

    metrics = {
        "well_id": str(well_id),
        "fold": int(outer_fold),
        "hidden_rows": number_of_hidden_rows,
        "sse_surface": sse["surface"],
        "sse_negative": sse["negative"],
        "sse_own_surface": sse["own_surface"],
        "sse_carry": sse["carry"],
        "sse_p2b00": sse["p2b00"],
        "surface_rmse": rmse_from_sse(sse["surface"], number_of_hidden_rows),
        "negative_rmse": rmse_from_sse(sse["negative"], number_of_hidden_rows),
        "own_surface_rmse": rmse_from_sse(sse["own_surface"], number_of_hidden_rows),
        "carry_rmse": rmse_from_sse(sse["carry"], number_of_hidden_rows),
        "p2b00_rmse": rmse_from_sse(sse["p2b00"], number_of_hidden_rows),
        "surface_p2b00_residual_correlation": residual_correlation,
        "median_nearest_source_distance": float(real_path["nearest_source_distance"].median()),
        "median_kth_source_distance": float(real_path["kth_source_distance"].median()),
        "median_local_plane_rmse": float(real_path["local_plane_rmse"].median()),
        "prefix_backtest_rmse": float(real_path["prefix_backtest_rmse"].iloc[0]),
        "gradient_cap": float(gradient_cap),
        "gradient_clipped_fraction": float(real_path["gradient_was_clipped"].mean()),
        "outside_support_fraction": float(real_path["outside_support"].mean()),
        "median_neighbor_change_fraction": float(real_path["neighbor_change_fraction"].median()),
    }
    write_json_atomic(
        runtime_output,
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "well_id": str(well_id),
            "fold": int(outer_fold),
            "gradient_cap": float(gradient_cap),
            "validation_fold_excluded": True,
            "elapsed_seconds": float(time.perf_counter() - started),
            "metrics": metrics,
        },
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke/all、配置和独立产物目录。"""

    parser = argparse.ArgumentParser(description="运行 P2-S02 dense EGFDU 相对梯度诊断")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """构建 dense source、逐井积分路径并汇总预注册证据。"""

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

    dense_points = load_or_build_dense_source_points(
        registry=registry,
        raw_train_dir=raw_train_dir,
        config=config,
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
    )

    real_index_by_fold: dict[int, DenseSurfaceGradientIndex] = {}
    negative_index_by_fold: dict[int, DenseSurfaceGradientIndex] = {}
    gradient_cap_by_fold: dict[int, float] = {}
    fold_lineage: dict[str, Any] = {}
    for fold_id in range(5):
        fold_points = build_fold_source_points(registry, dense_points, fold_id)
        gradient_cap = compute_outer_train_gradient_cap(
            fold_points,
            quantile=float(config["gradient_cap_quantile"]),
        )
        negative_points = reverse_source_profile_gradients(fold_points)
        real_index_by_fold[fold_id] = DenseSurfaceGradientIndex(fold_points)
        negative_index_by_fold[fold_id] = DenseSurfaceGradientIndex(negative_points)
        gradient_cap_by_fold[fold_id] = gradient_cap
        valid_ids = set(registry.loc[registry["fold"].eq(fold_id), "well_id"].astype(str))
        source_ids = set(fold_points["source_well_id"].astype(str))
        fold_lineage[str(fold_id)] = {
            "outer_fold": fold_id,
            "source_wells": int(len(source_ids)),
            "source_points": int(len(fold_points)),
            "source_points_hash": stable_frame_hash(fold_points),
            "gradient_cap": float(gradient_cap),
            "outer_valid_intersection_count": int(len(valid_ids & source_ids)),
            "validation_fold_excluded": bool(valid_ids.isdisjoint(source_ids)),
        }
    if not all(item["validation_fold_excluded"] for item in fold_lineage.values()):
        raise ValueError("P2-S02 lineage 没有完整排除验证折")
    write_json_atomic(
        artifact_dir / "lineage.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "legal_target_columns": list(LEGAL_TARGET_COLUMNS),
            "target_tvt_or_surface_in_legal_path": False,
            "folds": fold_lineage,
        },
    )

    baseline, baseline_positions = load_baseline_predictions(config)
    selected_registry = (
        registry.head(int(config["smoke_wells"])).copy()
        if args.mode == "smoke"
        else registry.copy()
    )
    print(
        f"P2-S02 {args.mode}：{len(selected_registry)} 口井，dense 点 {len(dense_points)}，"
        f"每折梯度上限 {[round(gradient_cap_by_fold[f], 6) for f in range(5)]}，"
        f"输出 {artifact_dir}",
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
        return process_one_well(
            well_id=well_id,
            outer_fold=fold_id,
            raw_train_dir=raw_train_dir,
            real_index=real_index_by_fold[fold_id],
            negative_index=negative_index_by_fold[fold_id],
            gradient_cap=gradient_cap_by_fold[fold_id],
            baseline_rows=baseline.iloc[positions].copy(),
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
            except Exception as error:  # noqa: BLE001 - 保存逐井错误后统一中止。
                errors.append({"well_id": well_id, "error": repr(error)})
            if completed_number % 10 == 0 or completed_number == len(selected_registry):
                print(
                    f"P2-S02 路径：{completed_number}/{len(selected_registry)}，失败 {len(errors)}",
                    flush=True,
                )

    wall_seconds = float(time.perf_counter() - started)
    if errors:
        write_json_atomic(
            artifact_dir / f"errors_{args.mode}.json",
            {"experiment_fingerprint": fingerprint, "errors": errors},
        )
        raise RuntimeError(f"P2-S02 有 {len(errors)} 口井失败")

    per_well = pd.DataFrame(metrics_rows).sort_values("well_id", kind="stable").reset_index(drop=True)
    summary = build_summary(per_well, config, wall_seconds)
    summary["mode"] = str(args.mode)
    summary["experiment_fingerprint"] = fingerprint
    summary["gradient_cap_by_fold"] = {
        str(fold_id): float(value) for fold_id, value in gradient_cap_by_fold.items()
    }
    suffix = "smoke" if args.mode == "smoke" else "all"
    per_well.to_csv(artifact_dir / f"per_well_{suffix}.csv", index=False)
    build_distance_slices(per_well).to_csv(
        artifact_dir / f"distance_slices_{suffix}.csv",
        index=False,
    )
    write_json_atomic(artifact_dir / f"summary_{suffix}.json", summary)
    write_json_atomic(
        artifact_dir / f"runtime_{suffix}.json",
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
