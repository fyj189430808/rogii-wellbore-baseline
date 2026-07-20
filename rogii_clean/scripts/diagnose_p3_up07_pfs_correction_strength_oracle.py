"""诊断 PFS 修正方向的逐井最佳强度及其合法可辨识性。

阶段 A 只读取测试时合法的路径、PFS 运行摘要和无标签井登记信息，并先把
井级合法量落盘。阶段 B 才读取 657 口开发井的隐藏真值做 oracle 诊断。
本脚本不训练模型、不读取 shadow 真值，也不更新实验总表或当前状态。
"""

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


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pfs02_conservative_postblend import (  # noqa: E402
    development_registry,
    file_sha256,
    load_base_legal_rows,
    load_pfs_legal_rows,
    load_targets,
    read_json,
    resolve_path,
    shadow_ids,
    write_json,
)
from scripts.run_p3_up03_up01_plus_pfs_correction import load_up01_legal  # noqa: E402
from src.p3_up07_pfs_correction_strength_oracle import (  # noqa: E402
    grid_oracle,
    linear_slope_per_1000ft,
    make_alpha_grid,
    rmse_curve_from_sufficient_statistics,
)


EXPERIMENT_ID = "P3_UP07_pfs_correction_strength_oracle_v1"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
ALPHA_GRID = make_alpha_grid(-0.5, 1.0, 0.05)
UP03_ALPHA = 0.25
POTENTIAL_THRESHOLD_FT = 0.30
IDENTIFIABILITY_MIN_ABS_RHO = 0.15
IDENTIFIABILITY_MIN_SAME_SIGN_FOLDS = 4
EXPECTED_DEVELOPMENT_WELLS = 657
EXPECTED_DEVELOPMENT_ROWS = 3_211_872
PFS_SEEDS = 128

UP01_PATH = CLEAN_ROOT / "artifacts/P3_UP01_robust_u_projection_v1/legal_candidates.parquet"
UP01_MANIFEST_PATH = CLEAN_ROOT / "artifacts/P3_UP01_robust_u_projection_v1/legal_generation_manifest.json"
P3B00_PATH = CLEAN_ROOT / "artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet"
UP03_CONFIG_PATH = CLEAN_ROOT / "configs/p3_up03_up01_plus_pfs_correction_v1.json"
FOLD_PATH = CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv"
SHADOW_PATH = CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv"
SHADOW_METADATA_PATH = CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/metadata.csv"
PFS_CACHE_DIR = CLEAN_ROOT / "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_cache"
PFS_RUNTIME_DIR = CLEAN_ROOT / "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_runtime"
PFS_RUNTIME_TABLE = CLEAN_ROOT / "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/per_well_runtime_all.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def rmse_from_sse(row_count: int, sse: float) -> float:
    return float(np.sqrt(max(float(sse), 0.0) / float(row_count)))


def legal_configuration(up03_config: dict[str, Any]) -> dict[str, Any]:
    """冻结本次只读诊断合同；该对象会在读取真值前先落盘。"""

    return {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "model_training": False,
        "base_path": "UP01 degree2_blend50",
        "direction_formula": "d = pfs_lag1000_abs_tvt - p3b00_pred_tvt",
        "candidate_formula": "prediction(alpha) = UP01_degree2_blend50 + alpha * d",
        "up03_fixed_alpha": UP03_ALPHA,
        "alpha_grid": [float(value) for value in ALPHA_GRID],
        "potential_threshold_ft": POTENTIAL_THRESHOLD_FT,
        "identifiability_screen": {
            "minimum_absolute_overall_spearman": IDENTIFIABILITY_MIN_ABS_RHO,
            "minimum_same_sign_folds": IDENTIFIABILITY_MIN_SAME_SIGN_FOLDS,
            "note": "This is only a descriptive screen; no predictive model is trained.",
        },
        "fold_registry": str(FOLD_PATH.relative_to(CLEAN_ROOT)),
        "shadow_registry": str(SHADOW_PATH.relative_to(CLEAN_ROOT)),
        "source_up01_legal_candidates": str(UP01_PATH.relative_to(CLEAN_ROOT)),
        "source_up01_sha256": up03_config["source_up01_sha256"],
        "source_p3b00_predictions": str(P3B00_PATH.relative_to(CLEAN_ROOT)),
        "source_p3b00_sha256": up03_config["source_predictions_sha256"],
        "source_pfs_cache_dir": str(PFS_CACHE_DIR.relative_to(CLEAN_ROOT)),
        "source_pfs_runtime_dir": str(PFS_RUNTIME_DIR.relative_to(CLEAN_ROOT)),
        "pfs_fingerprint": up03_config["pfs_fingerprint"],
        "development_wells": EXPECTED_DEVELOPMENT_WELLS,
        "development_rows": EXPECTED_DEVELOPMENT_ROWS,
        "shadow_target_access": False,
        "registry_update": False,
        "current_state_update": False,
    }


def load_runtime_features(registry: pd.DataFrame, expected_fingerprint: str) -> pd.DataFrame:
    """读取已生成的 PFS 逐井运行摘要，并验证其仍是合法缓存。"""

    runtime = pd.read_csv(PFS_RUNTIME_TABLE, dtype={"well_id": str})
    runtime = runtime.loc[runtime["well_id"].isin(set(registry["well_id"]))].copy()
    if len(runtime) != EXPECTED_DEVELOPMENT_WELLS or runtime["well_id"].nunique() != EXPECTED_DEVELOPMENT_WELLS:
        raise RuntimeError("PFS runtime 没有完整覆盖 657 口开发井")
    if runtime["hidden_tvt_read"].astype(bool).any():
        raise RuntimeError("PFS runtime 表明读取过隐藏 TVT")
    if set(runtime["experiment_fingerprint"].astype(str)) != {expected_fingerprint}:
        raise RuntimeError("PFS runtime 指纹不匹配")
    runtime["resample_per_seed_per_hidden_row"] = (
        runtime["resample_count"].astype(float)
        / (PFS_SEEDS * runtime["hidden_rows"].astype(float))
    )
    runtime["history_reordered_per_seed_per_hidden_row"] = (
        runtime["history_rows_reordered_total"].astype(float)
        / (PFS_SEEDS * runtime["hidden_rows"].astype(float))
    )
    for lag in (250, 500, 1000):
        runtime[f"lag{lag}_smooth_fraction"] = (
            runtime[f"lag{lag}_smooth_rows"].astype(float)
            / (PFS_SEEDS * runtime["hidden_rows"].astype(float))
        )
        runtime[f"lag{lag}_fallback_fraction"] = (
            runtime[f"lag{lag}_fallback_rows"].astype(float)
            / (PFS_SEEDS * runtime["hidden_rows"].astype(float))
        )
    keep = [
        "well_id",
        "resample_count",
        "resample_per_seed_per_hidden_row",
        "history_rows_reordered_total",
        "history_reordered_per_seed_per_hidden_row",
        "max_active_rows",
        "lag250_smooth_fraction",
        "lag250_fallback_fraction",
        "lag500_smooth_fraction",
        "lag500_fallback_fraction",
        "lag1000_smooth_fraction",
        "lag1000_fallback_fraction",
    ]
    return runtime[keep]


def build_legal_stage(output: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成逐行合法路径和逐井合法摘要；绝不读取 target_tvt。"""

    excluded = shadow_ids(SHADOW_PATH)
    registry = development_registry(FOLD_PATH, SHADOW_PATH)
    if len(registry) != EXPECTED_DEVELOPMENT_WELLS or int(registry["hidden_rows"].sum()) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError("开发井数量或隐藏行数偏离冻结合同")
    if file_sha256(UP01_PATH) != config["source_up01_sha256"]:
        raise RuntimeError("UP01 合法路径哈希不匹配")
    if file_sha256(P3B00_PATH) != config["source_p3b00_sha256"]:
        raise RuntimeError("P3B00 OOF 哈希不匹配")
    up01_manifest = read_json(UP01_MANIFEST_PATH)
    if up01_manifest.get("hidden_target_read") is not False or int(up01_manifest.get("shadow_overlap_wells", -1)) != 0:
        raise RuntimeError("UP01 合法路径血缘不合格")

    folds = [0, 1, 2, 3, 4]
    up01 = load_up01_legal(UP01_PATH, folds)
    p3b00 = load_base_legal_rows(P3B00_PATH, folds, excluded)
    pfs = load_pfs_legal_rows(
        registry,
        folds,
        PFS_CACHE_DIR,
        PFS_RUNTIME_DIR,
        str(config["pfs_fingerprint"]),
    )
    keys = ["well_id", "fold", "row_index"]
    legal_rows = up01[keys + ["md", "p2_pred_tvt", "degree2_blend50_pred_tvt"]].merge(
        p3b00[keys + ["p3b00_pred_tvt"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    ).merge(pfs, on=keys, how="inner", validate="one_to_one")
    if len(legal_rows) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError("三类合法路径未完整对齐到 3,211,872 行")
    if set(legal_rows["well_id"]).intersection(excluded):
        raise RuntimeError("合法阶段混入影子井")
    if not np.allclose(
        legal_rows["p2_pred_tvt"].to_numpy(dtype=np.float64),
        legal_rows["p3b00_pred_tvt"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("UP01 内的 P3B00 路径与冻结 OOF 不一致")

    last_visible = legal_rows["last_visible_tvt"].to_numpy(dtype=np.float64)
    for lag in (250, 500, 1000):
        legal_rows[f"pfs_lag{lag}_abs_tvt"] = (
            last_visible + legal_rows[f"pfs_lag{lag}_delta"].to_numpy(dtype=np.float64)
        )
    legal_rows["correction_direction"] = (
        legal_rows["pfs_lag1000_abs_tvt"].to_numpy(dtype=np.float64)
        - legal_rows["p3b00_pred_tvt"].to_numpy(dtype=np.float64)
    )
    legal_rows = legal_rows.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for (well_id, fold), frame in legal_rows.groupby(["well_id", "fold"], sort=True):
        direction = frame["correction_direction"].to_numpy(dtype=np.float64)
        md = frame["md"].to_numpy(dtype=np.float64)
        lag250 = frame["pfs_lag250_abs_tvt"].to_numpy(dtype=np.float64)
        lag500 = frame["pfs_lag500_abs_tvt"].to_numpy(dtype=np.float64)
        lag1000 = frame["pfs_lag1000_abs_tvt"].to_numpy(dtype=np.float64)
        lag_stack = np.vstack([lag250, lag500, lag1000])
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "hidden_rows": int(len(frame)),
                "hidden_md_span_ft": float(np.max(md) - np.min(md)),
                "d_rms_ft": float(np.sqrt(np.mean(direction * direction))),
                "d_mean_ft": float(np.mean(direction)),
                "d_abs_mean_ft": float(np.mean(np.abs(direction))),
                "d_std_ft": float(np.std(direction)),
                "d_end_ft": float(direction[-1]),
                "d_slope_ft_per_1000ft": linear_slope_per_1000ft(md, direction),
                "lag250_vs500_rms_ft": float(np.sqrt(np.mean((lag250 - lag500) ** 2))),
                "lag500_vs1000_rms_ft": float(np.sqrt(np.mean((lag500 - lag1000) ** 2))),
                "lag250_vs1000_rms_ft": float(np.sqrt(np.mean((lag250 - lag1000) ** 2))),
                "three_lag_spread_rms_ft": float(
                    np.sqrt(np.mean((np.max(lag_stack, axis=0) - np.min(lag_stack, axis=0)) ** 2))
                ),
                "three_lag_mean_row_std_ft": float(np.mean(np.std(lag_stack, axis=0))),
            }
        )
    legal_well = pd.DataFrame(rows)

    # shadow_holdout.csv 只保存 116 口影子井；完整的无标签井级元数据在 metadata.csv。
    no_label_registry = pd.read_csv(SHADOW_METADATA_PATH, dtype={"well_id": str})
    no_label_registry = no_label_registry.loc[~no_label_registry["well_id"].isin(excluded)].copy()
    if len(no_label_registry) != EXPECTED_DEVELOPMENT_WELLS:
        raise RuntimeError("无标签井级元数据没有完整覆盖 657 口开发井")
    no_label_registry["gr_observed_rows"] = np.rint(
        no_label_registry["hidden_gr_observed_rate"].astype(float)
        * no_label_registry["hidden_row_count"].astype(float)
    ).astype(np.int64)
    legal_well = legal_well.merge(
        no_label_registry[["well_id", "hidden_gr_observed_rate", "gr_observed_rows"]],
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    runtime = load_runtime_features(registry, str(config["pfs_fingerprint"]))
    legal_well = legal_well.merge(runtime, on="well_id", how="left", validate="one_to_one")
    if len(legal_well) != EXPECTED_DEVELOPMENT_WELLS or legal_well.isna().any().any():
        raise RuntimeError("逐井合法量不完整")

    legal_path = output / "legal_well_features.parquet"
    write_parquet_atomic(legal_well, legal_path)
    write_json(
        output / "legal_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "legal_generation_complete": True,
            "hidden_target_read": False,
            "shadow_target_read": False,
            "shadow_overlap_wells": 0,
            "wells": int(len(legal_well)),
            "rows_represented": int(legal_well["hidden_rows"].sum()),
            "legal_feature_columns": list(legal_well.columns),
            "legal_well_features_sha256": file_sha256(legal_path),
            "gr_support_source": "P3_shadow_holdout_v1 hidden_gr_observed_rate (constructed without labels)",
            "pfs_runtime_validated_per_well": True,
        },
    )
    return legal_rows, legal_well


def sufficient_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    error = (
        frame["target_tvt"].to_numpy(dtype=np.float64)
        - frame["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64)
    )
    direction = frame["correction_direction"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(frame)),
        "sum_error_squared": float(np.dot(error, error)),
        "sum_error_times_direction": float(np.dot(error, direction)),
        "sum_direction_squared": float(np.dot(direction, direction)),
    }


def curve_from_statistics(stats: dict[str, float | int], fold: int | None) -> pd.DataFrame:
    rmse_values = rmse_curve_from_sufficient_statistics(
        int(stats["rows"]),
        float(stats["sum_error_squared"]),
        float(stats["sum_error_times_direction"]),
        float(stats["sum_direction_squared"]),
        ALPHA_GRID,
    )
    return pd.DataFrame(
        {
            "scope": "overall" if fold is None else "fold",
            "fold": pd.Series([pd.NA if fold is None else int(fold)] * len(ALPHA_GRID), dtype="Int64"),
            "alpha": ALPHA_GRID,
            "rmse": rmse_values,
            "rows": int(stats["rows"]),
        }
    )


def spearman_table(merged: pd.DataFrame, legal_features: list[str]) -> pd.DataFrame:
    targets = ["oracle_alpha", "up03_gain_vs_up01_ft", "oracle_gain_vs_up03_ft"]
    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, int | None, pd.DataFrame]] = [("overall", None, merged)]
    scopes.extend(("fold", int(fold), frame) for fold, frame in merged.groupby("fold", sort=True))
    for scope, fold, frame in scopes:
        for feature in legal_features:
            for target in targets:
                rho = frame[feature].rank(method="average").corr(frame[target].rank(method="average"))
                records.append(
                    {
                        "scope": scope,
                        "fold": pd.NA if fold is None else fold,
                        "feature": feature,
                        "target": target,
                        "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                        "wells": int(len(frame)),
                    }
                )
    result = pd.DataFrame(records)
    result["fold"] = result["fold"].astype("Int64")
    return result


def identifiability_summary(correlations: pd.DataFrame, target: str) -> dict[str, Any]:
    overall = correlations.loc[
        (correlations["scope"] == "overall") & (correlations["target"] == target)
    ].dropna(subset=["spearman_rho"])
    if overall.empty:
        return {"target": target, "screen_passed": False, "reason": "no finite correlation"}
    best = overall.loc[overall["spearman_rho"].abs().idxmax()]
    fold_rows = correlations.loc[
        (correlations["scope"] == "fold")
        & (correlations["target"] == target)
        & (correlations["feature"] == best["feature"])
    ].dropna(subset=["spearman_rho"])
    sign = np.sign(float(best["spearman_rho"]))
    same_sign = int((np.sign(fold_rows["spearman_rho"].to_numpy(dtype=float)) == sign).sum())
    return {
        "target": target,
        "strongest_feature": str(best["feature"]),
        "overall_spearman_rho": float(best["spearman_rho"]),
        "fold_spearman_rhos": [float(value) for value in fold_rows.sort_values("fold")["spearman_rho"]],
        "same_sign_folds": same_sign,
        "screen_passed": bool(
            abs(float(best["spearman_rho"])) >= IDENTIFIABILITY_MIN_ABS_RHO
            and same_sign >= IDENTIFIABILITY_MIN_SAME_SIGN_FOLDS
        ),
    }


def score_oracle_stage(
    legal_rows: pd.DataFrame,
    legal_well: pd.DataFrame,
    output: Path,
    up03_config: dict[str, Any],
) -> dict[str, Any]:
    """在合法量落盘后读取开发真值，计算固定 alpha 与逐井 oracle。"""

    excluded = shadow_ids(SHADOW_PATH)
    targets = load_targets(P3B00_PATH, [0, 1, 2, 3, 4], excluded)
    keys = ["well_id", "fold", "row_index"]
    scored = legal_rows.merge(targets, on=keys, how="inner", validate="one_to_one")
    if len(scored) != EXPECTED_DEVELOPMENT_ROWS or set(scored["well_id"]).intersection(excluded):
        raise RuntimeError("评分阶段行数错误或混入影子井")

    per_well_records: list[dict[str, Any]] = []
    per_well_stats: list[dict[str, Any]] = []
    for (well_id, fold), frame in scored.groupby(["well_id", "fold"], sort=True):
        stats = sufficient_statistics(frame)
        best_alpha, best_rmse, best_index = grid_oracle(
            int(stats["rows"]),
            float(stats["sum_error_squared"]),
            float(stats["sum_error_times_direction"]),
            float(stats["sum_direction_squared"]),
            ALPHA_GRID,
        )
        up01_rmse = rmse_from_sse(int(stats["rows"]), float(stats["sum_error_squared"]))
        up03_curve = rmse_curve_from_sufficient_statistics(
            int(stats["rows"]),
            float(stats["sum_error_squared"]),
            float(stats["sum_error_times_direction"]),
            float(stats["sum_direction_squared"]),
            np.array([UP03_ALPHA]),
        )
        up03_rmse = float(up03_curve[0])
        per_well_records.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "rows": int(stats["rows"]),
                "up01_rmse": up01_rmse,
                "up03_alpha025_rmse": up03_rmse,
                "oracle_alpha": best_alpha,
                "oracle_grid_index": best_index,
                "oracle_rmse": best_rmse,
                "up03_gain_vs_up01_ft": up01_rmse - up03_rmse,
                "oracle_gain_vs_up03_ft": up03_rmse - best_rmse,
                "oracle_gain_vs_up01_ft": up01_rmse - best_rmse,
            }
        )
        per_well_stats.append({"well_id": str(well_id), "fold": int(fold), **stats})
    per_well = pd.DataFrame(per_well_records)
    stats_frame = pd.DataFrame(per_well_stats)
    write_csv_atomic(per_well, output / "per_well_oracle.csv")

    overall_stats = {
        "rows": int(stats_frame["rows"].sum()),
        "sum_error_squared": float(stats_frame["sum_error_squared"].sum()),
        "sum_error_times_direction": float(stats_frame["sum_error_times_direction"].sum()),
        "sum_direction_squared": float(stats_frame["sum_direction_squared"].sum()),
    }
    curves = [curve_from_statistics(overall_stats, None)]
    fold_best_records: list[dict[str, Any]] = []
    for fold, frame in stats_frame.groupby("fold", sort=True):
        fold_stats = {
            "rows": int(frame["rows"].sum()),
            "sum_error_squared": float(frame["sum_error_squared"].sum()),
            "sum_error_times_direction": float(frame["sum_error_times_direction"].sum()),
            "sum_direction_squared": float(frame["sum_direction_squared"].sum()),
        }
        curve = curve_from_statistics(fold_stats, int(fold))
        curves.append(curve)
        best = curve.loc[curve["rmse"].idxmin()]
        alpha025_rmse = float(curve.loc[np.isclose(curve["alpha"], UP03_ALPHA), "rmse"].iloc[0])
        fold_oracle = per_well.loc[per_well["fold"] == int(fold)]
        fold_oracle_pooled = rmse_from_sse(
            int(fold_oracle["rows"].sum()),
            float(np.sum(fold_oracle["rows"] * fold_oracle["oracle_rmse"] ** 2)),
        )
        fold_best_records.append(
            {
                "fold": int(fold),
                "rows": int(fold_stats["rows"]),
                "fixed_best_alpha": float(best["alpha"]),
                "fixed_best_rmse": float(best["rmse"]),
                "up03_alpha025_rmse": alpha025_rmse,
                "fixed_best_gain_vs_up03_ft": alpha025_rmse - float(best["rmse"]),
                "per_well_oracle_rmse": fold_oracle_pooled,
                "per_well_oracle_gain_vs_up03_ft": alpha025_rmse - fold_oracle_pooled,
            }
        )
    alpha_curves = pd.concat(curves, ignore_index=True)
    per_fold_best = pd.DataFrame(fold_best_records)
    write_csv_atomic(alpha_curves, output / "fixed_alpha_rmse_curves.csv")
    write_csv_atomic(per_fold_best, output / "per_fold_best_alpha.csv")

    alpha_distribution = (
        per_well.groupby("oracle_alpha", as_index=False)
        .agg(wells=("well_id", "size"), rows=("rows", "sum"))
        .sort_values("oracle_alpha")
    )
    alpha_distribution["well_fraction"] = alpha_distribution["wells"] / len(per_well)
    alpha_distribution["row_fraction"] = alpha_distribution["rows"] / per_well["rows"].sum()
    write_csv_atomic(alpha_distribution, output / "oracle_alpha_distribution.csv")

    merged_well = legal_well.merge(per_well, on=["well_id", "fold"], how="inner", validate="one_to_one")
    legal_features = [
        column
        for column in legal_well.columns
        if column not in {"well_id", "fold"}
        and pd.api.types.is_numeric_dtype(legal_well[column])
    ]
    correlations = spearman_table(merged_well, legal_features)
    write_csv_atomic(correlations, output / "spearman_correlations.csv")

    overall_curve = alpha_curves.loc[alpha_curves["scope"] == "overall"]
    overall_best = overall_curve.loc[overall_curve["rmse"].idxmin()]
    up01_rmse = float(overall_curve.loc[np.isclose(overall_curve["alpha"], 0.0), "rmse"].iloc[0])
    up03_rmse = float(overall_curve.loc[np.isclose(overall_curve["alpha"], UP03_ALPHA), "rmse"].iloc[0])
    oracle_pooled_rmse = rmse_from_sse(
        int(per_well["rows"].sum()),
        float(np.sum(per_well["rows"] * per_well["oracle_rmse"] ** 2)),
    )
    # 独立复算应与已经完成的 UP03 657 井结果逐位一致。
    up03_metrics = read_json(CLEAN_ROOT / "artifacts/P3_UP03_up01_plus_pfs_correction_v1/metrics.json")
    if abs(up01_rmse - float(up03_metrics["up01_pooled_rmse"])) > 1.0e-10:
        raise RuntimeError("UP01 RMSE 未复现既有 UP03 结果")
    if abs(up03_rmse - float(up03_metrics["candidate_pooled_rmse"])) > 1.0e-10:
        raise RuntimeError("alpha=0.25 未复现既有 UP03 结果")

    alpha_identifiability = identifiability_summary(correlations, "oracle_alpha")
    gain_identifiability = identifiability_summary(correlations, "up03_gain_vs_up01_ft")
    sign_counts = {
        "negative_wells": int((per_well["oracle_alpha"] < 0).sum()),
        "zero_wells": int((per_well["oracle_alpha"] == 0).sum()),
        "positive_wells": int((per_well["oracle_alpha"] > 0).sum()),
        "lower_boundary_wells": int((per_well["oracle_alpha"] == ALPHA_GRID[0]).sum()),
        "upper_boundary_wells": int((per_well["oracle_alpha"] == ALPHA_GRID[-1]).sum()),
    }
    sign_counts.update({f"{key}_fraction": value / len(per_well) for key, value in list(sign_counts.items())})
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "model_training": False,
        "shadow_target_read": False,
        "development_wells": int(per_well["well_id"].nunique()),
        "development_rows": int(per_well["rows"].sum()),
        "up01_alpha0_rmse": up01_rmse,
        "up03_alpha025_rmse": up03_rmse,
        "up03_gain_vs_up01_ft": up01_rmse - up03_rmse,
        "overall_fixed_best_alpha": float(overall_best["alpha"]),
        "overall_fixed_best_rmse": float(overall_best["rmse"]),
        "overall_fixed_best_gain_vs_up03_ft": up03_rmse - float(overall_best["rmse"]),
        "per_well_grid_oracle_rmse": oracle_pooled_rmse,
        "per_well_grid_oracle_gain_vs_up03_ft": up03_rmse - oracle_pooled_rmse,
        "per_well_grid_oracle_gain_vs_up01_ft": up01_rmse - oracle_pooled_rmse,
        "adaptive_weight_potential_threshold_ft": POTENTIAL_THRESHOLD_FT,
        "adaptive_weight_potential_ge_030ft": bool(up03_rmse - oracle_pooled_rmse >= POTENTIAL_THRESHOLD_FT),
        "per_fold_best_alpha": per_fold_best.to_dict(orient="records"),
        "oracle_alpha_sign_distribution": sign_counts,
        "oracle_alpha_quantiles": {
            str(quantile): float(per_well["oracle_alpha"].quantile(quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "oracle_alpha_identifiability_screen": alpha_identifiability,
        "up03_gain_identifiability_screen": gain_identifiability,
        "identifiability_is_not_model_validation": True,
        "selection_lineage": up03_config["selection_lineage"],
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def conclusion_text(metrics: dict[str, Any]) -> str:
    potential = "达到" if metrics["adaptive_weight_potential_ge_030ft"] else "没有达到"
    alpha_screen = metrics["oracle_alpha_identifiability_screen"]
    gain_screen = metrics["up03_gain_identifiability_screen"]
    lines = [
        "# P3-UP07：PFS 修正强度 oracle 诊断",
        "",
        "## 数据直接证明的事实",
        "",
        f"- 657 口开发井、{metrics['development_rows']:,} 行；影子集真值未读取，且没有训练模型。",
        f"- UP01（alpha=0）RMSE：{metrics['up01_alpha0_rmse']:.6f} ft。",
        f"- 固定 UP03（alpha=0.25）RMSE：{metrics['up03_alpha025_rmse']:.6f} ft。",
        f"- 全局固定 alpha 网格的最佳值为 {metrics['overall_fixed_best_alpha']:+.2f}，RMSE {metrics['overall_fixed_best_rmse']:.6f} ft；相对 UP03 改善 {metrics['overall_fixed_best_gain_vs_up03_ft']:+.6f} ft。",
        f"- 每口井分别事后选择 alpha 的网格 oracle RMSE：{metrics['per_well_grid_oracle_rmse']:.6f} ft；相对 UP03 的 oracle 差距为 {metrics['per_well_grid_oracle_gain_vs_up03_ft']:+.6f} ft。",
        f"- 因而自适应修正强度的理论空间{potential} 0.30 ft 诊断门槛。",
        f"- oracle alpha 正/零/负井数：{metrics['oracle_alpha_sign_distribution']['positive_wells']} / {metrics['oracle_alpha_sign_distribution']['zero_wells']} / {metrics['oracle_alpha_sign_distribution']['negative_wells']}。",
        f"- 与 oracle alpha 相关性最强的合法量是 `{alpha_screen.get('strongest_feature')}`，总体 Spearman={alpha_screen.get('overall_spearman_rho', float('nan')):+.4f}，同号折数={alpha_screen.get('same_sign_folds', 0)}/5。",
        f"- 与 UP03 相对 UP01 收益相关性最强的合法量是 `{gain_screen.get('strongest_feature')}`，总体 Spearman={gain_screen.get('overall_spearman_rho', float('nan')):+.4f}，同号折数={gain_screen.get('same_sign_folds', 0)}/5。",
        "",
        "## 基于事实的合理推断",
        "",
        f"- 预先写死的可辨识性描述筛查中，oracle alpha 为 {'通过' if alpha_screen.get('screen_passed') else '未通过'}；这只说明单个合法量是否呈现稳定单调关系，不等于学习模型一定能取得该收益。",
        "- 若逐井 oracle 很强但合法相关性弱，主要瓶颈是判断每口井应该信多少 PFS，而不是缺少候选方向。",
        "",
        "## 仍然没有验证的猜测",
        "",
        "- 本诊断没有训练严格 OOF 权重模型，因此 oracle 差距全部不可提交，也不能写成 CV 提升。",
        "- 多个合法量联合起来是否能预测 alpha，仍需另开严格 outer-fold 实验验证。",
        "",
        "## 当前实验只能否定的具体实现",
        "",
        "- 若固定 alpha 曲线很平或最优值接近 0.25，只能说明继续微调一个全局常数收益有限；不能否定逐井或分段权重。",
        "",
        "## 下一步最便宜的验证",
        "",
        "- 只有在 oracle 空间达到 0.30 ft 且合法量显示可辨识信号时，才值得用严格外层 OOF 的极小井级模型预测 alpha；否则先寻找更有方向性的可靠性特征。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() and not args.force:
        print(f"已有结果：{output / 'metrics.json'}", flush=True)
        return

    start = time.perf_counter()
    up03_config = read_json(UP03_CONFIG_PATH)
    config = legal_configuration(up03_config)
    write_json(output / "config.json", config)
    print("阶段 A：生成并落盘 657 口井的合法井级量，不读取任何隐藏真值。", flush=True)
    legal_rows, legal_well = build_legal_stage(output, config)
    print("阶段 A 完成：legal_well_features.parquet 与 legal_manifest.json 已落盘。", flush=True)
    print("阶段 B：现在才读取 657 口开发井真值，计算 alpha 网格 oracle。", flush=True)
    metrics = score_oracle_stage(legal_rows, legal_well, output, up03_config)
    (output / "conclusion.md").write_text(conclusion_text(metrics), encoding="utf-8")
    write_json(
        output / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start),
            "legal_stage_completed_before_truth_read": True,
            "model_training": False,
            "shadow_target_read": False,
            "registry_updated": False,
            "current_state_updated": False,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
