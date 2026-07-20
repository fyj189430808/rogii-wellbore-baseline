"""PFS01 固定滞后路径与 P3B00 OOF 预测的确定性保守后融合。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """返回逐行 micro RMSE，单位为 ft。"""

    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def candidate_name(lag_distance: int, blend_fraction: float) -> str:
    """把预登记的滞后距离和融合比例变成稳定候选名。"""

    return f"lag{int(lag_distance)}_alpha{int(round(100.0 * blend_fraction)):02d}"


def build_postblend_candidates(
    legal_rows: pd.DataFrame,
    lag_distances: Iterable[int],
    blend_fractions: Iterable[float],
) -> tuple[pd.DataFrame, list[str]]:
    """在不接触隐藏真值时生成固定网格的九条候选路径。"""

    output = legal_rows.copy()
    baseline = output["p3b00_pred_tvt"].to_numpy(dtype=np.float64)
    last_visible = output["last_visible_tvt"].to_numpy(dtype=np.float64)
    names: list[str] = []
    for lag_distance in lag_distances:
        lag = int(lag_distance)
        pfs_path = last_visible + output[f"pfs_lag{lag}_delta"].to_numpy(dtype=np.float64)
        for blend_fraction in blend_fractions:
            alpha = float(blend_fraction)
            name = candidate_name(lag, alpha)
            names.append(name)
            output[f"{name}_pred_tvt"] = (1.0 - alpha) * baseline + alpha * pfs_path
    candidate_values = output[[f"{name}_pred_tvt" for name in names]].to_numpy(dtype=np.float64)
    if not np.isfinite(candidate_values).all():
        raise ValueError("后融合候选含 NaN 或无穷值")
    return output, names


def score_candidates(
    legal_candidates: pd.DataFrame,
    targets: pd.DataFrame,
    candidate_names: list[str],
    minimum_pooled_improvement_ft: float,
    maximum_any_fold_degradation_ft: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """统一评分全部候选，并仅按合并 micro 改善锁定一个候选。"""

    key_columns = ["well_id", "fold", "row_index"]
    scored = legal_candidates.merge(targets, on=key_columns, how="inner", validate="one_to_one")
    if len(scored) != len(legal_candidates) or len(scored) != len(targets):
        raise ValueError("合法候选与评分真值行键不完全一致")
    target = scored["target_tvt"].to_numpy(dtype=np.float64)
    baseline = scored["p3b00_pred_tvt"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(target, baseline)

    per_fold_rows: list[dict[str, Any]] = []
    method_columns = {"baseline": "p3b00_pred_tvt"}
    method_columns.update({name: f"{name}_pred_tvt" for name in candidate_names})
    for fold, fold_frame in scored.groupby("fold", sort=True):
        fold_target = fold_frame["target_tvt"].to_numpy(dtype=np.float64)
        fold_baseline = rmse(fold_target, fold_frame["p3b00_pred_tvt"].to_numpy(dtype=np.float64))
        for method, column in method_columns.items():
            value = rmse(fold_target, fold_frame[column].to_numpy(dtype=np.float64))
            per_fold_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "rows": int(len(fold_frame)),
                    "baseline_rmse": fold_baseline,
                    "rmse": value,
                    "improvement_ft": fold_baseline - value,
                    "degradation_ft": value - fold_baseline,
                }
            )
    per_fold = pd.DataFrame(per_fold_rows)

    per_well_rows: list[dict[str, Any]] = []
    for (well_id, fold), well_frame in scored.groupby(["well_id", "fold"], sort=True):
        well_target = well_frame["target_tvt"].to_numpy(dtype=np.float64)
        well_baseline = rmse(
            well_target,
            well_frame["p3b00_pred_tvt"].to_numpy(dtype=np.float64),
        )
        row: dict[str, Any] = {
            "well_id": str(well_id),
            "fold": int(fold),
            "rows": int(len(well_frame)),
            "baseline_rmse": well_baseline,
        }
        for name in candidate_names:
            value = rmse(well_target, well_frame[f"{name}_pred_tvt"].to_numpy(dtype=np.float64))
            row[f"{name}_rmse"] = value
            row[f"{name}_improvement_ft"] = well_baseline - value
        per_well_rows.append(row)
    per_well = pd.DataFrame(per_well_rows)

    candidate_metrics: dict[str, Any] = {}
    for name in candidate_names:
        value = rmse(target, scored[f"{name}_pred_tvt"].to_numpy(dtype=np.float64))
        fold_rows = per_fold.loc[per_fold["method"] == name]
        worst_degradation = float(fold_rows["degradation_ft"].max())
        pooled_improvement = baseline_rmse - value
        candidate_metrics[name] = {
            "pooled_rmse": value,
            "pooled_improvement_ft": pooled_improvement,
            "maximum_fold_degradation_ft": worst_degradation,
            "fold_improvements_ft": [float(v) for v in fold_rows["improvement_ft"]],
            "gate_passed": bool(
                pooled_improvement >= minimum_pooled_improvement_ft
                and worst_degradation <= maximum_any_fold_degradation_ft
            ),
        }

    selected = max(candidate_names, key=lambda name: candidate_metrics[name]["pooled_improvement_ft"])
    selected_metrics = candidate_metrics[selected]
    gate_passed = bool(selected_metrics["gate_passed"])
    metrics = {
        "folds": sorted(int(v) for v in scored["fold"].unique()),
        "wells": int(scored["well_id"].nunique()),
        "rows": int(len(scored)),
        "baseline_pooled_rmse": baseline_rmse,
        "candidates": candidate_metrics,
        "selected_candidate": selected,
        "selection_rule": "maximum pooled micro RMSE improvement on preregistered screening folds",
        "gate": {
            "minimum_pooled_improvement_ft": float(minimum_pooled_improvement_ft),
            "maximum_any_fold_degradation_ft": float(maximum_any_fold_degradation_ft),
            "passed": gate_passed,
            "continue_other_folds": gate_passed,
        },
    }
    return metrics, per_fold, per_well
