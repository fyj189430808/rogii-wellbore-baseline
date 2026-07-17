"""运行 RF01a 的合法可见前缀回放，并比较 Huber 与 OLS 多窗口倾角。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 当前脚本位于 rogii_clean/scripts，父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.f01_features import fit_visible_u_trend
from src.lgbm_data import load_and_validate_registry
from src.rf01_diagnostics import build_prefix_holdout_predictions
from src.rf01a_features import fit_visible_u_huber_slope


WINDOWS_FT = [50, 100, 200, 500, 1000]


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """返回逐行 RMSE。"""

    error = prediction - target
    return float(np.sqrt(np.mean(error * error)))


def parse_args() -> argparse.Namespace:
    """解析数据目录、切分比例和输出目录。"""

    parser = argparse.ArgumentParser(description="诊断 RF01 Huber/OLS 前缀倾角")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=PROJECT_ROOT / "input" / "data" / "raw" / "train",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv",
    )
    parser.add_argument("--cut-fraction", type=float, default=0.75)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "RF01a_huber_slopes_v1",
    )
    return parser.parse_args()


def main() -> None:
    """逐井构造合法回放，保存行级、井级和汇总诊断。"""

    args = parse_args()
    registry = load_and_validate_registry(args.registry.resolve())
    train_dir = args.train_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prediction_parts: list[pd.DataFrame] = []
    slope_rows: list[dict] = []
    total_wells = len(registry)

    for well_number, registry_row in enumerate(
        registry.itertuples(index=False),
        start=1,
    ):
        well_id = str(registry_row.well_id)
        well_path = train_dir / f"{well_id}__horizontal_well.csv"
        horizontal_df = pd.read_csv(well_path)

        well_predictions = build_prefix_holdout_predictions(
            horizontal_df,
            cut_fraction=float(args.cut_fraction),
            windows_ft=WINDOWS_FT,
        )
        well_predictions.insert(0, "well_id", well_id)
        well_predictions.insert(1, "fold", int(registry_row.fold))
        prediction_parts.append(well_predictions)

        slope_row: dict[str, float | int | str] = {
            "well_id": well_id,
            "fold": int(registry_row.fold),
        }
        for window_ft in WINDOWS_FT:
            huber_slope = fit_visible_u_huber_slope(
                horizontal_df,
                window_ft=float(window_ft),
            )
            ols_slope, _ = fit_visible_u_trend(
                horizontal_df,
                window_ft=float(window_ft),
            )
            slope_row[f"huber_slope_{window_ft}"] = huber_slope
            slope_row[f"ols_slope_{window_ft}"] = ols_slope
            slope_row[f"huber_minus_ols_{window_ft}"] = huber_slope - ols_slope
        slope_rows.append(slope_row)

        if well_number % 50 == 0 or well_number == total_wells:
            print(f"RF01 诊断进度：{well_number}/{total_wells} 井", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    slope_comparison = pd.DataFrame(slope_rows)
    predictions.to_parquet(
        artifact_dir / "prefix_holdout_predictions.parquet",
        index=False,
        compression="zstd",
    )
    slope_comparison.to_csv(
        artifact_dir / "huber_vs_ols_slopes.csv",
        index=False,
    )

    target = predictions["target_tvt"].to_numpy(dtype=np.float64)
    prediction_columns = [
        "carry_tvt",
        "zero_u_slope_tvt",
    ]
    for window_ft in WINDOWS_FT:
        prediction_columns.append(f"huber_{window_ft}_tvt")
        prediction_columns.append(f"ols_{window_ft}_tvt")

    metrics: dict[str, object] = {
        "cut_fraction": float(args.cut_fraction),
        "wells": int(predictions["well_id"].nunique()),
        "rows": int(len(predictions)),
        "windows_ft": WINDOWS_FT,
        "micro_rmse": {},
        "well_win_rate_vs_carry": {},
    }

    per_well_rows: list[dict] = []
    for well_id, well_data in predictions.groupby("well_id", observed=True):
        well_target = well_data["target_tvt"].to_numpy(dtype=np.float64)
        well_row: dict[str, float | int | str] = {
            "well_id": str(well_id),
            "fold": int(well_data["fold"].iloc[0]),
        }
        for column_name in prediction_columns:
            well_prediction = well_data[column_name].to_numpy(dtype=np.float64)
            well_row[f"{column_name}_rmse"] = rmse(well_target, well_prediction)
        per_well_rows.append(well_row)
    per_well = pd.DataFrame(per_well_rows)
    per_well.to_csv(artifact_dir / "prefix_holdout_per_well.csv", index=False)

    carry_well_rmse = per_well["carry_tvt_rmse"].to_numpy(dtype=np.float64)
    for column_name in prediction_columns:
        prediction = predictions[column_name].to_numpy(dtype=np.float64)
        column_micro_rmse = rmse(target, prediction)
        metrics["micro_rmse"][column_name] = column_micro_rmse

        method_well_rmse = per_well[f"{column_name}_rmse"].to_numpy(
            dtype=np.float64
        )
        metrics["well_win_rate_vs_carry"][column_name] = float(
            np.mean(method_well_rmse < carry_well_rmse)
        )

    slope_difference_quantiles: dict[str, dict[str, float]] = {}
    for window_ft in WINDOWS_FT:
        difference = slope_comparison[
            f"huber_minus_ols_{window_ft}"
        ].abs()
        slope_difference_quantiles[str(window_ft)] = {
            "median_abs": float(difference.median()),
            "p90_abs": float(difference.quantile(0.90)),
            "max_abs": float(difference.max()),
        }
    metrics["huber_vs_ols_difference"] = slope_difference_quantiles

    metrics_path = artifact_dir / "prefix_holdout_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"诊断输出：{metrics_path}", flush=True)


if __name__ == "__main__":
    main()

