"""P3-R01a：outer fold 0 严格嵌套 OOF 的两系数线性残差实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p2_p02_multiscale_pf_paths_cv import FROZEN_MODEL_PARAMS  # noqa: E402
from scripts.run_p3_pf02_target_ess_lgbm_cv import (  # noqa: E402
    P3B00_FEATURES,
    load_development_feature_table,
    load_development_registry,
    merge_existing_p3b00_pf_cache,
)
from scripts.run_simple_lgbm_cv import train_fold, write_json  # noqa: E402
from src.p3_r01a_linear_residual import (  # noqa: E402
    apply_linear_residual,
    fit_linear_residual,
    fit_predict_ridge_coefficients,
    normalized_progress,
    paired_well_bootstrap_delta,
)


EXPERIMENT_ID = "P3_R01a_nested_linear_residual_v1"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_r01a_nested_linear_residual_v1.json"
P01_FINGERPRINT = "ac1dbefd59923585671156b7d3e8b4fc7faca95c20f5f1ae6fdab92a8665954a"
PATH_COLUMNS = [
    "pf128_mean_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
    "beam_mean_d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if len(x_values) < 2:
        return 0.0
    centered = x_values - float(np.mean(x_values))
    denominator = float(np.sum(centered * centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(centered * (y_values - float(np.mean(y_values)))) / denominator)


def load_feature_table() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    registry, shadow_ids = load_development_registry(
        CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv",
        CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    )
    table = load_development_feature_table(
        CLEAN_ROOT / "artifacts/B00_simple_lgbm_v1/feature_cache.parquet",
        CLEAN_ROOT
        / "artifacts/F05a_deterministic_candidate_cache_v1/candidate_feature_cache.parquet",
        registry,
    )
    table = merge_existing_p3b00_pf_cache(
        table,
        registry,
        CLEAN_ROOT / "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache",
        P01_FINGERPRINT,
    )
    if set(table["well_id"].astype(str)).intersection(shadow_ids):
        raise RuntimeError("开发特征表含影子井")
    return table, registry, shadow_ids


def train_nested_base_models(
    table: pd.DataFrame,
    registry: pd.DataFrame,
    output_dir: Path,
    fingerprint: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("R01a：训练 dev-only outer0 base（folds1-4 → fold0）", flush=True)
    train_fold(
        table,
        registry,
        0,
        FROZEN_MODEL_PARAMS,
        output_dir / "base_models/outer_0/base",
        fingerprint,
        P3B00_FEATURES,
    )
    outer_prediction = pd.read_parquet(
        output_dir / "base_models/outer_0/base/fold_0/predictions.parquet"
    )

    outer_train_table = table.loc[table["fold"].astype(int).ne(0)].copy()
    outer_train_registry = registry.loc[registry["fold"].astype(int).ne(0)].copy()
    inner_predictions: list[pd.DataFrame] = []
    for inner_fold in [1, 2, 3, 4]:
        print(
            f"R01a：inner held fold {inner_fold}，只用另外三个开发折训练 P2-P02",
            flush=True,
        )
        train_fold(
            outer_train_table,
            outer_train_registry,
            inner_fold,
            FROZEN_MODEL_PARAMS,
            output_dir / "base_models/outer_0/inner_oof",
            fingerprint,
            P3B00_FEATURES,
        )
        inner_predictions.append(
            pd.read_parquet(
                output_dir
                / f"base_models/outer_0/inner_oof/fold_{inner_fold}/predictions.parquet"
            )
        )
    inner_prediction = pd.concat(inner_predictions, ignore_index=True)
    if inner_prediction["well_id"].astype(str).isin(
        registry.loc[registry["fold"].eq(0), "well_id"].astype(str)
    ).any():
        raise RuntimeError("inner OOF 意外包含 outer fold0")
    return inner_prediction, outer_prediction


def build_coefficient_targets(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for well_id, well in predictions.groupby("well_id", sort=False):
        well = well.sort_values("row_index")
        coefficients = fit_linear_residual(
            well["md"].to_numpy(dtype=np.float64),
            well["pred_tvt"].to_numpy(dtype=np.float64),
            well["target_tvt"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(well["fold"].iloc[0]),
                "a0": float(coefficients[0]),
                "a1": float(coefficients[1]),
            }
        )
    return pd.DataFrame(rows)


def build_well_features(
    table: pd.DataFrame,
    base_predictions: pd.DataFrame,
) -> pd.DataFrame:
    prediction = base_predictions[["well_id", "row_index", "pred_tvt"]].copy()
    prediction["well_id"] = prediction["well_id"].astype(str)
    source = table.copy()
    source["well_id"] = source["well_id"].astype(str)
    source = source.merge(
        prediction,
        on=["well_id", "row_index"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, float | int | str]] = []
    for well_id, well in source.groupby("well_id", sort=False):
        well = well.sort_values("row_index")
        md = well["md"].to_numpy(dtype=np.float64)
        progress = normalized_progress(md)
        feature: dict[str, float | int | str] = {
            "well_id": str(well_id),
            "fold": int(well["fold"].iloc[0]),
            "hidden_rows_log": float(np.log1p(len(well))),
            "md_span_log": float(np.log1p(float(np.max(md) - np.min(md)))),
            "gr_observed_fraction": float(1.0 - well["gr_missing"].mean()),
            "azimuth_sin": float(np.sin(np.arctan2(
                float(well["y_current"].iloc[-1] - well["y_current"].iloc[0]),
                float(well["x_current"].iloc[-1] - well["x_current"].iloc[0]),
            ))),
            "azimuth_cos": float(np.cos(np.arctan2(
                float(well["y_current"].iloc[-1] - well["y_current"].iloc[0]),
                float(well["x_current"].iloc[-1] - well["x_current"].iloc[0]),
            ))),
            "dz_per_md": slope(md, well["z_current"].to_numpy(dtype=np.float64)),
            "sc_trust_mean": float(np.nanmean(well["sc_trust"])),
        }
        pf_endpoints: list[float] = []
        pf_slopes: list[float] = []
        for path_name in PATH_COLUMNS[:5]:
            values = well[path_name].to_numpy(dtype=np.float64)
            feature[f"{path_name}_end"] = float(values[-1])
            pf_endpoints.append(float(values[-1]))
            pf_slopes.append(slope(progress, values))
        feature["pf_mean_slope"] = pf_slopes[0]
        feature["pf_endpoint_range"] = float(np.ptp(pf_endpoints))
        feature["pf_slope_range"] = float(np.ptp(pf_slopes))
        beam = well["beam_mean_d"].to_numpy(dtype=np.float64)
        pf_mean = well["pf128_mean_delta"].to_numpy(dtype=np.float64)
        midpoint = int(np.argmin(np.abs(progress - 0.5)))
        gap = pf_mean - beam
        feature["pf_beam_end_gap"] = float(gap[-1])
        feature["pf_beam_second_half_growth"] = float(gap[-1] - gap[midpoint])

        predicted_u = (
            well["pred_tvt"].to_numpy(dtype=np.float64)
            + well["z_current"].to_numpy(dtype=np.float64)
        )
        base_slopes: dict[int, float] = {}
        for window_ft in [100, 200, 500]:
            mask = md <= float(np.min(md)) + float(window_ft)
            base_slopes[window_ft] = slope(md[mask], predicted_u[mask])
            feature[f"base_u_slope_{window_ft}"] = base_slopes[window_ft]
        feature["base_u_slope_100_minus_500"] = (
            base_slopes[100] - base_slopes[500]
        )
        rows.append(feature)
    result = pd.DataFrame(rows)
    numeric = [name for name in result.columns if name not in {"well_id", "fold"}]
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    return result


def impute_from_outer_train(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_output = train[feature_names].copy()
    validation_output = validation[feature_names].copy()
    indicator_columns: list[str] = []
    for name in feature_names:
        indicator = f"{name}__missing"
        train_output[indicator] = train_output[name].isna().astype(np.float64)
        validation_output[indicator] = validation_output[name].isna().astype(np.float64)
        indicator_columns.append(indicator)
        median = float(train_output[name].median())
        if not np.isfinite(median):
            median = 0.0
        train_output[name] = train_output[name].fillna(median)
        validation_output[name] = validation_output[name].fillna(median)
    return train_output, validation_output, [*feature_names, *indicator_columns]


def score_paths(
    outer_predictions: pd.DataFrame,
    coefficient_predictions: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = outer_predictions.merge(
        coefficient_predictions,
        on=["well_id", "fold"],
        how="inner",
        validate="many_to_one",
    )
    output_rows: list[pd.DataFrame] = []
    per_well_rows: list[dict[str, float | int | str]] = []
    for well_id, well in merged.groupby("well_id", sort=False):
        well = well.sort_values("row_index").copy()
        base = well["pred_tvt"].to_numpy(dtype=np.float64)
        target = well["target_tvt"].to_numpy(dtype=np.float64)
        candidate = apply_linear_residual(
            well["md"].to_numpy(dtype=np.float64),
            base,
            float(well["pred_a0"].iloc[0]),
            float(well["pred_a1"].iloc[0]),
        )
        shuffled = apply_linear_residual(
            well["md"].to_numpy(dtype=np.float64),
            base,
            float(well["shuffle_a0"].iloc[0]),
            float(well["shuffle_a1"].iloc[0]),
        )
        base_error = target - base
        candidate_error = target - candidate
        shuffled_error = target - shuffled
        well["base_tvt"] = base
        well["corrected_tvt"] = candidate
        well["shuffle_tvt"] = shuffled
        output_rows.append(well)
        per_well_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(well["fold"].iloc[0]),
                "rows": len(well),
                "base_sse": float(np.sum(base_error**2)),
                "candidate_sse": float(np.sum(candidate_error**2)),
                "shuffle_sse": float(np.sum(shuffled_error**2)),
                "base_rmse": float(np.sqrt(np.mean(base_error**2))),
                "candidate_rmse": float(np.sqrt(np.mean(candidate_error**2))),
                "shuffle_rmse": float(np.sqrt(np.mean(shuffled_error**2))),
            }
        )
    paths = pd.concat(output_rows, ignore_index=True)
    per_well = pd.DataFrame(per_well_rows)
    total_rows = int(per_well["rows"].sum())
    base_rmse = float(np.sqrt(per_well["base_sse"].sum() / total_rows))
    candidate_rmse = float(np.sqrt(per_well["candidate_sse"].sum() / total_rows))
    shuffle_rmse = float(np.sqrt(per_well["shuffle_sse"].sum() / total_rows))
    bootstrap = paired_well_bootstrap_delta(per_well, 2000, 42)
    metrics = {
        "outer_fold": 0,
        "wells": int(len(per_well)),
        "rows": total_rows,
        "base_rmse": base_rmse,
        "corrected_rmse": candidate_rmse,
        "improvement_ft": base_rmse - candidate_rmse,
        "shuffle_rmse": shuffle_rmse,
        "shuffle_improvement_ft": base_rmse - shuffle_rmse,
        "well_win_rate": float(np.mean(per_well["candidate_rmse"] < per_well["base_rmse"])),
        "paired_well_bootstrap": bootstrap,
    }
    paths.to_parquet(output_dir / "outer0_predictions.parquet", index=False)
    per_well.to_csv(output_dir / "outer0_per_well.csv", index=False)
    return per_well, metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    start = time.perf_counter()
    table, registry, shadow_ids = load_feature_table()
    fingerprint = stable_hash(
        {
            "experiment": EXPERIMENT_ID,
            "outer_fold": 0,
            "dev_wells": sorted(registry["well_id"].astype(str)),
            "shadow_wells": sorted(shadow_ids),
            "features": P3B00_FEATURES,
            "model": FROZEN_MODEL_PARAMS,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    inner_prediction, outer_prediction = train_nested_base_models(
        table, registry, output_dir, fingerprint
    )
    train_targets = build_coefficient_targets(inner_prediction)
    train_feature = build_well_features(
        table.loc[table["fold"].astype(int).ne(0)], inner_prediction
    )
    validation_feature = build_well_features(
        table.loc[table["fold"].astype(int).eq(0)], outer_prediction
    )
    train = train_feature.merge(train_targets, on=["well_id", "fold"], validate="one_to_one")
    feature_names = [
        name
        for name in train_feature.columns
        if name not in {"well_id", "fold"}
    ]
    x_train, x_validation, final_features = impute_from_outer_train(
        train, validation_feature, feature_names
    )
    y_train = train[["a0", "a1"]].to_numpy(dtype=np.float64)
    coefficient_prediction, model = fit_predict_ridge_coefficients(
        x_train, y_train, x_validation, alpha=10.0
    )
    generator = np.random.default_rng(20260719)
    shuffled_y = y_train[generator.permutation(len(y_train))]
    shuffled_prediction, _ = fit_predict_ridge_coefficients(
        x_train, shuffled_y, x_validation, alpha=10.0
    )
    coefficient_table = validation_feature[["well_id", "fold"]].copy()
    coefficient_table[["pred_a0", "pred_a1"]] = coefficient_prediction
    coefficient_table[["shuffle_a0", "shuffle_a1"]] = shuffled_prediction
    coefficient_table.to_csv(output_dir / "outer0_predicted_coefficients.csv", index=False)
    _, metrics = score_paths(outer_prediction, coefficient_table, output_dir)

    true_validation = build_coefficient_targets(outer_prediction)
    coefficient_audit = coefficient_table.merge(
        true_validation, on=["well_id", "fold"], validate="one_to_one"
    )
    coefficient_audit.to_csv(output_dir / "outer0_coefficient_audit.csv", index=False)
    metrics["a0_correlation"] = float(
        np.corrcoef(coefficient_audit["pred_a0"], coefficient_audit["a0"])[0, 1]
    )
    metrics["a1_correlation"] = float(
        np.corrcoef(coefficient_audit["pred_a1"], coefficient_audit["a1"])[0, 1]
    )
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "feature_list.json", {"features": final_features})
    write_json(output_dir / "metrics_outer0.json", metrics)
    write_json(
        output_dir / "runtime_outer0.json",
        {"seconds": time.perf_counter() - start, "fingerprint": fingerprint},
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
