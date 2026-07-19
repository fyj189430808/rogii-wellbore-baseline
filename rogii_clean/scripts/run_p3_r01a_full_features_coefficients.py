"""读取已完成的 outer0/inner OOF 路径，只运行 R01a 32 特征系数层。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pf02_target_ess_lgbm_cv import load_development_registry  # noqa: E402
from scripts.run_p3_r01a_nested_linear_residual import (  # noqa: E402
    build_coefficient_targets,
    score_paths,
    slope,
)
from scripts.run_simple_lgbm_cv import write_json  # noqa: E402
from src.f01_features import fit_visible_u_trend  # noqa: E402
from src.f03a_prefix_reliability_features import (  # noqa: E402
    build_well_prefix_reliability_features,
)


EXPERIMENT_ID = "P3_R01a_nested_linear_residual_full_features_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs/p3_r01a_nested_linear_residual_full_features_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SOURCE_DIR = CLEAN_ROOT / "artifacts/P3_R01a_nested_linear_residual_v1/base_models/outer_0"
PF_CACHE_DIR = CLEAN_ROOT / "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache"
BEAM_CACHE_DIR = CLEAN_ROOT / "artifacts/F05a_deterministic_candidate_cache_v1/per_well"
RAW_TRAIN_DIR = PROJECT_ROOT / "input/data/raw/train"

FEATURE_NAMES = [
    "pf_mean_end_delta",
    "pf_s3_end_delta",
    "pf_s5_end_delta",
    "pf_s8_end_delta",
    "pf_s12_end_delta",
    "pf_mean_average_slope",
    "pf_end_delta_range",
    "pf_average_slope_range",
    "pf_beam_end_gap",
    "pf_beam_abs_gap_growth",
    "pf_seed_std_end",
    "gr_observed_fraction",
    "gr_longest_gap_fraction",
    "gr_effective_evidence_fraction",
    "pf_scale8_normalized_ess",
    "visible_u_slope_100",
    "visible_u_slope_200",
    "visible_u_slope_500",
    "visible_u_slope_100_minus_500",
    "base_u_slope_100",
    "base_u_slope_200",
    "base_u_slope_500",
    "base_u_slope_100_minus_500",
    "azimuth_sin",
    "azimuth_cos",
    "hidden_dxy_per_md",
    "hidden_abs_dz_per_md",
    "hidden_tortuosity",
    "f03a_tail_zero_raw_ncc",
    "f03a_tail_zero_affine_mae",
    "f03a_tail_valid_pair_fraction",
    "f03a_tail_ncc_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_nested_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_parts = [
        pd.read_parquet(SOURCE_DIR / f"inner_oof/fold_{fold}/predictions.parquet")
        for fold in [1, 2, 3, 4]
    ]
    inner = pd.concat(inner_parts, ignore_index=True)
    outer = pd.read_parquet(SOURCE_DIR / "base/fold_0/predictions.parquet")
    for frame in [inner, outer]:
        frame["well_id"] = frame["well_id"].astype(str)
        frame["row_index"] = frame["row_index"].astype(np.int64)
        if frame.duplicated(["well_id", "row_index"]).any():
            raise ValueError("嵌套预测含重复行键")
    return inner, outer


def _load_typewell_features(registry: pd.DataFrame) -> pd.DataFrame:
    well_ids = set(registry["well_id"].astype(str))
    current_fold = registry.set_index("well_id")["fold"].astype(int).to_dict()
    score_columns = [
        "well_id", "fold", "scope", "offset_ft", "n_points", "raw_ncc",
        "affine_median_ae",
    ]
    margin_columns = [
        "well_id", "fold", "scope", "n_points", "ncc_margin_vs_best_wrong",
        "mae_margin_vs_best_wrong",
    ]
    scores = pd.read_csv(
        CLEAN_ROOT / "artifacts/RF03_D0_prefix_alignment_v1/offset_scores.csv",
        usecols=score_columns,
        dtype={"well_id": str},
    )
    margins = pd.read_csv(
        CLEAN_ROOT / "artifacts/RF03_D0_prefix_alignment_v1/per_well_margins.csv",
        usecols=margin_columns,
        dtype={"well_id": str},
    )
    # D0 缓存的 fold 只是旧 CV 元数据，特征值完全来自本井可见前缀。
    # 先过滤开发井，再把该元数据映射到当前 balanced fold，避免旧 fold 阻断合法合并。
    scores = scores.loc[scores["well_id"].isin(well_ids)].copy()
    margins = margins.loc[margins["well_id"].isin(well_ids)].copy()
    scores["fold"] = scores["well_id"].map(current_fold).astype(int)
    margins["fold"] = margins["well_id"].map(current_fold).astype(int)
    features = build_well_prefix_reliability_features(scores, margins, registry)
    keep = ["well_id", *FEATURE_NAMES[-4:]]
    return features[keep]


def _safe_tortuosity(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    chord = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2 + (z[-1] - z[0]) ** 2))
    if chord <= 1e-12:
        return 1.0
    return float(max(np.sum(step) / chord, 1.0))


def build_static_features(
    registry: pd.DataFrame,
    predicted_row_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    d01_columns = [
        "well_id", "observed_gr_fraction", "longest_gr_gap_md_ft",
        "effective_observation_count", "hidden_rows",
        "scale_8_normalized_effective_sample_size",
    ]
    d01 = pd.read_csv(
        CLEAN_ROOT / "artifacts/P3_D01_pf_observation_weight_audit_v1/per_well.csv",
        usecols=d01_columns,
        dtype={"well_id": str},
    ).set_index("well_id")
    metadata = pd.read_csv(
        CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/metadata.csv",
        usecols=["well_id", "azimuth_deg"],
        dtype={"well_id": str},
    ).set_index("well_id")
    rows: list[dict[str, float | int | str]] = []
    for number, registry_row in enumerate(registry.itertuples(index=False), start=1):
        well_id = str(registry_row.well_id)
        horizontal = pd.read_csv(
            RAW_TRAIN_DIR / f"{well_id}__horizontal_well.csv",
            usecols=["MD", "X", "Y", "Z", "TVT_input"],
        )
        pf = pd.read_parquet(
            PF_CACHE_DIR / f"{well_id}.parquet",
            columns=[
                "row_index", "last_visible_tvt", "pf128_mean_tvt",
                "pf128_scale_3_delta", "pf128_scale_5_delta",
                "pf128_scale_8_delta", "pf128_scale_12_delta", "pf128_seed_std",
            ],
        ).sort_values("row_index")
        beam = pd.read_parquet(
            BEAM_CACHE_DIR / f"{well_id}.parquet",
            columns=["row_index", "beam_mean_d"],
        ).sort_values("row_index")
        if not np.array_equal(
            pf["row_index"].to_numpy(dtype=np.int64),
            beam["row_index"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"{well_id} PF 与 Beam 行键不一致")
        hidden_indices = pf["row_index"].to_numpy(dtype=np.int64)
        if well_id not in predicted_row_indices or not np.array_equal(
            hidden_indices,
            np.sort(predicted_row_indices[well_id].astype(np.int64, copy=False)),
        ):
            raise ValueError(f"{well_id} 嵌套预测没有完整覆盖 PF 自然隐藏行键")
        hidden = horizontal.iloc[hidden_indices]
        md = hidden["MD"].to_numpy(dtype=np.float64)
        md_span = max(float(md[-1] - md[0]), 1.0)
        anchor = float(pf["last_visible_tvt"].iloc[0])
        path_deltas = [
            pf["pf128_mean_tvt"].to_numpy(dtype=np.float64) - anchor,
            pf["pf128_scale_3_delta"].to_numpy(dtype=np.float64),
            pf["pf128_scale_5_delta"].to_numpy(dtype=np.float64),
            pf["pf128_scale_8_delta"].to_numpy(dtype=np.float64),
            pf["pf128_scale_12_delta"].to_numpy(dtype=np.float64),
        ]
        endpoints = [float(path[-1]) for path in path_deltas]
        average_slopes = [float((path[-1] - path[0]) / md_span) for path in path_deltas]
        gap = path_deltas[0] - beam["beam_mean_d"].to_numpy(dtype=np.float64)
        midpoint = len(gap) // 2
        prefix_slopes = {
            window: fit_visible_u_trend(horizontal, float(window))[0]
            for window in [100, 200, 500]
        }
        x = hidden["X"].to_numpy(dtype=np.float64)
        y = hidden["Y"].to_numpy(dtype=np.float64)
        z = hidden["Z"].to_numpy(dtype=np.float64)
        azimuth = np.deg2rad(float(metadata.loc[well_id, "azimuth_deg"]))
        d01_row = d01.loc[well_id]
        row = {
            "well_id": well_id,
            "fold": int(registry_row.fold),
            "pf_mean_end_delta": endpoints[0],
            "pf_s3_end_delta": endpoints[1],
            "pf_s5_end_delta": endpoints[2],
            "pf_s8_end_delta": endpoints[3],
            "pf_s12_end_delta": endpoints[4],
            "pf_mean_average_slope": average_slopes[0],
            "pf_end_delta_range": float(np.ptp(endpoints)),
            "pf_average_slope_range": float(np.ptp(average_slopes)),
            "pf_beam_end_gap": float(gap[-1]),
            "pf_beam_abs_gap_growth": float(np.mean(np.abs(gap[midpoint:])) - np.mean(np.abs(gap[:midpoint]))),
            "pf_seed_std_end": float(pf["pf128_seed_std"].iloc[-1]),
            "gr_observed_fraction": float(d01_row["observed_gr_fraction"]),
            "gr_longest_gap_fraction": float(d01_row["longest_gr_gap_md_ft"]) / md_span,
            "gr_effective_evidence_fraction": float(d01_row["effective_observation_count"]) / max(float(d01_row["hidden_rows"]), 1.0),
            "pf_scale8_normalized_ess": float(d01_row["scale_8_normalized_effective_sample_size"]),
            "visible_u_slope_100": prefix_slopes[100],
            "visible_u_slope_200": prefix_slopes[200],
            "visible_u_slope_500": prefix_slopes[500],
            "visible_u_slope_100_minus_500": prefix_slopes[100] - prefix_slopes[500],
            "azimuth_sin": float(np.sin(azimuth)),
            "azimuth_cos": float(np.cos(azimuth)),
            "hidden_dxy_per_md": float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / md_span),
            "hidden_abs_dz_per_md": float(abs(z[-1] - z[0]) / md_span),
            "hidden_tortuosity": _safe_tortuosity(x, y, z),
        }
        rows.append(row)
        if number % 100 == 0 or number == len(registry):
            print(f"静态井级特征 {number}/{len(registry)}", flush=True)
    result = pd.DataFrame(rows)
    return result.merge(_load_typewell_features(registry), on="well_id", validate="one_to_one")


def build_dynamic_base_features(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for well_id, well in predictions.groupby("well_id", sort=False):
        well = well.sort_values("row_index")
        horizontal = pd.read_csv(
            RAW_TRAIN_DIR / f"{well_id}__horizontal_well.csv",
            usecols=["Z"],
        )
        indices = well["row_index"].to_numpy(dtype=np.int64)
        md = well["md"].to_numpy(dtype=np.float64)
        z = horizontal.iloc[indices]["Z"].to_numpy(dtype=np.float64)
        predicted_u = well["pred_tvt"].to_numpy(dtype=np.float64) + z
        slopes: dict[int, float] = {}
        for window in [100, 200, 500]:
            mask = md <= float(md[0]) + float(window)
            slopes[window] = slope(md[mask], predicted_u[mask])
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(well["fold"].iloc[0]),
                "base_u_slope_100": slopes[100],
                "base_u_slope_200": slopes[200],
                "base_u_slope_500": slopes[500],
                "base_u_slope_100_minus_500": slopes[100] - slopes[500],
            }
        )
    return pd.DataFrame(rows)


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    registry, shadow_ids = load_development_registry(
        CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv",
        CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    )
    inner, outer = load_nested_predictions()
    inner_wells = set(inner["well_id"])
    outer_wells = set(outer["well_id"])
    if len(inner_wells) != 526 or len(outer_wells) != 131 or inner_wells & outer_wells:
        raise ValueError("inner/outer 井覆盖不是严格 526/131")
    if (inner_wells | outer_wells) != set(registry["well_id"].astype(str)):
        raise ValueError("嵌套预测没有完整覆盖 657 口开发井")
    if (inner_wells | outer_wells) & shadow_ids:
        raise ValueError("嵌套预测含影子井")
    expected_inner_rows = int(registry.loc[registry["fold"].ne(0), "hidden_rows"].sum())
    expected_outer_rows = int(registry.loc[registry["fold"].eq(0), "hidden_rows"].sum())
    if len(inner) != expected_inner_rows or len(outer) != expected_outer_rows:
        raise ValueError("嵌套预测行数没有完整覆盖开发集自然隐藏区")

    all_predictions = pd.concat(
        [inner[["well_id", "row_index"]], outer[["well_id", "row_index"]]],
        ignore_index=True,
    )
    predicted_row_indices = {
        str(well_id): well["row_index"].to_numpy(dtype=np.int64)
        for well_id, well in all_predictions.groupby("well_id", sort=False)
    }
    static = build_static_features(registry, predicted_row_indices)
    dynamic = pd.concat(
        [build_dynamic_base_features(inner), build_dynamic_base_features(outer)],
        ignore_index=True,
    )
    features = static.merge(dynamic, on=["well_id", "fold"], validate="one_to_one")
    if len(FEATURE_NAMES) != 32 or features[FEATURE_NAMES].shape != (657, 32):
        raise ValueError("正式井级特征不是 657×32")
    train_targets = build_coefficient_targets(inner)
    train = features.loc[features["fold"].ne(0)].merge(
        train_targets, on=["well_id", "fold"], validate="one_to_one"
    )
    validation = features.loc[features["fold"].eq(0)].copy()
    if train.shape[0] != 526 or validation.shape[0] != 131:
        raise ValueError("系数层训练/验证井数不是 526/131")
    x_train = train[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    x_validation = validation[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    predictions: dict[str, np.ndarray] = {}
    for target_name in ["a0", "a1"]:
        model = make_pipeline(float(config["ridge_alpha"]))
        model.fit(x_train, train[target_name].to_numpy(dtype=np.float64))
        predictions[target_name] = model.predict(x_validation)
    generator = np.random.default_rng(int(config["shuffle_seed"]))
    shuffled_pairs = train[["a0", "a1"]].to_numpy(dtype=np.float64)[
        generator.permutation(len(train))
    ]
    shuffled_predictions: dict[str, np.ndarray] = {}
    for column_index, target_name in enumerate(["a0", "a1"]):
        model = make_pipeline(float(config["ridge_alpha"]))
        model.fit(x_train, shuffled_pairs[:, column_index])
        shuffled_predictions[target_name] = model.predict(x_validation)
    coefficients = validation[["well_id", "fold"]].copy()
    coefficients["pred_a0"] = predictions["a0"]
    coefficients["pred_a1"] = predictions["a1"]
    coefficients["shuffle_a0"] = shuffled_predictions["a0"]
    coefficients["shuffle_a1"] = shuffled_predictions["a1"]
    coefficients.to_csv(output_dir / "outer0_predicted_coefficients.csv", index=False)
    _, metrics = score_paths(outer, coefficients, output_dir)
    true_coefficients = build_coefficient_targets(outer)
    audit = coefficients.merge(
        true_coefficients, on=["well_id", "fold"], validate="one_to_one"
    )
    audit.to_csv(output_dir / "outer0_coefficient_audit.csv", index=False)
    metrics["a0_correlation"] = float(np.corrcoef(audit["pred_a0"], audit["a0"])[0, 1])
    metrics["a1_correlation"] = float(np.corrcoef(audit["pred_a1"], audit["a1"])[0, 1])
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "feature_list.json", {"feature_count": 32, "features": FEATURE_NAMES})
    features.to_parquet(output_dir / "well_features.parquet", index=False)
    write_json(output_dir / "metrics_outer0.json", metrics)
    write_json(output_dir / "runtime.json", {"seconds": time.perf_counter() - start})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
