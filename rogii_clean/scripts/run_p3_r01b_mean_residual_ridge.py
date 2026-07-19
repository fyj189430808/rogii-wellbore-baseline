"""P3-R01b：复用 R01a 严格嵌套预测，拟合整井平均残差 Ridge10。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_r01a_nested_linear_residual import (  # noqa: E402
    build_well_features,
    load_feature_table,
)
from scripts.run_simple_lgbm_cv import write_json  # noqa: E402
from src.p3_r01a_linear_residual import paired_well_bootstrap_delta  # noqa: E402


EXPERIMENT_ID = "P3_R01b_mean_residual_ridge_v1"
R01A_DIR = CLEAN_ROOT / "artifacts/P3_R01a_nested_linear_residual_v1"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
OUTER_FOLD = 0
INNER_FOLDS = (1, 2, 3, 4)
RIDGE_ALPHA = 10.0
SHUFFLE_SEED = 20260719
EXPECTED_TRAIN_WELLS = 526
EXPECTED_VALIDATION_WELLS = 131
EXPECTED_VALIDATION_ROWS = 651_881
OUTER_FEATURE_COLUMNS = ["well_id", "fold", "row_index", "md", "pred_tvt"]
SOURCE_PREDICTION_PATHS = {
    "outer0": Path("base_models/outer_0/base/fold_0/predictions.parquet"),
    **{
        f"inner{fold}": Path(
            f"base_models/outer_0/inner_oof/fold_{fold}/predictions.parquet"
        )
        for fold in INNER_FOLDS
    },
}
SOURCE_PREDICTION_SHA256 = {
    "outer0": "cb0d1b778a2193507204fd6b104efbdb4fd44f20a4a96d3f9eff139759df9bad",
    "inner1": "3e87cf380084fab3eb8b70487ca58f1b82158f68848ed68be842fda5a24d5a8c",
    "inner2": "0937be78c54a0357034ff39fc90b887e832bdf3003af95f7cb04141130dfecab",
    "inner3": "3032458872d966ff166782a621e7afeb5799be12623edf006be42433123d6942",
    "inner4": "1aec1c40e3e0e510bd6deda9d7ea0e77cf179ee1146e4850e08d6d108e7447ff",
}
FEATURE_NAMES = [
    "hidden_rows_log",
    "md_span_log",
    "gr_observed_fraction",
    "azimuth_sin",
    "azimuth_cos",
    "dz_per_md",
    "sc_trust_mean",
    "pf128_mean_delta_end",
    "pf128_scale_3_delta_end",
    "pf128_scale_5_delta_end",
    "pf128_scale_8_delta_end",
    "pf128_scale_12_delta_end",
    "pf_mean_slope",
    "pf_endpoint_range",
    "pf_slope_range",
    "pf_beam_end_gap",
    "pf_beam_second_half_growth",
    "base_u_slope_100",
    "base_u_slope_200",
    "base_u_slope_500",
    "base_u_slope_100_minus_500",
]


def read_outer_feature_predictions(path: Path) -> pd.DataFrame:
    """特征阶段只读取无标签 outer0 预测列。"""

    return pd.read_parquet(path, columns=OUTER_FEATURE_COLUMNS)


def read_outer_scoring_predictions(
    path: Path,
    feature_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """预测完成后读取标签，并核对其行键及基础预测未发生变化。"""

    scoring = pd.read_parquet(path)
    required = {*OUTER_FEATURE_COLUMNS, "target_tvt"}
    if missing := required.difference(scoring.columns):
        raise RuntimeError(f"outer0 评分预测缺列：{sorted(missing)}")
    keys = ["well_id", "row_index"]
    feature = feature_predictions[OUTER_FEATURE_COLUMNS].copy()
    feature["well_id"] = feature["well_id"].astype(str)
    scoring["well_id"] = scoring["well_id"].astype(str)
    if feature.duplicated(keys).any() or scoring.duplicated(keys).any():
        raise RuntimeError("outer0 特征/评分预测行键重复")
    audit = feature.merge(
        scoring[[*keys, "fold", "md", "pred_tvt"]],
        on=keys,
        how="outer",
        suffixes=("_feature", "_score"),
        indicator=True,
        validate="one_to_one",
    )
    if len(audit) != len(feature) or not audit["_merge"].eq("both").all():
        raise RuntimeError("outer0 二次读取的行键与特征阶段不一致")
    for name in ["fold", "md", "pred_tvt"]:
        left = audit[f"{name}_feature"].to_numpy(dtype=np.float64)
        right = audit[f"{name}_score"].to_numpy(dtype=np.float64)
        if not np.array_equal(left, right):
            raise RuntimeError(f"outer0 二次读取的 {name} 与特征阶段不一致")
    return scoring


def verify_prediction_sources(
    source_dir: Path,
    source_paths: dict[str, Path] = SOURCE_PREDICTION_PATHS,
    expected_hashes: dict[str, str] = SOURCE_PREDICTION_SHA256,
) -> dict[str, str]:
    """逐字节验证严格嵌套预测来源，任一不符立即停止。"""

    if set(source_paths) != set(expected_hashes):
        raise RuntimeError("预测来源路径与 SHA256 清单不一致")
    actual_hashes: dict[str, str] = {}
    for name, relative_path in source_paths.items():
        path = source_dir / relative_path
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except FileNotFoundError as exc:
            raise RuntimeError(f"{name} 预测来源不存在：{path}") from exc
        actual = digest.hexdigest()
        if actual != expected_hashes[name]:
            raise RuntimeError(
                f"{name} 预测来源 SHA256 不符：expected={expected_hashes[name]} actual={actual}"
            )
        actual_hashes[name] = actual
    return actual_hashes


def fit_mean_residual_target(predictions: pd.DataFrame) -> pd.DataFrame:
    """把逐行严格 OOF 预测转换成一井一行的平均残差目标。"""

    required = {"well_id", "fold", "pred_tvt", "target_tvt"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"预测表缺列：{sorted(missing)}")
    if predictions.empty:
        raise ValueError("预测表不能为空")
    if predictions.groupby("well_id", sort=False)["fold"].nunique().gt(1).any():
        raise ValueError("同一口井出现多个 fold")
    pred = predictions["pred_tvt"].to_numpy(dtype=np.float64)
    target = predictions["target_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("预测或目标含 NaN/Inf")
    working = predictions[["well_id", "fold"]].copy()
    working["mean_residual"] = target - pred
    return (
        working.groupby(["well_id", "fold"], sort=False, as_index=False)[
            "mean_residual"
        ]
        .mean()
        .astype({"well_id": str, "fold": np.int64, "mean_residual": np.float64})
    )


def apply_mean_residual(
    predictions: pd.DataFrame,
    corrections: pd.DataFrame,
    correction_column: str = "pred_mean_residual",
    output_column: str = "corrected_tvt",
) -> pd.DataFrame:
    """对一口井的每一行加同一个预测平均残差。"""

    required_prediction = {"well_id", "fold", "pred_tvt"}
    required_correction = {"well_id", "fold", correction_column}
    if missing := required_prediction.difference(predictions.columns):
        raise ValueError(f"逐行预测表缺列：{sorted(missing)}")
    if missing := required_correction.difference(corrections.columns):
        raise ValueError(f"井级修正表缺列：{sorted(missing)}")
    left = predictions.copy()
    right = corrections[["well_id", "fold", correction_column]].copy()
    left["well_id"] = left["well_id"].astype(str)
    right["well_id"] = right["well_id"].astype(str)
    merged = left.merge(
        right,
        on=["well_id", "fold"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if merged[correction_column].isna().any():
        raise ValueError("部分预测行没有对应的井级平均残差")
    merged[output_column] = (
        merged["pred_tvt"].to_numpy(dtype=np.float64)
        + merged[correction_column].to_numpy(dtype=np.float64)
    )
    return merged


def shuffle_mean_residual_targets(
    targets: pd.DataFrame,
    seed: int = SHUFFLE_SEED,
) -> pd.DataFrame:
    """固定井身份，仅打乱井与平均残差目标的对应关系。"""

    required = {"well_id", "fold", "mean_residual"}
    if missing := required.difference(targets.columns):
        raise ValueError(f"平均残差目标表缺列：{sorted(missing)}")
    if len(targets) < 2:
        raise ValueError("至少需要两口井才能打乱目标")
    output = targets.copy().reset_index(drop=True)
    values = output["mean_residual"].to_numpy(dtype=np.float64)
    permutation = np.random.default_rng(int(seed)).permutation(len(output))
    if np.array_equal(permutation, np.arange(len(output))):
        permutation = np.roll(permutation, 1)
    output["mean_residual"] = values[permutation]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r01a-dir", type=Path, default=R01A_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _assert_prediction_keys(
    name: str,
    predictions: pd.DataFrame,
    table: pd.DataFrame,
    registry: pd.DataFrame,
    expected_wells: set[str],
) -> None:
    required = {"well_id", "fold", "row_index", "pred_tvt"}
    if missing := required.difference(predictions.columns):
        raise RuntimeError(f"{name} 缺列：{sorted(missing)}")
    predictions["well_id"] = predictions["well_id"].astype(str)
    if predictions.duplicated(["well_id", "row_index"]).any():
        raise RuntimeError(f"{name} 行键重复")
    actual_wells = set(predictions["well_id"])
    if actual_wells != expected_wells:
        raise RuntimeError(
            f"{name} 井集合不符：expected={len(expected_wells)} actual={len(actual_wells)}"
        )
    registry_fold = (
        registry[["well_id", "fold"]]
        .assign(well_id=lambda frame: frame["well_id"].astype(str))
        .drop_duplicates()
    )
    if registry_fold["well_id"].duplicated().any():
        raise RuntimeError("registry 中同一口井出现多个 fold")
    observed_fold = predictions[["well_id", "fold"]].drop_duplicates()
    if observed_fold["well_id"].duplicated().any():
        raise RuntimeError(f"{name} 同一口井出现多个 prediction fold")
    fold_audit = observed_fold.merge(
        registry_fold.rename(columns={"fold": "registry_fold"}),
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    if fold_audit["registry_fold"].isna().any() or not np.array_equal(
        fold_audit["fold"].to_numpy(dtype=np.int64),
        fold_audit["registry_fold"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError(f"{name} prediction fold 与 registry fold 不一致")
    expected_keys = table.loc[
        table["well_id"].astype(str).isin(expected_wells), ["well_id", "row_index"]
    ].copy()
    expected_keys["well_id"] = expected_keys["well_id"].astype(str)
    actual_keys = predictions[["well_id", "row_index"]]
    if len(actual_keys) != len(expected_keys) or not pd.MultiIndex.from_frame(
        actual_keys
    ).equals(pd.MultiIndex.from_frame(expected_keys)):
        expected_index = pd.MultiIndex.from_frame(expected_keys)
        actual_index = pd.MultiIndex.from_frame(actual_keys)
        if set(expected_index) != set(actual_index):
            raise RuntimeError(f"{name} 未精确覆盖开发特征表行键")


def _assert_one_row_per_expected_well(
    name: str,
    frame: pd.DataFrame,
    expected_wells: set[str],
) -> None:
    if "well_id" not in frame.columns:
        raise RuntimeError(f"{name} 缺少 well_id")
    actual = frame["well_id"].astype(str)
    actual_wells = set(actual)
    if (
        len(frame) != len(expected_wells)
        or actual.duplicated().any()
        or actual_wells != expected_wells
    ):
        raise RuntimeError(
            f"{name} 井数/唯一性不符：expected={len(expected_wells)} actual={len(actual_wells)}"
        )


def _median_impute(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_features[FEATURE_NAMES].copy()
    validation = validation_features[FEATURE_NAMES].copy()
    for name in FEATURE_NAMES:
        median = float(train[name].median())
        if not np.isfinite(median):
            median = 0.0
        train[name] = train[name].fillna(median)
        validation[name] = validation[name].fillna(median)
    if not np.isfinite(train.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("训练井级特征中位数填充后仍含 NaN/Inf")
    if not np.isfinite(validation.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("验证井级特征中位数填充后仍含 NaN/Inf")
    return train, validation


def _score(
    outer_predictions: pd.DataFrame,
    corrections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    paths = apply_mean_residual(outer_predictions, corrections)
    paths = apply_mean_residual(
        paths,
        corrections,
        correction_column="shuffle_mean_residual",
        output_column="shuffle_tvt",
    )
    paths = apply_mean_residual(
        paths,
        corrections,
        correction_column="oracle_mean_residual",
        output_column="oracle_tvt",
    )
    paths["base_tvt"] = paths["pred_tvt"].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    for well_id, well in paths.groupby("well_id", sort=False):
        target = well["target_tvt"].to_numpy(dtype=np.float64)
        row: dict[str, float | int | str] = {
            "well_id": str(well_id),
            "fold": int(well["fold"].iloc[0]),
            "rows": int(len(well)),
        }
        for label, column in [
            ("base", "base_tvt"),
            ("candidate", "corrected_tvt"),
            ("shuffle", "shuffle_tvt"),
            ("oracle", "oracle_tvt"),
        ]:
            error = target - well[column].to_numpy(dtype=np.float64)
            row[f"{label}_sse"] = float(np.sum(error**2))
            row[f"{label}_rmse"] = float(np.sqrt(np.mean(error**2)))
        rows.append(row)
    per_well = pd.DataFrame(rows)
    total_rows = int(per_well["rows"].sum())

    def micro(label: str) -> float:
        return float(np.sqrt(per_well[f"{label}_sse"].sum() / total_rows))

    base_rmse = micro("base")
    corrected_rmse = micro("candidate")
    metrics: dict[str, object] = {
        "outer_fold": OUTER_FOLD,
        "wells": int(len(per_well)),
        "rows": total_rows,
        "feature_count": len(FEATURE_NAMES),
        "base_rmse": base_rmse,
        "corrected_rmse": corrected_rmse,
        "improvement_ft": base_rmse - corrected_rmse,
        "shuffle_rmse": micro("shuffle"),
        "oracle_rmse": micro("oracle"),
        "well_win_rate": float(
            np.mean(per_well["candidate_rmse"] < per_well["base_rmse"])
        ),
        "paired_well_bootstrap": paired_well_bootstrap_delta(
            per_well, n_resamples=2000, seed=42
        ),
    }
    return paths, per_well, metrics


def main() -> None:
    args = parse_args()
    source_dir = args.r01a_dir.resolve()
    output_dir = args.output_dir.resolve()
    print(
        "R01b：526 口训练井、131 口 outer0 井；主要耗时为特征表读取与 Ridge；"
        f"输出 {output_dir}；中断后重跑本命令（不训练 LightGBM）。",
        flush=True,
    )
    start = time.perf_counter()
    source_hashes = verify_prediction_sources(source_dir)
    table, registry, shadow_ids = load_feature_table()
    table = table.copy()
    registry = registry.copy()
    table["well_id"] = table["well_id"].astype(str)
    registry["well_id"] = registry["well_id"].astype(str)
    train_wells = set(registry.loc[registry["fold"].astype(int).isin(INNER_FOLDS), "well_id"])
    validation_wells = set(registry.loc[registry["fold"].astype(int).eq(OUTER_FOLD), "well_id"])
    if len(train_wells) != EXPECTED_TRAIN_WELLS or len(validation_wells) != EXPECTED_VALIDATION_WELLS:
        raise RuntimeError("outer0 开发井数不是冻结的 526/131")
    if train_wells.intersection(shadow_ids) or validation_wells.intersection(shadow_ids):
        raise RuntimeError("严格嵌套井集合与影子井相交")

    inner_parts: list[pd.DataFrame] = []
    for fold in INNER_FOLDS:
        part = pd.read_parquet(
            source_dir / f"base_models/outer_0/inner_oof/fold_{fold}/predictions.parquet"
        )
        if set(part["fold"].astype(int)) != {fold}:
            raise RuntimeError(f"inner fold {fold} 预测含其他 fold")
        inner_parts.append(part)
    inner_predictions = pd.concat(inner_parts, ignore_index=True)
    outer_path = source_dir / SOURCE_PREDICTION_PATHS["outer0"]
    outer_feature_predictions = read_outer_feature_predictions(outer_path)
    if set(outer_feature_predictions["fold"].astype(int)) != {OUTER_FOLD}:
        raise RuntimeError("outer0 预测含其他 fold")
    _assert_prediction_keys(
        "inner OOF", inner_predictions, table, registry, train_wells
    )
    _assert_prediction_keys(
        "outer0", outer_feature_predictions, table, registry, validation_wells
    )
    if len(outer_feature_predictions) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("outer0 评价行数不是冻结的 651881")

    train_targets = fit_mean_residual_target(inner_predictions)
    _assert_one_row_per_expected_well("train targets", train_targets, train_wells)
    train_features = build_well_features(
        table.loc[table["well_id"].isin(train_wells)], inner_predictions
    )
    validation_features = build_well_features(
        table.loc[table["well_id"].isin(validation_wells)],
        outer_feature_predictions,
    )
    _assert_one_row_per_expected_well("train features", train_features, train_wells)
    _assert_one_row_per_expected_well(
        "validation features", validation_features, validation_wells
    )
    actual_features = [
        name for name in train_features.columns if name not in {"well_id", "fold"}
    ]
    if actual_features != FEATURE_NAMES:
        raise RuntimeError(f"井级特征不是冻结的 21 列：{actual_features}")
    train = train_features.merge(
        train_targets, on=["well_id", "fold"], validate="one_to_one"
    )
    _assert_one_row_per_expected_well("train", train, train_wells)
    x_train, x_validation = _median_impute(train, validation_features)
    y_train = train["mean_residual"].to_numpy(dtype=np.float64)
    model = Pipeline(
        [("standardize", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))]
    )
    model.fit(x_train, y_train)
    shuffled_targets = shuffle_mean_residual_targets(
        train[["well_id", "fold", "mean_residual"]], seed=SHUFFLE_SEED
    )
    shuffle_model = Pipeline(
        [("standardize", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))]
    )
    shuffle_model.fit(
        x_train, shuffled_targets["mean_residual"].to_numpy(dtype=np.float64)
    )
    corrections = validation_features[["well_id", "fold"]].copy()
    corrections["pred_mean_residual"] = model.predict(x_validation)
    corrections["shuffle_mean_residual"] = shuffle_model.predict(x_validation)
    outer_predictions = read_outer_scoring_predictions(
        outer_path, outer_feature_predictions
    )
    validation_targets = fit_mean_residual_target(outer_predictions)
    _assert_one_row_per_expected_well(
        "validation targets", validation_targets, validation_wells
    )
    corrections = corrections.merge(
        validation_targets.rename(columns={"mean_residual": "oracle_mean_residual"}),
        on=["well_id", "fold"],
        validate="one_to_one",
    )
    paths, per_well, metrics = _score(outer_predictions, corrections)
    metrics["target_correlation"] = float(
        np.corrcoef(
            corrections["pred_mean_residual"], corrections["oracle_mean_residual"]
        )[0, 1]
    )
    metrics["shuffle_improvement_ft"] = float(
        metrics["base_rmse"] - metrics["shuffle_rmse"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths.to_parquet(output_dir / "outer0_predictions.parquet", index=False)
    per_well.to_csv(output_dir / "outer0_per_well.csv", index=False)
    corrections.to_csv(output_dir / "outer0_mean_residual_audit.csv", index=False)
    write_json(output_dir / "feature_list.json", {"features": FEATURE_NAMES})
    write_json(
        output_dir / "parameters.json",
        {"ridge_alpha": RIDGE_ALPHA, "shuffle_seed": SHUFFLE_SEED},
    )
    write_json(output_dir / "metrics_outer0.json", metrics)
    write_json(
        output_dir / "runtime_outer0.json",
        {
            "seconds": time.perf_counter() - start,
            "source": str(source_dir),
            "source_prediction_sha256": source_hashes,
        },
    )
    (output_dir / "conclusion.md").write_text(
        "# P3-R01b 结论\n\n"
        "数据直接证明的事实：见 metrics_outer0.json。\n\n"
        "基于事实的合理推断：仅在成功门槛通过时成立。\n\n"
        "仍然没有验证的猜测：其他 outer folds 的泛化。\n\n"
        "当前实验只能否定的具体实现：固定 21 特征 + Ridge10 的平均残差修正。\n\n"
        "下一步最便宜的验证：按实验卡门槛决定是否停止。\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
