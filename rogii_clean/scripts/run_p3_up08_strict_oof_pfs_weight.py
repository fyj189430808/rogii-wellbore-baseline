"""UP08：只用既有 outer0 嵌套预测训练严格 OOF 的 PFS 修正权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_r01a_linear_residual import paired_well_bootstrap_delta
from src.p3_up01_robust_u_projection import robust_polynomial_projection
from src.p3_up08_strict_oof_pfs_weight import build_well_features, optimal_alpha, shrink_alpha


EXPERIMENT_ID = "P3_UP08_strict_oof_pfs_weight_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up08_strict_oof_pfs_weight_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
NESTED_ROOT = CLEAN_ROOT / "artifacts/P3_R01a_nested_linear_residual_v1/base_models/outer_0"
PFS_CACHE_DIR = CLEAN_ROOT / "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_cache"
PFS_RUNTIME_DIR = CLEAN_ROOT / "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_runtime"
PFS_DEVELOPMENT_MANIFEST = CLEAN_ROOT / "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/per_well_runtime_all.csv"
B00_FEATURE_CACHE = CLEAN_ROOT / "artifacts/B00_simple_lgbm_v1/feature_cache.parquet"
KEYS = ["well_id", "fold", "row_index"]
PFS_FINGERPRINT = "4bf62f495ef530bb3ca53d65d15475f7bc3784fe5abaebbbb1053584704c4480"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-wells", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_outer0_nested_prediction_path(path: str | Path) -> Path:
    value = Path(path)
    normalized = value.as_posix()
    if "P3_R01a_nested_linear_residual_v1" not in normalized or "/outer_0/" not in normalized:
        raise ValueError("UP08 只接受 P3_R01a outer_0 的嵌套预测缓存")
    if value.name != "predictions.parquet":
        raise ValueError("嵌套缓存必须是 predictions.parquet")
    return value


def validate_nested_partition(
    inner: pd.DataFrame,
    outer: pd.DataFrame,
    *,
    forbidden_well_ids: set[str],
    allowed_well_ids: set[str] | None = None,
) -> None:
    for name, frame in (("inner", inner), ("outer", outer)):
        if not {"well_id", "fold"}.issubset(frame.columns):
            raise ValueError(f"{name} 缺少 well_id/fold")
        if frame.empty:
            raise ValueError(f"{name} 预测为空")
    inner_ids = set(inner["well_id"].astype(str))
    outer_ids = set(outer["well_id"].astype(str))
    if (inner_ids | outer_ids).intersection(forbidden_well_ids):
        raise ValueError("嵌套预测包含 shadow 井")
    if allowed_well_ids is not None and not (inner_ids | outer_ids).issubset(allowed_well_ids):
        raise ValueError("嵌套预测包含不在冻结 PFS 开发井清单中的井")
    if inner_ids.intersection(outer_ids):
        raise ValueError("outer 训练井与验证井发生交叉")
    if set(outer["fold"].astype(int)) != {0}:
        raise ValueError("outer 验证预测必须严格属于 fold 0")
    if not set(inner["fold"].astype(int)).issubset({1, 2, 3, 4}):
        raise ValueError("inner OOF 只能来自 folds 1-4")


def outer_train_preprocess(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if list(train.columns) != list(validation.columns):
        raise ValueError("outer-train 与验证特征列不一致")
    medians: dict[str, float] = {}
    train_filled = train.copy()
    validation_filled = validation.copy()
    for name in train.columns:
        median = float(train[name].median())
        if not np.isfinite(median):
            median = 0.0
        medians[name] = median
        train_filled[name] = train[name].replace([np.inf, -np.inf], np.nan).fillna(median)
        validation_filled[name] = validation[name].replace([np.inf, -np.inf], np.nan).fillna(median)
    if not np.isfinite(train_filled.to_numpy(dtype=float)).all() or not np.isfinite(validation_filled.to_numpy(dtype=float)).all():
        raise ValueError("outer-train 中位数填补后仍有非有限特征")
    scaler = StandardScaler().fit(train_filled)
    return (
        pd.DataFrame(scaler.transform(train_filled), columns=train.columns, index=train.index),
        pd.DataFrame(scaler.transform(validation_filled), columns=train.columns, index=validation.index),
        {"medians": medians, "scaler_mean": dict(zip(train.columns, scaler.mean_.tolist())), "scaler_scale": dict(zip(train.columns, scaler.scale_.tolist()))},
    )


def shuffle_training_targets(targets: np.ndarray, seed: int = 42) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    return np.random.default_rng(seed).permutation(values)


def validate_nested_runtime_fingerprint(runtime: dict[str, Any], expected: str) -> None:
    if str(runtime.get("fingerprint")) != str(expected):
        raise RuntimeError("嵌套 runtime fingerprint 不符合冻结合同")


def load_frozen_pfs_development_wells(config: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    manifest_path = CLEAN_ROOT / str(config.get("pfs_frozen_development_manifest", PFS_DEVELOPMENT_MANIFEST.relative_to(CLEAN_ROOT)))
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(
        manifest_path,
        usecols=["well_id", "hidden_tvt_read", "experiment_fingerprint"],
        dtype={"well_id": str},
    )
    expected_fingerprint = str(config["pfs_fingerprint"])
    expected_wells = int(config["expected_development_wells"])
    if len(manifest) != expected_wells or manifest["well_id"].nunique() != expected_wells:
        raise RuntimeError("冻结 PFS 开发井清单不是 657 个唯一井")
    if manifest["hidden_tvt_read"].astype(bool).any():
        raise RuntimeError("冻结 PFS 开发井清单表明读取过隐藏 TVT")
    if set(manifest["experiment_fingerprint"].astype(str)) != {expected_fingerprint}:
        raise RuntimeError("冻结 PFS 开发井清单 fingerprint 不匹配")
    return set(manifest["well_id"].astype(str)), {
        "path": str(manifest_path.relative_to(CLEAN_ROOT)),
        "sha256": sha256(manifest_path),
        "wells": expected_wells,
        "hidden_tvt_read": False,
        "fingerprint": expected_fingerprint,
        "columns_read": ["well_id", "hidden_tvt_read", "experiment_fingerprint"],
    }


def _prediction_paths() -> tuple[Path, list[Path]]:
    outer = NESTED_ROOT / "base/fold_0/predictions.parquet"
    inner = [NESTED_ROOT / f"inner_oof/fold_{fold}/predictions.parquet" for fold in range(1, 5)]
    return outer, inner


def _read_runtime(path: Path, expected_fold: int, config: dict[str, Any]) -> dict[str, Any]:
    runtime = json.loads(path.read_text(encoding="utf-8"))
    contract = config["model_contract"]
    if int(runtime.get("fold", -1)) != expected_fold:
        raise RuntimeError(f"嵌套 runtime fold 不匹配：{path}")
    if int(runtime.get("features", -1)) != int(contract["features"]) or int(runtime.get("trees", -1)) != int(contract["trees"]):
        raise RuntimeError(f"嵌套 runtime 的 features/trees 不符合 P3B00：{path}")
    return runtime


def load_nested_predictions(config: dict[str, Any], include_outer_target: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outer_path, inner_paths = _prediction_paths()
    all_paths = [outer_path, *inner_paths]
    frozen_well_ids, frozen_manifest_audit = load_frozen_pfs_development_wells(config)
    runtimes: list[dict[str, Any]] = []
    for expected_fold, path in enumerate(all_paths):
        require_outer0_nested_prediction_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        runtime = _read_runtime(path.parent / "runtime.json", expected_fold, config)
        validate_nested_runtime_fingerprint(runtime, str(config["expected_nested_fingerprint"]))
        runtimes.append(runtime)
    fingerprints = {str(item.get("fingerprint")) for item in runtimes}
    if len(fingerprints) != 1:
        raise RuntimeError("五个嵌套 runtime 的 fingerprint 不一致")
    outer_columns = ["well_id", "fold", "row_index", "md", "pred_tvt"]
    if include_outer_target:
        outer_columns.append("target_tvt")
    outer = pd.read_parquet(outer_path, columns=outer_columns)
    inner = pd.concat([
        pd.read_parquet(path, columns=["well_id", "fold", "row_index", "md", "pred_tvt", "target_tvt"])
        for path in inner_paths
    ], ignore_index=True)
    for frame in (inner, outer):
        frame["well_id"] = frame["well_id"].astype(str)
        if frame.duplicated(KEYS).any():
            raise RuntimeError("嵌套预测存在重复行键")
    validate_nested_partition(
        inner,
        outer,
        forbidden_well_ids=set(),
        allowed_well_ids=frozen_well_ids,
    )
    if outer["well_id"].nunique() != int(config["expected_outer_wells"]) or inner["well_id"].nunique() != int(config["expected_inner_wells"]):
        raise RuntimeError("嵌套预测井数不符合 131 outer + 526 inner 合同")
    nested_well_ids = set(inner["well_id"]) | set(outer["well_id"])
    if nested_well_ids != frozen_well_ids:
        raise RuntimeError("嵌套预测井集合不等于冻结 PFS 的 657 开发井清单；拒绝可能的 shadow 输入")
    for runtime, frame in zip(runtimes, [outer, *[inner.loc[inner["fold"].eq(fold)] for fold in range(1, 5)]]):
        if int(runtime["validation_rows"]) != len(frame) or int(runtime["validation_wells"]) != frame["well_id"].nunique():
            raise RuntimeError("嵌套 runtime 行/井数与 predictions 不一致")
    return inner, outer, {
        "runtime_fingerprint": next(iter(fingerprints)),
        "expected_runtime_fingerprint": str(config["expected_nested_fingerprint"]),
        "paths": [str(p.relative_to(CLEAN_ROOT)) for p in all_paths],
        "sha256": {str(p.relative_to(CLEAN_ROOT)): sha256(p) for p in all_paths},
        "frozen_pfs_development_manifest": frozen_manifest_audit,
    }


def load_b00_context(well_ids: set[str], keys: pd.DataFrame) -> pd.DataFrame:
    table = ds.dataset(B00_FEATURE_CACHE, format="parquet").to_table(
        columns=["well_id", "row_index", "z_current", "gr_missing"],
        filter=ds.field("well_id").isin(sorted(well_ids)),
    ).to_pandas()
    table["well_id"] = table["well_id"].astype(str)
    result = keys[["well_id", "row_index"]].merge(table, on=["well_id", "row_index"], how="inner", validate="one_to_one")
    if len(result) != len(keys) or result.duplicated(["well_id", "row_index"]).any():
        raise RuntimeError("B00 feature_cache 的 z/GR 行键不完整")
    return result


def load_pfs_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    audit: dict[str, Any] = {"fingerprint": PFS_FINGERPRINT, "hidden_tvt_read": False, "files": {}}
    for (well_id, fold), frame in predictions.groupby(["well_id", "fold"], sort=True):
        cache_path = PFS_CACHE_DIR / f"{well_id}.parquet"
        runtime_path = PFS_RUNTIME_DIR / f"{well_id}.json"
        if not cache_path.exists() or not runtime_path.exists():
            raise FileNotFoundError(f"缺少 PFS cache/runtime：{well_id}")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("hidden_tvt_read") is not False or runtime.get("experiment_fingerprint") != PFS_FINGERPRINT:
            raise RuntimeError(f"PFS runtime 血缘不合格：{well_id}")
        if int(runtime.get("fold", -1)) != int(fold) or int(runtime.get("hidden_rows", -1)) != len(frame):
            raise RuntimeError(f"PFS fold/hidden_rows 不匹配：{well_id}")
        cache_hash = sha256(cache_path)
        if runtime.get("cache_sha256") != cache_hash:
            raise RuntimeError(f"PFS cache sha256 不匹配：{well_id}")
        part = pd.read_parquet(cache_path, columns=[*KEYS, "last_visible_tvt", "pfs_lag250_delta", "pfs_lag500_delta", "pfs_lag1000_delta", "_cache_fingerprint"])
        if set(part["_cache_fingerprint"].astype(str)) != {PFS_FINGERPRINT}:
            raise RuntimeError(f"PFS parquet 指纹不匹配：{well_id}")
        parts.append(part.drop(columns=["_cache_fingerprint"]))
        audit["files"][str(well_id)] = {"cache_sha256": cache_hash, "fold": int(fold), "hidden_rows": int(len(frame))}
    result = pd.concat(parts, ignore_index=True)
    result["well_id"] = result["well_id"].astype(str)
    if result.duplicated(KEYS).any() or set(map(tuple, result[KEYS].to_numpy())) != set(map(tuple, predictions[KEYS].to_numpy())):
        raise RuntimeError("PFS 与嵌套预测行键不完全一致")
    return result, audit


def add_paths_and_features(predictions: pd.DataFrame, pfs: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = predictions.merge(pfs, on=KEYS, how="inner", validate="one_to_one").merge(context, on=["well_id", "row_index"], how="inner", validate="one_to_one")
    if len(merged) != len(predictions):
        raise RuntimeError("nested/PFS/B00 不能一一对齐")
    merged["up01_tvt"] = np.nan
    merged["pfs_lag250_abs_tvt"] = merged["last_visible_tvt"] + merged["pfs_lag250_delta"]
    merged["pfs_lag500_abs_tvt"] = merged["last_visible_tvt"] + merged["pfs_lag500_delta"]
    merged["pfs_lag1000_abs_tvt"] = merged["last_visible_tvt"] + merged["pfs_lag1000_delta"]
    feature_rows: list[dict[str, Any]] = []
    for (well_id, fold), frame in merged.groupby(["well_id", "fold"], sort=True):
        ordered = frame.sort_values("row_index")
        md = ordered["md"].to_numpy(dtype=float)
        pred = ordered["pred_tvt"].to_numpy(dtype=float)
        z = ordered["z_current"].to_numpy(dtype=float)
        projected_u = robust_polynomial_projection(md, pred + z, degree=2)
        up01 = pred + 0.5 * (projected_u - (pred + z))
        merged.loc[ordered.index, "up01_tvt"] = up01
        runtime = json.loads((PFS_RUNTIME_DIR / f"{well_id}.json").read_text(encoding="utf-8"))
        runtime["gr_observed_fraction"] = float(1.0 - ordered["gr_missing"].astype(float).mean())
        paths = {lag: ordered[f"pfs_{lag}_abs_tvt"].to_numpy(dtype=float) for lag in ("lag250", "lag500", "lag1000")}
        feature_rows.append({"well_id": str(well_id), "fold": int(fold), **build_well_features(md, pred, paths, runtime)})
    if not np.isfinite(merged[["up01_tvt", "pfs_lag1000_abs_tvt"]].to_numpy(dtype=float)).all():
        raise RuntimeError("UP01/PFS 路径出现非有限值")
    return merged, pd.DataFrame(feature_rows)


def build_targets(train_rows: pd.DataFrame, train_features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (well_id, fold), frame in train_rows.groupby(["well_id", "fold"], sort=True):
        ordered = frame.sort_values("row_index")
        direction = ordered["pfs_lag1000_abs_tvt"].to_numpy(float) - ordered["pred_tvt"].to_numpy(float)
        alpha, energy = optimal_alpha(ordered["target_tvt"].to_numpy(float), ordered["up01_tvt"].to_numpy(float), direction)
        records.append({"well_id": str(well_id), "fold": int(fold), "alpha_raw": alpha, "direction_energy": energy, "rows": int(len(ordered)), "raw_sample_weight": float(np.mean(direction * direction) * len(ordered))})
    targets = pd.DataFrame(records).merge(train_features, on=["well_id", "fold"], validate="one_to_one")
    raw = targets["raw_sample_weight"].to_numpy(float)
    normalized = raw / float(np.mean(raw))
    cap = 10.0 * float(np.median(normalized))
    targets["sample_weight"] = np.minimum(normalized, cap)
    return targets


def _score(outer: pd.DataFrame, alpha_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scored = outer.merge(alpha_table, on=["well_id", "fold"], validate="many_to_one")
    scored["fixed_alpha025_tvt"] = scored["up01_tvt"] + 0.25 * (scored["pfs_lag1000_abs_tvt"] - scored["pred_tvt"])
    direction = scored["pfs_lag1000_abs_tvt"] - scored["pred_tvt"]
    scored["predicted_alpha_tvt"] = scored["up01_tvt"] + scored["predicted_alpha"] * direction
    scored["shuffle_alpha_tvt"] = scored["up01_tvt"] + scored["shuffle_alpha"] * direction
    rows: list[dict[str, Any]] = []
    true_alpha: list[dict[str, Any]] = []
    for (well_id, fold), frame in scored.groupby(["well_id", "fold"], sort=True):
        target = frame["target_tvt"].to_numpy(float)
        record: dict[str, Any] = {"well_id": str(well_id), "fold": int(fold), "rows": len(frame)}
        for name, column in (("fixed", "fixed_alpha025_tvt"), ("predicted", "predicted_alpha_tvt"), ("shuffle", "shuffle_alpha_tvt")):
            error = target - frame[column].to_numpy(float)
            record[f"{name}_sse"] = float(np.dot(error, error))
            record[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
        alpha, _ = optimal_alpha(target, frame["up01_tvt"].to_numpy(float), (frame["pfs_lag1000_abs_tvt"] - frame["pred_tvt"]).to_numpy(float))
        true_alpha.append({"well_id": str(well_id), "fold": int(fold), "true_alpha_diagnostic": alpha})
        rows.append(record)
    per_well = pd.DataFrame(rows)
    total_rows = int(per_well["rows"].sum())
    rmse = {name: float(np.sqrt(per_well[f"{name}_sse"].sum() / total_rows)) for name in ("fixed", "predicted", "shuffle")}
    bootstrap_input = per_well.rename(columns={"fixed_sse": "base_sse", "predicted_sse": "candidate_sse"})
    return scored, per_well, {"rows": total_rows, "rmse": rmse, "bootstrap": paired_well_bootstrap_delta(bootstrap_input, 2000, 42), "true_alpha": pd.DataFrame(true_alpha)}


def conclusion(metrics: dict[str, Any]) -> str:
    gate = metrics["gate"]
    return "\n".join([
        "# P3-UP08 严格 OOF PFS 权重结论", "", "数据直接证明的事实：", "",
        f"- outer fold 0：{metrics['wells']} 口井、{metrics['rows']:,} 行；固定 alpha=0.25 RMSE 为 {metrics['fixed_alpha025_rmse']:.6f} ft。",
        f"- Ridge 预测 alpha 的 RMSE 为 {metrics['predicted_alpha_rmse']:.6f} ft，改善 {metrics['predicted_improvement_ft']:+.6f} ft，胜井率 {metrics['well_win_rate']:.2%}。",
        f"- shuffle 对照改善 {metrics['shuffle_improvement_ft']:+.6f} ft。", "", "基于事实的合理推断：", "",
        f"- 预注册 outer0 gate {'通过' if gate['passed'] else '未通过'}。", "", "仍然没有验证的猜测：", "", "- outer fold 0 的结果不能替代完整五折验证。", "", "当前实验只能否定的具体实现：", "", "- 未通过时，只能否定当前严格 OOF Ridge 权重实现。", "", "下一步最便宜的验证：", "", "- 仅在 gate 通过后再设计完整五折严格 OOF 验证。", ""
    ])


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    start = time.perf_counter()
    inner, outer, nested_audit = load_nested_predictions(config, include_outer_target=False)
    # inner OOF 的真值是 outer-train 的合法 alpha 训练目标；只有 outer fold0
    # 的 target_tvt 必须在三条路径预落盘之后才读取。
    all_predictions = pd.concat([inner, outer], ignore_index=True)
    if args.smoke_wells:
        selected = sorted(all_predictions["well_id"].unique())[:args.smoke_wells]
        smoke = all_predictions.loc[all_predictions["well_id"].isin(selected)].copy()
        pfs, _ = load_pfs_rows(smoke)
        context = load_b00_context(set(selected), smoke)
        paths, features = add_paths_and_features(smoke, pfs, context)
        if len(paths) != len(smoke) or not np.isfinite(features.drop(columns=["well_id", "fold"]).to_numpy(float)).all():
            raise RuntimeError("smoke shape/finite 检查失败")
        print(json.dumps({"smoke_wells": len(selected), "rows": len(smoke), "outer_target_read": False, "shadow_files_read": False}, ensure_ascii=False), flush=True)
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    pfs, pfs_audit = load_pfs_rows(all_predictions)
    context = load_b00_context(set(all_predictions["well_id"]), all_predictions)
    all_paths, all_features = add_paths_and_features(all_predictions, pfs, context)
    train_rows = all_paths.loc[all_paths["fold"].ne(0)].copy()
    outer_rows = all_paths.loc[all_paths["fold"].eq(0)].copy()
    train_features = all_features.loc[all_features["fold"].ne(0)].copy()
    outer_features = all_features.loc[all_features["fold"].eq(0)].copy()
    targets = build_targets(train_rows, train_features)
    feature_names = [name for name in train_features.columns if name not in {"well_id", "fold"}]
    x_train, x_outer, preprocess_audit = outer_train_preprocess(targets[feature_names], outer_features[feature_names])
    ridge = Ridge(alpha=float(config["ridge_alpha"])).fit(x_train, targets["alpha_raw"].to_numpy(float) - 0.25, sample_weight=targets["sample_weight"].to_numpy(float))
    shuffle_ridge = Ridge(alpha=float(config["ridge_alpha"])).fit(x_train, shuffle_training_targets(targets["alpha_raw"].to_numpy(float) - 0.25), sample_weight=targets["sample_weight"].to_numpy(float))
    alpha_table = outer_features[["well_id", "fold"]].copy()
    alpha_table["ridge_raw_alpha"] = ridge.predict(x_outer) + 0.25
    alpha_table["shuffle_raw_alpha"] = shuffle_ridge.predict(x_outer) + 0.25
    alpha_table["predicted_alpha"] = shrink_alpha(alpha_table["ridge_raw_alpha"].to_numpy(float))
    alpha_table["shuffle_alpha"] = shrink_alpha(alpha_table["shuffle_raw_alpha"].to_numpy(float))
    pretruth = outer_rows[KEYS + ["up01_tvt", "pred_tvt", "pfs_lag1000_abs_tvt"]].merge(alpha_table, on=["well_id", "fold"], validate="many_to_one")
    pretruth["fixed_alpha025_tvt"] = pretruth["up01_tvt"] + .25 * (pretruth["pfs_lag1000_abs_tvt"] - pretruth["pred_tvt"])
    pretruth["predicted_alpha_tvt"] = pretruth["up01_tvt"] + pretruth["predicted_alpha"] * (pretruth["pfs_lag1000_abs_tvt"] - pretruth["pred_tvt"])
    pretruth["shuffle_alpha_tvt"] = pretruth["up01_tvt"] + pretruth["shuffle_alpha"] * (pretruth["pfs_lag1000_abs_tvt"] - pretruth["pred_tvt"])
    pretruth.to_parquet(output / "pretruth_predictions.parquet", index=False)
    write_json(output / "config.json", config)
    write_json(output / "lineage_audit.json", {"nested": nested_audit, "pfs": pfs_audit, "b00_columns_read": ["well_id", "row_index", "z_current", "gr_missing"], "outer_fold0_target_read_before_pretruth": False, "shadow_files_read": False, "preprocess": preprocess_audit})
    targets[["well_id", "fold", "rows", "direction_energy", "alpha_raw", "raw_sample_weight", "sample_weight"]].to_csv(output / "train_targets.csv", index=False)
    alpha_table.to_csv(output / "predicted_alpha.csv", index=False)
    outer_target = pd.read_parquet(_prediction_paths()[0], columns=[*KEYS, "target_tvt"])
    outer_target["well_id"] = outer_target["well_id"].astype(str)
    # concat 保留 inner 的训练标签时，会给 outer 行带来全空 target_tvt
    # 占位列；评分前丢弃它，确保这里合入的是刚刚延后读取的 fold0 真值。
    scored_input = outer_rows.drop(columns=["target_tvt"], errors="ignore").merge(
        outer_target, on=KEYS, validate="one_to_one"
    )
    scored, per_well, summary = _score(scored_input, alpha_table)
    true_alpha = summary.pop("true_alpha")
    alpha_table = alpha_table.merge(true_alpha, on=["well_id", "fold"], validate="one_to_one")
    alpha_table.to_csv(output / "predicted_alpha.csv", index=False)
    correlation = float(alpha_table["predicted_alpha"].corr(alpha_table["true_alpha_diagnostic"]))
    fixed_rmse, predicted_rmse, shuffle_rmse = (summary["rmse"][name] for name in ("fixed", "predicted", "shuffle"))
    predicted_improvement = fixed_rmse - predicted_rmse
    shuffle_improvement = fixed_rmse - shuffle_rmse
    metrics = {"experiment_id": EXPERIMENT_ID, "outer_fold": 0, "wells": int(len(per_well)), "rows": summary["rows"], "fixed_alpha025_rmse": fixed_rmse, "predicted_alpha_rmse": predicted_rmse, "shuffle_alpha_rmse": shuffle_rmse, "predicted_improvement_ft": predicted_improvement, "shuffle_improvement_ft": shuffle_improvement, "well_win_rate": float(np.mean(per_well["predicted_rmse"] < per_well["fixed_rmse"])), "paired_well_bootstrap": summary["bootstrap"], "predicted_vs_true_outer_alpha_correlation_diagnostic": correlation, "alpha_distribution": {name: {str(q): float(alpha_table[name].quantile(q)) for q in (0, .1, .5, .9, 1)} for name in ("predicted_alpha", "shuffle_alpha")}, "alpha_clipping_fraction": {"predicted": float(np.mean(np.isin(alpha_table["predicted_alpha"], [-.25, .75]))), "shuffle": float(np.mean(np.isin(alpha_table["shuffle_alpha"], [-.25, .75])))} }
    metrics["gate"] = {"improvement_ge_010": bool(predicted_improvement >= .10), "win_rate_ge_055": bool(metrics["well_win_rate"] >= .55), "shuffle_improvement_le_0": bool(shuffle_improvement <= 0), "passed": bool(predicted_improvement >= .10 and metrics["well_win_rate"] >= .55 and shuffle_improvement <= 0)}
    scored.to_parquet(output / "predictions.parquet", index=False)
    per_well.to_csv(output / "per_well.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "runtime.json", {"experiment_id": EXPERIMENT_ID, "elapsed_seconds": time.perf_counter() - start, "base_model_retrained": False, "shadow_files_read": False, "outer_fold0_target_read_after_pretruth": True})
    (output / "conclusion.md").write_text(conclusion(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
