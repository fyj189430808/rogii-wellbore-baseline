"""运行 F03b：B00 加固定几何路径的逐行 Typewell offset 得分面特征。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
CONFIG_PATH = CLEAN_ROOT / "configs" / "f03b_geometry_path_offset_landscape_v1.json"
TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f03b_geometry_landscape_features import (  # noqa: E402
    BEST_BASIN_RADIUS_FT,
    F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    MINIMUM_VALID_PAIRS,
    NCC_WINDOW_WIDTH_FT,
    OFFSET_GRID_FT,
    SECOND_PEAK_MINIMUM_DISTANCE_FT,
    SMOOTHING_WIDTH_FT,
    SOFTMAX_TEMPERATURE,
    build_geometry_landscape_features,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS as B00_FEATURE_COLUMNS  # noqa: E402


RAW_SOURCE_CONTRACT = {
    "horizontal_pattern": "{well_id}__horizontal_well.csv",
    "typewell_pattern": "{well_id}__typewell.csv",
    "horizontal_usecols": ["MD", "Z", "GR", "TVT_input"],
    "typewell_usecols": ["TVT", "GR"],
    "candidate_path_formula": "last_visible_tvt - (Z_current - Z_visible_end)",
    "hidden_tvt_feature_access": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F03b 固定单模 LightGBM")
    parser.add_argument(
        "--mode",
        choices=["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"],
        required=True,
    )
    return parser.parse_args()


def stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config: dict) -> list[str]:
    feature_columns = [
        *B00_FEATURE_COLUMNS,
        *F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    ]
    if config["feature_columns"] != feature_columns:
        raise ValueError("F03b 配置中的特征顺序与代码不一致")
    fixed_values = {
        "offset_grid_ft": OFFSET_GRID_FT.tolist(),
        "gr_smoothing_width_ft": SMOOTHING_WIDTH_FT,
        "ncc_window_width_ft": NCC_WINDOW_WIDTH_FT,
        "minimum_valid_pairs": MINIMUM_VALID_PAIRS,
        "second_peak_minimum_distance_ft": SECOND_PEAK_MINIMUM_DISTANCE_FT,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "best_basin_radius_ft": BEST_BASIN_RADIUS_FT,
    }
    for key, expected_value in fixed_values.items():
        if config[key] != expected_value:
            raise ValueError(f"F03b 冻结参数 {key} 与代码不一致")
    return feature_columns


def build_experiment_fingerprint(
    config: dict,
    registry_hash: str,
    model_config_path: Path,
) -> tuple[str, dict]:
    baseline_metadata_path = CLEAN_ROOT / config["baseline_feature_cache_metadata"]
    manifest = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_code_sha256": file_sha256(Path(__file__).resolve()),
        "feature_code_sha256": file_sha256(
            CLEAN_ROOT / "src" / "f03b_geometry_landscape_features.py"
        ),
        "fold_registry_sha256": registry_hash,
        "baseline_cache_metadata_sha256": file_sha256(baseline_metadata_path),
        "model_config_sha256": file_sha256(model_config_path),
        "train_dir": str(TRAIN_DIR.resolve()),
        "raw_source_contract": RAW_SOURCE_CONTRACT,
    }
    payload = {
        "experiment_id": config["experiment_id"],
        "feature_columns": config["feature_columns"],
        "manifest": manifest,
    }
    return stable_hash(payload), manifest


def build_one_well_features(well_id: str) -> pd.DataFrame:
    horizontal_path = TRAIN_DIR / f"{well_id}__horizontal_well.csv"
    typewell_path = TRAIN_DIR / f"{well_id}__typewell.csv"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        raise FileNotFoundError(f"井 {well_id} 缺少 horizontal 或 typewell 文件")

    horizontal_df = pd.read_csv(
        horizontal_path,
        usecols=RAW_SOURCE_CONTRACT["horizontal_usecols"],
    )
    typewell_df = pd.read_csv(
        typewell_path,
        usecols=RAW_SOURCE_CONTRACT["typewell_usecols"],
    )
    features = build_geometry_landscape_features(horizontal_df, typewell_df)
    features.insert(0, "well_id", str(well_id))
    features["row_index"] = features["row_index"].astype(np.int32)
    for feature_name in F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS:
        features[feature_name] = features[feature_name].astype(np.float32)
    return features


def validate_landscape_table(
    table: pd.DataFrame,
    expected_rows: int | None = None,
) -> None:
    expected_columns = [
        "well_id",
        "row_index",
        *F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    ]
    if table.columns.tolist() != expected_columns:
        raise ValueError("F03b 逐行缓存列或顺序不正确")
    if expected_rows is not None and len(table) != int(expected_rows):
        raise ValueError("F03b 逐行缓存评价行数不正确")
    if table.duplicated(["well_id", "row_index"]).any():
        raise ValueError("F03b 逐行缓存含重复键")
    values = table[F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )
    if np.isinf(values).any():
        raise ValueError("F03b 新特征含 Inf")
    for feature_name in ["f03b_valid_pair_count", "f03b_valid_pair_fraction"]:
        if not np.isfinite(table[feature_name].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"F03b 支持量特征 {feature_name} 含缺失或 Inf")


def build_selected_features(
    registry: pd.DataFrame,
    selected_wells: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for well_id in selected_wells:
        frames.append(build_one_well_features(str(well_id)))
    table = pd.concat(frames, ignore_index=True)
    validate_landscape_table(table)
    return table


def load_or_build_landscape_cache(
    config: dict,
    registry: pd.DataFrame,
    artifact_dir: Path,
    fingerprint: str,
    manifest: dict,
) -> pd.DataFrame:
    cache_path = artifact_dir / "landscape_feature_cache.parquet"
    metadata_path = artifact_dir / "landscape_feature_cache.meta.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    cache_matches = (
        cache_path.is_file() and metadata.get("fingerprint") == fingerprint
    )
    if not cache_matches:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = artifact_dir / "landscape_feature_cache.tmp.parquet"
        temporary_path.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        built_rows = 0
        started = time.perf_counter()
        try:
            total_wells = len(registry)
            for well_number, row in enumerate(registry.itertuples(index=False), start=1):
                well_table = build_one_well_features(str(row.well_id))
                arrow_table = pa.Table.from_pandas(
                    well_table,
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        arrow_table.schema,
                        compression="zstd",
                    )
                writer.write_table(arrow_table)
                built_rows += len(well_table)
                if well_number % 25 == 0 or well_number == total_wells:
                    print(
                        f"F03b 逐井特征：{well_number}/{total_wells}，"
                        f"累计 {built_rows:,} 行",
                        flush=True,
                    )
        except Exception:
            if writer is not None:
                writer.close()
            temporary_path.unlink(missing_ok=True)
            raise
        if writer is None:
            raise ValueError("F03b 没有生成任何井特征")
        writer.close()
        if built_rows != int(config["expected_rows"]):
            temporary_path.unlink(missing_ok=True)
            raise ValueError("F03b 构造行数与 B00 不一致")
        temporary_path.replace(cache_path)
        write_json(
            metadata_path,
            {
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rows": built_rows,
                "wells": int(len(registry)),
                "seconds": time.perf_counter() - started,
                "manifest": manifest,
            },
        )
    else:
        print("F03b 逐行特征：复用指纹匹配缓存", flush=True)

    table = pd.read_parquet(cache_path)
    table["well_id"] = table["well_id"].astype(str)
    validate_landscape_table(table, expected_rows=int(config["expected_rows"]))
    return table


def load_and_merge_b00(
    baseline_cache_path: Path,
    landscape_features: pd.DataFrame,
    selected_wells: list[str] | None = None,
) -> pd.DataFrame:
    required_columns = [
        "well_id",
        "fold",
        "row_index",
        "md",
        "target_tvt",
        "carry_tvt",
        "target_delta",
        *B00_FEATURE_COLUMNS,
    ]
    parquet_filters = None
    if selected_wells is not None:
        parquet_filters = [("well_id", "in", selected_wells)]
    base_table = pd.read_parquet(
        baseline_cache_path,
        columns=required_columns,
        filters=parquet_filters,
    )
    base_table["well_id"] = base_table["well_id"].astype(str)
    original_keys = base_table[["well_id", "fold", "row_index"]].copy()

    merge_source = landscape_features.copy()
    merge_source["well_id"] = merge_source["well_id"].astype(str)
    merged = base_table.merge(
        merge_source,
        on=["well_id", "row_index"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("部分 B00 评价行没有匹配到 F03b 特征")
    merged = merged.drop(columns="_merge")
    if not original_keys.equals(merged[["well_id", "fold", "row_index"]]):
        raise ValueError("合并 F03b 后 B00 评价行键或顺序发生变化")
    new_values = merged[F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )
    if np.isinf(new_values).any():
        raise ValueError("合并后的 F03b 新特征含 Inf")
    for feature_name in F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS:
        merged[feature_name] = merged[feature_name].astype(np.float32)
    merged["well_id"] = merged["well_id"].astype("category")
    return merged


def run_smoke(
    config: dict,
    registry: pd.DataFrame,
    baseline_cache_path: Path,
    fingerprint: str,
) -> None:
    smoke_wells = sorted(registry["well_id"].astype(str).tolist())[:3]
    landscape_features = build_selected_features(registry, smoke_wells)
    smoke_table = load_and_merge_b00(
        baseline_cache_path,
        landscape_features,
        selected_wells=smoke_wells,
    )
    finite_rates = np.isfinite(
        smoke_table[F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )
    ).mean(axis=0)
    smoke_dir = CLEAN_ROOT / "artifacts" / "_smoke" / config["experiment_id"]
    smoke_dir.mkdir(parents=True, exist_ok=True)
    landscape_features.head(200).to_csv(smoke_dir / "feature_sample.csv", index=False)
    log_lines = [
        f"shape={smoke_table.shape}",
        f"wells={','.join(smoke_wells)}",
        f"fingerprint={fingerprint}",
        "finite_rates="
        + json.dumps(
            dict(zip(F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS, finite_rates.tolist())),
            ensure_ascii=False,
            sort_keys=True,
        ),
    ]
    log_text = "\n".join(log_lines) + "\n"
    (smoke_dir / "smoke.log").write_text(log_text, encoding="utf-8")
    print(log_text, flush=True)


def save_fold_summary(
    config: dict,
    artifact_dir: Path,
    fingerprint: str,
    manifest: dict,
) -> None:
    metric_rows: list[dict[str, float | int]] = []
    for fold_id in range(5):
        runtime_path = artifact_dir / f"fold_{fold_id}" / "runtime.json"
        baseline_path = (
            CLEAN_ROOT
            / "artifacts"
            / config["baseline_id"]
            / f"fold_{fold_id}"
            / "runtime.json"
        )
        if not runtime_path.is_file():
            continue
        runtime = read_json(runtime_path)
        if runtime.get("fingerprint") != fingerprint:
            continue
        baseline_runtime = read_json(baseline_path)
        baseline_rmse = float(baseline_runtime["micro_rmse"])
        experiment_rmse = float(runtime["micro_rmse"])
        metric_rows.append(
            {
                "fold": fold_id,
                "baseline_rmse": baseline_rmse,
                "experiment_rmse": experiment_rmse,
                "improvement_ft": baseline_rmse - experiment_rmse,
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("fold")
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(config["feature_columns"]), "features": config["feature_columns"]},
    )
    write_json(
        artifact_dir / "runtime.json",
        {"fingerprint": fingerprint, "manifest": manifest, "folds": metric_rows},
    )

    latest = metric_rows[-1]
    improvement = float(latest["improvement_ft"])
    fold_id = int(latest["fold"])
    if fold_id == 0:
        decision = "进入 fold 1" if improvement >= 0.15 else "fold 0 停止"
    elif fold_id == 1:
        decision = "继续 folds 2-4" if improvement > 0.0 else "fold 1 反向停止"
    else:
        decision = "已保存当前折"
    conclusion = (
        "事实：只在 B00 的 12 个特征上新增 12 个固定常 U 几何路径 offset 得分面特征。\n\n"
        f"结果：fold {fold_id} RMSE={float(latest['experiment_rmse']):.6f}，"
        f"B00={float(latest['baseline_rmse']):.6f}，改善 {improvement:.6f} ft。\n\n"
        "当前只能否定：本次固定常 U 几何路径实现，不能否定冻结 PF 路径或整个 Typewell 得分面方向。\n\n"
        f"下一步：{decision}。\n"
    )
    (artifact_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    print(f"{decision}；改善 {improvement:.6f} ft", flush=True)


def main() -> None:
    args = parse_args()
    config = read_json(CONFIG_PATH)
    feature_columns = validate_config(config)
    registry_path = CLEAN_ROOT / config["fold_registry"]
    registry_hash = file_sha256(registry_path)
    if registry_hash != config["fold_registry_sha256"]:
        raise ValueError("F03b 固定 fold hash 与 B00 不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(config["expected_wells"]),
        expected_rows=int(config["expected_rows"]),
    )
    baseline_cache_path = CLEAN_ROOT / config["baseline_feature_cache"]
    model_config_path = CLEAN_ROOT / config["model_config"]
    model_params = read_json(model_config_path)["params"]
    fingerprint, manifest = build_experiment_fingerprint(
        config,
        registry_hash,
        model_config_path,
    )

    print(
        f"experiment_id={config['experiment_id']} | baseline_id={config['baseline_id']} | "
        f"唯一新增特征组=12个固定几何路径offset得分面特征 | mode={args.mode}",
        flush=True,
    )
    if args.mode == "smoke":
        run_smoke(config, registry, baseline_cache_path, fingerprint)
        return

    artifact_dir = CLEAN_ROOT / "artifacts" / config["experiment_id"]
    landscape_features = load_or_build_landscape_cache(
        config,
        registry,
        artifact_dir,
        fingerprint,
        manifest,
    )
    feature_table = load_and_merge_b00(baseline_cache_path, landscape_features)
    if len(feature_table) != int(config["expected_rows"]):
        raise ValueError("F03b 评价行数与 B00 不一致")

    fold_id = int(args.mode[-1])
    train_fold(
        feature_table=feature_table,
        registry=registry,
        fold_id=fold_id,
        model_params=model_params,
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
        feature_columns=feature_columns,
    )
    save_fold_summary(config, artifact_dir, fingerprint, manifest)
    finalize_complete_cv(
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
        expected_rows=int(config["expected_rows"]),
        experiment_config=config,
        model_params=model_params,
        feature_columns=feature_columns,
    )


if __name__ == "__main__":
    main()
