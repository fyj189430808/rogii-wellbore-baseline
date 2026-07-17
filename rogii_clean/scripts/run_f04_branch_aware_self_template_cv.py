"""运行 F04：冻结 B00 加当前井 branch-aware 自模板特征。"""

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
CONFIG_PATH = CLEAN_ROOT / "configs" / "f04_branch_aware_self_template_v1.json"
TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f04_branch_aware_self_template import (  # noqa: E402
    F04_FEATURE_COLUMNS,
    MINIMUM_SEGMENT_ROWS,
    MINIMUM_WINDOW_PAIRS,
    NCC_WINDOW_WIDTH_FT,
    OFFSET_GRID_FT,
    TVT_BIN_WIDTH_FT,
    build_f04_self_template_rows,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS as B00_FEATURE_COLUMNS  # noqa: E402


RAW_SOURCE_CONTRACT = {
    "horizontal_pattern": "{well_id}__horizontal_well.csv",
    "horizontal_usecols": ["MD", "Z", "GR", "TVT_input"],
    "candidate_path_formula": "last_visible_tvt - (Z_current - Z_visible_end)",
    "hidden_tvt_feature_access": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F04 固定单模 LightGBM")
    parser.add_argument(
        "--mode",
        choices=["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"],
        required=True,
    )
    return parser.parse_args()


def validate_config(config: dict) -> list[str]:
    feature_columns = [*B00_FEATURE_COLUMNS, *F04_FEATURE_COLUMNS]
    if config["feature_columns"] != feature_columns:
        raise ValueError("F04 配置中的特征顺序与代码不一致")
    fixed_values = {
        "tvt_bin_width_ft": TVT_BIN_WIDTH_FT,
        "minimum_segment_rows": MINIMUM_SEGMENT_ROWS,
        "ncc_window_width_ft": NCC_WINDOW_WIDTH_FT,
        "minimum_window_pairs": MINIMUM_WINDOW_PAIRS,
        "offset_grid_ft": OFFSET_GRID_FT.tolist(),
    }
    for key, expected in fixed_values.items():
        if config[key] != expected:
            raise ValueError(f"F04 冻结参数 {key} 与代码不一致")
    return feature_columns


def build_experiment_fingerprint(
    config: dict,
    registry_hash: str,
    model_config_path: Path,
) -> tuple[str, dict]:
    manifest = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_code_sha256": file_sha256(Path(__file__).resolve()),
        "feature_code_sha256": file_sha256(
            CLEAN_ROOT / "src" / "f04_branch_aware_self_template.py"
        ),
        "fold_registry_sha256": registry_hash,
        "baseline_cache_metadata_sha256": file_sha256(
            CLEAN_ROOT / config["baseline_feature_cache_metadata"]
        ),
        "model_config_sha256": file_sha256(model_config_path),
        "raw_source_contract": RAW_SOURCE_CONTRACT,
    }
    payload = {
        "experiment_id": config["experiment_id"],
        "feature_columns": config["feature_columns"],
        "manifest": manifest,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fingerprint, manifest


def build_one_well_features(well_id: str, fold: int) -> pd.DataFrame:
    horizontal_path = TRAIN_DIR / f"{well_id}__horizontal_well.csv"
    if not horizontal_path.is_file():
        raise FileNotFoundError(f"井 {well_id} 缺少 horizontal 文件")
    horizontal_df = pd.read_csv(
        horizontal_path,
        usecols=RAW_SOURCE_CONTRACT["horizontal_usecols"],
    )
    rows = build_f04_self_template_rows(horizontal_df, str(well_id), int(fold))
    rows["well_id"] = rows["well_id"].astype(str)
    rows["fold"] = rows["fold"].astype(np.int8)
    rows["row_index"] = rows["row_index"].astype(np.int32)
    for feature_name in F04_FEATURE_COLUMNS:
        rows[feature_name] = rows[feature_name].astype(np.float32)
    return rows


def validate_feature_table(
    table: pd.DataFrame,
    expected_rows: int | None = None,
) -> None:
    expected_columns = ["well_id", "fold", "row_index", *F04_FEATURE_COLUMNS]
    if table.columns.tolist() != expected_columns:
        raise ValueError("F04 逐行缓存列或顺序不正确")
    if expected_rows is not None and len(table) != int(expected_rows):
        raise ValueError("F04 逐行缓存评价行数不正确")
    if table.duplicated(["well_id", "row_index"]).any():
        raise ValueError("F04 逐行缓存含重复键")
    values = table[F04_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("F04 新特征含 Inf")
    required_finite = [
        "f04_coverage_flag",
        "f04_same_direction_support_count",
        "f04_opposite_direction_support_count",
        "f04_number_of_visits",
    ]
    if not np.isfinite(table[required_finite].to_numpy(dtype=np.float64)).all():
        raise ValueError("F04 覆盖或支持量特征含缺失")


def build_selected_features(
    registry: pd.DataFrame,
    selected_wells: list[str],
) -> pd.DataFrame:
    fold_lookup = registry.set_index("well_id")["fold"].astype(int).to_dict()
    frames = [
        build_one_well_features(well_id, fold_lookup[well_id])
        for well_id in selected_wells
    ]
    table = pd.concat(frames, ignore_index=True)
    validate_feature_table(table)
    return table


def load_or_build_feature_cache(
    config: dict,
    registry: pd.DataFrame,
    artifact_dir: Path,
    fingerprint: str,
    manifest: dict,
) -> pd.DataFrame:
    cache_path = artifact_dir / "self_template_feature_cache.parquet"
    metadata_path = artifact_dir / "self_template_feature_cache.meta.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    cache_matches = cache_path.is_file() and metadata.get("fingerprint") == fingerprint

    if not cache_matches:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = artifact_dir / "self_template_feature_cache.tmp.parquet"
        temporary_path.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        built_rows = 0
        started = time.perf_counter()
        try:
            for number, row in enumerate(registry.itertuples(index=False), start=1):
                well_table = build_one_well_features(str(row.well_id), int(row.fold))
                arrow_table = pa.Table.from_pandas(well_table, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        arrow_table.schema,
                        compression="zstd",
                    )
                writer.write_table(arrow_table)
                built_rows += len(well_table)
                if number % 25 == 0 or number == len(registry):
                    print(
                        f"F04 逐井特征：{number}/{len(registry)}，累计 {built_rows:,} 行",
                        flush=True,
                    )
        except Exception:
            if writer is not None:
                writer.close()
            temporary_path.unlink(missing_ok=True)
            raise
        if writer is None:
            raise ValueError("F04 没有生成任何特征")
        writer.close()
        if built_rows != int(config["expected_rows"]):
            temporary_path.unlink(missing_ok=True)
            raise ValueError("F04 构造行数与 B00 不一致")
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
        print("F04 逐行特征：复用指纹匹配缓存", flush=True)

    table = pd.read_parquet(cache_path)
    table["well_id"] = table["well_id"].astype(str)
    validate_feature_table(table, expected_rows=int(config["expected_rows"]))
    return table


def load_and_merge_b00(
    baseline_cache_path: Path,
    self_features: pd.DataFrame,
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
    filters = None if selected_wells is None else [("well_id", "in", selected_wells)]
    base_table = pd.read_parquet(
        baseline_cache_path,
        columns=required_columns,
        filters=filters,
    )
    base_table["well_id"] = base_table["well_id"].astype(str)
    original_keys = base_table[["well_id", "fold", "row_index"]].copy()
    merge_source = self_features.copy()
    merge_source["well_id"] = merge_source["well_id"].astype(str)
    merged = base_table.merge(
        merge_source.drop(columns="fold"),
        on=["well_id", "row_index"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("部分 B00 评价行没有匹配到 F04 特征")
    merged = merged.drop(columns="_merge")
    if not original_keys.equals(merged[["well_id", "fold", "row_index"]]):
        raise ValueError("合并 F04 后 B00 评价行键或顺序发生变化")
    if np.isinf(
        merged[F04_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).any():
        raise ValueError("合并后的 F04 特征含 Inf")
    for feature_name in F04_FEATURE_COLUMNS:
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
    self_features = build_selected_features(registry, smoke_wells)
    smoke_table = load_and_merge_b00(
        baseline_cache_path,
        self_features,
        selected_wells=smoke_wells,
    )
    finite_rates = np.isfinite(
        smoke_table[F04_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).mean(axis=0)
    coverage_rate = float(smoke_table["f04_coverage_flag"].mean())
    smoke_dir = CLEAN_ROOT / "artifacts" / "_smoke" / config["experiment_id"]
    smoke_dir.mkdir(parents=True, exist_ok=True)
    self_features.head(200).to_csv(smoke_dir / "feature_sample.csv", index=False)
    log_text = "\n".join(
        [
            f"shape={smoke_table.shape}",
            f"wells={','.join(smoke_wells)}",
            f"coverage_rate={coverage_rate:.6f}",
            "finite_rates="
            + json.dumps(
                dict(zip(F04_FEATURE_COLUMNS, finite_rates.tolist())),
                ensure_ascii=False,
                sort_keys=True,
            ),
            f"fingerprint={fingerprint}",
        ]
    ) + "\n"
    (smoke_dir / "smoke.log").write_text(log_text, encoding="utf-8")
    print(log_text, flush=True)


def save_fold_summary(
    config: dict,
    artifact_dir: Path,
    fingerprint: str,
    manifest: dict,
) -> None:
    metric_rows: list[dict] = []
    for fold_id in range(5):
        runtime_path = artifact_dir / f"fold_{fold_id}" / "runtime.json"
        if not runtime_path.is_file():
            continue
        runtime = read_json(runtime_path)
        if runtime.get("fingerprint") != fingerprint:
            continue
        baseline_runtime = read_json(
            CLEAN_ROOT
            / "artifacts"
            / config["baseline_id"]
            / f"fold_{fold_id}"
            / "runtime.json"
        )
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
    fold_id = int(latest["fold"])
    improvement = float(latest["improvement_ft"])
    if fold_id == 0:
        decision = "进入 fold 1" if improvement >= 0.15 else "fold 0 停止"
    elif fold_id == 1:
        decision = "继续 folds 2-4" if improvement > 0.0 else "fold 1 反向停止"
    else:
        decision = "已保存当前折"
    conclusion = (
        "事实：只在 B00 上新增 12 个当前井方向分支自模板特征。\n\n"
        f"结果：fold {fold_id} RMSE={float(latest['experiment_rmse']):.6f}，"
        f"B00={float(latest['baseline_rmse']):.6f}，改善={improvement:.6f} ft。\n\n"
        "当前只能否定：本次固定自模板实现，不能否定整个 self-template 方向。\n\n"
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
        raise ValueError("F04 固定 fold hash 与 B00 不一致")
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
        f"唯一新增特征组=12个branch-aware自模板特征 | mode={args.mode}",
        flush=True,
    )
    if args.mode == "smoke":
        run_smoke(config, registry, baseline_cache_path, fingerprint)
        return

    artifact_dir = CLEAN_ROOT / "artifacts" / config["experiment_id"]
    self_features = load_or_build_feature_cache(
        config,
        registry,
        artifact_dir,
        fingerprint,
        manifest,
    )
    feature_table = load_and_merge_b00(baseline_cache_path, self_features)
    if len(feature_table) != int(config["expected_rows"]):
        raise ValueError("F04 评价行数与 B00 不一致")
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
