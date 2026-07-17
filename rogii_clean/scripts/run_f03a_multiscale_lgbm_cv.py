"""运行 F03a v2：B00 的 12 个特征加 9 个可见前缀 Typewell 特征。"""

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
PROJECT_ROOT = CLEAN_ROOT.parent
CONFIG_PATH = CLEAN_ROOT / "configs" / "f03a_multiscale_prefix_reliability_v2.json"
TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import (
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f03a_multiscale_prefix_features import (
    F03A_MULTISCALE_FEATURE_COLUMNS,
    build_multiscale_prefix_features,
)
from src.lgbm_data import file_sha256, load_and_validate_registry
from src.lgbm_features import FEATURE_COLUMNS as B00_FEATURE_COLUMNS


RAW_SOURCE_CONTRACT = {
    "horizontal_pattern": "{well_id}__horizontal_well.csv",
    "typewell_pattern": "{well_id}__typewell.csv",
    "horizontal_usecols": ["MD", "GR", "TVT_input"],
    "typewell_usecols": ["TVT", "GR"],
    "visible_definition": "TVT_input.notna()",
    "hidden_tvt_feature_access": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F03a v2 固定单模 LightGBM")
    parser.add_argument(
        "--mode",
        choices=["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"],
        required=True,
    )
    return parser.parse_args()


def stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_well_feature_fingerprint(config: dict, registry_hash: str) -> tuple[str, dict]:
    manifest = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "feature_code_sha256": file_sha256(
            CLEAN_ROOT / "src" / "f03a_multiscale_prefix_features.py"
        ),
        "fold_registry_sha256": registry_hash,
        "raw_source_contract": RAW_SOURCE_CONTRACT,
        "expected_wells": int(config["expected_wells"]),
        "train_dir": str(TRAIN_DIR.resolve()),
    }
    return stable_hash(manifest), manifest


def build_experiment_fingerprint(
    config: dict,
    model_config_path: Path,
    registry_hash: str,
    well_feature_fingerprint: str,
) -> tuple[str, dict]:
    baseline_meta_path = (
        CLEAN_ROOT / "artifacts" / config["baseline_id"] / "feature_cache.meta.json"
    )
    manifest = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_code_sha256": file_sha256(Path(__file__).resolve()),
        "feature_code_sha256": file_sha256(
            CLEAN_ROOT / "src" / "f03a_multiscale_prefix_features.py"
        ),
        "model_config_sha256": file_sha256(model_config_path),
        "fold_registry_sha256": registry_hash,
        "baseline_cache_meta_sha256": file_sha256(baseline_meta_path),
        "well_feature_fingerprint": well_feature_fingerprint,
        "raw_source_contract": RAW_SOURCE_CONTRACT,
    }
    payload = {
        "experiment_id": config["experiment_id"],
        "feature_columns": config["feature_columns"],
        "manifest": manifest,
    }
    return stable_hash(payload), manifest


def build_selected_well_features(
    registry: pd.DataFrame,
    selected_wells: list[str] | None = None,
    progress_interval: int = 100,
) -> pd.DataFrame:
    selected_registry = registry
    if selected_wells is not None:
        selected_registry = registry.loc[
            registry["well_id"].astype(str).isin(selected_wells)
        ]

    rows: list[dict[str, float | int | str]] = []
    total_wells = len(selected_registry)
    for well_number, registry_row in enumerate(
        selected_registry.itertuples(index=False), start=1
    ):
        well_id = str(registry_row.well_id)
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
        feature_values = build_multiscale_prefix_features(horizontal_df, typewell_df)
        rows.append(
            {
                "well_id": well_id,
                "fold": int(registry_row.fold),
                **feature_values,
            }
        )
        if progress_interval > 0 and (
            well_number % progress_interval == 0 or well_number == total_wells
        ):
            print(f"F03a 井级特征：{well_number}/{total_wells}", flush=True)

    return pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)


def validate_well_feature_cache(table: pd.DataFrame, registry: pd.DataFrame) -> None:
    expected_columns = ["well_id", "fold", *F03A_MULTISCALE_FEATURE_COLUMNS]
    if table.columns.tolist() != expected_columns:
        raise ValueError("F03a 井级缓存列或顺序不正确")
    if len(table) != len(registry) or table["well_id"].duplicated().any():
        raise ValueError("F03a 井级缓存井数或井号不正确")
    actual_keys = table[["well_id", "fold"]].copy()
    actual_keys["well_id"] = actual_keys["well_id"].astype(str)
    expected_keys = registry[["well_id", "fold"]].copy()
    expected_keys["well_id"] = expected_keys["well_id"].astype(str)
    if not actual_keys.reset_index(drop=True).equals(expected_keys.reset_index(drop=True)):
        raise ValueError("F03a 井级缓存与固定 fold 注册表不一致")
    feature_array = table[F03A_MULTISCALE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if np.isinf(feature_array).any():
        raise ValueError("F03a 井级缓存含 Inf")


def load_or_build_well_feature_cache(
    config: dict,
    registry: pd.DataFrame,
    fingerprint: str,
    source_manifest: dict,
    artifact_dir: Path,
) -> pd.DataFrame:
    cache_path = artifact_dir / "well_feature_cache.csv"
    metadata_path = artifact_dir / "well_feature_cache.meta.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    cache_matches = (
        cache_path.is_file() and metadata.get("fingerprint") == fingerprint
    )
    if cache_matches:
        print("F03a 井级特征：复用匹配缓存", flush=True)
        table = pd.read_csv(cache_path, dtype={"well_id": str})
    else:
        started = time.perf_counter()
        table = build_selected_well_features(registry)
        validate_well_feature_cache(table, registry)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
        write_json(
            metadata_path,
            {
                "fingerprint": fingerprint,
                "source_manifest": source_manifest,
                "wells": int(len(table)),
                "seconds": time.perf_counter() - started,
            },
        )
    validate_well_feature_cache(table, registry)
    return table


def load_and_merge_b00(
    baseline_cache_path: Path,
    well_features: pd.DataFrame,
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

    merge_source = well_features.copy()
    merge_source["well_id"] = merge_source["well_id"].astype(str)
    merged = base_table.merge(
        merge_source,
        on=["well_id", "fold"],
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("部分 B00 评价行没有匹配到 F03a 井级特征")
    merged = merged.drop(columns="_merge")
    if not original_keys.equals(merged[["well_id", "fold", "row_index"]]):
        raise ValueError("合并 F03a 后 B00 的评价行键或顺序发生变化")
    for feature_name in F03A_MULTISCALE_FEATURE_COLUMNS:
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
    well_features = build_selected_well_features(
        registry,
        selected_wells=smoke_wells,
        progress_interval=1,
    )
    smoke_table = load_and_merge_b00(
        baseline_cache_path,
        well_features,
        selected_wells=smoke_wells,
    )
    finite_rates = (
        np.isfinite(
            smoke_table[F03A_MULTISCALE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        )
        .mean(axis=0)
        .tolist()
    )
    smoke_dir = CLEAN_ROOT / "artifacts" / "_smoke" / config["experiment_id"]
    smoke_dir.mkdir(parents=True, exist_ok=True)
    well_features.to_csv(smoke_dir / "feature_sample.csv", index=False)
    log_lines = [
        f"shape={smoke_table.shape}",
        f"wells={','.join(smoke_wells)}",
        f"fingerprint={fingerprint}",
        "finite_rates="
        + json.dumps(
            dict(zip(F03A_MULTISCALE_FEATURE_COLUMNS, finite_rates)),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "sample=" + well_features.to_json(orient="records", force_ascii=False),
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
        {
            "feature_count": len(config["feature_columns"]),
            "features": config["feature_columns"],
        },
    )
    write_json(
        artifact_dir / "runtime.json",
        {"fingerprint": fingerprint, "manifest": manifest, "folds": metric_rows},
    )

    latest = metric_rows[-1]
    improvement = float(latest["improvement_ft"])
    if int(latest["fold"]) == 0:
        decision = "进入 fold 1" if improvement >= 0.15 else "fold 0 停止"
    elif int(latest["fold"]) == 1:
        decision = "继续 folds 2-4" if improvement > 0.0 else "fold 1 反向停止"
    else:
        decision = "已保存当前折"
    conclusion = (
        "事实：只在 B00 的 12 个特征上新增 9 个可见前缀 Typewell 多尺度特征。\n\n"
        f"结果：fold {int(latest['fold'])} RMSE={float(latest['experiment_rmse']):.6f}，"
        f"B00={float(latest['baseline_rmse']):.6f}，改善={improvement:.6f} ft。\n\n"
        f"下一步：{decision}。\n"
    )
    (artifact_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    print(f"{decision}；改善 {improvement:.6f} ft", flush=True)


def main() -> None:
    args = parse_args()
    config = read_json(CONFIG_PATH)
    feature_columns = [*B00_FEATURE_COLUMNS, *F03A_MULTISCALE_FEATURE_COLUMNS]
    if config["feature_columns"] != feature_columns:
        raise ValueError("F03a 配置中的特征顺序与代码不一致")

    registry_path = CLEAN_ROOT / config["fold_registry"]
    registry_hash = file_sha256(registry_path)
    if registry_hash != config["fold_registry_sha256"]:
        raise ValueError("F03a 固定 fold hash 与 B00 不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(config["expected_wells"]),
        expected_rows=int(config["expected_rows"]),
    )
    baseline_cache_path = CLEAN_ROOT / config["baseline_feature_cache"]
    model_config_path = CLEAN_ROOT / config["model_config"]
    model_params = read_json(model_config_path)["params"]
    well_fingerprint, source_manifest = build_well_feature_fingerprint(
        config,
        registry_hash,
    )
    fingerprint, manifest = build_experiment_fingerprint(
        config,
        model_config_path,
        registry_hash,
        well_fingerprint,
    )

    print(
        f"experiment_id={config['experiment_id']} | baseline_id={config['baseline_id']} | "
        f"唯一新增特征组=9个可见前缀Typewell多尺度特征 | mode={args.mode}",
        flush=True,
    )
    if args.mode == "smoke":
        run_smoke(config, registry, baseline_cache_path, fingerprint)
        return

    artifact_dir = CLEAN_ROOT / "artifacts" / config["experiment_id"]
    well_features = load_or_build_well_feature_cache(
        config,
        registry,
        well_fingerprint,
        source_manifest,
        artifact_dir,
    )
    feature_table = load_and_merge_b00(baseline_cache_path, well_features)
    if len(feature_table) != int(config["expected_rows"]):
        raise ValueError("F03a 评价行数与 B00 不一致")

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
