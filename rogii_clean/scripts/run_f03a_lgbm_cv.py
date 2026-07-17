"""在冻结 B00 上增加可见前缀 Typewell 可信度特征，并运行 fold 0。"""

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

from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json
from src.f03a_prefix_reliability_features import (
    F03A_FEATURE_COLUMNS,
    build_well_prefix_reliability_features,
)
from src.lgbm_data import file_sha256, load_and_validate_registry
from src.lgbm_features import FEATURE_COLUMNS as B00_FEATURE_COLUMNS


CONFIG_PATH = CLEAN_ROOT / "configs" / "f03a_typewell_prefix_reliability_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F03a Typewell 前缀可信度特征")
    parser.add_argument("--mode", choices=["smoke", "fold0"], required=True)
    return parser.parse_args()


def build_fingerprint(
    config: dict,
    model_config_path: Path,
    registry_path: Path,
    baseline_cache_path: Path,
    offset_scores_path: Path,
    margins_path: Path,
) -> tuple[str, dict]:
    """把所有会改变结果的输入文件纳入实验指纹。"""

    hash_manifest = {
        "config": file_sha256(CONFIG_PATH),
        "model_config": file_sha256(model_config_path),
        "fold_registry": file_sha256(registry_path),
        "baseline_feature_cache": file_sha256(baseline_cache_path),
        "d0_offset_scores": file_sha256(offset_scores_path),
        "d0_per_well_margins": file_sha256(margins_path),
        "feature_code": file_sha256(
            CLEAN_ROOT / "src" / "f03a_prefix_reliability_features.py"
        ),
        "runner_code": file_sha256(Path(__file__).resolve()),
    }
    payload = {
        "experiment_id": config["experiment_id"],
        "feature_columns": config["feature_columns"],
        "hash_manifest": hash_manifest,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fingerprint, hash_manifest


def load_well_features(
    offset_scores_path: Path,
    margins_path: Path,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """读取已经完成的 D0 结果，不再重读 773 口原始井。"""

    offset_scores = pd.read_csv(offset_scores_path, dtype={"well_id": str})
    margins = pd.read_csv(margins_path, dtype={"well_id": str})
    return build_well_prefix_reliability_features(
        offset_scores=offset_scores,
        margins=margins,
        registry=registry,
    )


def load_and_merge_feature_table(
    baseline_cache_path: Path,
    well_features: pd.DataFrame,
    selected_wells: list[str] | None = None,
) -> pd.DataFrame:
    """把一井一行的 F03a 常量特征合并到 B00 的每个隐藏评价行。"""

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
    merged = base_table.merge(
        well_features,
        on=["well_id", "fold"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    merged_keys = merged[["well_id", "fold", "row_index"]]
    if not original_keys.equals(merged_keys):
        raise ValueError("合并 F03a 后 B00 的评价行键或顺序发生变化")
    if merged[F03A_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("部分 B00 行没有匹配到 F03a 井级特征")

    merged["well_id"] = merged["well_id"].astype("category")
    return merged


def run_smoke(
    config: dict,
    registry: pd.DataFrame,
    baseline_cache_path: Path,
    well_features: pd.DataFrame,
    fingerprint: str,
) -> None:
    """只检查排序前三口井的合并结果，不训练模型。"""

    smoke_wells = sorted(registry["well_id"].astype(str).tolist())[:3]
    smoke_table = load_and_merge_feature_table(
        baseline_cache_path=baseline_cache_path,
        well_features=well_features,
        selected_wells=smoke_wells,
    )
    feature_sample = (
        smoke_table[["well_id", *F03A_FEATURE_COLUMNS]]
        .drop_duplicates("well_id")
        .sort_values("well_id")
    )
    finite_rate = float(
        np.isfinite(
            smoke_table[F03A_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        ).mean()
    )
    if len(feature_sample) != 3 or finite_rate != 1.0:
        raise ValueError("F03a smoke 的井数或有限值比例不正确")

    smoke_dir = CLEAN_ROOT / "artifacts" / "_smoke" / config["experiment_id"]
    smoke_dir.mkdir(parents=True, exist_ok=True)
    feature_sample.to_csv(smoke_dir / "feature_sample.csv", index=False)
    log_text = (
        f"experiment_id={config['experiment_id']}\n"
        f"baseline_id={config['baseline_id']}\n"
        f"wells={','.join(smoke_wells)}\n"
        f"rows={len(smoke_table)}\n"
        f"features={len(F03A_FEATURE_COLUMNS)}\n"
        f"finite_rate={finite_rate:.6f}\n"
        f"fingerprint={fingerprint}\n"
    )
    (smoke_dir / "smoke.log").write_text(log_text, encoding="utf-8")
    print(log_text, flush=True)


def run_fold0(
    config: dict,
    model_params: dict,
    registry: pd.DataFrame,
    baseline_cache_path: Path,
    well_features: pd.DataFrame,
    fingerprint: str,
    hash_manifest: dict,
) -> None:
    """加载冻结 B00 缓存，只训练一个固定参数的 fold 0 模型。"""

    started = time.perf_counter()
    feature_table = load_and_merge_feature_table(
        baseline_cache_path=baseline_cache_path,
        well_features=well_features,
    )
    if len(feature_table) != int(config["expected_rows"]):
        raise ValueError("F03a 特征表行数与冻结评价行数不一致")

    feature_columns = [*B00_FEATURE_COLUMNS, *F03A_FEATURE_COLUMNS]
    if feature_columns != config["feature_columns"]:
        raise ValueError("F03a 配置中的特征顺序与代码不一致")

    artifact_dir = CLEAN_ROOT / "artifacts" / config["experiment_id"]
    runtime = train_fold(
        feature_table=feature_table,
        registry=registry,
        fold_id=0,
        model_params=model_params,
        artifact_dir=artifact_dir,
        fingerprint=fingerprint,
        feature_columns=feature_columns,
    )

    baseline_runtime = read_json(
        CLEAN_ROOT / "artifacts" / config["baseline_id"] / "fold_0" / "runtime.json"
    )
    improvement = float(baseline_runtime["micro_rmse"]) - float(
        runtime["micro_rmse"]
    )
    metrics = pd.DataFrame(
        [
            {
                "experiment_id": config["experiment_id"],
                "fold": 0,
                "baseline_rmse": baseline_runtime["micro_rmse"],
                "experiment_rmse": runtime["micro_rmse"],
                "improvement_ft": improvement,
            }
        ]
    )
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(feature_columns), "features": feature_columns},
    )
    write_json(
        artifact_dir / "runtime.json",
        {
            "fingerprint": fingerprint,
            "hash_manifest": hash_manifest,
            "fold0": runtime,
            "total_seconds": time.perf_counter() - started,
        },
    )
    decision = "进入 fold 1" if improvement >= 0.15 else "fold 0 停止"
    conclusion = (
        f"事实：仅在 B00 增加 18 个可见前缀 Typewell 可信度特征。\n\n"
        f"结果：fold 0 RMSE={runtime['micro_rmse']:.6f}，"
        f"B00={float(baseline_runtime['micro_rmse']):.6f}，"
        f"改善={improvement:.6f} ft。\n\n"
        f"停止规则：{decision}。\n"
    )
    (artifact_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    print(f"F03a 决策：{decision}，改善 {improvement:.6f} ft", flush=True)


def main() -> None:
    args = parse_args()
    config = read_json(CONFIG_PATH)
    model_config_path = CLEAN_ROOT / config["model_config"]
    registry_path = CLEAN_ROOT / config["fold_registry"]
    baseline_cache_path = CLEAN_ROOT / config["baseline_feature_cache"]
    offset_scores_path = CLEAN_ROOT / config["d0_offset_scores"]
    margins_path = CLEAN_ROOT / config["d0_per_well_margins"]

    current_registry_hash = file_sha256(registry_path)
    if current_registry_hash != config["fold_registry_sha256"]:
        raise ValueError("F03a 的固定 fold hash 与 B00 不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(config["expected_wells"]),
        expected_rows=int(config["expected_rows"]),
    )
    model_params = read_json(model_config_path)["params"]
    fingerprint, hash_manifest = build_fingerprint(
        config=config,
        model_config_path=model_config_path,
        registry_path=registry_path,
        baseline_cache_path=baseline_cache_path,
        offset_scores_path=offset_scores_path,
        margins_path=margins_path,
    )
    well_features = load_well_features(
        offset_scores_path=offset_scores_path,
        margins_path=margins_path,
        registry=registry,
    )

    print(
        f"experiment_id={config['experiment_id']} | baseline_id={config['baseline_id']} | "
        f"唯一新增特征组=Typewell 可见前缀可信度 | mode={args.mode}",
        flush=True,
    )
    if args.mode == "smoke":
        run_smoke(
            config=config,
            registry=registry,
            baseline_cache_path=baseline_cache_path,
            well_features=well_features,
            fingerprint=fingerprint,
        )
        return
    run_fold0(
        config=config,
        model_params=model_params,
        registry=registry,
        baseline_cache_path=baseline_cache_path,
        well_features=well_features,
        fingerprint=fingerprint,
        hash_manifest=hash_manifest,
    )


if __name__ == "__main__":
    main()
