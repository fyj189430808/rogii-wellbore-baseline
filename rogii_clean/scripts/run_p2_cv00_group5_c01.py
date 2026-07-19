"""在二阶段按井五折上原样重跑 C01 的 36 特征单模 LightGBM。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 当前文件位于 rogii_clean/scripts，因此父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_f05a_deterministic_candidates_cv import (  # noqa: E402
    _validate_cache_metadata,
    load_aligned_feature_table,
    read_candidate_cache_provenance,
    require_candidate_cache_files,
)
from scripts.run_simple_lgbm_cv import (  # noqa: E402
    finalize_complete_cv,
    read_json,
    train_fold,
    write_json,
)
from src.f05a_direct_physical_candidates import (  # noqa: E402
    DIRECT_CANDIDATE_COLUMNS,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "P2_CV00_group5_c01_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p2_cv00_group5_c01_v1.json"
SUPPORTED_MODES = [
    "smoke",
    "fold0",
    "fold1",
    "fold2",
    "fold3",
    "fold4",
    "all",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析小样本检查、单折或连续完整五折运行模式。"""

    parser = argparse.ArgumentParser(description="运行 P2-CV00 按井五折 C01")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def stream_file_sha256(path: Path) -> str:
    """分块计算大缓存 SHA-256，避免一次把数百 MB 文件读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(experiment_config: dict[str, object]) -> list[str]:
    """确认二阶段配置仅改变 CV，仍是 C01 的原始 36 列和同一模型。"""

    expected_features = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
    if experiment_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("实验 ID 与 P2-CV00 runner 不一致")
    if experiment_config.get("feature_columns") != expected_features:
        raise ValueError("P2-CV00 特征不是冻结的 C01 36 列")
    if experiment_config.get("candidate_feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("P2-CV00 候选特征不是冻结的 24 列")
    if experiment_config.get("model_config") != "configs/lgbm_feature_baseline_v1.json":
        raise ValueError("P2-CV00 模型参数配置发生变化")
    if experiment_config.get("validation_unit") != "complete_well":
        raise ValueError("P2-CV00 验证单位必须是完整井")
    if experiment_config.get("fold_remapped_by_well_id") is not True:
        raise ValueError("P2-CV00 必须按 well_id 重映射折号")
    if experiment_config.get("cached_fold_column_ignored") is not True:
        raise ValueError("P2-CV00 必须忽略缓存内的一阶段折号")
    if experiment_config.get("candidate_cache_rebuild_allowed") is not False:
        raise ValueError("P2-CV00 runner 禁止自行重建候选缓存")
    if experiment_config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("配置中的运行模式与 P2-CV00 runner 不一致")
    return expected_features


def remap_feature_table_folds(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """按 well_id 覆盖缓存旧折号，并核对逐井评价行数量。"""

    required_registry_columns = {"well_id", "fold", "hidden_rows"}
    missing_columns = required_registry_columns - set(registry.columns)
    if missing_columns:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing_columns)}")
    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")
    if feature_table.duplicated(["well_id", "row_index"]).any():
        raise ValueError("特征缓存含重复评价行键")

    feature_wells = set(feature_table["well_id"].astype(str).unique())
    registry_wells = set(registry["well_id"].astype(str).unique())
    if feature_wells != registry_wells:
        missing_in_registry = sorted(feature_wells - registry_wells)
        missing_in_cache = sorted(registry_wells - feature_wells)
        raise ValueError(
            "特征缓存与新 fold 注册表井集合不一致："
            f"注册表缺少={missing_in_registry[:5]}，缓存缺少={missing_in_cache[:5]}"
        )

    actual_rows = (
        feature_table.assign(well_id=feature_table["well_id"].astype(str))
        .groupby("well_id")
        .size()
        .sort_index()
    )
    expected_rows = (
        registry.assign(well_id=registry["well_id"].astype(str))
        .set_index("well_id")["hidden_rows"]
        .astype(np.int64)
        .sort_index()
    )
    actual_rows.name = "hidden_rows"
    if not actual_rows.astype(np.int64).equals(expected_rows):
        raise ValueError("特征缓存逐井评价行数与新 fold 注册表不一致")

    fold_mapping = (
        registry.assign(well_id=registry["well_id"].astype(str))
        .set_index("well_id")["fold"]
        .astype(np.int64)
    )
    remapped_table = feature_table.copy()
    remapped_table["fold"] = (
        remapped_table["well_id"].astype(str).map(fold_mapping).astype(np.int64)
    )

    expected_fold_rows = (
        registry.groupby("fold")["hidden_rows"].sum().astype(np.int64).sort_index()
    )
    actual_fold_rows = (
        remapped_table.groupby("fold").size().astype(np.int64).sort_index()
    )
    actual_fold_rows.name = "hidden_rows"
    if not actual_fold_rows.equals(expected_fold_rows):
        raise ValueError("折号重映射后的逐折评价行数不正确")
    return remapped_table


def build_experiment_fingerprint(
    experiment_config: dict[str, object],
    model_params: dict[str, object],
    registry_sha256: str,
    base_cache_sha256: str,
    candidate_cache_sha256: str,
    candidate_cache_provenance: dict[str, object],
) -> str:
    """用新注册表、两个缓存、模型参数和执行代码构造完整实验指纹。"""

    payload = {
        "experiment_config": experiment_config,
        "model_params": model_params,
        "registry_sha256": registry_sha256,
        "base_cache_sha256": base_cache_sha256,
        "candidate_cache_sha256": candidate_cache_sha256,
        "candidate_cache_provenance": candidate_cache_provenance,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "cache_loader_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_f05a_deterministic_candidates_cv.py"
        ),
        "trainer_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_smoke(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    model_features: list[str],
) -> None:
    """不训练模型，只检查前三口井、36 列和完整折号重映射。"""

    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    if len(smoke_wells) != 3:
        raise ValueError("smoke 没有取得三口井")
    if feature_table[model_features].replace([np.inf, -np.inf], np.nan).isna().any().any():
        # 原始 GR 允许缺失，除此之外不接受无穷；下一段单独区分合法 GR NaN。
        invalid_columns = []
        for column_name in model_features:
            column_values = feature_table[column_name].to_numpy(dtype=np.float64)
            has_infinite = np.isinf(column_values).any()
            has_illegal_nan = column_name != "gr_raw" and np.isnan(column_values).any()
            if has_infinite or has_illegal_nan:
                invalid_columns.append(column_name)
        if invalid_columns:
            raise ValueError(f"特征缓存含非法 NaN 或 Inf：{invalid_columns}")

    actual_fold_rows = feature_table.groupby("fold").size().astype(int).to_dict()
    expected_fold_rows = (
        registry.groupby("fold")["hidden_rows"].sum().astype(int).to_dict()
    )
    if actual_fold_rows != expected_fold_rows:
        raise ValueError("smoke 检查发现折号重映射行数不一致")
    print(
        "smoke 完成："
        f"井={len(registry)}，评价行={len(feature_table):,}，特征={len(model_features)}，"
        f"前三井={smoke_wells}，逐折行={actual_fold_rows}",
        flush=True,
    )


def save_fold_runtime_row(artifact_dir: Path, runtime: dict[str, object]) -> None:
    """把单折摘要更新到可中断恢复的 metrics.csv。"""

    fold_id = int(runtime["fold"])
    metrics_path = artifact_dir / "metrics.csv"
    previous = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    if not previous.empty:
        previous = previous.loc[previous["fold"].astype(int) != fold_id]
    combined = pd.concat([previous, pd.DataFrame([runtime])], ignore_index=True)
    combined.sort_values("fold").to_csv(metrics_path, index=False)


def validate_complete_predictions(
    artifact_dir: Path,
    registry: pd.DataFrame,
    expected_rows: int,
) -> None:
    """核对完整 OOF 的井、折和键，并另存每折指标表。"""

    prediction_path = artifact_dir / "predictions.parquet"
    if not prediction_path.is_file():
        raise FileNotFoundError("完整五折结束后没有 predictions.parquet")
    predictions = pd.read_parquet(prediction_path)
    if len(predictions) != int(expected_rows):
        raise ValueError("完整 OOF 评价行数不正确")
    if predictions.duplicated(["well_id", "row_index"]).any():
        raise ValueError("完整 OOF 含重复评价行键")

    expected_well_folds = (
        registry.assign(well_id=registry["well_id"].astype(str))
        .set_index("well_id")["fold"]
        .astype(np.int64)
    )
    actual_well_folds = (
        predictions.assign(well_id=predictions["well_id"].astype(str))
        .groupby("well_id")["fold"]
        .agg(["nunique", "first"])
    )
    if int(actual_well_folds["nunique"].max()) != 1:
        raise ValueError("完整 OOF 中同一口井出现多个 fold")
    actual_mapping = actual_well_folds["first"].astype(np.int64).sort_index()
    expected_mapping = expected_well_folds.sort_index()
    actual_mapping.name = "fold"
    if not actual_mapping.equals(expected_mapping):
        raise ValueError("完整 OOF 的逐井 fold 与新注册表不一致")

    metrics = read_json(artifact_dir / "metrics.json")
    pd.DataFrame(metrics["folds"]).to_csv(artifact_dir / "per_fold.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    """加载冻结缓存、覆盖旧折号，并运行 smoke、单折或完整五折。"""

    args = parse_args(argv)
    config_path = args.config.resolve()
    experiment_config = read_json(config_path)
    model_features = validate_config(experiment_config)
    expected_wells = int(experiment_config["expected_wells"])
    expected_rows = int(experiment_config["expected_rows"])

    registry_path = CLEAN_ROOT / str(experiment_config["fold_registry"])
    actual_registry_sha = stream_file_sha256(registry_path)
    if actual_registry_sha != experiment_config["fold_registry_sha256"]:
        raise ValueError("按井 fold 注册表 SHA-256 与冻结配置不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )

    base_cache_path = CLEAN_ROOT / str(experiment_config["base_feature_cache"])
    candidate_cache_path = CLEAN_ROOT / str(
        experiment_config["candidate_feature_cache"]
    )
    candidate_metadata_path = CLEAN_ROOT / str(
        experiment_config["candidate_cache_metadata"]
    )
    require_candidate_cache_files(candidate_cache_path, candidate_metadata_path)
    candidate_cache_provenance = read_candidate_cache_provenance(
        candidate_cache_path,
        candidate_metadata_path,
    )
    candidate_metadata = candidate_cache_provenance["metadata"]
    if not isinstance(candidate_metadata, dict):
        raise ValueError("候选缓存 metadata 必须是 JSON 对象")
    _validate_cache_metadata(candidate_metadata, expected_wells, expected_rows)

    actual_base_cache_sha = stream_file_sha256(base_cache_path)
    actual_candidate_cache_sha = stream_file_sha256(candidate_cache_path)
    if actual_base_cache_sha != experiment_config["base_feature_cache_sha256"]:
        raise ValueError("B00 特征缓存 SHA-256 与冻结配置不一致")
    if actual_candidate_cache_sha != experiment_config[
        "candidate_feature_cache_sha256"
    ]:
        raise ValueError("C01 候选缓存 SHA-256 与冻结配置不一致")
    if actual_candidate_cache_sha != candidate_metadata["candidate_cache_sha256"]:
        raise ValueError("C01 候选缓存实际 SHA 与其 metadata 不一致")

    feature_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    original_fold_counts = (
        feature_table.groupby("fold").size().astype(int).to_dict()
    )
    feature_table = remap_feature_table_folds(feature_table, registry)
    new_fold_counts = feature_table.groupby("fold").size().astype(int).to_dict()

    if args.mode == "smoke":
        run_smoke(feature_table, registry, model_features)
        return

    model_params = read_json(
        CLEAN_ROOT / str(experiment_config["model_config"])
    )["params"]
    fingerprint = build_experiment_fingerprint(
        experiment_config,
        model_params,
        actual_registry_sha,
        actual_base_cache_sha,
        actual_candidate_cache_sha,
        candidate_cache_provenance,
    )
    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", experiment_config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(
        artifact_dir / "cache_and_fold_provenance.json",
        {
            "fingerprint": fingerprint,
            "fold_remapped": True,
            "remap_key": "well_id",
            "cached_fold_column_ignored": True,
            "old_fold_row_counts": original_fold_counts,
            "new_fold_row_counts": new_fold_counts,
            "registry_sha256": actual_registry_sha,
            "base_cache_sha256": actual_base_cache_sha,
            "candidate_cache_sha256": actual_candidate_cache_sha,
            "candidate_cache_old_registry_is_lineage_only": True,
            "candidate_cache_provenance": candidate_cache_provenance,
        },
    )

    fold_ids = list(range(5)) if args.mode == "all" else [int(args.mode[-1])]
    for fold_id in fold_ids:
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            artifact_dir,
            fingerprint,
            model_features,
        )
        save_fold_runtime_row(artifact_dir, runtime)

    complete_metrics = finalize_complete_cv(
        artifact_dir,
        fingerprint,
        expected_rows,
        experiment_config,
        model_params,
        model_features,
    )
    if complete_metrics is not None:
        validate_complete_predictions(artifact_dir, registry, expected_rows)


if __name__ == "__main__":
    main()

