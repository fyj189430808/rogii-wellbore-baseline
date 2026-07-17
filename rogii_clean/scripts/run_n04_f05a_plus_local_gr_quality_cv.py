"""运行冻结 F05a 36 列加 RF02b 九列局部 GR 质量特征的 LightGBM CV。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_f05a_deterministic_candidates_cv import (  # noqa: E402
    _validate_cache_metadata as validate_candidate_cache_metadata,
    load_aligned_feature_table,
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


EXPERIMENT_ID = "N04_f05a_plus_local_gr_quality_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "n04_f05a_plus_local_gr_quality_v1.json"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
KEY_COLUMNS = ["well_id", "row_index"]
LOCAL_GR_COLUMNS = [
    "gr_local_observed_count_50",
    "gr_local_mad_50",
    "gr_local_variance_50",
    "gr_local_observed_count_100",
    "gr_local_mad_100",
    "gr_local_variance_100",
    "gr_local_observed_count_200",
    "gr_local_mad_200",
    "gr_local_variance_200",
]
COUNT_COLUMNS = [
    "gr_local_observed_count_50",
    "gr_local_observed_count_100",
    "gr_local_observed_count_200",
]
ALLOW_NAN_COLUMNS = [column for column in LOCAL_GR_COLUMNS if column not in COUNT_COLUMNS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 N04 局部 GR 质量组合实验")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _validate_config(config: dict[str, object]) -> list[str]:
    """锁定 F05a 36 列，并只允许增加指定的九列局部 GR 特征。"""

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *LOCAL_GR_COLUMNS,
    ]
    required_values = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "F05a_deterministic_candidates_v1",
        "feature_count": len(expected_features),
        "feature_columns": expected_features,
        "candidate_feature_columns": DIRECT_CANDIDATE_COLUMNS,
        "new_feature_columns": LOCAL_GR_COLUMNS,
        "allow_nan_features": ALLOW_NAN_COLUMNS,
        "model_config": "configs/lgbm_feature_baseline_v1.json",
        "n_estimators": 1734,
        "model_seed": 29,
        "early_stopping": False,
        "cache_seed": 42,
        "candidate_cache_rebuild_allowed": False,
        "target": "target_delta",
        "supported_modes": SUPPORTED_MODES,
    }
    for key, expected in required_values.items():
        if config.get(key) != expected:
            raise ValueError(f"N04 配置项 {key} 已偏离冻结合同")
    return expected_features


def _normalised_keys(rows: pd.DataFrame) -> pd.DataFrame:
    missing = set(KEY_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"缓存缺少连接键：{sorted(missing)}")
    return pd.DataFrame(
        {
            "well_id": rows["well_id"].astype(str).to_numpy(),
            "row_index": pd.to_numeric(
                rows["row_index"], errors="raise"
            ).to_numpy(dtype=np.int64),
        }
    )


def add_local_gr_features(
    f05a_rows: pd.DataFrame,
    local_gr_rows: pd.DataFrame,
) -> pd.DataFrame:
    """按复合键一一连接九列，并保持 F05a 原始行序。"""

    missing_features = set(LOCAL_GR_COLUMNS) - set(local_gr_rows.columns)
    if missing_features:
        raise ValueError(f"RF02b 缓存缺少特征：{sorted(missing_features)}")
    if len(f05a_rows) != len(local_gr_rows):
        raise ValueError("F05a 与 RF02b 的键不能一一对应：行数不同")

    left_keys = _normalised_keys(f05a_rows)
    right_keys = _normalised_keys(local_gr_rows)
    if left_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("F05a 特征表含重复键")
    if right_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("RF02b 缓存含重复键")

    right_values = local_gr_rows[LOCAL_GR_COLUMNS].reset_index(drop=True)
    if left_keys.equals(right_keys):
        aligned_values = right_values
    else:
        right_selected = pd.concat([right_keys, right_values], axis=1)
        left_lookup = left_keys.copy()
        left_lookup["__n04_row_order"] = np.arange(
            len(left_lookup), dtype=np.int64
        )
        aligned = left_lookup.merge(
            right_selected,
            on=KEY_COLUMNS,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if not aligned["_merge"].eq("both").all():
            raise ValueError("F05a 与 RF02b 的键不能一一对应：存在缺失或额外键")
        aligned = aligned.sort_values("__n04_row_order", kind="stable")
        aligned_values = aligned[LOCAL_GR_COLUMNS].reset_index(drop=True)

    return pd.concat(
        [f05a_rows.reset_index(drop=True), aligned_values],
        axis=1,
    )


def _validate_local_gr_values(rows: pd.DataFrame) -> None:
    """count 必须有限，MAD/variance 只允许支持不足产生的 NaN。"""

    missing = set(LOCAL_GR_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"局部 GR 表缺少特征：{sorted(missing)}")
    for column in LOCAL_GR_COLUMNS:
        values = rows[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"局部 GR 特征 {column} 含 Inf")
    if rows[COUNT_COLUMNS].isna().any().any():
        raise ValueError("局部 GR count 特征含 NaN")
    if (rows[COUNT_COLUMNS].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError("局部 GR count 特征含负数")


def require_cache_bundle(*paths: Path) -> None:
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 N04 冻结上游文件；runner 不会重建缓存：\n"
            + "\n".join(missing)
        )


def _require_sha(path: Path, expected_sha256: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 与 N04 冻结配置不一致")
    return actual


def _validate_local_gr_metadata(
    metadata: dict[str, object],
    expected_wells: int,
    expected_rows: int,
) -> None:
    if int(metadata.get("wells", -1)) != expected_wells:
        raise ValueError("RF02b 缓存元数据井数不正确")
    if int(metadata.get("rows", -1)) != expected_rows:
        raise ValueError("RF02b 缓存元数据行数不正确")
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("RF02b 缓存元数据缺少合法 fingerprint")


def _validate_model_params(model_params: dict[str, object]) -> None:
    if int(model_params.get("n_estimators", -1)) != 1734:
        raise ValueError("N04 必须固定使用 1734 棵树")
    for seed_key in (
        "random_state",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
    ):
        if int(model_params.get(seed_key, -1)) != 29:
            raise ValueError(f"N04 模型参数 {seed_key} 必须固定为 29")


def build_experiment_fingerprint(
    *,
    config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    runner_hash: str,
    candidate_cache_sha256: str,
    candidate_metadata_sha256: str,
    local_gr_cache_sha256: str,
    local_gr_metadata_sha256: str,
) -> str:
    payload = {
        "config": config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "runner_sha256": runner_hash,
        "candidate_cache_sha256": candidate_cache_sha256,
        "candidate_metadata_sha256": candidate_metadata_sha256,
        "local_gr_cache_sha256": local_gr_cache_sha256,
        "local_gr_metadata_sha256": local_gr_metadata_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_smoke(feature_table: pd.DataFrame) -> None:
    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    smoke_rows = feature_table.loc[
        feature_table["well_id"].astype(str).isin(smoke_wells)
    ]
    _validate_local_gr_values(smoke_rows)
    if len(smoke_wells) != 3:
        raise ValueError("N04 smoke 没有取到三口井")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，新增特征=9",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_json(args.config.resolve())
    model_features = _validate_config(config)
    expected_wells = int(config["expected_wells"])
    expected_rows = int(config["expected_rows"])

    base_cache_path = CLEAN_ROOT / str(config["base_feature_cache"])
    candidate_cache_path = CLEAN_ROOT / str(config["candidate_feature_cache"])
    candidate_metadata_path = CLEAN_ROOT / str(config["candidate_cache_metadata"])
    candidate_parameter_path = CLEAN_ROOT / str(config["candidate_parameter_config"])
    local_gr_cache_path = CLEAN_ROOT / str(config["local_gr_feature_cache"])
    local_gr_metadata_path = CLEAN_ROOT / str(
        config["local_gr_feature_cache_metadata"]
    )
    local_gr_parameter_path = CLEAN_ROOT / str(config["local_gr_parameter_config"])
    model_config_path = CLEAN_ROOT / str(config["model_config"])
    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    require_cache_bundle(
        base_cache_path,
        candidate_cache_path,
        candidate_metadata_path,
        candidate_parameter_path,
        local_gr_cache_path,
        local_gr_metadata_path,
        local_gr_parameter_path,
        model_config_path,
        registry_path,
    )

    base_cache_sha = _require_sha(
        base_cache_path, str(config["base_feature_cache_sha256"]), "B00 特征缓存"
    )
    candidate_cache_sha = _require_sha(
        candidate_cache_path, str(config["candidate_cache_sha256"]), "F05a 候选缓存"
    )
    candidate_metadata_sha = _require_sha(
        candidate_metadata_path,
        str(config["candidate_cache_metadata_sha256"]),
        "F05a 缓存元数据",
    )
    _require_sha(
        candidate_parameter_path,
        str(config["candidate_parameter_config_sha256"]),
        "F05a 参数配置",
    )
    local_gr_cache_sha = _require_sha(
        local_gr_cache_path,
        str(config["local_gr_feature_cache_sha256"]),
        "RF02b 特征缓存",
    )
    local_gr_metadata_sha = _require_sha(
        local_gr_metadata_path,
        str(config["local_gr_feature_cache_metadata_sha256"]),
        "RF02b 缓存元数据",
    )
    _require_sha(
        local_gr_parameter_path,
        str(config["local_gr_parameter_config_sha256"]),
        "RF02b 参数配置",
    )
    _require_sha(
        model_config_path,
        str(config["model_config_sha256"]),
        "LightGBM 参数配置",
    )
    registry_hash = _require_sha(
        registry_path, str(config["fold_registry_sha256"]), "fold 注册表"
    )

    candidate_metadata = read_json(candidate_metadata_path)
    validate_candidate_cache_metadata(
        candidate_metadata, expected_wells, expected_rows
    )
    local_gr_metadata = read_json(local_gr_metadata_path)
    _validate_local_gr_metadata(local_gr_metadata, expected_wells, expected_rows)

    f05a_table = load_aligned_feature_table(
        base_cache_path, candidate_cache_path, expected_rows
    )
    local_gr_rows = pd.read_parquet(
        local_gr_cache_path, columns=[*KEY_COLUMNS, *LOCAL_GR_COLUMNS]
    )
    if len(local_gr_rows) != expected_rows:
        raise ValueError("RF02b 缓存行数与固定评价行不一致")
    _validate_local_gr_values(local_gr_rows)
    feature_table = add_local_gr_features(f05a_table, local_gr_rows)
    if args.mode == "smoke":
        _run_smoke(feature_table)
        return

    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    model_params = read_json(model_config_path)["params"]
    _validate_model_params(model_params)
    fingerprint = build_experiment_fingerprint(
        config=config,
        model_params=model_params,
        registry_hash=registry_hash,
        runner_hash=file_sha256(Path(__file__).resolve()),
        candidate_cache_sha256=candidate_cache_sha,
        candidate_metadata_sha256=candidate_metadata_sha,
        local_gr_cache_sha256=local_gr_cache_sha,
        local_gr_metadata_sha256=local_gr_metadata_sha,
    )

    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(
        artifact_dir / "cache_provenance.json",
        {
            "base_cache_sha256": base_cache_sha,
            "candidate_cache_sha256": candidate_cache_sha,
            "candidate_metadata_sha256": candidate_metadata_sha,
            "candidate_metadata": candidate_metadata,
            "local_gr_cache_sha256": local_gr_cache_sha,
            "local_gr_metadata_sha256": local_gr_metadata_sha,
            "local_gr_metadata": local_gr_metadata,
        },
    )

    fold_id = int(args.mode[-1])
    runtime = train_fold(
        feature_table,
        registry,
        fold_id,
        model_params,
        artifact_dir,
        fingerprint,
        model_features,
    )
    metrics_path = artifact_dir / "metrics.csv"
    previous = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    if not previous.empty:
        previous = previous.loc[previous["fold"] != fold_id]
    pd.concat([previous, pd.DataFrame([runtime])], ignore_index=True).sort_values(
        "fold"
    ).to_csv(metrics_path, index=False)
    finalize_complete_cv(
        artifact_dir,
        fingerprint,
        expected_rows,
        config,
        model_params,
        model_features,
    )


if __name__ == "__main__":
    main()
