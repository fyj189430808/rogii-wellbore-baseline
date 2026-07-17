"""运行冻结 F05a 36 列加 F03b 12 列几何偏移景观特征的 CV。"""

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
from src.f03b_geometry_landscape_features import (  # noqa: E402
    F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
)
from src.f05a_direct_physical_candidates import (  # noqa: E402
    DIRECT_CANDIDATE_COLUMNS,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "N06_f05a_plus_geometry_offset_landscape_v1"
DEFAULT_CONFIG = (
    CLEAN_ROOT / "configs" / "n06_f05a_plus_geometry_offset_landscape_v1.json"
)
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
KEY_COLUMNS = ["well_id", "row_index"]
LANDSCAPE_COLUMNS = list(F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS)
ALLOW_NAN_COLUMNS = [
    "f03b_best_offset",
    "f03b_best_ncc",
    "f03b_second_offset",
    "f03b_second_ncc",
    "f03b_peak_separation",
    "f03b_peak_gap",
    "f03b_soft_offset_mean",
    "f03b_soft_offset_std",
    "f03b_offset_entropy",
    "f03b_best_basin_mass",
]
REQUIRE_FINITE_COLUMNS = [
    column for column in LANDSCAPE_COLUMNS if column not in ALLOW_NAN_COLUMNS
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 N06 几何偏移景观组合实验")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _validate_config(config: dict[str, object]) -> list[str]:
    """锁定 F05a 36 列，只允许追加既有 F03b 十二列。"""

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *LANDSCAPE_COLUMNS,
    ]
    required_values = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "F05a_deterministic_candidates_v1",
        "feature_count": len(expected_features),
        "feature_columns": expected_features,
        "candidate_feature_columns": DIRECT_CANDIDATE_COLUMNS,
        "new_feature_columns": LANDSCAPE_COLUMNS,
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
            raise ValueError(f"N06 配置项 {key} 已偏离冻结组合合同")
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


def add_landscape_features(
    f05a_rows: pd.DataFrame,
    landscape_rows: pd.DataFrame,
) -> pd.DataFrame:
    """严格按 well_id、row_index 一一连接，并恢复 F05a 原始行序。"""

    missing_features = set(LANDSCAPE_COLUMNS) - set(landscape_rows.columns)
    if missing_features:
        raise ValueError(f"F03b 缓存缺少特征：{sorted(missing_features)}")
    if len(f05a_rows) != len(landscape_rows):
        raise ValueError("F05a 与 F03b 的键不能一一对应：行数不同")

    left_keys = _normalised_keys(f05a_rows)
    right_keys = _normalised_keys(landscape_rows)
    if left_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("F05a 特征表含重复键")
    if right_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("F03b 缓存含重复键")

    right_values = landscape_rows[LANDSCAPE_COLUMNS].reset_index(drop=True)
    if left_keys.equals(right_keys):
        aligned_values = right_values
    else:
        right_selected = pd.concat([right_keys, right_values], axis=1)
        left_lookup = left_keys.copy()
        left_lookup["__n06_row_order"] = np.arange(
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
            raise ValueError("F05a 与 F03b 的键不能一一对应：存在缺失或额外键")
        aligned = aligned.sort_values("__n06_row_order", kind="stable")
        aligned_values = aligned[LANDSCAPE_COLUMNS].reset_index(drop=True)

    return pd.concat(
        [f05a_rows.reset_index(drop=True), aligned_values],
        axis=1,
    )


def _validate_landscape_values(rows: pd.DataFrame) -> None:
    """沿用 F03b 原配置的 NaN 许可，但任何列都不允许 Inf。"""

    missing = set(LANDSCAPE_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"F03b 表缺少特征：{sorted(missing)}")
    for column in LANDSCAPE_COLUMNS:
        values = rows[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"F03b 特征 {column} 含 Inf")
    if rows[REQUIRE_FINITE_COLUMNS].isna().any().any():
        raise ValueError("F03b 有效配对数量或比例含 NaN")


def require_cache_bundle(*paths: Path) -> None:
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 N06 冻结上游文件，runner 不会重建缓存：\n"
            + "\n".join(missing)
        )


def _require_sha(path: Path, expected_sha256: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 与 N06 冻结配置不一致")
    return actual


def _validate_landscape_metadata(
    metadata: dict[str, object],
    expected_wells: int,
    expected_rows: int,
    expected_config_sha: str,
    expected_registry_sha: str,
    expected_model_sha: str,
) -> None:
    if int(metadata.get("wells", -1)) != expected_wells:
        raise ValueError("F03b 缓存元数据井数不正确")
    if int(metadata.get("rows", -1)) != expected_rows:
        raise ValueError("F03b 缓存元数据行数不正确")
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("F03b 缓存元数据缺少合法 fingerprint")
    manifest = metadata.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("F03b 缓存元数据缺少 manifest")
    expected_manifest = {
        "config_sha256": expected_config_sha,
        "fold_registry_sha256": expected_registry_sha,
        "model_config_sha256": expected_model_sha,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"F03b 缓存元数据 {key} 已变化")
    raw_contract = manifest.get("raw_source_contract")
    if not isinstance(raw_contract, dict) or raw_contract.get(
        "hidden_tvt_feature_access"
    ) is not False:
        raise ValueError("F03b 缓存没有声明禁止读取隐藏 TVT")


def _validate_model_params(model_params: dict[str, object]) -> None:
    if int(model_params.get("n_estimators", -1)) != 1734:
        raise ValueError("N06 必须固定使用 1734 棵树")
    for seed_key in (
        "random_state",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
    ):
        if int(model_params.get(seed_key, -1)) != 29:
            raise ValueError(f"N06 模型参数 {seed_key} 必须固定为 29")


def build_experiment_fingerprint(
    *,
    config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    runner_hash: str,
    candidate_cache_sha256: str,
    candidate_metadata_sha256: str,
    landscape_cache_sha256: str,
    landscape_metadata_sha256: str,
) -> str:
    payload = {
        "config": config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "runner_sha256": runner_hash,
        "candidate_cache_sha256": candidate_cache_sha256,
        "candidate_metadata_sha256": candidate_metadata_sha256,
        "landscape_cache_sha256": landscape_cache_sha256,
        "landscape_metadata_sha256": landscape_metadata_sha256,
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
    _validate_landscape_values(smoke_rows)
    if len(smoke_wells) != 3:
        raise ValueError("N06 smoke 没有取到三口井")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，新增特征=12",
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
    landscape_cache_path = CLEAN_ROOT / str(config["landscape_feature_cache"])
    landscape_metadata_path = CLEAN_ROOT / str(
        config["landscape_feature_cache_metadata"]
    )
    landscape_parameter_path = CLEAN_ROOT / str(
        config["landscape_parameter_config"]
    )
    model_config_path = CLEAN_ROOT / str(config["model_config"])
    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    require_cache_bundle(
        base_cache_path,
        candidate_cache_path,
        candidate_metadata_path,
        candidate_parameter_path,
        landscape_cache_path,
        landscape_metadata_path,
        landscape_parameter_path,
        model_config_path,
        registry_path,
    )

    base_cache_sha = _require_sha(
        base_cache_path,
        str(config["base_feature_cache_sha256"]),
        "B00 特征缓存",
    )
    candidate_cache_sha = _require_sha(
        candidate_cache_path,
        str(config["candidate_cache_sha256"]),
        "F05a 候选缓存",
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
    landscape_cache_sha = _require_sha(
        landscape_cache_path,
        str(config["landscape_feature_cache_sha256"]),
        "F03b 景观特征缓存",
    )
    landscape_metadata_sha = _require_sha(
        landscape_metadata_path,
        str(config["landscape_feature_cache_metadata_sha256"]),
        "F03b 缓存元数据",
    )
    _require_sha(
        landscape_parameter_path,
        str(config["landscape_parameter_config_sha256"]),
        "F03b 参数配置",
    )
    model_config_sha = _require_sha(
        model_config_path,
        str(config["model_config_sha256"]),
        "LightGBM 参数配置",
    )
    registry_hash = _require_sha(
        registry_path,
        str(config["fold_registry_sha256"]),
        "fold 注册表",
    )

    candidate_metadata = read_json(candidate_metadata_path)
    validate_candidate_cache_metadata(
        candidate_metadata,
        expected_wells,
        expected_rows,
    )
    landscape_metadata = read_json(landscape_metadata_path)
    _validate_landscape_metadata(
        landscape_metadata,
        expected_wells,
        expected_rows,
        str(config["landscape_parameter_config_sha256"]),
        registry_hash,
        model_config_sha,
    )

    f05a_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    landscape_rows = pd.read_parquet(
        landscape_cache_path,
        columns=[*KEY_COLUMNS, *LANDSCAPE_COLUMNS],
    )
    if len(landscape_rows) != expected_rows:
        raise ValueError("F03b 缓存行数与固定评价行不一致")
    _validate_landscape_values(landscape_rows)
    feature_table = add_landscape_features(f05a_table, landscape_rows)
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
        landscape_cache_sha256=landscape_cache_sha,
        landscape_metadata_sha256=landscape_metadata_sha,
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
            "landscape_cache_sha256": landscape_cache_sha,
            "landscape_metadata_sha256": landscape_metadata_sha,
            "landscape_metadata": landscape_metadata,
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
