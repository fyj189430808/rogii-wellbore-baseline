"""运行冻结 F05a 36 列加 F05b 10 列内部统计的固定 LightGBM CV。"""

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
from scripts.run_f05b_internal_stats_cv import (  # noqa: E402
    _validate_cache_metadata as validate_internal_stats_cache_metadata,
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
from src.f05b_internal_stats import F05B_FEATURE_COLUMNS  # noqa: E402
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "N03_f05a_plus_pf_internal_stats_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "n03_f05a_plus_pf_internal_stats_v1.json"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
KEY_COLUMNS = ["well_id", "row_index"]
F05B_COLUMNS = list(F05B_FEATURE_COLUMNS)
EXPECTED_INTERNAL_STATS_COLUMNS = [*KEY_COLUMNS, *F05B_COLUMNS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke 或一个固定折；本 runner 没有重建缓存选项。"""

    parser = argparse.ArgumentParser(description="运行 N03 PF 内部统计组合实验")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _validate_config(config: dict[str, object]) -> list[str]:
    """冻结 F05a 36 列，仅允许追加既有 F05b 十列。"""

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *F05B_COLUMNS,
    ]
    required_values = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "F05a_deterministic_candidates_v1",
        "feature_count": len(expected_features),
        "feature_columns": expected_features,
        "candidate_feature_columns": DIRECT_CANDIDATE_COLUMNS,
        "new_feature_columns": F05B_COLUMNS,
        "model_config": "configs/lgbm_feature_baseline_v1.json",
        "n_estimators": 1734,
        "model_seed": 29,
        "early_stopping": False,
        "cache_seed": 42,
        "cache_rebuild_allowed": False,
        "target": "target_delta",
        "supported_modes": SUPPORTED_MODES,
    }
    for key, expected in required_values.items():
        if config.get(key) != expected:
            raise ValueError(f"N03 配置项 {key} 已偏离冻结合同")
    return expected_features


def _normalised_keys(rows: pd.DataFrame) -> pd.DataFrame:
    """将复合键统一为字符串井号和 int64 行号。"""

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


def add_internal_stats_features(
    f05a_rows: pd.DataFrame,
    internal_stats_rows: pd.DataFrame,
) -> pd.DataFrame:
    """按 well_id、row_index 一一连接十列，并保持 F05a 原始行序。"""

    missing_features = set(F05B_COLUMNS) - set(internal_stats_rows.columns)
    if missing_features:
        raise ValueError(f"F05b 缓存缺少特征：{sorted(missing_features)}")
    if len(f05a_rows) != len(internal_stats_rows):
        raise ValueError("F05a 与 F05b 的键不能一一对应：行数不同")

    left_keys = _normalised_keys(f05a_rows)
    right_keys = _normalised_keys(internal_stats_rows)
    if left_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("F05a 特征表含重复键")
    if right_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("F05b 缓存含重复键")

    right_values = internal_stats_rows[F05B_COLUMNS].reset_index(drop=True)
    if left_keys.equals(right_keys):
        aligned_values = right_values
    else:
        right_selected = pd.concat([right_keys, right_values], axis=1)
        left_lookup = left_keys.copy()
        left_lookup["__n03_row_order"] = np.arange(
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
            raise ValueError("F05a 与 F05b 的键不能一一对应：存在缺失或额外键")
        aligned = aligned.sort_values("__n03_row_order", kind="stable")
        aligned_values = aligned[F05B_COLUMNS].reset_index(drop=True)

    return pd.concat(
        [f05a_rows.reset_index(drop=True), aligned_values],
        axis=1,
    )


def require_cache_bundle(*paths: Path) -> None:
    """要求所有上游缓存均已存在；训练和 smoke 都不会重建。"""

    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 N03 冻结上游文件；runner 不会重建缓存：\n"
            + "\n".join(missing)
        )


def _require_sha(path: Path, expected_sha256: str, label: str) -> str:
    """计算并核对一个冻结文件的 SHA-256。"""

    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 与 N03 冻结配置不一致")
    return actual


def _validate_cache_lineage(
    metadata: dict[str, object],
    expected_parameter_sha256: str,
    expected_registry_sha256: str,
    label: str,
) -> None:
    """核对生成缓存时使用的参数版本、seed 和 fold 注册表。"""

    if int(metadata.get("seed", -1)) != 42:
        raise ValueError(f"{label} 不是固定 seed=42")
    lineage = metadata.get("base_lineage")
    if not isinstance(lineage, dict):
        raise ValueError(f"{label} 缺少 base_lineage")
    if int(lineage.get("seed", -1)) != 42:
        raise ValueError(f"{label} lineage 不是固定 seed=42")
    if lineage.get("parameter_json_sha256") != expected_parameter_sha256:
        raise ValueError(f"{label} 参数版本已变化")
    if lineage.get("registry_sha256") != expected_registry_sha256:
        raise ValueError(f"{label} 使用的 fold 注册表已变化")


def build_experiment_fingerprint(
    *,
    config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    runner_hash: str,
    candidate_cache_sha256: str,
    candidate_metadata_sha256: str,
    internal_stats_cache_sha256: str,
    internal_stats_metadata_sha256: str,
) -> str:
    """把固定模型、fold 和两个缓存/元数据 SHA 一起写入实验指纹。"""

    payload = {
        "config": config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "runner_sha256": runner_hash,
        "candidate_cache_sha256": candidate_cache_sha256,
        "candidate_metadata_sha256": candidate_metadata_sha256,
        "internal_stats_cache_sha256": internal_stats_cache_sha256,
        "internal_stats_metadata_sha256": internal_stats_metadata_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_internal_stats_cache(cache_path: Path, expected_rows: int) -> pd.DataFrame:
    """读取并检查既有 F05b 十列缓存，不接受其他正式输入列。"""

    rows = pd.read_parquet(cache_path)
    if list(rows.columns) != EXPECTED_INTERNAL_STATS_COLUMNS:
        raise ValueError("F05b 缓存列或顺序不正确")
    if len(rows) != expected_rows:
        raise ValueError("F05b 缓存行数与固定评价行不一致")
    if rows.duplicated(KEY_COLUMNS).any():
        raise ValueError("F05b 缓存含重复键")
    values = rows[F05B_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("F05b 十列含 NaN 或 Inf")
    return rows


def _validate_model_params(model_params: dict[str, object]) -> None:
    """确认实际送入训练器的树数和所有随机种子均未改变。"""

    if int(model_params.get("n_estimators", -1)) != 1734:
        raise ValueError("N03 必须固定使用 1734 棵树")
    for seed_key in (
        "random_state",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
    ):
        if int(model_params.get(seed_key, -1)) != 29:
            raise ValueError(f"N03 模型参数 {seed_key} 必须固定为 29")


def _run_smoke(feature_table: pd.DataFrame) -> None:
    """检查前三口井的新十列、键和有限值，不训练模型。"""

    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    smoke_rows = feature_table.loc[
        feature_table["well_id"].astype(str).isin(smoke_wells)
    ]
    values = smoke_rows[F05B_COLUMNS].to_numpy(dtype=np.float32)
    if len(smoke_wells) != 3 or not np.isfinite(values).all():
        raise ValueError("N03 smoke 检查失败")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，新增特征=10",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """只读两个冻结缓存，运行 smoke 或一个指定 fold。"""

    args = parse_args(argv)
    config = read_json(args.config.resolve())
    model_features = _validate_config(config)
    expected_wells = int(config["expected_wells"])
    expected_rows = int(config["expected_rows"])

    base_cache_path = CLEAN_ROOT / str(config["base_feature_cache"])
    candidate_cache_path = CLEAN_ROOT / str(config["candidate_feature_cache"])
    candidate_metadata_path = CLEAN_ROOT / str(config["candidate_cache_metadata"])
    internal_stats_cache_path = CLEAN_ROOT / str(config["internal_stats_cache"])
    internal_stats_metadata_path = CLEAN_ROOT / str(
        config["internal_stats_cache_metadata"]
    )
    candidate_parameter_path = CLEAN_ROOT / str(
        config["candidate_parameter_config"]
    )
    internal_stats_parameter_path = CLEAN_ROOT / str(
        config["internal_stats_parameter_config"]
    )
    model_config_path = CLEAN_ROOT / str(config["model_config"])
    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    require_cache_bundle(
        base_cache_path,
        candidate_cache_path,
        candidate_metadata_path,
        internal_stats_cache_path,
        internal_stats_metadata_path,
        candidate_parameter_path,
        internal_stats_parameter_path,
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
    internal_stats_cache_sha = _require_sha(
        internal_stats_cache_path,
        str(config["internal_stats_cache_sha256"]),
        "F05b 内部统计缓存",
    )
    internal_stats_metadata_sha = _require_sha(
        internal_stats_metadata_path,
        str(config["internal_stats_cache_metadata_sha256"]),
        "F05b 缓存元数据",
    )
    _require_sha(
        candidate_parameter_path,
        str(config["candidate_parameter_config_sha256"]),
        "F05a 参数配置",
    )
    _require_sha(
        internal_stats_parameter_path,
        str(config["internal_stats_parameter_config_sha256"]),
        "F05b 参数配置",
    )
    _require_sha(
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
    internal_stats_metadata = read_json(internal_stats_metadata_path)
    validate_candidate_cache_metadata(
        candidate_metadata,
        expected_wells,
        expected_rows,
    )
    validate_internal_stats_cache_metadata(
        internal_stats_metadata,
        expected_wells,
        expected_rows,
    )
    _validate_cache_lineage(
        candidate_metadata,
        str(config["candidate_parameter_config_sha256"]),
        registry_hash,
        "F05a 候选缓存",
    )
    _validate_cache_lineage(
        internal_stats_metadata,
        str(config["internal_stats_parameter_config_sha256"]),
        registry_hash,
        "F05b 内部统计缓存",
    )

    f05a_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    internal_stats_rows = _load_internal_stats_cache(
        internal_stats_cache_path,
        expected_rows,
    )
    feature_table = add_internal_stats_features(f05a_table, internal_stats_rows)
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
        internal_stats_cache_sha256=internal_stats_cache_sha,
        internal_stats_metadata_sha256=internal_stats_metadata_sha,
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
            "internal_stats_cache_sha256": internal_stats_cache_sha,
            "internal_stats_metadata_sha256": internal_stats_metadata_sha,
            "internal_stats_metadata": internal_stats_metadata,
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
