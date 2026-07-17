"""运行确定性 F05a 加九个 GR 缺失段几何特征的固定 LightGBM CV。"""

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


EXPERIMENT_ID = "N02_f05a_plus_gr_gap_geometry_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "n02_f05a_plus_gr_gap_geometry_v1.json"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]
KEY_COLUMNS = ["well_id", "row_index"]
GR_GAP_COLUMNS = [
    "gr_gap_length_ft",
    "gr_distance_to_left_observed_ft",
    "gr_distance_to_right_observed_ft",
    "gr_relative_position_inside_gap",
    "gr_valid_fraction_50",
    "gr_valid_fraction_100",
    "gr_valid_fraction_200",
    "well_hidden_gr_valid_fraction",
    "well_hidden_longest_gr_gap_ft",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke 或一个固定折；两个上游缓存始终只读。"""

    parser = argparse.ArgumentParser(description="运行 N02 GR 缺失段几何组合实验")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _validate_config(experiment_config: dict[str, object]) -> list[str]:
    """锁定 F05a 的 36 列，并只允许追加约定的九列。"""

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *GR_GAP_COLUMNS,
    ]
    if experiment_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("实验 ID 与 N02 runner 不一致")
    if experiment_config.get("baseline_id") != "F05a_deterministic_candidates_v1":
        raise ValueError("N02 基线不是冻结的确定性 F05a")
    if experiment_config.get("feature_columns") != expected_features:
        raise ValueError("N02 特征顺序不是 F05a 36 列加固定九列")
    if experiment_config.get("candidate_feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("F05a 的 24 个候选特征或顺序发生变化")
    if experiment_config.get("new_feature_columns") != GR_GAP_COLUMNS:
        raise ValueError("N02 的九个新增特征或顺序发生变化")
    if experiment_config.get("candidate_cache_rebuild_allowed") is not False:
        raise ValueError("N02 禁止重建上游缓存")
    if experiment_config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("配置支持的运行模式与 runner 不一致")
    return expected_features


def _normalised_keys(rows: pd.DataFrame) -> pd.DataFrame:
    """把两个缓存的连接键统一为字符串井号和 int64 行号。"""

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


def add_gr_gap_features(
    f05a_rows: pd.DataFrame,
    gr_gap_cache_rows: pd.DataFrame,
) -> pd.DataFrame:
    """按 well_id、row_index 一一连接九列，并保持 F05a 原始行序。"""

    missing_features = set(GR_GAP_COLUMNS) - set(gr_gap_cache_rows.columns)
    if missing_features:
        raise ValueError(f"GR 缺失段缓存缺少特征：{sorted(missing_features)}")
    if len(f05a_rows) != len(gr_gap_cache_rows):
        raise ValueError("两个缓存的键不能一一对应：行数不同")

    left_keys = _normalised_keys(f05a_rows)
    right_keys = _normalised_keys(gr_gap_cache_rows)
    if left_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("F05a 缓存含重复键")
    if right_keys.duplicated(KEY_COLUMNS).any():
        raise ValueError("GR 缺失段缓存含重复键")

    right_values = gr_gap_cache_rows[GR_GAP_COLUMNS].reset_index(drop=True)
    same_order = left_keys.equals(right_keys)
    if same_order:
        aligned_values = right_values
    else:
        right_selected = pd.concat([right_keys, right_values], axis=1)
        left_lookup = left_keys.copy()
        left_lookup["__n02_row_order"] = np.arange(len(left_lookup), dtype=np.int64)
        aligned = left_lookup.merge(
            right_selected,
            on=KEY_COLUMNS,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if not aligned["_merge"].eq("both").all():
            raise ValueError("两个缓存的键不能一一对应：存在缺失键")
        aligned = aligned.sort_values("__n02_row_order", kind="stable")
        aligned_values = aligned[GR_GAP_COLUMNS].reset_index(drop=True)

    return pd.concat(
        [f05a_rows.reset_index(drop=True), aligned_values],
        axis=1,
    )


def _validate_gr_gap_values(
    feature_table: pd.DataFrame,
    allow_nan_features: list[str],
) -> None:
    """允许约定的边界 NaN，但所有新增特征都禁止正负无穷。"""

    values = feature_table[GR_GAP_COLUMNS].to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("GR 缺失段特征含 Inf")
    disallowed_nan_columns = [
        column for column in GR_GAP_COLUMNS if column not in allow_nan_features
    ]
    if feature_table[disallowed_nan_columns].isna().any().any():
        raise ValueError("不允许缺失的 GR 缺失段特征含 NaN")


def read_gr_gap_cache_provenance(
    cache_path: Path,
    metadata_path: Path,
    expected_cache_sha256: str,
    expected_metadata_sha256: str,
) -> dict[str, object]:
    """核对 RF02a 缓存内容 SHA，并返回进入实验指纹的来源信息。"""

    for path in (cache_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 RF02a 冻结缓存文件：{path.resolve()}")
    actual_cache_sha256 = file_sha256(cache_path)
    actual_metadata_sha256 = file_sha256(metadata_path)
    if actual_cache_sha256 != expected_cache_sha256:
        raise ValueError("RF02a 特征缓存 SHA-256 与冻结配置不一致")
    if actual_metadata_sha256 != expected_metadata_sha256:
        raise ValueError("RF02a 缓存元数据 SHA-256 与冻结配置不一致")
    return {
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": actual_cache_sha256,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": actual_metadata_sha256,
        "metadata": read_json(metadata_path),
    }


def _experiment_fingerprint(
    experiment_config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    candidate_cache_provenance: dict[str, object],
    gr_gap_cache_provenance: dict[str, object],
) -> str:
    """把配置、模型、fold 和两个缓存的来源共同写入 SHA-256 指纹。"""

    payload = {
        "experiment_config": experiment_config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "candidate_cache_provenance": candidate_cache_provenance,
        "gr_gap_cache_provenance": gr_gap_cache_provenance,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_smoke(feature_table: pd.DataFrame, allow_nan_features: list[str]) -> None:
    """只检查前三口井的九个新增特征和连接结果，不训练模型。"""

    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    smoke_rows = feature_table.loc[
        feature_table["well_id"].astype(str).isin(smoke_wells)
    ]
    _validate_gr_gap_values(smoke_rows, allow_nan_features)
    if len(smoke_wells) != 3:
        raise ValueError("N02 smoke 没有取到三口井")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，新增特征=9",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """只读 F05a 和 RF02a 冻结缓存，运行 smoke 或一个指定 fold。"""

    args = parse_args(argv)
    experiment_config = read_json(args.config.resolve())
    model_features = _validate_config(experiment_config)
    expected_wells = int(experiment_config["expected_wells"])
    expected_rows = int(experiment_config["expected_rows"])
    allow_nan_features = list(experiment_config["allow_nan_features"])

    candidate_cache_path = CLEAN_ROOT / str(
        experiment_config["candidate_feature_cache"]
    )
    candidate_metadata_path = CLEAN_ROOT / str(
        experiment_config["candidate_cache_metadata"]
    )
    require_candidate_cache_files(candidate_cache_path, candidate_metadata_path)
    candidate_provenance = read_candidate_cache_provenance(
        candidate_cache_path,
        candidate_metadata_path,
    )
    candidate_metadata = candidate_provenance["metadata"]
    if not isinstance(candidate_metadata, dict):
        raise ValueError("候选缓存 meta 必须是 JSON 对象")
    _validate_cache_metadata(candidate_metadata, expected_wells, expected_rows)

    base_cache_path = CLEAN_ROOT / str(experiment_config["base_feature_cache"])
    f05a_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )

    gr_gap_cache_path = CLEAN_ROOT / str(experiment_config["gr_gap_feature_cache"])
    gr_gap_metadata_path = CLEAN_ROOT / str(
        experiment_config["gr_gap_feature_cache_metadata"]
    )
    gr_gap_provenance = read_gr_gap_cache_provenance(
        gr_gap_cache_path,
        gr_gap_metadata_path,
        str(experiment_config["gr_gap_feature_cache_sha256"]),
        str(experiment_config["gr_gap_feature_cache_metadata_sha256"]),
    )
    gr_gap_rows = pd.read_parquet(
        gr_gap_cache_path,
        columns=[*KEY_COLUMNS, *GR_GAP_COLUMNS],
    )
    feature_table = add_gr_gap_features(f05a_table, gr_gap_rows)
    _validate_gr_gap_values(feature_table, allow_nan_features)
    if args.mode == "smoke":
        _run_smoke(feature_table, allow_nan_features)
        return

    registry_path = CLEAN_ROOT / str(experiment_config["fold_registry"])
    registry_hash = file_sha256(registry_path)
    if registry_hash != experiment_config["fold_registry_sha256"]:
        raise ValueError("fold 注册表 SHA-256 与冻结配置不一致")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    model_params = read_json(
        CLEAN_ROOT / str(experiment_config["model_config"])
    )["params"]
    fingerprint = _experiment_fingerprint(
        experiment_config,
        model_params,
        registry_hash,
        candidate_provenance,
        gr_gap_provenance,
    )

    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", experiment_config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(
        artifact_dir / "candidate_cache_provenance.json",
        candidate_provenance,
    )
    write_json(artifact_dir / "gr_gap_cache_provenance.json", gr_gap_provenance)

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
        experiment_config,
        model_params,
        model_features,
    )


if __name__ == "__main__":
    main()
