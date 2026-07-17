"""运行冻结 F05a 36 列加 F03a 18 列 Typewell 前缀可靠性特征的 CV。"""

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
from src.f03a_prefix_reliability_features import (  # noqa: E402
    F03A_FEATURE_COLUMNS,
    build_well_prefix_reliability_features,
)
from src.f05a_direct_physical_candidates import (  # noqa: E402
    DIRECT_CANDIDATE_COLUMNS,
)
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "N05_f05a_plus_typewell_prefix_reliability_v1"
DEFAULT_CONFIG = (
    CLEAN_ROOT / "configs" / "n05_f05a_plus_typewell_prefix_reliability_v1.json"
)
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 N05 Typewell 前缀可靠性组合实验")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _validate_config(config: dict[str, object]) -> list[str]:
    """锁定 F05a 36 列，只允许追加既有 F03a 十八列。"""

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *F03A_FEATURE_COLUMNS,
    ]
    required_values = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "F05a_deterministic_candidates_v1",
        "feature_count": len(expected_features),
        "feature_columns": expected_features,
        "candidate_feature_columns": DIRECT_CANDIDATE_COLUMNS,
        "new_feature_columns": F03A_FEATURE_COLUMNS,
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
            raise ValueError(f"N05 配置项 {key} 已偏离冻结组合合同")
    return expected_features


def require_input_bundle(*paths: Path) -> None:
    """只接受已经存在的冻结输入，runner 不生成或修改上游缓存。"""

    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 N05 冻结上游文件；runner 不会重建缓存：\n"
            + "\n".join(missing)
        )


def _require_sha(path: Path, expected_sha256: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 与 N05 冻结配置不一致")
    return actual


def _validate_candidate_lineage(
    metadata: dict[str, object],
    expected_parameter_sha256: str,
    expected_registry_sha256: str,
) -> None:
    """核对 F05a 缓存的参数、seed 和 fold 来源。"""

    if int(metadata.get("seed", -1)) != 42:
        raise ValueError("F05a 候选缓存不是固定 seed=42")
    lineage = metadata.get("base_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("F05a 候选缓存缺少 base_lineage")
    if int(lineage.get("seed", -1)) != 42:
        raise ValueError("F05a 候选缓存 lineage 不是固定 seed=42")
    if lineage.get("parameter_json_sha256") != expected_parameter_sha256:
        raise ValueError("F05a 候选缓存的参数版本已变化")
    if lineage.get("registry_sha256") != expected_registry_sha256:
        raise ValueError("F05a 候选缓存使用的 fold 注册表已变化")


def _validate_model_params(model_params: dict[str, object]) -> None:
    if int(model_params.get("n_estimators", -1)) != 1734:
        raise ValueError("N05 必须固定使用 1734 棵树")
    for seed_key in (
        "random_state",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
    ):
        if int(model_params.get(seed_key, -1)) != 29:
            raise ValueError(f"N05 模型参数 {seed_key} 必须固定为 29")


def load_well_prefix_reliability_features(
    offset_scores_path: Path,
    margins_path: Path,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """从已有 RF03-D0 CSV 构造一井一行特征，不读取原始井数据。"""

    offset_scores = pd.read_csv(offset_scores_path, dtype={"well_id": str})
    margins = pd.read_csv(margins_path, dtype={"well_id": str})
    rows = build_well_prefix_reliability_features(
        offset_scores=offset_scores,
        margins=margins,
        registry=registry,
    )
    values = rows[F03A_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("F03a 十八列井级特征含 NaN 或 Inf")
    return rows


def add_prefix_reliability_features(
    f05a_rows: pd.DataFrame,
    well_features: pd.DataFrame,
) -> pd.DataFrame:
    """严格按 well_id 连接，并把十八列井级特征复制到该井隐藏行。"""

    required_left = {"well_id", "fold", "row_index"}
    required_right = {"well_id", "fold", *F03A_FEATURE_COLUMNS}
    if missing := required_left - set(f05a_rows.columns):
        raise ValueError(f"F05a 特征表缺少连接列：{sorted(missing)}")
    if missing := required_right - set(well_features.columns):
        raise ValueError(f"井级特征表缺少列：{sorted(missing)}")

    left = f05a_rows.reset_index(drop=True).copy()
    right = well_features[["well_id", "fold", *F03A_FEATURE_COLUMNS]].copy()
    left["well_id"] = left["well_id"].astype(str)
    right["well_id"] = right["well_id"].astype(str)
    if right["well_id"].duplicated().any():
        raise ValueError("井级特征表存在重复 well_id")
    if left.groupby("well_id", observed=True)["fold"].nunique().gt(1).any():
        raise ValueError("F05a 特征表中同一 well_id 对应多个 fold")

    left_wells = set(left["well_id"].unique())
    right_wells = set(right["well_id"].unique())
    if left_wells != right_wells:
        missing_wells = sorted(left_wells - right_wells)
        extra_wells = sorted(right_wells - left_wells)
        raise ValueError(
            "井级特征与 F05a 井集合不一致："
            f"missing={missing_wells[:3]}, extra={extra_wells[:3]}"
        )

    right = right.rename(columns={"fold": "__n05_feature_fold"})
    left["__n05_row_order"] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(
        right,
        on="well_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(merged) != len(left):
        raise ValueError("井级特征连接改变了隐藏评价行数")
    if not np.array_equal(
        merged["__n05_row_order"].to_numpy(dtype=np.int64),
        np.arange(len(left), dtype=np.int64),
    ):
        raise ValueError("井级特征连接改变了隐藏评价行顺序")
    if not np.array_equal(
        pd.to_numeric(merged["fold"], errors="raise").to_numpy(dtype=np.int64),
        pd.to_numeric(
            merged["__n05_feature_fold"], errors="raise"
        ).to_numpy(dtype=np.int64),
    ):
        raise ValueError("井级特征的 fold 与 F05a 隐藏行不一致")

    values = merged[F03A_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("连接后的 F03a 十八列含 NaN 或 Inf")
    return merged.drop(columns=["__n05_row_order", "__n05_feature_fold"])


def build_experiment_fingerprint(
    *,
    config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    runner_hash: str,
    candidate_cache_sha256: str,
    candidate_metadata_sha256: str,
    offset_scores_sha256: str,
    per_well_margins_sha256: str,
    feature_code_sha256: str,
) -> str:
    payload = {
        "config": config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "runner_sha256": runner_hash,
        "candidate_cache_sha256": candidate_cache_sha256,
        "candidate_metadata_sha256": candidate_metadata_sha256,
        "offset_scores_sha256": offset_scores_sha256,
        "per_well_margins_sha256": per_well_margins_sha256,
        "feature_code_sha256": feature_code_sha256,
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
    values = smoke_rows[F03A_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if len(smoke_wells) != 3 or not np.isfinite(values).all():
        raise ValueError("N05 smoke 检查失败")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，新增特征=18",
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
    offset_scores_path = CLEAN_ROOT / str(config["offset_scores"])
    margins_path = CLEAN_ROOT / str(config["per_well_margins"])
    feature_code_path = CLEAN_ROOT / str(config["f03a_feature_code"])
    model_config_path = CLEAN_ROOT / str(config["model_config"])
    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    require_input_bundle(
        base_cache_path,
        candidate_cache_path,
        candidate_metadata_path,
        candidate_parameter_path,
        offset_scores_path,
        margins_path,
        feature_code_path,
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
    offset_scores_sha = _require_sha(
        offset_scores_path,
        str(config["offset_scores_sha256"]),
        "RF03-D0 offset_scores",
    )
    margins_sha = _require_sha(
        margins_path,
        str(config["per_well_margins_sha256"]),
        "RF03-D0 per_well_margins",
    )
    feature_code_sha = _require_sha(
        feature_code_path,
        str(config["f03a_feature_code_sha256"]),
        "F03a 特征公式代码",
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
    validate_candidate_cache_metadata(
        candidate_metadata,
        expected_wells,
        expected_rows,
    )
    _validate_candidate_lineage(
        candidate_metadata,
        str(config["candidate_parameter_config_sha256"]),
        registry_hash,
    )
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=expected_wells,
        expected_rows=expected_rows,
    )
    f05a_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    well_features = load_well_prefix_reliability_features(
        offset_scores_path,
        margins_path,
        registry,
    )
    feature_table = add_prefix_reliability_features(f05a_table, well_features)
    if args.mode == "smoke":
        _run_smoke(feature_table)
        return

    model_params = read_json(model_config_path)["params"]
    _validate_model_params(model_params)
    fingerprint = build_experiment_fingerprint(
        config=config,
        model_params=model_params,
        registry_hash=registry_hash,
        runner_hash=file_sha256(Path(__file__).resolve()),
        candidate_cache_sha256=candidate_cache_sha,
        candidate_metadata_sha256=candidate_metadata_sha,
        offset_scores_sha256=offset_scores_sha,
        per_well_margins_sha256=margins_sha,
        feature_code_sha256=feature_code_sha,
    )

    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(
        artifact_dir / "input_provenance.json",
        {
            "base_cache_sha256": base_cache_sha,
            "candidate_cache_sha256": candidate_cache_sha,
            "candidate_metadata_sha256": candidate_metadata_sha,
            "candidate_metadata": candidate_metadata,
            "offset_scores_sha256": offset_scores_sha,
            "per_well_margins_sha256": margins_sha,
            "feature_code_sha256": feature_code_sha,
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
