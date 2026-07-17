"""运行 F05a 确定性候选加九个路径族分歧特征的固定 LightGBM CV。"""

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
from src.n01_f05a_family_disagreement import (  # noqa: E402
    FAMILY_DISAGREEMENT_COLUMNS,
    build_family_disagreement_features,
)


EXPERIMENT_ID = "N01_f05a_family_disagreement_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "n01_f05a_family_disagreement_v1.json"
SUPPORTED_MODES = ["smoke", "fold0", "fold1", "fold2", "fold3", "fold4"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke 或一个固定折；本入口没有重建候选缓存的参数。"""

    parser = argparse.ArgumentParser(description="运行 N01 路径族分歧特征实验")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def _validate_config(experiment_config: dict[str, object]) -> list[str]:
    """锁定 F05a 的 36 列，并只允许追加约定的九列。"""

    expected_features = [
        *FEATURE_COLUMNS,
        *DIRECT_CANDIDATE_COLUMNS,
        *FAMILY_DISAGREEMENT_COLUMNS,
    ]
    if experiment_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("实验 ID 与 N01 runner 不一致")
    if experiment_config.get("baseline_id") != "F05a_deterministic_candidates_v1":
        raise ValueError("N01 基线不是冻结的确定性 F05a")
    if experiment_config.get("feature_columns") != expected_features:
        raise ValueError("N01 特征顺序不是 F05a 36 列加固定九列")
    if experiment_config.get("candidate_feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("F05a 的 24 个候选特征或顺序发生变化")
    if experiment_config.get("new_feature_columns") != FAMILY_DISAGREEMENT_COLUMNS:
        raise ValueError("N01 的九个新增特征或顺序发生变化")
    if experiment_config.get("candidate_cache_rebuild_allowed") is not False:
        raise ValueError("N01 禁止重建确定性候选缓存")
    if experiment_config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("配置支持的运行模式与 runner 不一致")
    return expected_features


def add_family_disagreement_features(feature_table: pd.DataFrame) -> pd.DataFrame:
    """在已经对齐的 B00+F05a 表末尾追加九列，不改变原有列。"""

    disagreement = build_family_disagreement_features(feature_table)
    return pd.concat(
        [
            feature_table.reset_index(drop=True),
            disagreement[FAMILY_DISAGREEMENT_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )


def _experiment_fingerprint(
    experiment_config: dict[str, object],
    model_params: dict[str, object],
    registry_hash: str,
    cache_provenance: dict[str, object],
) -> str:
    """把配置、固定模型、fold、候选缓存和九列公式共同写入实验指纹。"""

    payload = {
        "experiment_config": experiment_config,
        "model_params": model_params,
        "registry_sha256": registry_hash,
        "candidate_cache_provenance": cache_provenance,
        "formula_sha256": file_sha256(
            CLEAN_ROOT / "src" / "n01_f05a_family_disagreement.py"
        ),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_smoke(feature_table: pd.DataFrame) -> None:
    """只检查前三口井的九个新增特征是否完整且有限。"""

    smoke_wells = (
        feature_table["well_id"].astype(str).drop_duplicates().iloc[:3].tolist()
    )
    smoke_rows = feature_table.loc[
        feature_table["well_id"].astype(str).isin(smoke_wells)
    ]
    values = smoke_rows[FAMILY_DISAGREEMENT_COLUMNS].to_numpy(dtype=np.float32)
    if len(smoke_wells) != 3 or not np.isfinite(values).all():
        raise ValueError("N01 smoke 检查失败")
    print(
        f"smoke 完成：井={smoke_wells}，行={len(smoke_rows):,}，新增特征=9",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """只读冻结缓存，运行 smoke 或一个指定 fold。"""

    args = parse_args(argv)
    experiment_config = read_json(args.config.resolve())
    model_features = _validate_config(experiment_config)
    expected_wells = int(experiment_config["expected_wells"])
    expected_rows = int(experiment_config["expected_rows"])

    candidate_cache_path = CLEAN_ROOT / str(
        experiment_config["candidate_feature_cache"]
    )
    metadata_path = CLEAN_ROOT / str(experiment_config["candidate_cache_metadata"])
    require_candidate_cache_files(candidate_cache_path, metadata_path)
    cache_provenance = read_candidate_cache_provenance(
        candidate_cache_path,
        metadata_path,
    )
    metadata = cache_provenance["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("候选缓存 meta 必须是 JSON 对象")
    _validate_cache_metadata(metadata, expected_wells, expected_rows)

    base_cache_path = CLEAN_ROOT / str(experiment_config["base_feature_cache"])
    f05a_table = load_aligned_feature_table(
        base_cache_path,
        candidate_cache_path,
        expected_rows,
    )
    feature_table = add_family_disagreement_features(f05a_table)
    if args.mode == "smoke":
        _run_smoke(feature_table)
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
        cache_provenance,
    )

    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", experiment_config)
    write_json(
        artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    write_json(artifact_dir / "candidate_cache_provenance.json", cache_provenance)

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

