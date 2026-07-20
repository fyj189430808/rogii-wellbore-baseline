"""评价 PFS01 三条固定滞后路径，并运行冻结的 44 列单模 LightGBM。

这个入口与路径生成入口物理分开。它首先只读取逐井 ``legal_cache`` 与
``legal_runtime``，确认 657 口开发井全部齐全、指纹一致，而且生成阶段从未读取
隐藏 TVT。只有这道门通过后，才允许读取含训练目标的旧特征缓存，先评价三条路径
自身，再运行 folds 0～1。完整五折必须通过预注册的 0.20/0.25 门槛。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pf02_target_ess_lgbm_cv as pf02_runner  # noqa: E402
from scripts import run_p3_pfs01_fixed_lag_particle_smoothing as generation  # noqa: E402
from scripts.run_p2_cv00_group5_c01 import save_fold_runtime_row  # noqa: E402
from scripts.run_p2_p02_multiscale_pf_paths_cv import (  # noqa: E402
    FROZEN_MODEL_FEATURES as _P3B00_FEATURES,
    FROZEN_MODEL_PARAMS,
)
from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json  # noqa: E402


EXPERIMENT_ID = "P3_PFS01_fixed_lag_particle_smoothing_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pfs01_fixed_lag_model_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

# 复制列表，防止调用者意外改写 P2-P02 模块中的冻结常量。
P3B00_FEATURES = list(_P3B00_FEATURES)
NEW_PFS_FEATURES = [
    "pfs_lag250_delta",
    "pfs_lag500_delta",
    "pfs_lag1000_delta",
]
PATH_DIAGNOSTIC_FEATURES = ["pf128_scale_8_delta", *NEW_PFS_FEATURES]

# 正式缓存必须逐列、逐类型匹配生成器的输出，不能容忍额外摘要列混入模型阶段。
PFS_CACHE_SCHEMA = pa.schema(
    [
        pa.field("well_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("row_index", pa.int64()),
        pa.field("last_visible_tvt", pa.float32()),
        pa.field("pfs_lag250_delta", pa.float32()),
        pa.field("pfs_lag500_delta", pa.float32()),
        pa.field("pfs_lag1000_delta", pa.float32()),
        pa.field("_cache_fingerprint", pa.string()),
    ]
)

FROZEN_TRAINING_POLICY = {
    "early_stopping": False,
    "uniform_row_weight": True,
    "native_missing_values": True,
    "well_id_is_feature": False,
    "row_id_is_feature": False,
    "test_overlap_wells_removed_from_training": True,
}
FROZEN_SUCCESS_CONDITIONS = {
    "fold01_minimum_combined_improvement_ft": 0.20,
    "fold01_maximum_single_fold_degradation_ft": 0.25,
    "full5_minimum_improvement_ft": 0.10,
    "full5_minimum_improved_folds": 4,
    "full5_minimum_improved_folds_2_to_4": 2,
    "full5_maximum_single_fold_degradation_ft": 0.25,
    "full5_maximum_bootstrap_ci_upper": 0.0,
    "full5_minimum_well_win_rate": 0.55,
    "full5_maximum_p90_degradation_ft": 0.20,
    "full5_maximum_top5_percent_positive_gain_share": 0.60,
}
FROZEN_CONFIG_CONTRACT: dict[str, Any] = {
    "experiment_id": EXPERIMENT_ID,
    "baseline_id": "P3B00_group5_p2p02_v1",
    "fold_version": "balanced_well_5fold_v1",
    "fold_registry": "artifacts/folds/balanced_well_5fold_v1.csv",
    "fold_registry_sha256": "a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c",
    "shadow_registry": "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    "shadow_registry_sha256": "7fde7c16895e02f74c96a70a1ca3d9935e0e9bfb24c7744a1992c025aad483c1",
    "base_feature_cache": "artifacts/B00_simple_lgbm_v1/feature_cache.parquet",
    "base_feature_cache_sha256": "8801493752e03f40a3957843e8efa25c091d04244dd341c64c9329b50d5de625",
    "candidate_feature_cache": "artifacts/F05a_deterministic_candidate_cache_v1/candidate_feature_cache.parquet",
    "candidate_feature_cache_sha256": "66b32f8ed790ea53184d43bc02d6899031348deec418d7933366bbef64105076",
    "source_p01_legal_cache_dir": "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache",
    "source_p01_fingerprint": "ac1dbefd59923585671156b7d3e8b4fc7faca95c20f5f1ae6fdab92a8665954a",
    "source_pfs_generation_config": "configs/p3_pfs01_fixed_lag_particle_smoothing_v1.json",
    "source_pfs_generation_config_sha256": "109bfa082e3b5b512e0dcc8951af1ec363ec1e287ae7cb8ab9bc36c5247b9dbf",
    "source_pfs_fingerprint": "4bf62f495ef530bb3ca53d65d15475f7bc3784fe5abaebbbb1053584704c4480",
    "source_pfs_legal_cache_dir": "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_cache",
    "source_pfs_runtime_dir": "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_runtime",
    "source_p3b00_predictions": "artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet",
    "source_p3b00_predictions_sha256": "8109514eb125f8beda2559f39e1d9f54396d41fd38fa76114423c8dd6bca4d37",
    "source_p3b00_feature_list": "artifacts/P2_P02_multiscale_pf_paths_v1/feature_list.json",
    "source_p3b00_feature_list_sha256": "2bf5992b45fe95e6ae38862da2408b306c0303806291d112ff8fa1ecf4b00172",
    "model_config": "configs/lgbm_feature_baseline_v1.json",
    "model_config_sha256": "02f6c4cb737132641c2dbbb6076d0e248371786559f2cd6f8581ce72146cb838",
    "new_feature_names": NEW_PFS_FEATURES,
    "baseline_feature_count": 41,
    "formal_feature_count": 44,
    "total_wells": 773,
    "shadow_wells": 116,
    "development_wells": 657,
    "development_hidden_rows": 3_211_872,
    "development_fold_well_counts": {
        "0": 131,
        "1": 132,
        "2": 131,
        "3": 132,
        "4": 131,
    },
    "development_fold_hidden_row_counts": {
        "0": 651_881,
        "1": 630_395,
        "2": 645_557,
        "3": 649_717,
        "4": 634_322,
    },
    "baseline_development_micro_rmse": 10.272146267501086,
    "path_reference_feature": "pf128_scale_8_delta",
    "model_training": True,
    "shadow_target_access": False,
    "success_conditions": FROZEN_SUCCESS_CONDITIONS,
}


# 这些函数只复用已经冻结并经过旧实验验证的数据读取、评分和训练实现。
resolve_clean_path = pf02_runner.resolve_clean_path
file_sha256 = pf02_runner.file_sha256
stable_hash = pf02_runner.stable_hash
load_development_registry = pf02_runner.load_development_registry
load_development_feature_table = pf02_runner.load_development_feature_table
merge_existing_p3b00_pf_cache = pf02_runner.merge_existing_p3b00_pf_cache
validate_feature_values = pf02_runner.validate_feature_values
read_baseline_predictions = pf02_runner.read_baseline_predictions
read_fold_predictions = pf02_runner.read_fold_predictions
save_stage_artifacts = pf02_runner.save_stage_artifacts
save_mean_feature_importance = pf02_runner.save_mean_feature_importance


@dataclass(frozen=True)
class LegalCacheBundle:
    """真值阶段唯一认可的合法缓存通行证及三条路径表。"""

    path_table: pd.DataFrame
    generator_fingerprint: str
    audit: dict[str, Any]


def build_formal_feature_names() -> list[str]:
    """返回顺序冻结、无重复的“原 41 列 + 三条 PFS 路径”。"""

    features = [*P3B00_FEATURES, *NEW_PFS_FEATURES]
    if len(features) != 44 or len(set(features)) != 44:
        raise ValueError("PFS01 正式特征必须是无重复的 41+3=44 列")
    return features


def parse_fold_spec(value: str) -> list[int]:
    """只允许预筛 folds0-1 或带自动门控的完整五折。"""

    normalized = str(value).strip().lower()
    if normalized == "0,1":
        return [0, 1]
    if normalized == "all":
        return [0, 1, 2, 3, 4]
    raise ValueError("--folds 只允许 0,1 或 all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """读取命令行参数；不提供改模型或改特征的入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="先筛 0,1；晋级后可用 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def validate_frozen_contract(
    config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """在接触任何训练目标前锁死配置、41 列清单和 LightGBM 参数。"""

    if config != FROZEN_CONFIG_CONTRACT:
        changed = sorted(
            key
            for key in set(config).union(FROZEN_CONFIG_CONTRACT)
            if config.get(key) != FROZEN_CONFIG_CONTRACT.get(key)
        )
        raise ValueError(f"PFS01 冻结配置合同发生变化：{changed}")

    manifest = read_json(resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41 or manifest.get("features") != P3B00_FEATURES:
        raise ValueError("P3B00 原 41 列清单发生变化")

    model_config = read_json(resolve_clean_path(config["model_config"]))
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("正式模型不是冻结的单模 LightGBM")
    if model_config.get("params") != FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 参数、1734 棵树或 seed29 发生变化")
    if model_config.get("training_policy") != FROZEN_TRAINING_POLICY:
        raise ValueError("LightGBM 训练策略发生变化")

    features = build_formal_feature_names()
    forbidden_tokens = (
        "direction",
        "mass",
        "seed_count",
        "p2_position",
        "separation",
        "oracle",
        "target",
        "truth",
        "surface",
    )
    forbidden = [
        name
        for name in features
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    if features[:41] != P3B00_FEATURES or forbidden:
        raise ValueError(f"PFS01 44 列冻结合同非法：{forbidden}")
    return features, dict(FROZEN_MODEL_PARAMS)


def _validate_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    """合法缓存门通过后，再核对所有含目标和不含目标的冻结来源。"""

    source_keys = [
        "fold_registry",
        "shadow_registry",
        "base_feature_cache",
        "candidate_feature_cache",
        "source_pfs_generation_config",
        "source_p3b00_predictions",
        "source_p3b00_feature_list",
        "model_config",
    ]
    observed: dict[str, str] = {}
    for key in source_keys:
        path = resolve_clean_path(config[key])
        actual = file_sha256(path)
        expected = str(config[f"{key}_sha256"]).lower()
        if actual.lower() != expected:
            raise ValueError(f"冻结来源 {key} 的 SHA-256 不匹配")
        observed[key] = actual
    return observed


def build_expected_generator_fingerprint(config: dict[str, Any]) -> str:
    """按路径生成器原公式，从无标签配置和代码哈希重算唯一合法指纹。"""

    generation_config_path = resolve_clean_path(config["source_pfs_generation_config"])
    observed_config_hash = file_sha256(generation_config_path)
    if observed_config_hash != str(config["source_pfs_generation_config_sha256"]):
        raise ValueError("PFS01 冻结生成配置 SHA-256 不匹配")
    generation_config = read_json(generation_config_path)
    if generation_config.get("experiment_id") != generation.EXPERIMENT_ID:
        raise ValueError("PFS01 生成配置实验编号错误")
    if generation_config.get("model_training") is not False:
        raise ValueError("PFS01 生成配置不得训练模型")
    if generation_config.get("hidden_tvt_read") is not False:
        raise ValueError("PFS01 生成配置不得读取隐藏 TVT")

    source_pf_config_path = generation._resolve_clean_path(
        generation_config["source_pf_config"]
    )
    folds_path = generation._resolve_clean_path(generation_config["fold_registry"])
    shadow_path = generation._resolve_clean_path(generation_config["shadow_registry"])
    source_hashes = {
        "runner": generation.FROZEN_GENERATION_RUNNER_SHA256,
        "pfs_core": file_sha256(CLEAN_ROOT / "src" / "p3_pfs01_fixed_lag_smoothing.py"),
        "p2_pf_core": file_sha256(CLEAN_ROOT / "src" / "p2_p01_multiseed_pf.py"),
        "source_pf_config": file_sha256(source_pf_config_path),
        "fold_registry": file_sha256(folds_path),
        "shadow_registry": file_sha256(shadow_path),
    }
    expected = generation.build_cache_fingerprint(generation_config, source_hashes)
    frozen = str(config["source_pfs_fingerprint"])
    if expected != frozen:
        raise ValueError(
            "按冻结生成配置重算的 PFS01 指纹与模型合同中的预期指纹不一致"
        )
    return expected


def _validate_runtime_record(
    runtime: dict[str, Any],
    cache_path: Path,
    well_id: str,
    fold: int,
    hidden_rows: int,
    fingerprint: str,
) -> None:
    """逐字段复核生成记录，尤其确认生成阶段没有读取隐藏 TVT。"""

    required = {
        "experiment_id",
        "well_id",
        "fold",
        "hidden_rows",
        "legal_horizontal_columns",
        "typewell_columns",
        "experiment_fingerprint",
        "hidden_tvt_read",
        "cache_sha256",
        "generation_seconds",
        "resume_check_seconds",
        *generation.RUNTIME_NONNEGATIVE_INTEGER_FIELDS,
    }
    missing = required.difference(runtime)
    if missing:
        raise ValueError(f"{well_id} 运行记录缺字段：{sorted(missing)}")
    if runtime["experiment_id"] != generation.EXPERIMENT_ID:
        raise ValueError(f"{well_id} 运行记录实验编号错误")
    if str(runtime["well_id"]) != well_id:
        raise ValueError(f"{well_id} 运行记录井号错误")
    if int(runtime["fold"]) != fold:
        raise ValueError(f"{well_id} 运行记录 fold 错误")
    if int(runtime["hidden_rows"]) != hidden_rows:
        raise ValueError(f"{well_id} 运行记录行数错误")
    if runtime["legal_horizontal_columns"] != list(generation.HORIZONTAL_USECOLS):
        raise ValueError(f"{well_id} 生成阶段水平井读取列非法")
    if runtime["typewell_columns"] != list(generation.TYPEWELL_USECOLS):
        raise ValueError(f"{well_id} 生成阶段 Typewell 读取列非法")
    if runtime["experiment_fingerprint"] != fingerprint:
        raise ValueError(f"{well_id} 运行记录与缓存指纹不一致")
    if runtime["hidden_tvt_read"] is not False:
        raise ValueError(f"{well_id} 路径生成阶段读取了隐藏 TVT")
    if runtime["cache_sha256"] != file_sha256(cache_path):
        raise ValueError(f"{well_id} 缓存 SHA-256 与运行记录不一致")

    for name in generation.RUNTIME_NONNEGATIVE_INTEGER_FIELDS:
        value = runtime[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{well_id} 运行记录 {name} 不是非负整数")
    for name in ("generation_seconds", "resume_check_seconds"):
        value = runtime[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{well_id} 运行时间 {name} 非法")
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{well_id} 运行时间 {name} 非法")


def validate_complete_legal_cache(
    registry: pd.DataFrame,
    cache_dir: Path,
    runtime_dir: Path,
    shadow_ids: set[str],
    expected_wells: int,
    expected_rows: int,
    expected_fingerprint: str,
) -> LegalCacheBundle:
    """只读核验全部开发井缓存，成功后返回进入真值阶段所需的通行证。"""

    if not cache_dir.is_dir():
        raise FileNotFoundError(f"缺少 PFS01 legal_cache：{cache_dir}")
    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"缺少 PFS01 legal_runtime：{runtime_dir}")
    required_registry_columns = {"well_id", "fold", "hidden_rows"}
    missing_registry_columns = required_registry_columns.difference(registry.columns)
    if missing_registry_columns:
        raise ValueError(f"开发井注册表缺列：{sorted(missing_registry_columns)}")
    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("开发井注册表含重复井号")
    if len(registry) != int(expected_wells):
        raise ValueError("开发井注册表井数与冻结合同不一致")
    if int(registry["hidden_rows"].sum()) != int(expected_rows):
        raise ValueError("开发井注册表总行数与冻结合同不一致")

    development_ids = set(registry["well_id"].astype(str))
    cache_ids = {path.stem for path in cache_dir.glob("*.parquet")}
    runtime_ids = {path.stem for path in runtime_dir.glob("*.json")}
    missing_cache = development_ids.difference(cache_ids)
    missing_runtime = development_ids.difference(runtime_ids)
    if missing_cache:
        raise FileNotFoundError(f"缺少开发井合法缓存：{sorted(missing_cache)[:3]}")
    if missing_runtime:
        raise FileNotFoundError(f"缺少开发井运行记录：{sorted(missing_runtime)[:3]}")
    extra_cache = cache_ids.difference(development_ids)
    extra_runtime = runtime_ids.difference(development_ids)
    shadow_overlap = (extra_cache | extra_runtime).intersection(shadow_ids)
    if shadow_overlap:
        raise ValueError(f"合法缓存目录混入影子井：{sorted(shadow_overlap)[:3]}")
    if extra_cache or extra_runtime:
        raise ValueError(
            "合法缓存目录出现未登记的额外井："
            f"cache={sorted(extra_cache)[:3]}, runtime={sorted(extra_runtime)[:3]}"
        )

    rows: list[pd.DataFrame] = []
    common_fingerprint: str | None = None
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        hidden_rows = int(registry_row.hidden_rows)
        cache_path = cache_dir / f"{well_id}.parquet"
        runtime_path = runtime_dir / f"{well_id}.json"

        observed_schema = parquet.ParquetFile(cache_path).schema_arrow
        if not observed_schema.equals(PFS_CACHE_SCHEMA):
            raise ValueError(f"{well_id} legal_cache schema/列类型不符合冻结合同")
        cache = pd.read_parquet(cache_path, columns=list(generation.CACHE_COLUMNS))
        if len(cache) != hidden_rows:
            raise ValueError(f"{well_id} legal_cache 行数错误")
        if cache.duplicated(["well_id", "row_index"]).any():
            raise ValueError(f"{well_id} legal_cache 含重复行键")
        if not cache["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"{well_id} legal_cache 井号错误")
        if not cache["fold"].astype(np.int64).eq(fold).all():
            raise ValueError(f"{well_id} legal_cache fold 错误")
        if cache["row_index"].astype(np.int64).duplicated().any():
            raise ValueError(f"{well_id} legal_cache row_index 重复")
        numeric_columns = ["last_visible_tvt", *NEW_PFS_FEATURES]
        if not np.isfinite(cache[numeric_columns].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{well_id} legal_cache 路径含 NaN/Inf，必须全部有限")
        if cache["last_visible_tvt"].nunique(dropna=False) != 1:
            raise ValueError(f"{well_id} legal_cache last_visible_tvt 井内不唯一")

        fingerprints = cache["_cache_fingerprint"].astype(str).unique().tolist()
        if len(fingerprints) != 1 or not fingerprints[0]:
            raise ValueError(f"{well_id} legal_cache 指纹为空或不唯一")
        fingerprint = fingerprints[0]
        if fingerprint != str(expected_fingerprint):
            raise ValueError(
                f"{well_id} legal_cache 不等于冻结生成配置的预期指纹"
            )
        if common_fingerprint is None:
            common_fingerprint = fingerprint
        elif fingerprint != common_fingerprint:
            raise ValueError("PFS01 各井合法缓存指纹不统一")

        runtime = read_json(runtime_path)
        _validate_runtime_record(
            runtime=runtime,
            cache_path=cache_path,
            well_id=well_id,
            fold=fold,
            hidden_rows=hidden_rows,
            fingerprint=fingerprint,
        )
        # 复用生成器的更细续跑校验，覆盖 lag 行数守恒和历史复制计数守恒。
        validated_runtime = generation.validate_cache_hit(
            cache_path=cache_path,
            runtime_path=runtime_path,
            well_id=well_id,
            fold=fold,
            expected_row_index=cache["row_index"].to_numpy(dtype=np.int64),
            fingerprint=fingerprint,
        )
        if validated_runtime is None:
            raise ValueError(f"{well_id} 未通过生成器的完整合法缓存复核")
        rows.append(
            cache[["well_id", "row_index", "last_visible_tvt", *NEW_PFS_FEATURES]]
        )

    if common_fingerprint is None:
        raise ValueError("PFS01 没有任何开发井合法缓存")
    path_table = pd.concat(rows, ignore_index=True)
    audit = {
        "legal_cache_complete": True,
        "hidden_tvt_read": False,
        "development_wells": int(len(registry)),
        "development_hidden_rows": int(len(path_table)),
        "shadow_overlap": 0,
        "generator_fingerprint": common_fingerprint,
        "cache_columns": list(generation.CACHE_COLUMNS),
        "truth_stage_opened": False,
    }
    return LegalCacheBundle(path_table, common_fingerprint, audit)


def require_truth_stage_permission(
    audit: dict[str, Any],
    expected_wells: int,
    expected_rows: int,
) -> None:
    """在任何含目标缓存读取前，再核对一次合法缓存通行条件。"""

    allowed = bool(
        audit.get("legal_cache_complete") is True
        and audit.get("hidden_tvt_read") is False
        and int(audit.get("development_wells", -1)) == int(expected_wells)
        and int(audit.get("development_hidden_rows", -1)) == int(expected_rows)
        and int(audit.get("shadow_overlap", -1)) == 0
    )
    if not allowed:
        raise RuntimeError("合法缓存或影子集审计未通过，禁止进入真值阶段")


def merge_pfs_paths(
    feature_table: pd.DataFrame,
    path_table: pd.DataFrame,
) -> pd.DataFrame:
    """按井号和自然行号一对一追加三列，并逐位核对井口锚点。"""

    return pf02_runner._merge_path_table(
        feature_table,
        path_table,
        NEW_PFS_FEATURES,
        anchor_name="pfs01_anchor",
    )


def score_path_candidates(
    feature_table: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """在训练 LightGBM 前，单独评价旧 scale8 与三条 PFS 路径自身。"""

    required = {
        "well_id",
        "fold",
        "row_index",
        "last_visible_tvt",
        "target_tvt",
        *PATH_DIAGNOSTIC_FEATURES,
    }
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"路径诊断表缺列：{sorted(missing)}")
    if feature_table.duplicated(["well_id", "row_index"]).any():
        raise ValueError("路径诊断表含重复评价行")

    target = feature_table["target_tvt"].to_numpy(dtype=np.float64)
    anchor = feature_table["last_visible_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(anchor).all():
        raise ValueError("路径诊断目标或锚点含 NaN/Inf")

    path_metrics: dict[str, dict[str, Any]] = {}
    per_well_rows: list[dict[str, Any]] = []
    per_fold_rows: list[dict[str, Any]] = []
    for path_name in PATH_DIAGNOSTIC_FEATURES:
        delta = feature_table[path_name].to_numpy(dtype=np.float64)
        if not np.isfinite(delta).all():
            raise ValueError(f"路径 {path_name} 含 NaN/Inf")
        prediction = anchor + delta
        squared_error = np.square(target - prediction)

        path_well_rmses: list[float] = []
        for well_id, indices in feature_table.groupby("well_id", sort=False).indices.items():
            positions = np.asarray(indices, dtype=np.int64)
            folds = feature_table.iloc[positions]["fold"].astype(int).unique()
            if len(folds) != 1:
                raise ValueError(f"井 {well_id} 跨越多个 fold")
            sse = float(squared_error[positions].sum())
            rmse = float(np.sqrt(sse / len(positions)))
            path_well_rmses.append(rmse)
            per_well_rows.append(
                {
                    "path": path_name,
                    "well_id": str(well_id),
                    "fold": int(folds[0]),
                    "rows": int(len(positions)),
                    "sse": sse,
                    "rmse": rmse,
                }
            )

        fold_metrics: list[dict[str, Any]] = []
        for fold, indices in feature_table.groupby("fold", sort=True).indices.items():
            positions = np.asarray(indices, dtype=np.int64)
            fold_well = [
                row["rmse"]
                for row in per_well_rows
                if row["path"] == path_name and row["fold"] == int(fold)
            ]
            fold_row = {
                "path": path_name,
                "fold": int(fold),
                "rows": int(len(positions)),
                "micro_rmse": float(np.sqrt(squared_error[positions].mean())),
                "macro_rmse": float(np.mean(fold_well)),
            }
            fold_metrics.append(fold_row)
            per_fold_rows.append(fold_row)

        path_metrics[path_name] = {
            "micro_rmse": float(np.sqrt(squared_error.mean())),
            "macro_rmse": float(np.mean(path_well_rmses)),
            "per_fold": fold_metrics,
        }

    reference_rmse = path_metrics["pf128_scale_8_delta"]["micro_rmse"]
    for path_name, values in path_metrics.items():
        values["improvement_vs_old_scale8_ft"] = float(
            reference_rmse - values["micro_rmse"]
        )
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "reference_path": "pf128_scale_8_delta",
        "paths": path_metrics,
    }
    return metrics, pd.DataFrame(per_well_rows), pd.DataFrame(per_fold_rows)


def compare_with_p3b00(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """复用统一指标实现，并把实验编号改回 PFS01。"""

    metrics, per_well, per_fold = pf02_runner.compare_with_p3b00(
        candidate,
        baseline,
        config,
        stage,
    )
    metrics["experiment_id"] = EXPERIMENT_ID
    return metrics, per_well, per_fold


def remaining_folds_after_screen(
    requested_folds: list[int],
    metrics_folds01: dict[str, Any],
) -> list[int]:
    """只有 ``all`` 且 folds0-1 通过预注册门槛时，才放行 folds2-4。"""

    if requested_folds == [0, 1]:
        return []
    if requested_folds != [0, 1, 2, 3, 4]:
        raise ValueError("请求折必须是 0,1 或 all")
    checks = metrics_folds01.get("success_checks", {})
    combined = float(checks.get("combined_improvement_ft", -np.inf))
    worst = float(checks.get("worst_fold_improvement_ft", -np.inf))
    passed = bool(
        checks.get("folds01_pass") is True
        and combined >= FROZEN_SUCCESS_CONDITIONS[
            "fold01_minimum_combined_improvement_ft"
        ]
        and worst
        >= -FROZEN_SUCCESS_CONDITIONS[
            "fold01_maximum_single_fold_degradation_ft"
        ]
    )
    return [2, 3, 4] if passed else []


def _save_path_diagnostics(
    output_dir: Path,
    metrics: dict[str, Any],
    per_well: pd.DataFrame,
    per_fold: pd.DataFrame,
) -> None:
    """把路径自身评分与模型产物分开保存，防止把 oracle 指标当特征。"""

    write_json(output_dir / "path_diagnostics.json", metrics)
    per_well.to_csv(output_dir / "path_diagnostics_per_well.csv", index=False)
    per_fold.to_csv(output_dir / "path_diagnostics_per_fold.csv", index=False)


def _write_conclusion(output_dir: Path, metrics: dict[str, Any]) -> None:
    """按三阶段要求限定结论只覆盖当前三条 fixed-lag 路径。"""

    checks = metrics["success_checks"]
    stage = str(metrics["stage"])
    if stage == "folds01":
        improvement = float(checks["combined_improvement_ft"])
        passed = bool(checks["folds01_pass"])
    else:
        improvement = float(checks["full5_improvement_ft"])
        passed = bool(checks["full5_pass"])
    text = f"""# P3-PFS01 结论

数据直接证明的事实：当前阶段相对 P3B00 的 micro RMSE 改善为 `{improvement:.6f} ft`，预注册门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：这里仅检验原 41 列后追加 fixed-lag 250/500/1000 三条路径的增量价值。

仍然没有验证的猜测：其他 lag、动态 lag 或稀疏控制点近似是否更好。

当前实验只能否定的具体实现：当前冻结参数下三条精确 fixed-lag 路径直接加入单模 LightGBM。

下一步最便宜的验证：{'按预注册路线继续完整五折。' if passed else '停止本实现，不据此否定粒子平滑这一信息源。'}
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """严格按“合法缓存门→路径评分→folds0-1→自动门控”执行。"""

    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config = read_json(args.config.resolve())
    model_features, model_params = validate_frozen_contract(config)

    # fold 和 shadow 表不含训练目标，可以在合法缓存门之前安全读取。
    registry, shadow_ids = load_development_registry(
        resolve_clean_path(config["fold_registry"]),
        resolve_clean_path(config["shadow_registry"]),
        config,
    )
    expected_generator_fingerprint = build_expected_generator_fingerprint(config)
    legal_bundle = validate_complete_legal_cache(
        registry=registry,
        cache_dir=resolve_clean_path(config["source_pfs_legal_cache_dir"]),
        runtime_dir=resolve_clean_path(config["source_pfs_runtime_dir"]),
        shadow_ids=shadow_ids,
        expected_wells=int(config["development_wells"]),
        expected_rows=int(config["development_hidden_rows"]),
        expected_fingerprint=expected_generator_fingerprint,
    )
    require_truth_stage_permission(
        legal_bundle.audit,
        expected_wells=int(config["development_wells"]),
        expected_rows=int(config["development_hidden_rows"]),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "legal_cache_audit.json", legal_bundle.audit)
    print(
        "PFS01 合法缓存门通过："
        f"{legal_bundle.audit['development_wells']} 口井，"
        f"{legal_bundle.audit['development_hidden_rows']:,} 行，"
        "hidden_tvt_read=false；现在才进入真值阶段。",
        flush=True,
    )

    # 从这一行开始才读取含 target_tvt/target_delta 的旧特征缓存。
    observed_hashes = _validate_source_hashes(config)
    feature_table = load_development_feature_table(
        resolve_clean_path(config["base_feature_cache"]),
        resolve_clean_path(config["candidate_feature_cache"]),
        registry,
    )
    feature_table = merge_existing_p3b00_pf_cache(
        feature_table,
        registry,
        resolve_clean_path(config["source_p01_legal_cache_dir"]),
        str(config["source_p01_fingerprint"]),
    )
    feature_table = merge_pfs_paths(feature_table, legal_bundle.path_table)
    if set(feature_table["well_id"].astype(str).unique()).intersection(shadow_ids):
        raise RuntimeError("正式训练表含影子井")
    validate_feature_values(feature_table, model_features)

    # 路径评分必须先落盘，然后才能训练任何 LightGBM。
    path_metrics, path_per_well, path_per_fold = score_path_candidates(feature_table)
    _save_path_diagnostics(output_dir, path_metrics, path_per_well, path_per_fold)

    fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "model_features": model_features,
            "model_params": model_params,
            "generator_fingerprint": legal_bundle.generator_fingerprint,
            "observed_source_hashes": observed_hashes,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"),
        }
    )
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "parameter_list.json", model_params)
    write_json(
        output_dir / "feature_list.json",
        {"feature_count": 44, "features": model_features},
    )
    write_json(
        output_dir / "leakage_audit.json",
        {
            **legal_bundle.audit,
            "truth_stage_opened": True,
            "shadow_target_access": False,
            "shadow_wells": len(shadow_ids),
            "shadow_feature_overlap": 0,
            "validation_unit": "complete_well",
            "fold_version": config["fold_version"],
            "old_41_features_preserved": model_features[:41] == P3B00_FEATURES,
            "new_features": NEW_PFS_FEATURES,
            "path_diagnostics_are_not_model_features": True,
            "cv_fingerprint": fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )
    print(
        f"PFS01 模型阶段：开发井={len(registry)}，行={len(feature_table):,}，"
        f"特征=44，树=1734，folds={requested_folds}",
        flush=True,
    )

    for fold_id in (0, 1):
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            model_features,
        )
        save_fold_runtime_row(output_dir, runtime)
    candidate_folds01 = read_fold_predictions(output_dir, [0, 1])
    registry_folds01 = registry.loc[registry["fold"].astype(int).isin([0, 1])].copy()
    baseline_folds01 = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]),
        registry_folds01,
    )
    metrics01, per_well01, per_fold01 = compare_with_p3b00(
        candidate_folds01,
        baseline_folds01,
        config,
        "folds01",
    )
    metrics01["cv_fingerprint"] = fingerprint
    save_stage_artifacts(
        output_dir,
        "folds01",
        candidate_folds01,
        metrics01,
        per_well01,
        per_fold01,
    )
    _write_conclusion(output_dir, metrics01)
    print(json.dumps(metrics01["success_checks"], ensure_ascii=False, indent=2), flush=True)

    late_folds = remaining_folds_after_screen(requested_folds, metrics01)
    if not late_folds:
        if requested_folds == [0, 1, 2, 3, 4]:
            print("PFS01 folds0-1 未晋级，停止且不训练 folds2-4。", flush=True)
        return

    for fold_id in late_folds:
        runtime = train_fold(
            feature_table,
            registry,
            fold_id,
            model_params,
            output_dir,
            fingerprint,
            model_features,
        )
        save_fold_runtime_row(output_dir, runtime)
    candidate_full = read_fold_predictions(output_dir, [0, 1, 2, 3, 4])
    baseline_full = read_baseline_predictions(
        resolve_clean_path(config["source_p3b00_predictions"]),
        registry,
    )
    metrics_full, per_well_full, per_fold_full = compare_with_p3b00(
        candidate_full,
        baseline_full,
        config,
        "full5",
    )
    metrics_full["cv_fingerprint"] = fingerprint
    save_stage_artifacts(
        output_dir,
        "full5",
        candidate_full,
        metrics_full,
        per_well_full,
        per_fold_full,
    )
    save_mean_feature_importance(output_dir)
    fold_runtimes = [
        read_json(output_dir / f"fold_{fold_id}" / "runtime.json")
        for fold_id in range(5)
    ]
    write_json(
        output_dir / "runtime.json",
        {
            "cv_fingerprint": fingerprint,
            "folds": fold_runtimes,
            "total_fold_seconds": float(
                sum(float(row["seconds"]) for row in fold_runtimes)
            ),
        },
    )
    _write_conclusion(output_dir, metrics_full)
    print(json.dumps(metrics_full["success_checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
