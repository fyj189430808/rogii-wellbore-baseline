"""把四条固定目标 ESS 路径加入 P3B00，运行开发集单模 LightGBM。

本脚本只使用 657 口开发井。先从井注册表中反连接 116 口影子井，再用
Arrow 过滤读取含目标的特征缓存和 P3B00 OOF，影子目标不会进入 pandas。
原 P3B00 的 41 列完整保留，四条 ESS 路径追加在末尾，总计 45 列。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p2_cv00_group5_c01 import (  # noqa: E402
    remap_feature_table_folds,
    save_fold_runtime_row,
)
from scripts.run_p2_p01_multiseed_pf_mean import (  # noqa: E402
    validate_legal_cache as validate_p01_legal_cache,
)
from scripts.run_p2_p02_multiscale_pf_paths_cv import (  # noqa: E402
    FROZEN_MODEL_FEATURES as _P3B00_FEATURES,
    FROZEN_MODEL_PARAMS,
)
from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json  # noqa: E402
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402
from src.metrics import (  # noqa: E402
    build_per_well_metrics,
    paired_well_bootstrap,
    summarize_by_fold,
    summarize_per_well_metrics,
)


EXPERIMENT_ID = "P3_PF02_target_ess_lgbm_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_pf02_target_ess_lgbm_v1.json"
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID

# 复制列表，避免任何调用者意外修改二阶段模块里的冻结常量。
P3B00_FEATURES = list(_P3B00_FEATURES)
NEW_ESS_FEATURES = [
    "pf128_ess2_delta",
    "pf128_ess8_delta",
    "pf128_ess32_delta",
    "pf128_ess96_delta",
]
OLD_PF_FEATURES = [
    "pf128_mean_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
]
PF02_CACHE_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    *NEW_ESS_FEATURES,
    "_cache_fingerprint",
]
BASE_METADATA_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "md",
    "target_tvt",
    "carry_tvt",
    "target_delta",
]


def resolve_clean_path(value: str | Path) -> Path:
    """把配置相对路径统一解释为相对 ``rogii_clean``。"""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (CLEAN_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    """流式计算文件哈希，避免大型 parquet 一次进入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """将实验合同稳定编码为可恢复训练的指纹。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique(values: Iterable[str]) -> list[str]:
    """保持顺序去重，供 parquet 列选择使用。"""

    return list(dict.fromkeys(str(value) for value in values))


def build_formal_feature_names() -> list[str]:
    """返回不替换旧路径的正式 45 列。"""

    features = [*P3B00_FEATURES, *NEW_ESS_FEATURES]
    if len(features) != 45 or len(set(features)) != 45:
        raise ValueError("PF02 正式特征必须是无重复的 41+4=45 列")
    return features


def parse_fold_spec(value: str) -> list[int]:
    """只允许预筛 ``0,1`` 或晋级后的 ``all``。"""

    normalized = str(value).strip().lower()
    if normalized == "0,1":
        return [0, 1]
    if normalized == "all":
        return [0, 1, 2, 3, 4]
    raise ValueError("--folds 只允许 0,1 或 all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="预筛传 0,1；晋级后传 all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def load_development_registry(
    fold_path: Path,
    shadow_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """先按井反连接影子集，只返回允许开发的井注册表。"""

    fold = pd.read_csv(
        fold_path,
        dtype={"well_id": str, "pad_id": str},
    )
    shadow = pd.read_csv(shadow_path, dtype={"well_id": str})
    required_fold = {"well_id", "pad_id", "fold", "hidden_rows"}
    if not required_fold.issubset(fold.columns):
        raise ValueError(f"fold 注册表缺列：{sorted(required_fold - set(fold.columns))}")
    if "well_id" not in shadow.columns:
        raise ValueError("影子注册表缺少 well_id")
    fold["well_id"] = fold["well_id"].astype(str)
    shadow["well_id"] = shadow["well_id"].astype(str)
    if fold["well_id"].duplicated().any() or shadow["well_id"].duplicated().any():
        raise ValueError("fold 或影子注册表含重复井")
    shadow_ids = set(shadow["well_id"].tolist())
    development = fold.loc[~fold["well_id"].isin(shadow_ids)].copy()
    development = development.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if set(development["well_id"]).intersection(shadow_ids):
        raise RuntimeError("影子井反连接失败")

    if config is not None:
        if len(fold) != int(config["total_wells"]):
            raise ValueError("完整 fold 井数不等于配置")
        if len(shadow_ids) != int(config["shadow_wells"]):
            raise ValueError("影子井数不等于配置")
        if len(development) != int(config["development_wells"]):
            raise ValueError("开发井数不等于配置")
        if int(development["hidden_rows"].sum()) != int(
            config["development_hidden_rows"]
        ):
            raise ValueError("开发集隐藏行数不等于配置")
        observed_wells = {
            str(int(fold_id)): int(count)
            for fold_id, count in development.groupby("fold").size().items()
        }
        observed_rows = {
            str(int(fold_id)): int(count)
            for fold_id, count in development.groupby("fold")["hidden_rows"].sum().items()
        }
        if observed_wells != {
            str(key): int(value)
            for key, value in config["development_fold_well_counts"].items()
        }:
            raise ValueError("开发集逐折井数不等于配置")
        if observed_rows != {
            str(key): int(value)
            for key, value in config["development_fold_hidden_row_counts"].items()
        }:
            raise ValueError("开发集逐折隐藏行数不等于配置")
    return development, shadow_ids


def read_development_parquet(
    path: Path,
    well_ids: list[str],
    columns: list[str],
) -> pd.DataFrame:
    """在 Arrow 扫描层按井过滤，之后才把开发行转成 pandas。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    data = arrow_dataset.dataset(str(path), format="parquet")
    missing = set(columns).difference(data.schema.names)
    if missing:
        raise ValueError(f"parquet {path.name} 缺列：{sorted(missing)}")
    allowed_ids = [str(well_id) for well_id in well_ids]
    table = data.to_table(
        columns=columns,
        filter=arrow_dataset.field("well_id").isin(allowed_ids),
    )
    frame = table.to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    outside = set(frame["well_id"].unique()).difference(allowed_ids)
    if outside:
        raise RuntimeError(f"Arrow 井过滤失败：{sorted(outside)[:3]}")
    return frame


def load_development_feature_table(
    base_path: Path,
    candidate_path: Path,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """只加载开发井的 B00 原始列与 C01 确定性路径列。"""

    well_ids = registry["well_id"].astype(str).tolist()
    base_columns = _unique([*BASE_METADATA_COLUMNS, *FEATURE_COLUMNS])
    candidate_columns = ["well_id", "row_index", *DIRECT_CANDIDATE_COLUMNS]
    base = read_development_parquet(base_path, well_ids, base_columns)
    candidate = read_development_parquet(candidate_path, well_ids, candidate_columns)
    for name, frame in (("基础", base), ("候选", candidate)):
        if frame.duplicated(["well_id", "row_index"]).any():
            raise ValueError(f"{name}开发缓存含重复行键")
    if len(base) != int(registry["hidden_rows"].sum()) or len(candidate) != len(base):
        raise ValueError("开发特征缓存总行数不正确")

    base["row_index"] = base["row_index"].astype(np.int64)
    candidate["row_index"] = candidate["row_index"].astype(np.int64)
    base["_original_order"] = np.arange(len(base), dtype=np.int64)
    merged = base.merge(
        candidate,
        on=["well_id", "row_index"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("B00 与 C01 开发行键不能一一对齐")
    merged = merged.sort_values("_original_order", kind="stable").drop(
        columns=["_original_order", "_merge"]
    )
    merged = merged.reset_index(drop=True)
    return remap_feature_table_folds(merged, registry)


def _merge_path_table(
    feature_table: pd.DataFrame,
    path_table: pd.DataFrame,
    path_features: list[str],
    anchor_name: str,
) -> pd.DataFrame:
    """按自然隐藏行键合并路径，并逐位核对路径起点。"""

    collisions = set(path_features).intersection(feature_table.columns)
    if collisions:
        raise ValueError(f"路径特征已存在，禁止覆盖：{sorted(collisions)}")
    base = feature_table.copy()
    base["_original_order"] = np.arange(len(base), dtype=np.int64)
    paths = path_table.rename(columns={"last_visible_tvt": anchor_name})
    merged = base.merge(
        paths[["well_id", "row_index", anchor_name, *path_features]],
        on=["well_id", "row_index"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or len(merged) != len(base):
        raise ValueError("路径缓存与开发特征行键不能一一对齐")
    merged = merged.sort_values("_original_order", kind="stable").reset_index(drop=True)
    base_anchor = merged["last_visible_tvt"].to_numpy(dtype=np.float32)
    path_anchor = merged[anchor_name].to_numpy(dtype=np.float32)
    if not np.array_equal(base_anchor, path_anchor):
        raise ValueError("路径 last_visible_tvt 与基础表不一致")
    values = merged[path_features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("路径特征含 NaN 或 Inf")
    return merged.drop(columns=[anchor_name, "_original_order", "_merge"])


def merge_existing_p3b00_pf_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    expected_fingerprint: str,
) -> pd.DataFrame:
    """从旧合法缓存加入 P3B00 已有五条 PF 路径，不读取影子井。"""

    rows: list[pd.DataFrame] = []
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        cache_path = cache_dir / f"{well_id}.parquet"
        cache = pd.read_parquet(cache_path)
        validate_p01_legal_cache(cache, well_id, int(registry_row.hidden_rows))
        fingerprints = cache["_cache_fingerprint"].astype(str).unique().tolist()
        if fingerprints != [expected_fingerprint]:
            raise ValueError(f"旧 PF 井 {well_id} 指纹不匹配")
        rows.append(cache[["well_id", "row_index", "last_visible_tvt", *OLD_PF_FEATURES]])
    paths = pd.concat(rows, ignore_index=True)
    return _merge_path_table(
        feature_table,
        paths,
        OLD_PF_FEATURES,
        anchor_name="p3b00_pf_anchor",
    )


def merge_pf02_legal_cache(
    feature_table: pd.DataFrame,
    registry: pd.DataFrame,
    cache_dir: Path,
    runtime_dir: Path,
    shadow_ids: set[str],
    require_runtime: bool = False,
) -> tuple[pd.DataFrame, str]:
    """验证并加入四条正式 ESS 路径；禁止目标列、影子井和静默换指纹。"""

    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)
    shadow_cache_overlap = {path.stem for path in cache_dir.glob("*.parquet")}.intersection(
        shadow_ids
    )
    if shadow_cache_overlap:
        raise ValueError(f"PF02 legal_cache 出现影子井：{sorted(shadow_cache_overlap)[:3]}")

    rows: list[pd.DataFrame] = []
    common_fingerprint: str | None = None
    expected_schema = set(PF02_CACHE_COLUMNS)
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        cache_path = cache_dir / f"{well_id}.parquet"
        if not cache_path.is_file():
            raise FileNotFoundError(f"缺少 PF02 井缓存：{cache_path}")
        observed_schema = set(parquet.ParquetFile(cache_path).schema_arrow.names)
        if observed_schema != expected_schema:
            raise ValueError(
                f"PF02 {well_id} schema 不等于冻结合法列："
                f"缺少={sorted(expected_schema-observed_schema)}，"
                f"多余={sorted(observed_schema-expected_schema)}"
            )
        cache = pd.read_parquet(cache_path, columns=PF02_CACHE_COLUMNS)
        if len(cache) != int(registry_row.hidden_rows):
            raise ValueError(f"PF02 {well_id} 行数不匹配")
        if cache.duplicated(["well_id", "row_index"]).any():
            raise ValueError(f"PF02 {well_id} 含重复行键")
        if not cache["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"PF02 {well_id} 井号不匹配")
        if not cache["fold"].astype(int).eq(int(registry_row.fold)).all():
            raise ValueError(f"PF02 {well_id} fold 不匹配")
        fingerprint_values = cache["_cache_fingerprint"].astype(str).unique().tolist()
        if len(fingerprint_values) != 1 or not fingerprint_values[0]:
            raise ValueError(f"PF02 {well_id} 缓存指纹不唯一")
        fingerprint = fingerprint_values[0]
        if common_fingerprint is None:
            common_fingerprint = fingerprint
        elif fingerprint != common_fingerprint:
            raise ValueError("PF02 各井缓存指纹不统一")
        if require_runtime:
            runtime_path = runtime_dir / f"{well_id}.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(f"缺少 PF02 井运行记录：{runtime_path}")
            runtime = read_json(runtime_path)
            if runtime.get("experiment_fingerprint") != fingerprint:
                raise ValueError(f"PF02 {well_id} 运行记录与缓存指纹不一致")
            if runtime.get("hidden_tvt_read") is not False:
                raise ValueError(f"PF02 {well_id} 路径生成读取了隐藏 TVT")
        rows.append(cache[["well_id", "row_index", "last_visible_tvt", *NEW_ESS_FEATURES]])

    if common_fingerprint is None:
        raise ValueError("PF02 没有开发井缓存")
    paths = pd.concat(rows, ignore_index=True)
    merged = _merge_path_table(
        feature_table,
        paths,
        NEW_ESS_FEATURES,
        anchor_name="pf02_anchor",
    )
    return merged, common_fingerprint


def validate_feature_values(feature_table: pd.DataFrame, features: list[str]) -> None:
    """LightGBM 只允许原始 GR 一列含 NaN，其余正式列必须有限。"""

    missing = set(features).difference(feature_table.columns)
    if missing:
        raise ValueError(f"正式特征表缺列：{sorted(missing)}")
    for feature in features:
        values = feature_table[feature].to_numpy(dtype=np.float64)
        if np.isinf(values).any() or (feature != "gr_raw" and np.isnan(values).any()):
            raise ValueError(f"正式特征 {feature} 含非法 NaN/Inf")


def validate_frozen_contract(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """确认唯一变化是追加四条 ESS 路径，模型和旧 41 列未变化。"""

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PF02 模型配置实验编号错误")
    if config.get("baseline_id") != "P3B00_group5_p2p02_v1":
        raise ValueError("PF02 模型基线不是 P3B00")
    if config.get("fold_version") != "balanced_well_5fold_v1":
        raise ValueError("PF02 模型 fold 版本错误")
    if config.get("shadow_target_access") is not False:
        raise ValueError("PF02 模型禁止打开影子目标")
    if config.get("model_training") is not True:
        raise ValueError("PF02 模型配置未声明单模训练")
    if config.get("new_feature_names") != NEW_ESS_FEATURES:
        raise ValueError("PF02 新特征不是冻结的 ESS 2/8/32/96 四列")

    manifest = read_json(resolve_clean_path(config["source_p3b00_feature_list"]))
    if manifest.get("feature_count") != 41 or manifest.get("features") != P3B00_FEATURES:
        raise ValueError("P3B00 41 列清单发生变化")
    model_config = read_json(resolve_clean_path(config["model_config"]))
    if model_config.get("model_family") != "lightgbm.LGBMRegressor":
        raise ValueError("模型不是冻结的单模 LightGBM")
    if model_config.get("params") != FROZEN_MODEL_PARAMS:
        raise ValueError("LightGBM 参数不是冻结的 1734 树 seed29 参数")
    features = build_formal_feature_names()
    if features[:41] != P3B00_FEATURES or any(
        old_feature not in features for old_feature in OLD_PF_FEATURES
    ):
        raise ValueError("PF02 替换或删除了 P3B00 旧 PF 路径")
    forbidden = [
        feature
        for feature in features
        if any(word in feature.lower() for word in ("target", "truth", "oracle", "surface"))
    ]
    if forbidden:
        raise ValueError(f"正式特征含禁止列：{forbidden}")
    return features, dict(FROZEN_MODEL_PARAMS)


def validate_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    """核对冻结来源；大型缓存也只顺序读取一次。"""

    source_keys = [
        "fold_registry",
        "shadow_registry",
        "base_feature_cache",
        "candidate_feature_cache",
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


def read_baseline_predictions(
    path: Path,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Arrow 层只读取当前待比较开发井的 P3B00 OOF。"""

    columns = ["well_id", "fold", "row_index", "target_tvt", "carry_tvt", "pred_tvt"]
    baseline = read_development_parquet(
        path,
        registry["well_id"].astype(str).tolist(),
        columns,
    )
    if len(baseline) != int(registry["hidden_rows"].sum()):
        raise ValueError("P3B00 开发 OOF 行数不正确")
    fold_map = registry.set_index("well_id")["fold"].astype(int)
    remapped = baseline["well_id"].map(fold_map).astype(int)
    if not np.array_equal(remapped.to_numpy(), baseline["fold"].astype(int).to_numpy()):
        raise ValueError("P3B00 OOF fold 与冻结注册表不一致")
    return baseline


def read_fold_predictions(output_dir: Path, fold_ids: list[int]) -> pd.DataFrame:
    """读取当前指纹下已完成折的候选 OOF。"""

    frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        path = output_dir / f"fold_{fold_id}" / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 fold {fold_id} 预测：{path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def compare_with_p3b00(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """严格配对同一开发行，计算统一指标与预登记晋级门槛。"""

    if stage not in {"folds01", "full5"}:
        raise ValueError("未知比较阶段")
    for name, frame in (("候选", candidate), ("基线", baseline)):
        if frame.duplicated(["well_id", "row_index"]).any():
            raise ValueError(f"{name}预测含重复行键")
    candidate_rows = candidate.rename(
        columns={
            "fold": "candidate_fold",
            "target_tvt": "candidate_target",
            "carry_tvt": "candidate_carry",
            "pred_tvt": "candidate_tvt",
        }
    )
    baseline_rows = baseline.rename(
        columns={
            "fold": "baseline_fold",
            "target_tvt": "baseline_target",
            "carry_tvt": "baseline_carry",
            "pred_tvt": "baseline_tvt",
        }
    )
    paired = candidate_rows.merge(
        baseline_rows,
        on=["well_id", "row_index"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(paired) != len(candidate) or not paired["_merge"].eq("both").all():
        raise ValueError("候选和 P3B00 开发行键不完全一致")
    for left, right, dtype, label in (
        ("candidate_fold", "baseline_fold", np.int64, "fold"),
        ("candidate_target", "baseline_target", np.float64, "target"),
        ("candidate_carry", "baseline_carry", np.float64, "carry"),
    ):
        if not np.array_equal(
            paired[left].to_numpy(dtype=dtype),
            paired[right].to_numpy(dtype=dtype),
        ):
            raise ValueError(f"候选与 P3B00 的 {label} 不一致")

    scoring = pd.DataFrame(
        {
            "well_id": paired["well_id"].astype(str),
            "fold": paired["candidate_fold"].astype(int),
            "target_tvt": paired["candidate_target"].astype(np.float64),
            "pred_tvt": paired["candidate_tvt"].astype(np.float64),
            "baseline_tvt": paired["baseline_tvt"].astype(np.float64),
        }
    )
    per_well = build_per_well_metrics(scoring, baseline_column="baseline_tvt")
    overall = summarize_per_well_metrics(per_well)
    overall["baseline_p90_well_rmse"] = float(per_well["baseline_rmse"].quantile(0.90))
    overall["p90_degradation_ft"] = float(
        overall["p90_well_rmse"] - overall["baseline_p90_well_rmse"]
    )
    fold_rows = summarize_by_fold(per_well)
    per_fold = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    per_fold["improvement_ft"] = (
        per_fold["baseline_micro_rmse"] - per_fold["micro_rmse"]
    )
    bootstrap = paired_well_bootstrap(per_well, n_resamples=2000, seed=42)

    positive_sse_gain = np.maximum(
        per_well["baseline_sse"].to_numpy(dtype=np.float64)
        - per_well["prediction_sse"].to_numpy(dtype=np.float64),
        0.0,
    )
    top_count = max(1, int(math.ceil(len(per_well) * 0.05)))
    total_positive_gain = float(positive_sse_gain.sum())
    top_share = (
        float(np.sort(positive_sse_gain)[-top_count:].sum() / total_positive_gain)
        if total_positive_gain > 0.0
        else 1.0
    )
    conditions = config["success_conditions"]
    combined_improvement = float(
        overall["baseline_micro_rmse"] - overall["micro_rmse"]
    )
    improvements = per_fold.set_index("fold")["improvement_ft"].to_dict()

    if stage == "folds01":
        checks = {
            "combined_improvement_ft": combined_improvement,
            "minimum_combined_improvement_ft": float(
                conditions["fold01_minimum_combined_improvement_ft"]
            ),
            "worst_fold_improvement_ft": float(min(improvements.values())),
            "maximum_single_fold_degradation_ft": float(
                conditions["fold01_maximum_single_fold_degradation_ft"]
            ),
        }
        checks["folds01_pass"] = bool(
            checks["combined_improvement_ft"]
            >= checks["minimum_combined_improvement_ft"]
            and checks["worst_fold_improvement_ft"]
            >= -checks["maximum_single_fold_degradation_ft"]
        )
    else:
        improved_folds = int(sum(value > 0.0 for value in improvements.values()))
        improved_late_folds = int(
            sum(improvements.get(fold_id, -np.inf) > 0.0 for fold_id in (2, 3, 4))
        )
        checks = {
            "full5_improvement_ft": combined_improvement,
            "improved_folds": improved_folds,
            "improved_folds_2_to_4": improved_late_folds,
            "worst_fold_improvement_ft": float(min(improvements.values())),
            "bootstrap_ci95_high": float(bootstrap["ci95_high"]),
            "well_win_rate": float(overall["well_win_rate"]),
            "p90_degradation_ft": float(overall["p90_degradation_ft"]),
            "top5_percent_positive_gain_share": top_share,
        }
        checks["full5_pass"] = bool(
            combined_improvement >= float(conditions["full5_minimum_improvement_ft"])
            and improved_folds >= int(conditions["full5_minimum_improved_folds"])
            and improved_late_folds
            >= int(conditions["full5_minimum_improved_folds_2_to_4"])
            and checks["worst_fold_improvement_ft"]
            >= -float(conditions["full5_maximum_single_fold_degradation_ft"])
            and checks["bootstrap_ci95_high"]
            < float(conditions["full5_maximum_bootstrap_ci_upper"])
            and checks["well_win_rate"]
            >= float(conditions["full5_minimum_well_win_rate"])
            and checks["p90_degradation_ft"]
            <= float(conditions["full5_maximum_p90_degradation_ft"])
            and top_share
            <= float(conditions["full5_maximum_top5_percent_positive_gain_share"])
        )

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": config["baseline_id"],
        "stage": stage,
        "overall": overall,
        "folds": per_fold.to_dict(orient="records"),
        "paired_well_bootstrap": bootstrap,
        "top5_percent_positive_gain_share": top_share,
        "success_checks": checks,
    }
    return metrics, per_well, per_fold


def save_stage_artifacts(
    output_dir: Path,
    stage: str,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    per_well: pd.DataFrame,
    per_fold: pd.DataFrame,
) -> None:
    """保存预筛或完整五折的预测、指标和逐井/逐折结果。"""

    if stage == "full5":
        prediction_name = "predictions.parquet"
        metrics_name = "metrics.json"
        per_well_name = "per_well.csv"
        per_fold_name = "per_fold.csv"
    else:
        prediction_name = "predictions_folds01.parquet"
        metrics_name = "metrics_folds01.json"
        per_well_name = "per_well_folds01.csv"
        per_fold_name = "per_fold_folds01.csv"
    predictions.sort_values(["well_id", "row_index"]).to_parquet(
        output_dir / prediction_name,
        index=False,
        compression="zstd",
    )
    write_json(output_dir / metrics_name, metrics)
    per_well.to_csv(output_dir / per_well_name, index=False)
    per_fold.to_csv(output_dir / per_fold_name, index=False)


def save_mean_feature_importance(output_dir: Path) -> None:
    """完整五折后保存平均特征重要性。"""

    tables: list[pd.DataFrame] = []
    for fold_id in range(5):
        table = pd.read_csv(output_dir / f"fold_{fold_id}" / "feature_importance.csv")
        table["fold"] = fold_id
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    mean_importance = (
        combined.groupby("feature", as_index=False)[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    mean_importance.to_csv(output_dir / "feature_importance.csv", index=False)


def write_conclusion(output_dir: Path, metrics: dict[str, Any]) -> None:
    """按三阶段要求，用事实和边界写简短结论。"""

    stage = str(metrics["stage"])
    checks = metrics["success_checks"]
    if stage == "folds01":
        improvement = checks["combined_improvement_ft"]
        passed = checks["folds01_pass"]
    else:
        improvement = checks["full5_improvement_ft"]
        passed = checks["full5_pass"]
    text = f"""# P3-PF02 单模 LightGBM 结论

数据直接证明的事实：本阶段相对 P3B00 的 micro RMSE 改善为 `{improvement:.6f} ft`，预登记门槛{'通过' if passed else '未通过'}。

基于事实的合理推断：这里只检验四条固定目标 ESS 路径对原 41 列的增量价值。

仍然没有验证的猜测：目标 ESS 是否需要沿井深动态变化。

当前实验只能否定的具体实现：将 ESS 2/8/32/96 四条整井路径直接追加到冻结单模 LightGBM。

下一步最便宜的验证：{'按路线进入下一阶段。' if passed else '停止当前 45 列实现，不训练或筛选其他 ESS 参数。'}
"""
    (output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    requested_folds = parse_fold_spec(args.folds)
    config = read_json(args.config.resolve())
    model_features, model_params = validate_frozen_contract(config)
    observed_hashes = validate_source_hashes(config)
    registry, shadow_ids = load_development_registry(
        resolve_clean_path(config["fold_registry"]),
        resolve_clean_path(config["shadow_registry"]),
        config,
    )

    # 这一步始终只读取 657 口开发井；训练 fold 0/1 也需要其余三折作为训练集。
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
    feature_table, pf02_fingerprint = merge_pf02_legal_cache(
        feature_table,
        registry,
        resolve_clean_path(config["source_pf02_legal_cache_dir"]),
        resolve_clean_path(config["source_pf02_runtime_dir"]),
        shadow_ids,
        require_runtime=True,
    )
    if set(feature_table["well_id"].astype(str).unique()).intersection(shadow_ids):
        raise RuntimeError("正式训练表含影子井")
    validate_feature_values(feature_table, model_features)

    fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "model_features": model_features,
            "model_params": model_params,
            "pf02_generator_fingerprint": pf02_fingerprint,
            "observed_source_hashes": observed_hashes,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(
                CLEAN_ROOT / "scripts" / "run_simple_lgbm_cv.py"
            ),
        }
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "parameter_list.json", model_params)
    write_json(
        output_dir / "feature_list.json",
        {"feature_count": 45, "features": model_features},
    )
    write_json(
        output_dir / "leakage_audit.json",
        {
            "shadow_target_access": False,
            "shadow_wells": len(shadow_ids),
            "development_wells": int(feature_table["well_id"].nunique()),
            "development_hidden_rows": len(feature_table),
            "shadow_feature_overlap": 0,
            "arrow_filter_before_target_to_pandas": True,
            "validation_unit": "complete_well",
            "fold_version": config["fold_version"],
            "old_41_features_preserved": model_features[:41] == P3B00_FEATURES,
            "new_features": NEW_ESS_FEATURES,
            "pf02_generator_fingerprint": pf02_fingerprint,
            "cv_fingerprint": fingerprint,
            "observed_source_hashes": observed_hashes,
        },
    )
    print(
        f"P3-PF02 模型：开发井={len(registry)}，行={len(feature_table):,}，"
        f"特征=45，树=1734，折={requested_folds}，输出={output_dir}",
        flush=True,
    )

    # all 也先执行/复用 folds 0～1，并重新按本次指纹判断晋级门槛。
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
    write_conclusion(output_dir, metrics01)
    print(json.dumps(metrics01["success_checks"], ensure_ascii=False, indent=2), flush=True)
    if requested_folds == [0, 1]:
        return
    if not metrics01["success_checks"]["folds01_pass"]:
        print("PF02 模型 folds 0～1 未晋级，停止，不训练 folds 2～4。", flush=True)
        return

    for fold_id in (2, 3, 4):
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
    fold_runtimes = [read_json(output_dir / f"fold_{fold_id}" / "runtime.json") for fold_id in range(5)]
    write_json(
        output_dir / "runtime.json",
        {
            "cv_fingerprint": fingerprint,
            "folds": fold_runtimes,
            "total_fold_seconds": float(sum(float(row["seconds"]) for row in fold_runtimes)),
        },
    )
    write_conclusion(output_dir, metrics_full)
    print(json.dumps(metrics_full["success_checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
