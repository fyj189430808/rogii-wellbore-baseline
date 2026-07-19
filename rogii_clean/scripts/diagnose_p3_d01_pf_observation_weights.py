"""P3-D01：仅用合法输入回放 PF，并保存 128 个种子的累计似然。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_d01_pf_observation_audit import (  # noqa: E402
    read_likelihood_cache,
    run_pf_likelihood_audit,
    summarize_weight_scale,
    write_likelihood_cache,
)


EXPERIMENT_ID = "P3_D01_pf_observation_weight_audit_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_d01_pf_observation_weight_audit_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "fold01", "all")
LEGAL_HORIZONTAL_COLUMNS = ["MD", "Z", "GR", "TVT_input"]
LEGAL_TYPEWELL_COLUMNS = ["TVT", "GR"]
PATH_COLUMNS = [
    "row_index",
    "last_visible_tvt",
    "pf128_mean_tvt",
    "pf128_mean_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
]
QUALITY_COLUMN_MAP = {
    "pf_best_ll_per_row": "pf_best_ll_per_row",
    "pf_ll_spread": "pf_ll_spread",
    "gr_sigma": "pf_gr_sigma",
}
FROZEN_SOURCE_CONTRACT = {
    "fold_registry": "artifacts/folds/balanced_well_5fold_v1.csv",
    "fold_registry_sha256": "a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c",
    "shadow_registry": "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    "shadow_registry_sha256": "7fde7c16895e02f74c96a70a1ca3d9935e0e9bfb24c7744a1992c025aad483c1",
    "raw_train_dir": "../input/data/raw/train",
    "source_pf_config": "configs/p2_p01_multiseed_pf_mean_v1.json",
    "source_pf_config_sha256": "f4e9ba4e59b985aec7dcc14d2c99a4590d1bcb9946def0630a46acc3f14a243b",
    "source_pf_core": "src/p2_p01_multiseed_pf.py",
    "source_pf_core_sha256": "b636982d24daa4f0f34adee578bc50783e5a1e1791e9929e96dcfa32490c0782",
    "audit_core": "src/p3_d01_pf_observation_audit.py",
    "source_model_predictions": "artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet",
    "source_model_predictions_sha256": "8109514eb125f8beda2559f39e1d9f54396d41fd38fa76114423c8dd6bca4d37",
    "source_pf_legal_cache_dir": "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache",
    "source_pf_legal_runtime_dir": "artifacts/P2_P01_multiseed_pf_mean_v1/legal_runtime",
    "source_pf_artifact_fingerprint": "ac1dbefd59923585671156b7d3e8b4fc7faca95c20f5f1ae6fdab92a8665954a",
}


def file_sha256(path: Path) -> str:
    """分块计算文件 SHA-256，避免一次读入大文件。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """把完整配置稳定序列化后计算实验指纹。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_ready(value: Any) -> Any:
    """把 NumPy 标量递归转换为 JSON 能直接保存的 Python 标量。"""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    """先写临时 JSON 再替换，避免中断留下半个文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存不含目标和误差的逐井合法诊断表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _resolve_from_clean(path_text: str) -> Path:
    """配置中的路径统一相对 rogii_clean 根目录解析。"""

    return (CLEAN_ROOT / path_text).resolve()


def validate_config(config: dict[str, Any]) -> None:
    """拒绝影子边界、井数、PF 来源或实验职责被静默修改。"""

    expected_values: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "P3B00_group5_p2p02_v1",
        "source_pf_experiment_id": "P2_P01_multiseed_pf_mean_v1",
        "source_model_experiment_id": "P2_P02_multiscale_pf_paths_v1",
        "fold_version": "balanced_well_5fold_v1",
        "total_wells": 773,
        "total_hidden_rows": 3_783_989,
        "shadow_wells": 116,
        "development_wells": 657,
        "development_hidden_rows": 3_211_872,
        "development_fold_well_counts": {"0": 131, "1": 132, "2": 131, "3": 132, "4": 131},
        "development_fold_hidden_row_counts": {
            "0": 651_881,
            "1": 630_395,
            "2": 645_557,
            "3": 649_717,
            "4": 634_322,
        },
        "workers": 8,
        "likelihood_scales": [3.0, 5.0, 8.0, 12.0],
        "model_training": False,
        "formal_feature_output": False,
        "shadow_target_access": False,
    }
    for name, expected in expected_values.items():
        if config.get(name) != expected:
            raise ValueError(f"P3-D01 {name} 不等于冻结值 {expected}")

    for name, expected in FROZEN_SOURCE_CONTRACT.items():
        if config.get(name) != expected:
            raise ValueError(f"P3-D01 {name} 不等于冻结来源 {expected}")

    required_strings = [
        "fold_registry",
        "fold_registry_sha256",
        "shadow_registry",
        "shadow_registry_sha256",
        "raw_train_dir",
        "source_pf_config",
        "source_pf_config_sha256",
        "source_pf_core",
        "source_pf_core_sha256",
        "audit_core",
        "source_model_predictions",
        "source_model_predictions_sha256",
        "source_pf_legal_cache_dir",
        "source_pf_legal_runtime_dir",
        "source_pf_artifact_fingerprint",
    ]
    for name in required_strings:
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"P3-D01 缺少字符串配置 {name}")


def rebuild_likelihood_report(
    seed_log_likelihoods: np.ndarray,
    hidden_rows: int,
    observed_gr_rows: int,
) -> dict[str, Any]:
    """只从 NPZ 累计似然重建全部 LL 和 ESS 相关报告字段。"""

    values = np.asarray(seed_log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("P3-D01 累计似然必须是一维有限数组")
    if hidden_rows <= 0 or observed_gr_rows < 0:
        raise ValueError("P3-D01 隐藏行数或 GR 观测行数非法")
    rebuilt: dict[str, Any] = {
        "seed_ll_min": float(np.min(values)),
        "seed_ll_max": float(np.max(values)),
        "seed_ll_mean": float(np.mean(values)),
        "seed_ll_median": float(np.median(values)),
        "seed_ll_std": float(np.std(values)),
        "seed_ll_range": float(np.ptp(values)),
        "seed_ll_iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
        "seed_ll_mean_per_hidden_row": float(np.mean(values) / hidden_rows),
        "ll_per_observed_row": float(np.mean(values) / max(observed_gr_rows, 1)),
        "best_seed_id": int(np.argmax(values)),
        "pf_best_ll_per_row": float(np.max(values) / hidden_rows),
        "pf_ll_spread": float(np.std(values)),
    }
    for scale in (3.0, 5.0, 8.0, 12.0):
        prefix = f"scale_{int(scale)}"
        summary = summarize_weight_scale(values, scale)
        rebuilt.update({f"{prefix}_{name}": value for name, value in summary.items()})
    return rebuilt


def _shadow_selection_mask(shadow: pd.DataFrame) -> pd.Series:
    """兼容布尔、0/1 和字符串形式的 is_shadow。"""

    if "is_shadow" not in shadow.columns:
        return pd.Series(True, index=shadow.index)
    values = shadow["is_shadow"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def load_development_registry(
    fold_path: Path,
    shadow_path: Path,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """先按井号反连接影子井，再返回可创建任务的开发井注册表。"""

    fold = pd.read_csv(
        fold_path,
        usecols=["well_id", "fold", "hidden_rows"],
        dtype={"well_id": str},
    )
    fold["well_id"] = fold["well_id"].astype(str)
    fold["fold"] = pd.to_numeric(fold["fold"], errors="raise").astype(int)
    fold["hidden_rows"] = pd.to_numeric(fold["hidden_rows"], errors="raise").astype(int)
    if fold["well_id"].duplicated().any():
        raise ValueError("P3-D01 fold 表含重复井号")
    if not set(fold["fold"]).issubset({0, 1, 2, 3, 4}):
        raise ValueError("P3-D01 fold 只能是 0～4")
    if (fold["hidden_rows"] <= 0).any():
        raise ValueError("P3-D01 hidden_rows 必须为正整数")

    shadow_header = pd.read_csv(shadow_path, nrows=0).columns.tolist()
    shadow_columns = ["well_id"]
    if "is_shadow" in shadow_header:
        shadow_columns.append("is_shadow")
    shadow = pd.read_csv(shadow_path, usecols=shadow_columns, dtype={"well_id": str})
    selected_shadow = shadow.loc[_shadow_selection_mask(shadow), ["well_id"]].copy()
    selected_shadow["well_id"] = selected_shadow["well_id"].astype(str)
    if selected_shadow["well_id"].duplicated().any():
        raise ValueError("P3-D01 shadow 表含重复井号")
    shadow_ids = set(selected_shadow["well_id"])
    if not shadow_ids.issubset(set(fold["well_id"])):
        raise ValueError("P3-D01 shadow 表出现 fold 表外井号")

    development = fold.loc[~fold["well_id"].isin(shadow_ids)].copy()
    development = development.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if not set(development["well_id"]).isdisjoint(shadow_ids):
        raise RuntimeError("P3-D01 开发井和影子井仍有交集")

    if config is not None:
        if len(fold) != int(config["total_wells"]):
            raise ValueError("P3-D01 总井数不等于冻结值")
        if int(fold["hidden_rows"].sum()) != int(config["total_hidden_rows"]):
            raise ValueError("P3-D01 总隐藏行数不等于冻结值")
        if len(shadow_ids) != int(config["shadow_wells"]):
            raise ValueError("P3-D01 影子井数不等于冻结值")
        if len(development) != int(config["development_wells"]):
            raise ValueError("P3-D01 开发井数不等于冻结值")
        if int(development["hidden_rows"].sum()) != int(config["development_hidden_rows"]):
            raise ValueError("P3-D01 开发隐藏行数不等于冻结值")
        observed_wells = development.groupby("fold").size().to_dict()
        observed_rows = development.groupby("fold")["hidden_rows"].sum().to_dict()
        expected_wells = {int(key): int(value) for key, value in config["development_fold_well_counts"].items()}
        expected_rows = {int(key): int(value) for key, value in config["development_fold_hidden_row_counts"].items()}
        if observed_wells != expected_wells:
            raise ValueError("P3-D01 开发集逐折井数不等于冻结值")
        if observed_rows != expected_rows:
            raise ValueError("P3-D01 开发集逐折隐藏行数不等于冻结值")
    return development


def select_mode_registry(development: pd.DataFrame, mode: str) -> pd.DataFrame:
    """smoke 固定第一口井，fold01 固定前两折，all 固定全部开发井。"""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"P3-D01 不支持模式 {mode}")
    selected = development.sort_values("well_id", kind="mergesort").copy()
    if mode == "smoke":
        selected = selected.iloc[:1]
    elif mode == "fold01":
        selected = selected.loc[selected["fold"].isin([0, 1])]
    return selected.reset_index(drop=True)


def read_legal_inputs(raw_train_dir: Path, well_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """物理限制 CSV 列，水平井隐藏 TVT 和 surface 从未进入内存。"""

    horizontal_path = Path(raw_train_dir) / f"{well_id}__horizontal_well.csv"
    typewell_path = Path(raw_train_dir) / f"{well_id}__typewell.csv"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        raise FileNotFoundError(f"P3-D01 找不到井 {well_id} 的原始 CSV")
    horizontal = pd.read_csv(horizontal_path, usecols=LEGAL_HORIZONTAL_COLUMNS)
    typewell = pd.read_csv(typewell_path, usecols=LEGAL_TYPEWELL_COLUMNS)
    return horizontal[LEGAL_HORIZONTAL_COLUMNS], typewell[LEGAL_TYPEWELL_COLUMNS]


def _validate_path_features(path_features: dict[str, np.ndarray], hidden_rows: int) -> None:
    """路径字段、行数、行键和有限值必须与冻结 P2-P01 合同一致。"""

    if set(path_features) != set(PATH_COLUMNS):
        raise ValueError("P3-D01 PF 回放路径字段不等于冻结字段")
    for name in PATH_COLUMNS:
        values = np.asarray(path_features[name])
        if values.ndim != 1 or len(values) != hidden_rows:
            raise ValueError(f"P3-D01 路径 {name} 行数不匹配")
        if not np.isfinite(values.astype(np.float64)).all():
            raise ValueError(f"P3-D01 路径 {name} 含非有限值")
    row_index = np.asarray(path_features["row_index"], dtype=np.int64)
    if len(row_index) > 1 and np.any(np.diff(row_index) <= 0):
        raise ValueError("P3-D01 row_index 必须严格递增")


def reconcile_legacy_outputs(
    *,
    well_id: str,
    hidden_rows: int,
    path_features: dict[str, np.ndarray],
    report: dict[str, Any],
    legacy_cache_path: Path,
    legacy_runtime_path: Path,
    expected_source_fingerprint: str | None = None,
) -> dict[str, Any]:
    """按 row_index 与旧合法缓存逐位比较；任何非零差异立即失败。"""

    if not legacy_cache_path.is_file() or not legacy_runtime_path.is_file():
        raise FileNotFoundError(f"P3-D01 找不到井 {well_id} 的旧合法产物")
    old = pd.read_parquet(
        legacy_cache_path,
        columns=["well_id", *PATH_COLUMNS, "_cache_fingerprint"],
    )
    if len(old) != hidden_rows or not old["well_id"].astype(str).eq(well_id).all():
        raise ValueError("P3-D01 旧合法缓存井号或行数不匹配")
    if expected_source_fingerprint is not None:
        fingerprints = old["_cache_fingerprint"].astype(str).unique().tolist()
        if fingerprints != [expected_source_fingerprint]:
            raise ValueError("P3-D01 旧合法缓存指纹不匹配")

    new = pd.DataFrame({name: np.asarray(path_features[name]) for name in PATH_COLUMNS})
    if new["row_index"].duplicated().any() or old["row_index"].duplicated().any():
        raise ValueError("P3-D01 新旧路径含重复 row_index")
    new = new.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    old = old.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    path_differences: dict[str, float] = {}
    for name in PATH_COLUMNS:
        new_values = new[name].to_numpy()
        old_values = old[name].to_numpy()
        if not np.array_equal(new_values, old_values):
            raise ValueError(f"P3-D01 {name} 未与旧合法缓存逐位一致")
        path_differences[name] = 0.0

    old_runtime = json.loads(legacy_runtime_path.read_text(encoding="utf-8"))
    if expected_source_fingerprint is not None:
        if old_runtime.get("experiment_fingerprint") != expected_source_fingerprint:
            raise ValueError("P3-D01 旧 runtime 指纹不匹配")
        if old_runtime.get("cache_sha256") != file_sha256(legacy_cache_path):
            raise ValueError("P3-D01 旧 runtime 记录的缓存 SHA 不匹配")
    old_quality = old_runtime.get("quality")
    if not isinstance(old_quality, dict):
        raise ValueError("P3-D01 旧 runtime 缺少质量诊断")
    quality_differences: dict[str, float] = {}
    for new_name, old_name in QUALITY_COLUMN_MAP.items():
        new_value = float(report[new_name])
        old_value = float(old_quality[old_name])
        difference = abs(new_value - old_value)
        if difference != 0.0:
            raise ValueError(f"P3-D01 质量诊断 {new_name} 未与旧值完全一致")
        quality_differences[new_name] = difference
    return {
        "enabled": True,
        "path_max_abs_difference": path_differences,
        "maximum_path_difference": 0.0,
        "quality_abs_difference": quality_differences,
        "maximum_quality_difference": 0.0,
        "source_legal_cache_sha256": file_sha256(legacy_cache_path),
        "source_legal_runtime_sha256": file_sha256(legacy_runtime_path),
    }


def _cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    """返回一口井的似然 NPZ 和合法 runtime 路径。"""

    return (
        artifact_dir / "legal_likelihood_cache" / f"{well_id}.npz",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def load_valid_cache_hit(task: dict[str, Any]) -> dict[str, Any] | None:
    """runtime、NPZ、SHA、指纹和旧对账全部有效时才允许恢复。"""

    well_id = str(task["well_id"])
    fold = int(task["fold"])
    hidden_rows = int(task["hidden_rows"])
    fingerprint = str(task["fingerprint"])
    number_of_seeds = int(task["expected_number_of_seeds"])
    verify_legacy = bool(task.get("verify_legacy_cache", True))
    artifact_dir = Path(str(task["artifact_dir"]))
    cache_path, runtime_path = _cache_paths(artifact_dir, well_id)
    if not cache_path.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        required = {
            "experiment_id",
            "experiment_fingerprint",
            "well_id",
            "fold",
            "hidden_rows",
            "legal_horizontal_columns",
            "typewell_columns",
            "hidden_tvt_read",
            "number_of_seeds",
            "cache_sha256",
            "legal_report",
            "legal_report_sha256",
            "legacy_reconciliation",
            "elapsed_seconds",
        }
        if not required.issubset(runtime):
            return None
        if runtime["experiment_id"] != EXPERIMENT_ID:
            return None
        if runtime["experiment_fingerprint"] != fingerprint:
            return None
        if runtime["well_id"] != well_id or int(runtime["fold"]) != fold:
            return None
        if int(runtime["hidden_rows"]) != hidden_rows:
            return None
        if runtime["legal_horizontal_columns"] != LEGAL_HORIZONTAL_COLUMNS:
            return None
        if runtime["typewell_columns"] != LEGAL_TYPEWELL_COLUMNS:
            return None
        if runtime["hidden_tvt_read"] is not False:
            return None
        if int(runtime["number_of_seeds"]) != number_of_seeds:
            return None
        if runtime["cache_sha256"] != file_sha256(cache_path):
            return None
        likelihoods = read_likelihood_cache(cache_path, fingerprint, number_of_seeds)
        if not isinstance(runtime["legal_report"], dict):
            return None
        report = runtime["legal_report"]
        if runtime["legal_report_sha256"] != stable_json_hash(report):
            return None
        observed_gr_rows = int(report["observed_gr_rows"])
        rebuilt_report = rebuild_likelihood_report(
            likelihoods,
            hidden_rows,
            observed_gr_rows,
        )
        for name, rebuilt_value in rebuilt_report.items():
            if name not in report or report[name] != rebuilt_value:
                return None
        reconciliation = runtime["legacy_reconciliation"]
        if verify_legacy:
            if not isinstance(reconciliation, dict) or reconciliation.get("enabled") is not True:
                return None
            if float(reconciliation.get("maximum_path_difference", np.inf)) != 0.0:
                return None
            if float(reconciliation.get("maximum_quality_difference", np.inf)) != 0.0:
                return None
            path_differences = reconciliation.get("path_max_abs_difference", {})
            quality_differences = reconciliation.get("quality_abs_difference", {})
            if set(path_differences) != set(PATH_COLUMNS) or any(float(value) != 0.0 for value in path_differences.values()):
                return None
            if set(quality_differences) != set(QUALITY_COLUMN_MAP) or any(float(value) != 0.0 for value in quality_differences.values()):
                return None
            old_cache = Path(str(task["source_legal_cache_dir"])) / f"{well_id}.parquet"
            old_runtime = Path(str(task["source_legal_runtime_dir"])) / f"{well_id}.json"
            if reconciliation.get("source_legal_cache_sha256") != file_sha256(old_cache):
                return None
            if reconciliation.get("source_legal_runtime_sha256") != file_sha256(old_runtime):
                return None
            old_runtime_data = json.loads(old_runtime.read_text(encoding="utf-8"))
            old_quality = old_runtime_data.get("quality")
            if not isinstance(old_quality, dict):
                return None
            if rebuilt_report["pf_best_ll_per_row"] != old_quality.get("pf_best_ll_per_row"):
                return None
            if rebuilt_report["pf_ll_spread"] != old_quality.get("pf_ll_spread"):
                return None
            if report.get("gr_sigma") != old_quality.get("pf_gr_sigma"):
                return None
        result = dict(runtime)
        result["cache_hit"] = True
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """只读合法列回放一口井；旧路径完全一致后原子保存似然。"""

    cached = load_valid_cache_hit(task)
    if cached is not None:
        return cached

    started = time.perf_counter()
    well_id = str(task["well_id"])
    fold = int(task["fold"])
    hidden_rows = int(task["hidden_rows"])
    fingerprint = str(task["fingerprint"])
    number_of_seeds = int(task["expected_number_of_seeds"])
    raw_train_dir = Path(str(task["raw_train_dir"]))
    artifact_dir = Path(str(task["artifact_dir"]))
    horizontal, typewell = read_legal_inputs(raw_train_dir, well_id)
    observed_hidden_rows = int(horizontal["TVT_input"].isna().sum())
    if observed_hidden_rows != hidden_rows:
        raise ValueError(f"P3-D01 井 {well_id} 原始隐藏行数不匹配")

    likelihoods, report, path_features = run_pf_likelihood_audit(
        horizontal,
        typewell,
        dict(task["particle_filter"]),
    )
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    if likelihoods.shape != (number_of_seeds,):
        raise ValueError(f"P3-D01 井 {well_id} 累计似然数量不匹配")
    _validate_path_features(path_features, hidden_rows)
    if int(report.get("hidden_rows", -1)) != hidden_rows:
        raise ValueError(f"P3-D01 井 {well_id} 合法报告隐藏行数不匹配")
    rebuilt_report = rebuild_likelihood_report(
        likelihoods,
        hidden_rows,
        int(report.get("observed_gr_rows", -1)),
    )
    for name, rebuilt_value in rebuilt_report.items():
        if name not in report or report[name] != rebuilt_value:
            raise ValueError(f"P3-D01 井 {well_id} 合法报告 {name} 无法由累计似然复算")

    verify_legacy = bool(task.get("verify_legacy_cache", True))
    if verify_legacy:
        legacy_cache_path = Path(str(task["source_legal_cache_dir"])) / f"{well_id}.parquet"
        legacy_runtime_path = Path(str(task["source_legal_runtime_dir"])) / f"{well_id}.json"
        reconciliation = reconcile_legacy_outputs(
            well_id=well_id,
            hidden_rows=hidden_rows,
            path_features=path_features,
            report=report,
            legacy_cache_path=legacy_cache_path,
            legacy_runtime_path=legacy_runtime_path,
            expected_source_fingerprint=str(task["source_pf_artifact_fingerprint"]),
        )
    else:
        reconciliation = {"enabled": False}

    cache_path, runtime_path = _cache_paths(artifact_dir, well_id)
    write_likelihood_cache(cache_path, likelihoods, fingerprint)
    runtime: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "well_id": well_id,
        "fold": fold,
        "hidden_rows": hidden_rows,
        "legal_horizontal_columns": LEGAL_HORIZONTAL_COLUMNS,
        "typewell_columns": LEGAL_TYPEWELL_COLUMNS,
        "hidden_tvt_read": False,
        "number_of_seeds": number_of_seeds,
        "cache_sha256": file_sha256(cache_path),
        "legal_report": report,
        "legal_report_sha256": stable_json_hash(report),
        "legacy_reconciliation": reconciliation,
        "cache_hit": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(runtime_path, runtime)
    return _json_ready(runtime)


def run_legal_generation(
    tasks: list[dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """并发逐井生成；异常只收集，不会被静默吞掉。"""

    if workers <= 0:
        raise ValueError("P3-D01 workers 必须为正整数")
    runtimes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_well = {
            executor.submit(generate_one_well, task): str(task["well_id"])
            for task in tasks
        }
        for completed, future in enumerate(as_completed(future_to_well), start=1):
            well_id = future_to_well[future]
            try:
                runtime = future.result()
                runtimes.append(runtime)
                state = "缓存" if bool(runtime.get("cache_hit")) else "新算"
                seconds = float(runtime.get("elapsed_seconds", 0.0))
                print(f"P3-D01 legal {completed}/{len(tasks)}：{well_id}（{state}，{seconds:.1f} 秒）", flush=True)
            except Exception as error:  # noqa: BLE001 - 必须保留井号和完整异常。
                errors.append({"well_id": well_id, "error": repr(error)})
                print(f"P3-D01 legal {completed}/{len(tasks)}：{well_id} 失败：{error}", flush=True)
    return runtimes, errors


def ensure_legal_generation_succeeded(errors: list[dict[str, str]]) -> None:
    """合法生成有任一错误就阻断后续阶段。"""

    if errors:
        raise RuntimeError(f"P3-D01 合法生成失败，共 {len(errors)} 口井")


def build_legal_per_well_table(runtimes: list[dict[str, Any]]) -> pd.DataFrame:
    """展开合法 report；列名中不允许出现目标、误差或 oracle。"""

    rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        row: dict[str, Any] = {
            "well_id": runtime["well_id"],
            "fold": runtime["fold"],
            "hidden_rows": runtime["hidden_rows"],
            "cache_hit": runtime["cache_hit"],
            "elapsed_seconds": runtime["elapsed_seconds"],
        }
        row.update(dict(runtime["legal_report"]))
        reconciliation = runtime["legacy_reconciliation"]
        row["maximum_path_difference"] = reconciliation.get("maximum_path_difference", np.nan)
        row["maximum_quality_difference"] = reconciliation.get("maximum_quality_difference", np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    forbidden_fragments = ("target", "truth", "error", "rmse", "oracle", "surface")
    forbidden = [column for column in table.columns if any(fragment in column.lower() for fragment in forbidden_fragments)]
    if forbidden:
        raise ValueError(f"P3-D01 合法逐井表出现禁止列：{forbidden}")
    return table


def validate_external_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """核对冻结来源 SHA，并从唯一 P2-P01 config 读取粒子参数。"""

    checked_files = {
        "fold_registry": (config["fold_registry"], config["fold_registry_sha256"]),
        "shadow_registry": (config["shadow_registry"], config["shadow_registry_sha256"]),
        "source_pf_config": (config["source_pf_config"], config["source_pf_config_sha256"]),
        "source_pf_core": (config["source_pf_core"], config["source_pf_core_sha256"]),
        "source_model_predictions": (
            config["source_model_predictions"],
            config["source_model_predictions_sha256"],
        ),
    }
    hashes: dict[str, str] = {}
    for name, (path_text, expected_hash) in checked_files.items():
        path = _resolve_from_clean(str(path_text))
        if not path.is_file():
            raise FileNotFoundError(f"P3-D01 找不到冻结输入 {path}")
        actual_hash = file_sha256(path)
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError(f"P3-D01 {name} SHA-256 不一致")
        hashes[name] = actual_hash
    raw_train_dir = _resolve_from_clean(str(config["raw_train_dir"]))
    source_cache_dir = _resolve_from_clean(str(config["source_pf_legal_cache_dir"]))
    source_runtime_dir = _resolve_from_clean(str(config["source_pf_legal_runtime_dir"]))
    if not raw_train_dir.is_dir() or not source_cache_dir.is_dir() or not source_runtime_dir.is_dir():
        raise FileNotFoundError("P3-D01 原始训练目录或旧合法缓存目录不存在")
    source_pf_config = json.loads(
        _resolve_from_clean(str(config["source_pf_config"])).read_text(encoding="utf-8")
    )
    particle_filter = source_pf_config.get("particle_filter")
    if not isinstance(particle_filter, dict):
        raise ValueError("P3-D01 冻结 P2-P01 config 缺少 particle_filter")
    if int(particle_filter.get("number_of_seeds", -1)) != 128:
        raise ValueError("P3-D01 冻结 PF 种子数不是 128")
    return {
        "hashes": hashes,
        "particle_filter": particle_filter,
        "raw_train_dir": raw_train_dir,
        "source_cache_dir": source_cache_dir,
        "source_runtime_dir": source_runtime_dir,
    }


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """指纹覆盖配置、runner、D01 核心、旧 PF 核心和冻结来源。"""

    return stable_json_hash(
        {
            "config": config,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "audit_core_sha256": file_sha256(_resolve_from_clean(str(config["audit_core"]))),
            "source_pf_core_sha256": config["source_pf_core_sha256"],
            "fold_registry_sha256": config["fold_registry_sha256"],
            "shadow_registry_sha256": config["shadow_registry_sha256"],
            "source_pf_config_sha256": config["source_pf_config_sha256"],
            "source_pf_artifact_fingerprint": config["source_pf_artifact_fingerprint"],
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析固定模式、配置和产物目录。"""

    parser = argparse.ArgumentParser(description="运行 P3-D01 PF 观测权重合法审计")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """完成合法回放和缓存；本脚本到此为止，不读取目标或计算 RMSE。"""

    arguments = parse_args(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    validate_config(config)
    external = validate_external_inputs(config)
    development = load_development_registry(
        _resolve_from_clean(str(config["fold_registry"])),
        _resolve_from_clean(str(config["shadow_registry"])),
        config,
    )
    selected = select_mode_registry(development, arguments.mode)
    fingerprint = experiment_fingerprint(config)
    artifact_dir = arguments.artifact_dir.resolve()
    tasks: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        tasks.append(
            {
                "well_id": str(row.well_id),
                "fold": int(row.fold),
                "hidden_rows": int(row.hidden_rows),
                "raw_train_dir": str(external["raw_train_dir"]),
                "artifact_dir": str(artifact_dir),
                "fingerprint": fingerprint,
                "particle_filter": external["particle_filter"],
                "expected_number_of_seeds": 128,
                "verify_legacy_cache": True,
                "source_legal_cache_dir": str(external["source_cache_dir"]),
                "source_legal_runtime_dir": str(external["source_runtime_dir"]),
                "source_pf_artifact_fingerprint": config["source_pf_artifact_fingerprint"],
            }
        )

    print(
        f"P3-D01 {arguments.mode}：{len(tasks)} 口开发井，"
        f"{int(selected['hidden_rows'].sum())} 个隐藏行，{config['workers']} 个线程",
        flush=True,
    )
    print(f"输出目录：{artifact_dir}", flush=True)
    print("中断后原命令重跑即可，完整有效的单井缓存会自动命中。", flush=True)
    started = time.perf_counter()
    runtimes, errors = run_legal_generation(tasks, int(config["workers"]))
    mode_runtime_path = artifact_dir / f"runtime_{arguments.mode}.json"
    if errors:
        write_json_atomic(
            mode_runtime_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "mode": arguments.mode,
                "completed": False,
                "experiment_fingerprint": fingerprint,
                "errors": errors,
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )
        ensure_legal_generation_succeeded(errors)

    legal_table = build_legal_per_well_table(runtimes)
    output_path = artifact_dir / "legal" / f"per_well_{arguments.mode}.csv"
    write_csv_atomic(output_path, legal_table)
    write_json_atomic(
        mode_runtime_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "mode": arguments.mode,
            "completed": True,
            "experiment_fingerprint": fingerprint,
            "wells": len(runtimes),
            "hidden_rows": int(selected["hidden_rows"].sum()),
            "cache_hits": int(sum(bool(runtime["cache_hit"]) for runtime in runtimes)),
            "hidden_tvt_read": False,
            "model_training": False,
            "formal_feature_output": False,
            "shadow_target_access": False,
            "elapsed_seconds": float(time.perf_counter() - started),
            "per_well_output": str(output_path),
        },
    )
    print(f"P3-D01 合法生成完成：{output_path}", flush=True)


if __name__ == "__main__":
    main()
