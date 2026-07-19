"""生成并审计 P2-P01 的 128-seed 粒子路径均值。

合法阶段只读取水平井 ``MD/Z/GR/TVT_input`` 和 Typewell ``TVT/GR``。
全部选中井的合法缓存落盘后，脚本才读取冻结 P2B00 真值和 F05a 单次 PF，
用于路径精度诊断。本脚本不训练 LightGBM。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import numba


CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_p01_multiseed_pf import build_multiseed_pf_features  # noqa: E402


EXPERIMENT_ID = "P2_P01_multiseed_pf_mean_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_p01_multiseed_pf_mean_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "fold0", "all")
EXPECTED_ALL_WELLS = 773
EXPECTED_ALL_HIDDEN_ROWS = 3_783_989
EXPECTED_FOLD0_WELLS = 155
EXPECTED_FOLD0_HIDDEN_ROWS = 757_738

LEGAL_FEATURE_COLUMNS = [
    "well_id",
    "row_index",
    "last_visible_tvt",
    "pf128_mean_tvt",
    "pf128_mean_delta",
    "pf128_seed0_delta",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
    "pf128_seed_std",
    "_cache_fingerprint",
]

FROZEN_PARTICLE_FILTER = {
    "number_of_particles": 500,
    "number_of_seeds": 128,
    "seed_base": 0,
    "typewell_grid_step_ft": 0.2,
    "initial_position_spread_ft": 4.5,
    "initial_rate_std": 0.01,
    "rate_momentum": 0.998,
    "rate_noise": 0.002,
    "position_noise_ft": 0.005,
    "position_limit_beyond_typewell_ft": 100.0,
    "minimum_md_step_ft": 1.0,
    "squared_gr_residual_cap": 600.0,
    "likelihood_floor": 1e-300,
    "gr_sigma_min_api": 10.0,
    "gr_sigma_max_api": 60.0,
    "initial_rate_visible_tail_rows": 30,
    "resample_effective_fraction": 0.5,
    "resample_position_noise_ft": 0.1,
    "resample_rate_noise": 0.001,
    "likelihood_scales": [3.0, 5.0, 8.0, 12.0],
    "formal_seed_aggregation": "unweighted_mean",
}

FROZEN_RUNTIME_VERSIONS = {
    "numpy": "1.26.4",
    "numba": "0.60.0",
}

FROZEN_NUMERICAL_CONVENTIONS = {
    "typewell_gr_missing": "fill_typewell_mean_after_stable_tvt_sort",
    "visible_gr_missing_for_sigma": "fill_zero",
    "hidden_gr_missing": "whole_well_linear_both_then_typewell_mean",
    "typewell_grid_upper": "numpy_arange_min_to_max_plus_step",
    "typewell_kernel_upper": "vmin_plus_grid_length_times_step",
    "md_step": "first_step_one_and_all_steps_minimum_one",
    "resample_comparison": "strict_effective_count_less_than_threshold",
    "legacy_feature_cast": (
        "absolute_path_float32_then_subtract_float32_last_visible"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析固定运行阶段、配置和产物目录。"""

    parser = argparse.ArgumentParser(description="运行 P2-P01 128-seed PF 路径实验")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    """分块计算文件 SHA-256，避免把大 parquet 一次读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """把字典按稳定顺序序列化后计算指纹。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    """先写临时 JSON 再替换，避免中断留下半个文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存单井合法特征缓存。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存逐井评分表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    """拒绝结果出现后修改 PF、正式特征或冻结数据合同。"""

    expected_top_level = {
        "experiment_id": EXPERIMENT_ID,
        "fold_version": "balanced_well_5fold_v1",
        "formal_feature_name": "pf128_mean_delta",
        "source_feature_name": "likpf_mean_d",
        "expected_wells": EXPECTED_ALL_WELLS,
        "expected_hidden_rows": EXPECTED_ALL_HIDDEN_ROWS,
        "expected_fold0_wells": EXPECTED_FOLD0_WELLS,
        "expected_fold0_hidden_rows": EXPECTED_FOLD0_HIDDEN_ROWS,
        "workers": 8,
    }
    for name, expected in expected_top_level.items():
        if config.get(name) != expected:
            raise ValueError(f"P2-P01 {name} 不等于冻结值 {expected}")

    if config.get("runtime_versions") != FROZEN_RUNTIME_VERSIONS:
        raise ValueError("P2-P01 runtime_versions 被修改")
    if config.get("numerical_conventions") != FROZEN_NUMERICAL_CONVENTIONS:
        raise ValueError("P2-P01 numerical_conventions 被修改")

    observed_parameters = config.get("particle_filter")
    if not isinstance(observed_parameters, dict):
        raise ValueError("P2-P01 particle_filter 配置缺失")
    for name, expected in FROZEN_PARTICLE_FILTER.items():
        if observed_parameters.get(name) != expected:
            raise ValueError(f"P2-P01 particle_filter.{name} 被修改")
    if set(observed_parameters) != set(FROZEN_PARTICLE_FILTER):
        raise ValueError("P2-P01 particle_filter 出现未登记参数")

    required_generation_conditions = {
        "maximum_repeat_difference_ft",
        "maximum_concurrent_difference_ft",
        "maximum_hidden_tvt_mutation_difference_ft",
    }
    observed_conditions = config.get("feature_generation_success_conditions")
    if not isinstance(observed_conditions, dict):
        raise ValueError("P2-P01 feature_generation_success_conditions 缺失")
    missing_conditions = required_generation_conditions.difference(
        observed_conditions
    )
    if missing_conditions:
        raise ValueError(f"P2-P01 特征生成门槛缺失：{sorted(missing_conditions)}")
    if set(observed_conditions) != required_generation_conditions:
        raise ValueError("P2-P01 特征生成门槛出现未登记字段")


def validate_runtime_versions(config: dict[str, Any]) -> None:
    """核对实际 NumPy/Numba 版本，防止随机数轨迹悄悄改变。"""

    actual = {
        "numpy": str(np.__version__),
        "numba": str(numba.__version__),
    }
    if actual != config["runtime_versions"]:
        raise RuntimeError(
            f"P2-P01 运行库版本不一致：配置 {config['runtime_versions']}，实际 {actual}"
        )


def select_mode_registry(registry: pd.DataFrame, mode: str) -> pd.DataFrame:
    """按注册表固定选择 smoke 第一口 fold0、完整 fold0 或全部井。"""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"P2-P01 不支持运行模式 {mode}")
    required = {"well_id", "fold", "hidden_rows"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"P2-P01 fold 表缺列：{sorted(missing)}")
    selected = registry.copy()
    selected["well_id"] = selected["well_id"].astype(str)
    if mode == "all":
        return selected.reset_index(drop=True)
    selected = selected.loc[selected["fold"].astype(int).eq(0)].copy()
    if mode == "smoke":
        selected = selected.iloc[:1].copy()
    return selected.reset_index(drop=True)


def validate_legal_cache(
    cache: pd.DataFrame,
    expected_well_id: str,
    expected_rows: int,
) -> None:
    """验证合法缓存的列边界、行键、有限值和逐井身份。"""

    forbidden_columns: list[str] = []
    for column in cache.columns:
        lowered = str(column).lower()
        if lowered == "tvt" or any(
            fragment in lowered
            for fragment in ("target", "truth", "oracle", "surface", "geology")
        ):
            forbidden_columns.append(str(column))
    if forbidden_columns:
        raise ValueError(f"P2-P01 合法缓存出现禁止列：{forbidden_columns}")

    missing = set(LEGAL_FEATURE_COLUMNS).difference(cache.columns)
    extra = set(cache.columns).difference(LEGAL_FEATURE_COLUMNS)
    if missing:
        raise ValueError(f"P2-P01 合法缓存缺列：{sorted(missing)}")
    if extra:
        raise ValueError(f"P2-P01 合法缓存出现未登记列：{sorted(extra)}")
    if len(cache) != int(expected_rows):
        raise ValueError(
            f"P2-P01 合法缓存行数 {len(cache)} 不等于预期 {expected_rows}"
        )
    if bool(cache.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("P2-P01 合法缓存含重复 well_id-row_index 键")
    if not cache["well_id"].astype(str).eq(str(expected_well_id)).all():
        raise ValueError("P2-P01 合法缓存 well_id 不匹配")
    row_index = pd.to_numeric(cache["row_index"], errors="raise").to_numpy()
    if len(row_index) > 1 and bool(np.any(np.diff(row_index) <= 0)):
        raise ValueError("P2-P01 合法缓存 row_index 不是严格递增")
    numeric_columns = [
        column
        for column in LEGAL_FEATURE_COLUMNS
        if column not in {"well_id", "_cache_fingerprint"}
    ]
    numeric_values = cache[numeric_columns].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(numeric_values).all()):
        raise ValueError("P2-P01 合法缓存出现非有限数值")
    fingerprints = cache["_cache_fingerprint"].astype(str).unique().tolist()
    if len(fingerprints) != 1 or not fingerprints[0]:
        raise ValueError("P2-P01 合法缓存指纹不唯一或为空")


def evaluate_path_success_checks(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    """只用确定性和隐藏真值隔离控制决定能否生成全量特征。

    fold0 RMSE、胜井率、P90 和反转路径都只保留为诊断，不参与本函数。
    """

    conditions = config["feature_generation_success_conditions"]
    checks = {
        "repeat_determinism_pass": bool(
            metrics["maximum_repeat_difference_ft"]
            <= float(conditions["maximum_repeat_difference_ft"])
        ),
        "concurrent_determinism_pass": bool(
            metrics["maximum_concurrent_difference_ft"]
            <= float(conditions["maximum_concurrent_difference_ft"])
        ),
        "hidden_tvt_independence_pass": bool(
            metrics["maximum_hidden_tvt_mutation_difference_ft"]
            <= float(conditions["maximum_hidden_tvt_mutation_difference_ft"])
        ),
    }
    checks["feature_generation_supported"] = bool(all(checks.values()))
    return checks


def _resolve_from_clean(path_text: str) -> Path:
    """配置路径统一相对 clean 根目录解析。"""

    return (CLEAN_ROOT / path_text).resolve()


def validate_external_inputs(config: dict[str, Any]) -> dict[str, str]:
    """运行前逐个核对冻结 fold、Notebook、基线和单 PF 来源。"""

    checked = {
        "fold_registry": (
            _resolve_from_clean(str(config["fold_registry"])),
            str(config["fold_registry_sha256"]),
        ),
        "source_notebook": (
            _resolve_from_clean(str(config["source_notebook"])),
            str(config["source_notebook_sha256"]),
        ),
        "baseline_feature_list": (
            _resolve_from_clean(str(config["baseline_feature_list"])),
            str(config["baseline_feature_list_sha256"]),
        ),
        "baseline_predictions": (
            _resolve_from_clean(str(config["baseline_predictions"])),
            str(config["baseline_predictions_sha256"]),
        ),
        "single_pf_meta": (
            _resolve_from_clean(str(config["single_pf_meta"])),
            str(config["single_pf_meta_sha256"]),
        ),
    }
    observed: dict[str, str] = {}
    for name, (path, expected_hash) in checked.items():
        if not path.is_file():
            raise FileNotFoundError(f"P2-P01 找不到冻结输入：{path}")
        actual_hash = file_sha256(path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(f"P2-P01 {name} SHA-256 不一致")
        observed[name] = actual_hash

    raw_train_dir = _resolve_from_clean(str(config["raw_train_dir"]))
    single_pf_dir = _resolve_from_clean(str(config["single_pf_cache_dir"]))
    if not raw_train_dir.is_dir():
        raise FileNotFoundError(f"P2-P01 找不到原始训练目录：{raw_train_dir}")
    if not single_pf_dir.is_dir():
        raise FileNotFoundError(f"P2-P01 找不到 F05a 单井缓存：{single_pf_dir}")

    meta = json.loads(checked["single_pf_meta"][0].read_text(encoding="utf-8"))
    if not bool(meta.get("completed")):
        raise ValueError("P2-P01 F05a 单 PF 缓存未完成")
    if int(meta.get("wells", -1)) != EXPECTED_ALL_WELLS:
        raise ValueError("P2-P01 F05a 单 PF 缓存井数不一致")
    if int(meta.get("rows", -1)) != EXPECTED_ALL_HIDDEN_ROWS:
        raise ValueError("P2-P01 F05a 单 PF 缓存行数不一致")
    return observed


def load_registry(config: dict[str, Any]) -> pd.DataFrame:
    """读取并严格检查固定按井五折注册表。"""

    registry = pd.read_csv(
        _resolve_from_clean(str(config["fold_registry"])),
        dtype={"well_id": str, "pad_id": str},
    )
    required = {"well_id", "fold", "hidden_rows"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"P2-P01 fold 表缺列：{sorted(missing)}")
    registry["well_id"] = registry["well_id"].astype(str)
    registry["fold"] = pd.to_numeric(registry["fold"], errors="raise").astype(int)
    registry["hidden_rows"] = pd.to_numeric(
        registry["hidden_rows"], errors="raise"
    ).astype(int)
    if bool(registry["well_id"].duplicated().any()):
        raise ValueError("P2-P01 fold 表含重复井")
    if sorted(registry["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("P2-P01 fold 表必须含 0～4")
    if len(registry) != EXPECTED_ALL_WELLS:
        raise ValueError("P2-P01 fold 表井数不等于 773")
    if int(registry["hidden_rows"].sum()) != EXPECTED_ALL_HIDDEN_ROWS:
        raise ValueError("P2-P01 fold 表隐藏行数不一致")
    return registry.reset_index(drop=True)


def validate_selected_registry(selected: pd.DataFrame, mode: str) -> None:
    """对 fold0/all 的预注册井数和行数做硬检查。"""

    wells = int(len(selected))
    rows = int(selected["hidden_rows"].sum())
    if mode == "fold0" and (
        wells != EXPECTED_FOLD0_WELLS or rows != EXPECTED_FOLD0_HIDDEN_ROWS
    ):
        raise ValueError("P2-P01 fold0 井数或隐藏行数不等于冻结值")
    if mode == "all" and (
        wells != EXPECTED_ALL_WELLS or rows != EXPECTED_ALL_HIDDEN_ROWS
    ):
        raise ValueError("P2-P01 all 井数或隐藏行数不等于冻结值")


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """指纹覆盖配置、runner、PF 核心和全部冻结外部哈希。"""

    return stable_json_hash(
        {
            "config": config,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "core_sha256": file_sha256(
                CLEAN_ROOT / "src" / "p2_p01_multiseed_pf.py"
            ),
            "fold_registry_sha256": config["fold_registry_sha256"],
            "source_notebook_sha256": config["source_notebook_sha256"],
            "baseline_feature_list_sha256": str(
                config["baseline_feature_list_sha256"]
            ).lower(),
            "baseline_predictions_sha256": config["baseline_predictions_sha256"],
            "single_pf_meta_sha256": config["single_pf_meta_sha256"],
        }
    )


def _cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    """返回一口井的合法 parquet 和 runtime 路径。"""

    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def _read_legal_inputs(
    raw_train_dir: Path,
    well_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """物理限制 CSV 读取列，保证正式特征生成看不到隐藏 TVT。"""

    horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        raise FileNotFoundError(f"P2-P01 找不到井 {well_id} 的原始文件")
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    return horizontal, typewell


def _load_valid_cache_hit(
    cache_path: Path,
    runtime_path: Path,
    well_id: str,
    expected_rows: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    """只有 runtime、parquet 内容和文件哈希全部一致才允许命中。"""

    if not cache_path.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("experiment_fingerprint") != fingerprint:
            return None
        if runtime.get("well_id") != str(well_id):
            return None
        if int(runtime.get("hidden_rows", -1)) != int(expected_rows):
            return None
        if runtime.get("cache_sha256") != file_sha256(cache_path):
            return None
        cache = pd.read_parquet(cache_path)
        validate_legal_cache(cache, well_id, expected_rows)
        if cache["_cache_fingerprint"].iloc[0] != fingerprint:
            return None
        runtime = dict(runtime)
        runtime["cache_hit"] = True
        return runtime
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """用合法输入生成一口井 128-seed 路径并原子落盘。"""

    well_id = str(task["well_id"])
    fold = int(task["fold"])
    expected_rows = int(task["hidden_rows"])
    raw_train_dir = Path(str(task["raw_train_dir"]))
    artifact_dir = Path(str(task["artifact_dir"]))
    config = dict(task["config"])
    fingerprint = str(task["fingerprint"])
    cache_path, runtime_path = _cache_paths(artifact_dir, well_id)

    cache_hit = _load_valid_cache_hit(
        cache_path,
        runtime_path,
        well_id,
        expected_rows,
        fingerprint,
    )
    if cache_hit is not None:
        return cache_hit

    started = time.perf_counter()
    horizontal, typewell = _read_legal_inputs(raw_train_dir, well_id)
    features, quality = build_multiseed_pf_features(
        horizontal,
        typewell,
        dict(config["particle_filter"]),
    )
    legal = features.copy()
    legal.insert(0, "well_id", well_id)
    legal["_cache_fingerprint"] = fingerprint
    legal = legal[LEGAL_FEATURE_COLUMNS]
    validate_legal_cache(legal, well_id, expected_rows)
    write_parquet_atomic(cache_path, legal)

    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "well_id": well_id,
        "fold": fold,
        "hidden_rows": expected_rows,
        "legal_horizontal_columns": ["MD", "Z", "GR", "TVT_input"],
        "typewell_columns": ["TVT", "GR"],
        "hidden_tvt_read": False,
        "quality": {name: float(value) for name, value in quality.items()},
        "cache_sha256": file_sha256(cache_path),
        "cache_hit": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(runtime_path, runtime)
    return runtime


def run_legal_generation(
    tasks: list[dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """用线程池逐井生成，完成顺序不影响每井固定 seed。"""

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
                print(
                    f"P2-P01 legal {completed}/{len(tasks)}：{well_id}（{state}）",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - 必须记录井号。
                errors.append({"well_id": well_id, "error": repr(error)})
                print(
                    f"P2-P01 legal {completed}/{len(tasks)}：{well_id} 失败",
                    flush=True,
                )
    return runtimes, errors


def _maximum_feature_difference(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """比较两份同键特征的最大绝对差；非有限或错键直接报错。"""

    left_sorted = left.sort_values("row_index", kind="stable").reset_index(drop=True)
    right_sorted = right.sort_values("row_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        left_sorted["row_index"].to_numpy(),
        right_sorted["row_index"].to_numpy(),
    ):
        raise ValueError("P2-P01 程序控制的 row_index 不一致")
    columns = [
        column
        for column in left_sorted.columns
        if column != "row_index" and column in right_sorted.columns
    ]
    difference = np.abs(
        left_sorted[columns].to_numpy(dtype=np.float64)
        - right_sorted[columns].to_numpy(dtype=np.float64)
    )
    return float(np.max(difference)) if difference.size else 0.0


def run_smoke_program_controls(
    selected_registry: pd.DataFrame,
    raw_train_dir: Path,
    artifact_dir: Path,
    config: dict[str, Any],
) -> dict[str, float | str | int]:
    """用 smoke 真井做重复、并发和隐藏 TVT 改写三项 exact 检查。"""

    registry_row = selected_registry.iloc[0]
    well_id = str(registry_row["well_id"])
    horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
    full_horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    legal_horizontal = full_horizontal[["MD", "Z", "GR", "TVT_input"]].copy()
    cached = pd.read_parquet(artifact_dir / "legal_cache" / f"{well_id}.parquet")
    cached_features = cached.drop(columns=["well_id", "_cache_fingerprint"])

    def build_original() -> pd.DataFrame:
        result, _ = build_multiseed_pf_features(
            legal_horizontal,
            typewell,
            dict(config["particle_filter"]),
        )
        return result

    mutated_horizontal = full_horizontal.copy()
    hidden_mask = mutated_horizontal["TVT_input"].isna()
    if "TVT" in mutated_horizontal.columns:
        hidden_count = int(hidden_mask.sum())
        mutation = 50_000.0 + np.arange(hidden_count, dtype=np.float64)
        mutated_horizontal.loc[hidden_mask, "TVT"] = mutation

    def build_mutated() -> pd.DataFrame:
        result, _ = build_multiseed_pf_features(
            mutated_horizontal,
            typewell,
            dict(config["particle_filter"]),
        )
        return result

    # 第四份输入物理删除 TVT 列；这比单纯改写数值更直接地证明生成器不依赖它。
    without_truth_horizontal = full_horizontal.drop(
        columns=["TVT"],
        errors="ignore",
    )

    def build_without_truth() -> pd.DataFrame:
        result, _ = build_multiseed_pf_features(
            without_truth_horizontal,
            typewell,
            dict(config["particle_filter"]),
        )
        return result

    # 四个调用同时运行，避免程序控制串行等待四次完整 128-seed。
    with ThreadPoolExecutor(max_workers=4) as executor:
        original_future_a = executor.submit(build_original)
        original_future_b = executor.submit(build_original)
        mutated_future = executor.submit(build_mutated)
        without_truth_future = executor.submit(build_without_truth)
        original_a = original_future_a.result()
        original_b = original_future_b.result()
        mutated = mutated_future.result()
        without_truth = without_truth_future.result()

    mutation_difference = _maximum_feature_difference(original_a, mutated)
    deletion_difference = _maximum_feature_difference(original_a, without_truth)

    return {
        "well_id": well_id,
        "rows": int(len(cached_features)),
        "maximum_repeat_difference_ft": _maximum_feature_difference(
            cached_features,
            original_a,
        ),
        "maximum_concurrent_difference_ft": _maximum_feature_difference(
            original_a,
            original_b,
        ),
        "maximum_hidden_tvt_mutation_difference_ft": max(
            mutation_difference,
            deletion_difference,
        ),
        "maximum_hidden_tvt_deletion_difference_ft": deletion_difference,
    }


def load_smoke_program_controls(
    artifact_dir: Path,
    fingerprint: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """fold0/all 只接受当前代码指纹下已通过的 smoke 程序控制。"""

    control_path = artifact_dir / "program_controls_smoke.json"
    if not control_path.is_file():
        raise FileNotFoundError("P2-P01 必须先运行 smoke 程序控制")
    controls = json.loads(control_path.read_text(encoding="utf-8"))
    if controls.get("experiment_fingerprint") != fingerprint:
        raise ValueError("P2-P01 smoke 程序控制指纹与当前代码不一致")
    checks = evaluate_path_success_checks(controls, config)
    if not checks["feature_generation_supported"]:
        raise RuntimeError("P2-P01 smoke 程序控制失败，不能生成后续特征")
    return controls


def _load_baseline_selected(selected_wells: set[str], config: dict[str, Any]) -> pd.DataFrame:
    """合法缓存完成后才读取含 target_tvt 的冻结 P2B00 文件。"""

    baseline = pd.read_parquet(_resolve_from_clean(str(config["baseline_predictions"])))
    required = {
        "well_id",
        "fold",
        "row_index",
        "target_tvt",
        "carry_tvt",
        "pred_tvt",
    }
    missing = required.difference(baseline.columns)
    if missing:
        raise ValueError(f"P2-P01 P2B00 缺列：{sorted(missing)}")
    baseline["well_id"] = baseline["well_id"].astype(str)
    selected = baseline.loc[baseline["well_id"].isin(selected_wells)].copy()
    if bool(selected.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("P2-P01 P2B00 含重复行键")
    return selected


def _read_single_pf(
    well_id: str,
    expected_row_index: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """读取冻结 F05a 单次 PF，并严格核对自然隐藏行键。"""

    path = _resolve_from_clean(str(config["single_pf_cache_dir"])) / f"{well_id}.parquet"
    single = pd.read_parquet(
        path,
        columns=["well_id", "row_index", "pf_ancc_delta", "_cache_fingerprint"],
    )
    single["well_id"] = single["well_id"].astype(str)
    single = single.sort_values("row_index", kind="stable").reset_index(drop=True)
    if not single["well_id"].eq(well_id).all():
        raise ValueError(f"P2-P01 {well_id} F05a well_id 不一致")
    if bool(single.duplicated(["well_id", "row_index"]).any()):
        raise ValueError(f"P2-P01 {well_id} F05a 行键重复")
    if not np.array_equal(
        single["row_index"].to_numpy(dtype=np.int64),
        expected_row_index,
    ):
        raise ValueError(f"P2-P01 {well_id} F05a 行键不一致")
    if single["_cache_fingerprint"].astype(str).nunique() != 1:
        raise ValueError(f"P2-P01 {well_id} F05a 指纹不唯一")
    return single["pf_ancc_delta"].to_numpy(dtype=np.float64)


def _rmse(sse: float, rows: int) -> float:
    """由平方误差和与行数计算 pooled RMSE。"""

    return float(math.sqrt(float(sse) / int(rows)))


PATH_NAMES = (
    "carry",
    "p2b00",
    "single_pf",
    "seed0",
    "pf128_mean",
    "scale_3",
    "scale_5",
    "scale_8",
    "scale_12",
    "reversed",
)


def validate_last_visible_matches_carry(
    carry_tvt: pd.Series,
    cached_last_visible_tvt: pd.Series,
) -> None:
    """按旧高分特征的 float32 精度，核对两边是否使用同一个路径起点。"""

    # P01 的旧代码语义会先把最后一个可见 TVT 转成 float32；
    # 因此这里也先做相同转换，再要求逐位完全相等，既允许正常舍入，也不放宽定义。
    expected_float32 = carry_tvt.to_numpy(dtype=np.float32)
    cached_float32 = cached_last_visible_tvt.to_numpy(dtype=np.float32)
    if not np.array_equal(expected_float32, cached_float32):
        raise ValueError("P2-P01 last_visible_tvt 与 carry 不一致")


def score_legal_cache(
    selected_registry: pd.DataFrame,
    artifact_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """在合法缓存全部落盘后，隔离读取真值并计算路径阶段指标。"""

    selected_wells = set(selected_registry["well_id"].astype(str))
    baseline = _load_baseline_selected(selected_wells, config)
    baseline_groups = {
        str(well_id): group.sort_values("row_index", kind="stable").reset_index(drop=True)
        for well_id, group in baseline.groupby("well_id", sort=False)
    }
    metric_rows: list[dict[str, Any]] = []

    for completed, registry_row in enumerate(
        selected_registry.itertuples(index=False),
        start=1,
    ):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        expected_rows = int(registry_row.hidden_rows)
        legal = pd.read_parquet(artifact_dir / "legal_cache" / f"{well_id}.parquet")
        validate_legal_cache(legal, well_id, expected_rows)
        legal = legal.sort_values("row_index", kind="stable").reset_index(drop=True)
        expected = baseline_groups.get(well_id)
        if expected is None:
            raise ValueError(f"P2-P01 P2B00 缺少井 {well_id}")
        if len(expected) != expected_rows:
            raise ValueError(f"P2-P01 {well_id} P2B00 行数不一致")
        row_index = legal["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            expected["row_index"].to_numpy(dtype=np.int64),
            row_index,
        ):
            raise ValueError(f"P2-P01 {well_id} P2B00 行键不一致")
        if not expected["fold"].astype(int).eq(fold).all():
            raise ValueError(f"P2-P01 {well_id} fold 不一致")

        validate_last_visible_matches_carry(
            expected["carry_tvt"],
            legal["last_visible_tvt"],
        )
        last_visible = legal["last_visible_tvt"].to_numpy(dtype=np.float64)
        single_delta = _read_single_pf(well_id, row_index, config)
        truth = expected["target_tvt"].to_numpy(dtype=np.float64)
        mean_delta = legal["pf128_mean_delta"].to_numpy(dtype=np.float64)
        predictions = {
            "carry": expected["carry_tvt"].to_numpy(dtype=np.float64),
            "p2b00": expected["pred_tvt"].to_numpy(dtype=np.float64),
            "single_pf": last_visible + single_delta,
            "seed0": last_visible
            + legal["pf128_seed0_delta"].to_numpy(dtype=np.float64),
            "pf128_mean": legal["pf128_mean_tvt"].to_numpy(dtype=np.float64),
            "scale_3": last_visible
            + legal["pf128_scale_3_delta"].to_numpy(dtype=np.float64),
            "scale_5": last_visible
            + legal["pf128_scale_5_delta"].to_numpy(dtype=np.float64),
            "scale_8": last_visible
            + legal["pf128_scale_8_delta"].to_numpy(dtype=np.float64),
            "scale_12": last_visible
            + legal["pf128_scale_12_delta"].to_numpy(dtype=np.float64),
            "reversed": last_visible + mean_delta[::-1],
        }
        metric_row: dict[str, Any] = {
            "well_id": well_id,
            "fold": fold,
            "hidden_rows": expected_rows,
            "pf128_seed_std_mean": float(
                legal["pf128_seed_std"].to_numpy(dtype=np.float64).mean()
            ),
        }
        for path_name, prediction in predictions.items():
            error = np.asarray(prediction, dtype=np.float64) - truth
            sse = float(np.sum(np.square(error)))
            metric_row[f"{path_name}_sse"] = sse
            metric_row[f"{path_name}_rmse"] = _rmse(sse, expected_rows)
        metric_row["pf128_wins_single_pf"] = bool(
            metric_row["pf128_mean_rmse"] < metric_row["single_pf_rmse"]
        )
        metric_rows.append(metric_row)
        if completed % 20 == 0 or completed == len(selected_registry):
            print(
                f"P2-P01 score {completed}/{len(selected_registry)}",
                flush=True,
            )

    per_well = pd.DataFrame(metric_rows)
    total_rows = int(per_well["hidden_rows"].sum())
    summary: dict[str, Any] = {
        "wells": int(len(per_well)),
        "hidden_rows": total_rows,
        "well_win_rate_vs_single_pf": float(per_well["pf128_wins_single_pf"].mean()),
        "mean_pf128_seed_std": float(
            np.average(per_well["pf128_seed_std_mean"], weights=per_well["hidden_rows"])
        ),
    }
    for path_name in PATH_NAMES:
        total_sse = float(per_well[f"{path_name}_sse"].sum())
        summary[f"{path_name}_rmse"] = _rmse(total_sse, total_rows)
        summary[f"{path_name}_p90_well_rmse"] = float(
            per_well[f"{path_name}_rmse"].quantile(0.90)
        )
    summary["fold0_improvement_vs_single_pf_ft"] = float(
        summary["single_pf_rmse"] - summary["pf128_mean_rmse"]
    )
    summary["p90_degradation_vs_single_pf_ft"] = float(
        summary["pf128_mean_p90_well_rmse"]
        - summary["single_pf_p90_well_rmse"]
    )
    summary["gain_vs_reversed_path_ft"] = float(
        summary["reversed_rmse"] - summary["pf128_mean_rmse"]
    )
    summary["fold_metrics"] = {}
    for fold, fold_frame in per_well.groupby("fold", sort=True):
        fold_rows = int(fold_frame["hidden_rows"].sum())
        summary["fold_metrics"][str(int(fold))] = {
            path_name: _rmse(float(fold_frame[f"{path_name}_sse"].sum()), fold_rows)
            for path_name in PATH_NAMES
        }
    return per_well, summary


def _write_stage_conclusion(
    artifact_dir: Path,
    mode: str,
    summary: dict[str, Any],
) -> None:
    """保存五段式简明结论，避免 smoke/fold0/all 相互覆盖。"""

    path = artifact_dir / f"conclusion_{mode}.md"
    feature_generation_supported = summary.get("success_checks", {}).get(
        "feature_generation_supported"
    )
    if mode == "fold0":
        fact = (
            f"fold 0 的 pf128 均值 RMSE 为 {summary['pf128_mean_rmse']:.6f}，"
            f"单次 PF 为 {summary['single_pf_rmse']:.6f}。"
        )
        next_step = (
            "程序控制通过，可生成 all 合法缓存；路径 RMSE 只作诊断。"
            if feature_generation_supported
            else "程序控制未通过，不能生成 all 合法缓存。"
        )
    else:
        fact = f"{mode} 完成 {summary['wells']} 口井、{summary['hidden_rows']} 行。"
        next_step = "等待 fold0 路径门槛。" if mode == "smoke" else "合法全量特征已生成。"
    text = (
        f"# P2-P01 {mode} 结论\n\n"
        f"事实：{fact}\n\n"
        "推断：只有预注册门槛能决定当前实现是否继续。\n\n"
        "仍未验证：把 pf128_mean_delta 加入固定 LightGBM 后的 CV。\n\n"
        "当前只能否定：当前冻结参数的 128-seed 均值路径实现。\n\n"
        f"下一步：{next_step}\n"
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def main(argv: list[str] | None = None) -> None:
    """先完成合法路径缓存，再隔离读取真值评分；不训练模型。"""

    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    validate_runtime_versions(config)
    observed_hashes = validate_external_inputs(config)
    registry = load_registry(config)
    selected = select_mode_registry(registry, str(args.mode))
    validate_selected_registry(selected, str(args.mode))
    fingerprint = experiment_fingerprint(config)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifact_dir / "config.json", config)
    write_json_atomic(
        artifact_dir / "lineage.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "legal_horizontal_columns": ["MD", "Z", "GR", "TVT_input"],
            "typewell_columns": ["TVT", "GR"],
            "formal_feature_name": "pf128_mean_delta",
            "formal_seed_aggregation": "unweighted_mean",
            "runtime_versions": config["runtime_versions"],
            "numerical_conventions": config["numerical_conventions"],
            "external_sha256": observed_hashes,
            "hidden_tvt_read_after_all_selected_legal_cache": True,
            "model_trained_by_this_script": False,
        },
    )
    write_json_atomic(
        artifact_dir / "feature_list.json",
        {
            "formal_new_feature": ["pf128_mean_delta"],
            "diagnostic_only_features": [
                "pf128_seed0_delta",
                "pf128_scale_3_delta",
                "pf128_scale_5_delta",
                "pf128_scale_8_delta",
                "pf128_scale_12_delta",
                "pf128_seed_std",
            ],
            "legal_cache_columns": LEGAL_FEATURE_COLUMNS,
        },
    )
    write_json_atomic(artifact_dir / "parameter_list.json", config["particle_filter"])

    raw_train_dir = _resolve_from_clean(str(config["raw_train_dir"]))
    # fold0/all 不再由路径 RMSE 决定是否运行；唯一前置条件是当前指纹的
    # smoke 确定性与隐藏真值隔离控制已经通过。
    controls: dict[str, Any] = {}
    if args.mode != "smoke":
        controls = load_smoke_program_controls(
            artifact_dir,
            fingerprint,
            config,
        )
    workers = 1 if args.mode == "smoke" else int(config["workers"])
    print(
        f"P2-P01 {args.mode}：{len(selected)} 口井，"
        f"每井 128 seed × 500 粒子，{workers} 线程；输出 {artifact_dir}",
        flush=True,
    )
    started = time.perf_counter()
    tasks = [
        {
            "well_id": str(row.well_id),
            "fold": int(row.fold),
            "hidden_rows": int(row.hidden_rows),
            "raw_train_dir": str(raw_train_dir),
            "artifact_dir": str(artifact_dir),
            "config": config,
            "fingerprint": fingerprint,
        }
        for row in selected.itertuples(index=False)
    ]
    runtimes, errors = run_legal_generation(tasks, workers)
    if errors:
        write_json_atomic(
            artifact_dir / f"errors_{args.mode}.json",
            {"experiment_fingerprint": fingerprint, "errors": errors},
        )
        raise RuntimeError(f"P2-P01 有 {len(errors)} 口井生成失败")

    if args.mode == "smoke":
        controls = run_smoke_program_controls(
            selected,
            raw_train_dir,
            artifact_dir,
            config,
        )
        controls["experiment_fingerprint"] = fingerprint
        controls["success_checks"] = evaluate_path_success_checks(controls, config)
        write_json_atomic(artifact_dir / "program_controls_smoke.json", controls)
        if not controls["success_checks"]["feature_generation_supported"]:
            raise RuntimeError("P2-P01 smoke 特征生成程序控制失败")

    # 到这里所有选中井合法缓存已经落盘，才允许载入 target 和 F05a 做诊断。
    per_well, summary = score_legal_cache(selected, artifact_dir, config)
    summary["mode"] = str(args.mode)
    summary["experiment_id"] = EXPERIMENT_ID
    summary["experiment_fingerprint"] = fingerprint
    summary.update(controls)
    summary["success_checks"] = evaluate_path_success_checks(summary, config)

    wall_seconds = float(time.perf_counter() - started)
    stage = str(args.mode)
    write_csv_atomic(artifact_dir / f"per_well_path_{stage}.csv", per_well)
    write_json_atomic(artifact_dir / f"path_metrics_{stage}.json", summary)
    write_json_atomic(
        artifact_dir / f"runtime_{stage}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "mode": stage,
            "wells": int(len(selected)),
            "hidden_rows": int(selected["hidden_rows"].sum()),
            "workers": workers,
            "wall_seconds": wall_seconds,
            "cache_hits": int(sum(bool(item.get("cache_hit")) for item in runtimes)),
            "per_well_elapsed_seconds_sum": float(
                sum(float(item.get("elapsed_seconds", 0.0)) for item in runtimes)
            ),
        },
    )
    _write_stage_conclusion(artifact_dir, stage, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
