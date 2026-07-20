"""生成 P3-PFS01 三条固定滞后粒子平滑路径的逐井合法缓存。

本脚本只读取测试时可取得的水平井 ``MD/Z/GR/TVT_input`` 和 Typewell
``TVT/GR``。它不会读取水平井隐藏 ``TVT``，也不训练模型。三条路径都先按
128 个 seed 独立完成尾段回退，再用冻结的 likelihood scale=8 聚合。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_p01_multiseed_pf import (  # noqa: E402
    _kernel_arguments,
    _likelihood_weighted_path,
    prepare_particle_filter_inputs,
)
from src.p3_pfs01_fixed_lag_smoothing import (  # noqa: E402
    DIAGNOSTIC_FIXED_COLUMN_COUNT,
    DIAGNOSTIC_HISTORY_ROWS_REORDERED,
    DIAGNOSTIC_MAX_ACTIVE_ROWS,
    DIAGNOSTIC_RESAMPLE_COUNT,
    particle_filter_fixed_lag_all_seeds_with_diagnostics_numba,
)


EXPERIMENT_ID = "P3_PFS01_fixed_lag_particle_smoothing_v1"
FORMAL_LAG_DISTANCES_FT = (250.0, 500.0, 1000.0)
FORMAL_LIKELIHOOD_SCALE = 8.0
FORMAL_NUMBER_OF_PARTICLES = 500
FORMAL_NUMBER_OF_SEEDS = 128
FORMAL_DEVELOPMENT_WELLS = 657
FORMAL_DEVELOPMENT_ROWS = 3_211_872
FORMAL_SHADOW_WELLS = 116

# 该哈希冻结的是已经通过三井 smoke 的“路径生成合同”。续跑校验代码的收紧
# 不改变路径数值，因此不应迫使 5 分钟的合法缓存重新计算。若以后修改输入准备、
# 内核调用或 scale8 聚合，必须显式更新此值并让旧缓存失效。
FROZEN_GENERATION_RUNNER_SHA256 = (
    "75d94d26cb3e2ca1cd31f524829fe4db"
    "5241a4e55cf260486b424cf9305a8183"
)

# smoke 井是预先按长度和 fold 固定的，不能用注册表 head(3) 替代。
SMOKE_WELLS = (
    ("fba7683c", 1, 407),
    ("cdc31d65", 4, 4840),
    ("ea3a0e38", 0, 10052),
)

# 常量同时作为代码审计和单元测试合同，任何隐藏 TVT 都不在读取列表中。
HORIZONTAL_USECOLS = ("MD", "Z", "GR", "TVT_input")
TYPEWELL_USECOLS = ("TVT", "GR")
CACHE_COLUMNS = (
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    "pfs_lag250_delta",
    "pfs_lag500_delta",
    "pfs_lag1000_delta",
    "_cache_fingerprint",
)

# 这些计数共同证明缓存来自完整的 PFS 内核运行，而不是缺字段的旧或半成品 runtime。
RUNTIME_NONNEGATIVE_INTEGER_FIELDS = (
    "resample_count",
    "history_rows_reordered_total",
    "history_particle_copies",
    "max_active_rows",
    "history_bytes_peak",
    "lag250_smooth_rows",
    "lag250_fallback_rows",
    "lag500_smooth_rows",
    "lag500_fallback_rows",
    "lag1000_smooth_rows",
    "lag1000_fallback_rows",
)


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大型 parquet 一次读进内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    """用稳定 JSON 序列化得到配置指纹。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_cache_fingerprint(
    config: dict[str, Any],
    source_hashes: dict[str, str],
) -> str:
    """生成只代表算法和数据血缘、与线程数和输出目录无关的缓存指纹。"""

    ignored_runtime_keys = {
        "workers",
        "output_dir",
        "artifact_dir",
    }
    algorithm_config = {
        key: value
        for key, value in config.items()
        if key not in ignored_runtime_keys
    }
    return stable_json_hash(
        {
            "algorithm_config": algorithm_config,
            "source_hashes": source_hashes,
        }
    )


def write_json_atomic(payload: Any, path: Path) -> None:
    """在目标目录先写临时文件，再用同盘原子替换正式 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """原子写 parquet，防止中断后留下看似完整的半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, path)


def _resolve_clean_path(value: str | Path) -> Path:
    """把配置中的相对路径统一解释为相对 ``rogii_clean``。"""

    path = Path(value)
    if path.is_absolute():
        return path
    return (CLEAN_ROOT / path).resolve()


def load_development_registry(
    folds_path: Path,
    shadow_path: Path,
    enforce_formal_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取固定 fold，先物理排除 shadow，再按预注册顺序返回固定 smoke 三井。"""

    folds = pd.read_csv(folds_path, dtype={"well_id": str})
    shadow = pd.read_csv(shadow_path, dtype={"well_id": str})
    required_fold_columns = {"well_id", "fold", "hidden_rows"}
    if missing := required_fold_columns.difference(folds.columns):
        raise ValueError(f"fold 注册表缺列：{sorted(missing)}")
    if "well_id" not in shadow.columns:
        raise ValueError("shadow 注册表缺少 well_id")
    if folds["well_id"].duplicated().any() or shadow["well_id"].duplicated().any():
        raise ValueError("fold 或 shadow 注册表含重复井")

    folds = folds[["well_id", "fold", "hidden_rows"]].copy()
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(np.int64)
    folds["hidden_rows"] = pd.to_numeric(
        folds["hidden_rows"], errors="raise"
    ).astype(np.int64)
    shadow_wells = set(shadow["well_id"].astype(str))
    development = folds.loc[~folds["well_id"].isin(shadow_wells)].copy()

    if set(development["well_id"]).intersection(shadow_wells):
        raise AssertionError("开发集仍与 shadow 重叠")
    if enforce_formal_counts:
        if len(shadow_wells) != FORMAL_SHADOW_WELLS:
            raise ValueError("shadow 井数不是冻结的 116")
        if len(development) != FORMAL_DEVELOPMENT_WELLS:
            raise ValueError("开发井数不是冻结的 657")
        if int(development["hidden_rows"].sum()) != FORMAL_DEVELOPMENT_ROWS:
            raise ValueError("开发集隐藏行数不是冻结的 3,211,872")

    smoke_rows: list[dict[str, Any]] = []
    indexed_development = development.set_index("well_id", drop=False)
    for well_id, expected_fold, expected_rows in SMOKE_WELLS:
        if well_id not in indexed_development.index:
            raise ValueError(f"固定 smoke 井不在开发集：{well_id}")
        selected = indexed_development.loc[well_id]
        if int(selected["fold"]) != expected_fold:
            raise ValueError(f"固定 smoke 井 fold 不匹配：{well_id}")
        if int(selected["hidden_rows"]) != expected_rows:
            raise ValueError(f"固定 smoke 井隐藏行数不匹配：{well_id}")
        smoke_rows.append(
            {
                "well_id": well_id,
                "fold": expected_fold,
                "hidden_rows": expected_rows,
            }
        )
    smoke = pd.DataFrame(smoke_rows, columns=["well_id", "fold", "hidden_rows"])
    return development.reset_index(drop=True), smoke


def read_legal_inputs(
    raw_train_dir: Path,
    well_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用 ``usecols`` 物理限制 CSV 读取列，保证看不到水平井隐藏 TVT。"""

    horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        raise FileNotFoundError(f"找不到井 {well_id} 的水平井或 Typewell CSV")
    horizontal = pd.read_csv(horizontal_path, usecols=list(HORIZONTAL_USECOLS))
    typewell = pd.read_csv(typewell_path, usecols=list(TYPEWELL_USECOLS))
    if tuple(horizontal.columns) != HORIZONTAL_USECOLS:
        horizontal = horizontal.loc[:, list(HORIZONTAL_USECOLS)]
    if tuple(typewell.columns) != TYPEWELL_USECOLS:
        typewell = typewell.loc[:, list(TYPEWELL_USECOLS)]
    return horizontal, typewell


def build_legal_cache_frame(
    well_id: str,
    fold: int,
    row_index: np.ndarray,
    last_visible_tvt: float,
    smoothed_tvt_paths: np.ndarray,
    fingerprint: str,
) -> pd.DataFrame:
    """把三条绝对 TVT 路径转成冻结的 float32 delta 合法表。"""

    row_index = np.asarray(row_index, dtype=np.int64)
    paths = np.asarray(smoothed_tvt_paths, dtype=np.float64)
    if paths.shape != (3, len(row_index)):
        raise ValueError("三条平滑路径形状必须为 [3, hidden_rows]")
    if not np.isfinite(paths).all() or not np.isfinite(last_visible_tvt):
        raise ValueError("平滑路径或 last_visible_tvt 含 NaN/Inf")
    if len(np.unique(row_index)) != len(row_index):
        raise ValueError("row_index 含重复值")

    last_visible_float = np.float32(last_visible_tvt)
    frame = pd.DataFrame(
        {
            "well_id": np.repeat(str(well_id), len(row_index)),
            "fold": np.full(len(row_index), int(fold), dtype=np.int64),
            "row_index": row_index,
            "last_visible_tvt": np.full(
                len(row_index), last_visible_float, dtype=np.float32
            ),
            "pfs_lag250_delta": paths[0].astype(np.float32) - last_visible_float,
            "pfs_lag500_delta": paths[1].astype(np.float32) - last_visible_float,
            "pfs_lag1000_delta": paths[2].astype(np.float32) - last_visible_float,
            "_cache_fingerprint": np.repeat(fingerprint, len(row_index)),
        }
    )
    return frame.loc[:, list(CACHE_COLUMNS)]


def _cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    """返回当前井合法 parquet 和逐井 runtime JSON 的固定位置。"""

    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def validate_cache_hit(
    cache_path: Path,
    runtime_path: Path,
    well_id: str,
    fold: int,
    expected_row_index: np.ndarray,
    fingerprint: str,
) -> dict[str, Any] | None:
    """只有 schema、行键、指纹、runtime 和 parquet 哈希全匹配才允许续跑命中。"""

    if not cache_path.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if not isinstance(runtime, dict):
            return None
        required_runtime_fields = {
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
            *RUNTIME_NONNEGATIVE_INTEGER_FIELDS,
        }
        if not required_runtime_fields.issubset(runtime):
            return None
        if runtime["experiment_id"] != EXPERIMENT_ID:
            return None
        if type(runtime["fold"]) is not int or runtime["fold"] != int(fold):
            return None
        expected_hidden_rows = len(np.asarray(expected_row_index))
        if (
            type(runtime["hidden_rows"]) is not int
            or runtime["hidden_rows"] != expected_hidden_rows
        ):
            return None
        if runtime["legal_horizontal_columns"] != list(HORIZONTAL_USECOLS):
            return None
        if runtime["typewell_columns"] != list(TYPEWELL_USECOLS):
            return None

        # 时间字段允许整数或浮点数，但 bool、NaN、Inf 和负数均非法。
        for time_field in ("generation_seconds", "resume_check_seconds"):
            time_value = runtime[time_field]
            if isinstance(time_value, bool) or not isinstance(time_value, (int, float)):
                return None
            if not np.isfinite(float(time_value)) or float(time_value) < 0.0:
                return None

        # 诊断字段必须是非负 JSON 整数，并满足两个可直接复核的守恒关系。
        for diagnostic_field in RUNTIME_NONNEGATIVE_INTEGER_FIELDS:
            diagnostic_value = runtime[diagnostic_field]
            if type(diagnostic_value) is not int or diagnostic_value < 0:
                return None
        if runtime["max_active_rows"] > expected_hidden_rows:
            return None
        if runtime["history_particle_copies"] != (
            runtime["history_rows_reordered_total"] * FORMAL_NUMBER_OF_PARTICLES
        ):
            return None
        expected_seed_rows = expected_hidden_rows * FORMAL_NUMBER_OF_SEEDS
        for lag_distance in FORMAL_LAG_DISTANCES_FT:
            lag_name = int(lag_distance)
            covered_rows = (
                runtime[f"lag{lag_name}_smooth_rows"]
                + runtime[f"lag{lag_name}_fallback_rows"]
            )
            if covered_rows != expected_seed_rows:
                return None
        if runtime.get("well_id") != str(well_id):
            return None
        if int(runtime.get("fold", -1)) != int(fold):
            return None
        if runtime.get("experiment_fingerprint") != fingerprint:
            return None
        if runtime.get("hidden_tvt_read") is not False:
            return None
        if runtime.get("cache_sha256") != file_sha256(cache_path):
            return None

        cache = pd.read_parquet(cache_path)
        if tuple(cache.columns) != CACHE_COLUMNS:
            return None
        expected_row_index = np.asarray(expected_row_index, dtype=np.int64)
        if len(cache) != len(expected_row_index):
            return None
        if cache["well_id"].astype(str).nunique() != 1:
            return None
        if str(cache["well_id"].iloc[0]) != str(well_id):
            return None
        if cache["fold"].astype(np.int64).nunique() != 1:
            return None
        if int(cache["fold"].iloc[0]) != int(fold):
            return None
        np.testing.assert_array_equal(
            cache["row_index"].to_numpy(dtype=np.int64),
            expected_row_index,
        )
        if cache["_cache_fingerprint"].astype(str).nunique() != 1:
            return None
        if str(cache["_cache_fingerprint"].iloc[0]) != fingerprint:
            return None
        numeric_columns = [
            "last_visible_tvt",
            "pfs_lag250_delta",
            "pfs_lag500_delta",
            "pfs_lag1000_delta",
        ]
        if not np.isfinite(cache[numeric_columns].to_numpy(dtype=np.float64)).all():
            return None
        return runtime
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        AssertionError,
    ):
        return None


def _summarize_diagnostics(
    diagnostics: np.ndarray,
    number_of_particles: int,
    number_of_hidden_rows: int,
) -> dict[str, Any]:
    """把逐 seed 诊断矩阵压成逐井运行审计字段。"""

    number_of_lags = len(FORMAL_LAG_DISTANCES_FT)
    smooth_start = DIAGNOSTIC_FIXED_COLUMN_COUNT
    fallback_start = smooth_start + number_of_lags
    smooth_counts = diagnostics[:, smooth_start:fallback_start]
    fallback_counts = diagnostics[:, fallback_start:fallback_start + number_of_lags]
    if not np.all(smooth_counts + fallback_counts == number_of_hidden_rows):
        raise ValueError("每个 seed 的平滑行与尾段回退行没有覆盖完整隐藏段")

    reordered_rows = int(diagnostics[:, DIAGNOSTIC_HISTORY_ROWS_REORDERED].sum())
    maximum_active_rows = int(diagnostics[:, DIAGNOSTIC_MAX_ACTIVE_ROWS].max())
    summary: dict[str, Any] = {
        "resample_count": int(diagnostics[:, DIAGNOSTIC_RESAMPLE_COUNT].sum()),
        "history_rows_reordered_total": reordered_rows,
        "history_particle_copies": int(reordered_rows * number_of_particles),
        "max_active_rows": maximum_active_rows,
        # 内核同时分配 history_u 和 history_scratch 两个 float64 [R,P] 缓冲区。
        "history_bytes_peak": int(maximum_active_rows * number_of_particles * 8 * 2),
    }
    for lag_index, lag_distance in enumerate(FORMAL_LAG_DISTANCES_FT):
        lag_name = int(lag_distance)
        summary[f"lag{lag_name}_smooth_rows"] = int(smooth_counts[:, lag_index].sum())
        summary[f"lag{lag_name}_fallback_rows"] = int(fallback_counts[:, lag_index].sum())
    return summary


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """生成一口井缓存；cache hit 也先用合法列复核自然隐藏行键。"""

    well_id = str(task["well_id"])
    fold = int(task["fold"])
    expected_rows = int(task["hidden_rows"])
    raw_train_dir = Path(str(task["raw_train_dir"]))
    artifact_dir = Path(str(task["artifact_dir"]))
    particle_filter_parameters = dict(task["particle_filter"])
    fingerprint = str(task["fingerprint"])
    force = bool(task.get("force", False))
    cache_path, runtime_path = _cache_paths(artifact_dir, well_id)

    # 即使续跑也只读合法列，并据此重新得到自然隐藏行键；不会信任旧 runtime 自报行数。
    horizontal, typewell = read_legal_inputs(raw_train_dir, well_id)
    prepared = prepare_particle_filter_inputs(
        horizontal,
        typewell,
        particle_filter_parameters,
    )
    row_index = np.asarray(prepared["row_index"], dtype=np.int64)
    if len(row_index) != expected_rows:
        raise ValueError(f"{well_id} 隐藏行数与 fold 注册表不一致")

    resume_started = time.perf_counter()
    if not force:
        cached_runtime = validate_cache_hit(
            cache_path=cache_path,
            runtime_path=runtime_path,
            well_id=well_id,
            fold=fold,
            expected_row_index=row_index,
            fingerprint=fingerprint,
        )
        if cached_runtime is not None:
            result = dict(cached_runtime)
            result["cache_hit"] = True
            result["resume_check_seconds"] = float(time.perf_counter() - resume_started)
            return result
    resume_check_seconds = float(time.perf_counter() - resume_started)

    generation_started = time.perf_counter()
    filtered_paths, smoothed_seed_paths, final_log_likelihoods, diagnostics = (
        particle_filter_fixed_lag_all_seeds_with_diagnostics_numba(
            **_kernel_arguments(prepared),
            lag_distances_ft=np.asarray(
                FORMAL_LAG_DISTANCES_FT,
                dtype=np.float64,
            ),
        )
    )
    if not (
        np.isfinite(filtered_paths).all()
        and np.isfinite(smoothed_seed_paths).all()
        and np.isfinite(final_log_likelihoods).all()
    ):
        raise ValueError(f"{well_id} PFS 内核输出含 NaN/Inf")

    # 每个 lag 独立用同一组最终 likelihood 做冻结 scale=8 聚合。
    aggregated_paths = np.empty((3, expected_rows), dtype=np.float64)
    for lag_index in range(3):
        aggregated_paths[lag_index] = _likelihood_weighted_path(
            smoothed_seed_paths[lag_index],
            final_log_likelihoods,
            FORMAL_LIKELIHOOD_SCALE,
        )
    legal_cache = build_legal_cache_frame(
        well_id=well_id,
        fold=fold,
        row_index=row_index,
        last_visible_tvt=float(prepared["last_visible_tvt"]),
        smoothed_tvt_paths=aggregated_paths,
        fingerprint=fingerprint,
    )
    write_parquet_atomic(legal_cache, cache_path)

    generation_seconds = float(time.perf_counter() - generation_started)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "well_id": well_id,
        "fold": fold,
        "hidden_rows": expected_rows,
        "legal_horizontal_columns": list(HORIZONTAL_USECOLS),
        "typewell_columns": list(TYPEWELL_USECOLS),
        "hidden_tvt_read": False,
        "cache_hit": False,
        "cache_sha256": file_sha256(cache_path),
        "generation_seconds": generation_seconds,
        "resume_check_seconds": resume_check_seconds,
        **_summarize_diagnostics(
            diagnostics,
            number_of_particles=int(particle_filter_parameters["number_of_particles"]),
            number_of_hidden_rows=expected_rows,
        ),
    }
    write_json_atomic(runtime, runtime_path)
    if validate_cache_hit(
        cache_path=cache_path,
        runtime_path=runtime_path,
        well_id=well_id,
        fold=fold,
        expected_row_index=row_index,
        fingerprint=fingerprint,
    ) is None:
        raise RuntimeError(f"{well_id} 刚写入的合法缓存未通过严格复核")
    return runtime


def _load_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取 PFS 配置和冻结 P2-P01 PF 参数，并验证关键数值没有漂移。"""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_config_path = _resolve_clean_path(config["source_pf_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    particle_filter = dict(source_config["particle_filter"])
    if int(particle_filter["number_of_particles"]) != FORMAL_NUMBER_OF_PARTICLES:
        raise ValueError("粒子数不是冻结的 500")
    if int(particle_filter["number_of_seeds"]) != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("seed 数不是冻结的 128")
    if int(particle_filter["seed_base"]) != 0:
        raise ValueError("seed_base 不是冻结的 0")
    if tuple(float(value) for value in config["lag_distances_ft"]) != FORMAL_LAG_DISTANCES_FT:
        raise ValueError("正式 lag 必须严格为 250/500/1000 ft")
    if float(config["likelihood_scale"]) != FORMAL_LIKELIHOOD_SCALE:
        raise ValueError("正式 seed 聚合 scale 必须为 8")
    return config, source_config


def run_generation(
    config_path: Path,
    mode: str,
    force: bool,
    workers_override: int | None = None,
) -> dict[str, Any]:
    """执行固定三井 smoke 或 657 开发井生成，并把逐井进度和 ETA 打印出来。"""

    run_started = time.perf_counter()
    config, source_config = _load_config(config_path)
    folds_path = _resolve_clean_path(config["fold_registry"])
    shadow_path = _resolve_clean_path(config["shadow_registry"])
    raw_train_dir = _resolve_clean_path(config["raw_train_dir"])
    artifact_dir = _resolve_clean_path(config["artifact_dir"])
    source_config_path = _resolve_clean_path(config["source_pf_config"])
    development, smoke = load_development_registry(folds_path, shadow_path)
    selected = smoke if mode == "smoke" else development
    workers = int(workers_override or config.get("workers", 1))
    if workers <= 0:
        raise ValueError("workers 必须为正整数")

    source_hashes = {
        "runner": FROZEN_GENERATION_RUNNER_SHA256,
        "pfs_core": file_sha256(CLEAN_ROOT / "src" / "p3_pfs01_fixed_lag_smoothing.py"),
        "p2_pf_core": file_sha256(CLEAN_ROOT / "src" / "p2_p01_multiseed_pf.py"),
        "source_pf_config": file_sha256(source_config_path),
        "fold_registry": file_sha256(folds_path),
        "shadow_registry": file_sha256(shadow_path),
    }
    fingerprint = build_cache_fingerprint(config, source_hashes)
    write_json_atomic(
        {
            **config,
            "experiment_fingerprint": fingerprint,
            "source_hashes": source_hashes,
            "hidden_tvt_read": False,
        },
        artifact_dir / "config.json",
    )
    write_json_atomic(
        ["pfs_lag250_delta", "pfs_lag500_delta", "pfs_lag1000_delta"],
        artifact_dir / "feature_list.json",
    )
    write_json_atomic(
        {
            "lag_distances_ft": list(FORMAL_LAG_DISTANCES_FT),
            "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
            "particle_filter": source_config["particle_filter"],
        },
        artifact_dir / "parameter_list.json",
    )

    tasks = [
        {
            "well_id": str(row.well_id),
            "fold": int(row.fold),
            "hidden_rows": int(row.hidden_rows),
            "raw_train_dir": str(raw_train_dir),
            "artifact_dir": str(artifact_dir),
            "particle_filter": source_config["particle_filter"],
            "fingerprint": fingerprint,
            "force": force,
        }
        for row in selected.itertuples(index=False)
    ]
    print(
        f"PFS01 {mode}: {len(tasks)}口井，{int(selected['hidden_rows'].sum()):,}隐藏行，"
        f"workers={workers}，hidden_tvt_read=false",
        flush=True,
    )

    runtimes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(generate_one_well, task): task
            for task in tasks
        }
        for completed_count, future in enumerate(as_completed(future_to_task), start=1):
            runtime = future.result()
            runtimes.append(runtime)
            elapsed = time.perf_counter() - run_started
            average_seconds = elapsed / completed_count
            remaining_seconds = average_seconds * (len(tasks) - completed_count)
            print(
                f"[{completed_count}/{len(tasks)}] {runtime['well_id']} "
                f"rows={runtime['hidden_rows']:,} "
                f"max_active_rows={runtime.get('max_active_rows', 'cache')} "
                f"generation={runtime.get('generation_seconds', 0.0):.1f}s "
                f"cache_hit={runtime.get('cache_hit', False)} "
                f"ETA={remaining_seconds / 60.0:.1f}min",
                flush=True,
            )

    runtimes.sort(key=lambda item: str(item["well_id"]))
    per_well = pd.DataFrame(runtimes)
    per_well_path = artifact_dir / f"per_well_runtime_{mode}.csv"
    per_well_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = per_well_path.with_name(
        f".{per_well_path.name}.{uuid.uuid4().hex}.tmp"
    )
    per_well.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, per_well_path)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "mode": mode,
        "wells": len(runtimes),
        "hidden_rows": int(sum(int(item["hidden_rows"]) for item in runtimes)),
        "cache_hits": int(sum(bool(item.get("cache_hit")) for item in runtimes)),
        "cache_misses": int(sum(not bool(item.get("cache_hit")) for item in runtimes)),
        "hidden_tvt_read": False,
        "generation_seconds": float(
            sum(float(item.get("generation_seconds", 0.0)) for item in runtimes)
        ),
        "resume_check_seconds": float(
            sum(float(item.get("resume_check_seconds", 0.0)) for item in runtimes)
        ),
        "wall_seconds": float(time.perf_counter() - run_started),
        "parquet_sha256": {
            str(item["well_id"]): str(item["cache_sha256"])
            for item in runtimes
        },
    }
    write_json_atomic(summary, artifact_dir / f"runtime_{mode}.json")
    # ``runtime.json`` 总是指向最近一次 scope，方便人工查看；逐 scope 文件仍完整保留。
    write_json_atomic(summary, artifact_dir / "runtime.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke/all、force 和可选线程覆盖，不提供任何模型或评分入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "all"), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "p3_pfs01_fixed_lag_particle_smoothing_v1.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""

    args = parse_args(argv)
    run_generation(
        config_path=args.config.resolve(),
        mode=str(args.mode),
        force=bool(args.force),
        workers_override=args.workers,
    )


if __name__ == "__main__":
    main()
