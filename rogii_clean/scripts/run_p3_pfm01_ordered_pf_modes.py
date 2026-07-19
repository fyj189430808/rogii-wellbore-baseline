"""运行 P3-PFM01：只读审计 PF128 的有方向 low/middle/high 三模式。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pf02_target_ess_lgbm_cv import (  # noqa: E402
    load_development_registry,
)
from src.p3_mode01_three_seed_path_modes import read_mode01_shared_cache  # noqa: E402
from src.p3_pfm01_ordered_pf_modes import (  # noqa: E402
    FORMAL_DEGENERATE_SEPARATION_FT,
    FORMAL_LIKELIHOOD_SCALE,
    FORMAL_NUMBER_OF_MODES,
    FORMAL_NUMBER_OF_SEEDS,
    FORMAL_ROTATION,
    MODE_NAMES,
    OLD_TEMPERATURE_NAMES,
    build_direction_quintiles,
    build_ordered_mode_features,
    compute_binary_slice_metrics,
    compute_direction_metrics,
    compute_three_vs_five_oracle,
)


EXPERIMENT_ID = "P3-PFM01_ordered_pf_modes_v1"
ARTIFACT_DIRECTORY_NAME = "P3_PFM01_ordered_pf_modes_v1"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / ARTIFACT_DIRECTORY_NAME
FOLD_REGISTRY = CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv"
SHADOW_REGISTRY = CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv"
SHARED_CACHE_DIR = CLEAN_ROOT / "artifacts/P3_shared_pf_seed_paths_v1"
SHARED_CACHE_FINGERPRINT = (
    "91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0"
)
OLD_PATH_CACHE_DIR = CLEAN_ROOT / "artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache"
OUTER_SOURCE = (
    CLEAN_ROOT
    / "artifacts/P3_R01a_nested_linear_residual_v1/base_models/outer_0/base/"
    "fold_0/predictions.parquet"
)
OUTER_SOURCE_SHA256 = (
    "cb0d1b778a2193507204fd6b104efbdb4fd44f20a4a96d3f9eff139759df9bad"
)
RAW_TRAIN_DIR = PROJECT_ROOT / "input/data/raw/train"
EXPECTED_OUTER_WELLS = 131
EXPECTED_OUTER_ROWS = 651_881
DERANGEMENT_SEED = 20260719
FORBIDDEN_LEGAL_TERMS = (
    "target",
    "true",
    "residual",
    "error",
    "rmse",
    "oracle",
    "best_path",
)
LEGAL_BASE_COLUMNS = ["well_id", "fold", "row_index", "md", "pred_tvt"]
LEGAL_PATH_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "last_visible_tvt",
    "pf_mode_low_delta",
    "pf_mode_middle_delta",
    "pf_mode_high_delta",
    "_cache_fingerprint",
]
OLD_PATH_COLUMNS = {
    "mean": "pf128_mean_delta",
    "scale3": "pf128_scale_3_delta",
    "scale5": "pf128_scale_5_delta",
    "scale8": "pf128_scale_8_delta",
    "scale12": "pf128_scale_12_delta",
}


def file_sha256(path: Path) -> str:
    """分块计算文件 SHA256，避免一次读入大型 parquet。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """把固定配置稳定编码为实验缓存指纹。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    """把 numpy 标量和非有限浮点转成严格 JSON 可以保存的类型。"""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    """先写临时文件再替换，避免中断后留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存 parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def assert_legal_columns_safe(columns: list[str] | pd.Index) -> None:
    """合法区列名若暗示标签或 oracle，立即拒绝落盘。"""

    unsafe: list[str] = []
    for column in [str(value) for value in columns]:
        normalized = column.lower()
        if any(term in normalized for term in FORBIDDEN_LEGAL_TERMS):
            unsafe.append(column)
    if unsafe:
        raise ValueError(f"合法缓存出现禁止字段：{unsafe}")


def validate_row_alignment(
    name: str,
    expected_row_index: np.ndarray,
    observed_row_index: np.ndarray,
) -> None:
    """要求两个来源的 row_index 数量、顺序和数值逐位一致。"""

    expected = np.asarray(expected_row_index, dtype=np.int64)
    observed = np.asarray(observed_row_index, dtype=np.int64)
    if not np.array_equal(expected, observed):
        raise ValueError(f"{name} 的 row_index 与 strict P2-P02 不一致")


def resolve_run_artifact_dir(base_dir: Path, max_wells: int | None) -> Path:
    """smoke 始终进入独立子目录，避免污染 131 口正式缓存。"""

    base = Path(base_dir)
    if max_wells is None:
        return base
    if max_wells < 1 or max_wells > 3:
        raise ValueError("smoke 的 --max-wells 只允许 1～3")
    return base / f"smoke_{max_wells}"


def build_deranged_indices(number_of_wells: int, seed: int) -> np.ndarray:
    """用 Sattolo 洗牌生成一个无固定点的跨井置换。"""

    if number_of_wells < 2:
        raise ValueError("跨井错配至少需要两口井")
    generator = np.random.default_rng(seed)
    indices = np.arange(number_of_wells, dtype=np.int64)
    for current in range(number_of_wells - 1, 0, -1):
        swap_with = int(generator.integers(0, current))
        indices[current], indices[swap_with] = indices[swap_with], indices[current]
    if np.any(indices == np.arange(number_of_wells)):
        raise RuntimeError("Sattolo 洗牌意外产生固定点")
    return indices


def validate_outer_contract(
    legal_outer: pd.DataFrame,
    selected_registry: pd.DataFrame,
    shadow_ids: set[str],
    formal: bool,
) -> None:
    """核对 strict outer0 的井、行、fold、有限值及影子隔离合同。"""

    required = set(LEGAL_BASE_COLUMNS)
    if missing := required.difference(legal_outer.columns):
        raise ValueError(f"strict outer0 缺列：{sorted(missing)}")
    if "target_tvt" in legal_outer.columns:
        raise ValueError("strict outer0 合法读取意外包含 target_tvt")
    frame = legal_outer.copy()
    frame["well_id"] = frame["well_id"].astype(str)
    registry = selected_registry.copy()
    registry["well_id"] = registry["well_id"].astype(str)
    actual_wells = set(frame["well_id"])
    expected_wells = set(registry["well_id"])
    if actual_wells.intersection(shadow_ids) or expected_wells.intersection(shadow_ids):
        raise ValueError("outer0 选择中出现影子井")
    if actual_wells != expected_wells:
        raise ValueError("strict outer0 没有精确覆盖所选井")
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("strict outer0 含重复自然隐藏行键")
    if not frame["fold"].astype(int).eq(0).all():
        raise ValueError("strict outer0 含非 fold0 行")
    if not np.isfinite(frame[["md", "pred_tvt"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("strict outer0 的 md/pred_tvt 含 NaN/Inf")
    expected_counts = registry.set_index("well_id")["hidden_rows"].astype(int).to_dict()
    actual_counts = frame.groupby("well_id").size().astype(int).to_dict()
    if actual_counts != expected_counts:
        raise ValueError("strict outer0 每井行数与 registry 不一致")
    if formal and (
        len(expected_wells) != EXPECTED_OUTER_WELLS
        or len(frame) != EXPECTED_OUTER_ROWS
    ):
        raise ValueError("正式 outer0 必须精确为 131 井和 651881 行")


def select_outer_registry(
    development: pd.DataFrame,
    max_wells: int | None,
) -> pd.DataFrame:
    """正式取完整 fold0；smoke 只取同时具有两类合法路径缓存的前 1～3 井。"""

    outer = development.loc[development["fold"].astype(int).eq(0)].copy()
    outer["well_id"] = outer["well_id"].astype(str)
    outer = outer.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if max_wells is None:
        if len(outer) != EXPECTED_OUTER_WELLS or int(outer["hidden_rows"].sum()) != EXPECTED_OUTER_ROWS:
            raise RuntimeError("开发注册表的 fold0 不是冻结的 131 井/651881 行")
        return outer
    if max_wells < 1 or max_wells > 3:
        raise ValueError("smoke 的 --max-wells 只允许 1～3")
    available = [
        well_id
        for well_id in outer["well_id"].tolist()
        if (SHARED_CACHE_DIR / f"{well_id}.npz").is_file()
        and (OLD_PATH_CACHE_DIR / f"{well_id}.parquet").is_file()
    ]
    if len(available) < max_wells:
        raise FileNotFoundError("可用于 PFM01 smoke 的完整缓存井不足")
    return outer.loc[outer["well_id"].isin(available[:max_wells])].reset_index(drop=True)


def load_strict_outer_legal(selected: pd.DataFrame) -> pd.DataFrame:
    """先验证冻结 SHA，再通过 Arrow 只投影五个无标签列。"""

    observed_hash = file_sha256(OUTER_SOURCE)
    if observed_hash != OUTER_SOURCE_SHA256:
        raise ValueError(
            f"strict P2-P02 SHA 不匹配：expected={OUTER_SOURCE_SHA256} actual={observed_hash}"
        )
    selected_ids = selected["well_id"].astype(str).tolist()
    dataset = arrow_dataset.dataset(str(OUTER_SOURCE), format="parquet")
    table = dataset.to_table(
        columns=LEGAL_BASE_COLUMNS,
        filter=arrow_dataset.field("well_id").isin(selected_ids),
    )
    frame = table.to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.sort_values(["well_id", "row_index"], kind="mergesort").reset_index(drop=True)
    return frame


def _load_hidden_gr_observed_fraction(
    well_id: str,
    expected_row_index: np.ndarray,
) -> float:
    """从原始水平井只读 GR/TVT_input，统计自然隐藏段真实 GR 非缺失比例。"""

    horizontal_path = RAW_TRAIN_DIR / f"{well_id}__horizontal_well.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
    hidden_positions = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    validate_row_alignment(f"井 {well_id} 原始自然隐藏区", expected_row_index, hidden_positions)
    hidden_gr = pd.to_numeric(
        horizontal.iloc[hidden_positions]["GR"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    return float(np.mean(np.isfinite(hidden_gr)))


def _legal_cache_paths(artifact_dir: Path, well_id: str) -> tuple[Path, Path]:
    """返回一口井的合法路径缓存和运行审计文件。"""

    return (
        artifact_dir / "legal_cache" / f"{well_id}.parquet",
        artifact_dir / "legal_runtime" / f"{well_id}.json",
    )


def _load_legal_cache_hit(
    artifact_dir: Path,
    well_id: str,
    hidden_rows: int,
    experiment_fingerprint: str,
    shared_path: Path,
) -> dict[str, Any] | None:
    """只有指纹、源文件哈希、合法文件哈希和行数全匹配时才复用。"""

    legal_path, runtime_path = _legal_cache_paths(artifact_dir, well_id)
    if not legal_path.is_file() or not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime["experiment_fingerprint"] != experiment_fingerprint:
            return None
        if runtime["source_shared_cache_sha256"] != file_sha256(shared_path):
            return None
        if runtime["legal_cache_sha256"] != file_sha256(legal_path):
            return None
        cached = pd.read_parquet(
            legal_path,
            columns=["well_id", "row_index", "_cache_fingerprint"],
        )
        if len(cached) != hidden_rows or set(cached["well_id"].astype(str)) != {well_id}:
            return None
        if cached["_cache_fingerprint"].astype(str).unique().tolist() != [experiment_fingerprint]:
            return None
        output = dict(runtime)
        output["cache_hit"] = True
        return output
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_legal_well(
    well_id: str,
    fold: int,
    base_well: pd.DataFrame,
    artifact_dir: Path,
    experiment_fingerprint: str,
) -> dict[str, Any]:
    """只用 PF128、strict P2 和原始 GR 缺失状态生成一口井的合法模式。"""

    started = time.perf_counter()
    base = base_well.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    expected_row_index = base["row_index"].to_numpy(dtype=np.int64)
    shared_path = SHARED_CACHE_DIR / f"{well_id}.npz"
    if not shared_path.is_file():
        raise FileNotFoundError(f"井 {well_id} 缺少 PF128 共享缓存")
    cached = _load_legal_cache_hit(
        artifact_dir,
        well_id,
        len(base),
        experiment_fingerprint,
        shared_path,
    )
    if cached is not None:
        return cached

    arrays = read_mode01_shared_cache(
        str(shared_path),
        expected_fingerprint=SHARED_CACHE_FINGERPRINT,
    )
    validate_row_alignment(
        f"井 {well_id} PF128",
        expected_row_index,
        arrays["row_index"],
    )
    last_visible_tvt = float(arrays["last_tvt"][0])
    row_features, legal_summary, diagnostics = build_ordered_mode_features(
        seed_delta=arrays["seed_delta"],
        final_ll=arrays["final_ll"],
        seed_ids=arrays["seed_ids"],
        p2_pred_tvt=base["pred_tvt"].to_numpy(dtype=np.float64),
        last_visible_tvt=last_visible_tvt,
    )
    legal_summary.update(
        {
            "well_id": well_id,
            "fold": int(fold),
            "hidden_rows": int(len(base)),
            "gr_observed_fraction": _load_hidden_gr_observed_fraction(
                well_id,
                expected_row_index,
            ),
        }
    )
    legal_path_frame = pd.DataFrame(
        {
            "well_id": np.full(len(base), well_id, dtype=object),
            "fold": np.full(len(base), int(fold), dtype=np.int8),
            "row_index": expected_row_index.astype(np.int32, copy=False),
            "last_visible_tvt": np.full(len(base), last_visible_tvt, dtype=np.float64),
            "pf_mode_low_delta": row_features["pf_mode_low_delta"].to_numpy(),
            "pf_mode_middle_delta": row_features["pf_mode_middle_delta"].to_numpy(),
            "pf_mode_high_delta": row_features["pf_mode_high_delta"].to_numpy(),
            "_cache_fingerprint": np.full(len(base), experiment_fingerprint, dtype=object),
        },
        columns=LEGAL_PATH_COLUMNS,
    )
    assert_legal_columns_safe(legal_path_frame.columns)
    assert_legal_columns_safe(list(legal_summary))
    legal_path, runtime_path = _legal_cache_paths(artifact_dir, well_id)
    write_parquet_atomic(legal_path, legal_path_frame)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "well_id": well_id,
        "fold": int(fold),
        "hidden_rows": int(len(base)),
        "experiment_fingerprint": experiment_fingerprint,
        "source_shared_cache_fingerprint": SHARED_CACHE_FINGERPRINT,
        "source_shared_cache_sha256": file_sha256(shared_path),
        "legal_cache_sha256": file_sha256(legal_path),
        "hidden_tvt_read": False,
        "legal_summary": legal_summary,
        "diagnostics": diagnostics,
        "elapsed_seconds": float(time.perf_counter() - started),
        "cache_hit": False,
    }
    write_json_atomic(runtime_path, runtime)
    return runtime


def assert_legal_complete(
    selected: pd.DataFrame,
    artifact_dir: Path,
) -> None:
    """目标读取前确认逐井缓存、总路径、井表和配置均已完整落盘。"""

    errors: list[str] = []
    for row in selected.itertuples(index=False):
        well_id = str(row.well_id)
        legal_path, runtime_path = _legal_cache_paths(artifact_dir, well_id)
        if not legal_path.is_file() or not runtime_path.is_file():
            errors.append(f"{well_id}: 文件缺失")
            continue
        try:
            keys = pd.read_parquet(legal_path, columns=["well_id", "row_index"])
            if len(keys) != int(row.hidden_rows):
                errors.append(f"{well_id}: 行数不符")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{well_id}: {error}")
    for required_path in (
        artifact_dir / "legal/per_well.csv",
        artifact_dir / "predictions.parquet",
        artifact_dir / "config.json",
        artifact_dir / "feature_list.json",
        artifact_dir / "parameter_list.json",
    ):
        if not required_path.is_file():
            errors.append(f"缺少 {required_path.name}")
    if errors:
        raise RuntimeError(f"合法缓存尚未完整，禁止读取目标：{errors[:5]}")


def guarded_target_read(
    selected: pd.DataFrame,
    artifact_dir: Path,
    reader: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """物理门闩：先验证合法产物完整，之后才执行可能读取 target 的回调。"""

    assert_legal_complete(selected, artifact_dir)
    return reader()


def read_targets_after_legal(selected: pd.DataFrame) -> pd.DataFrame:
    """合法区完整后，第二次投影 strict 来源中的评分列。"""

    selected_ids = selected["well_id"].astype(str).tolist()
    dataset = arrow_dataset.dataset(str(OUTER_SOURCE), format="parquet")
    columns = [*LEGAL_BASE_COLUMNS, "target_tvt"]
    table = dataset.to_table(
        columns=columns,
        filter=arrow_dataset.field("well_id").isin(selected_ids),
    )
    scoring = table.to_pandas()
    scoring["well_id"] = scoring["well_id"].astype(str)
    return scoring.sort_values(["well_id", "row_index"], kind="mergesort").reset_index(drop=True)


def _validate_second_read(base_legal: pd.DataFrame, scoring: pd.DataFrame) -> None:
    """核对目标读取没有改变井、行键、fold、MD 或 P2 路径。"""

    required = {*LEGAL_BASE_COLUMNS, "target_tvt"}
    if missing := required.difference(scoring.columns):
        raise RuntimeError(f"评分读取缺列：{sorted(missing)}")
    keys = ["well_id", "row_index"]
    if scoring.duplicated(keys).any() or base_legal.duplicated(keys).any():
        raise RuntimeError("评分读取或合法读取含重复行键")
    audit = base_legal.merge(
        scoring,
        on=keys,
        how="outer",
        suffixes=("_legal", "_score"),
        indicator=True,
        validate="one_to_one",
    )
    if len(audit) != len(base_legal) or not audit["_merge"].eq("both").all():
        raise RuntimeError("评分读取行键与合法阶段不一致")
    for column in ("fold", "md", "pred_tvt"):
        left = audit[f"{column}_legal"].to_numpy(dtype=np.float64)
        right = audit[f"{column}_score"].to_numpy(dtype=np.float64)
        if not np.array_equal(left, right):
            raise RuntimeError(f"评分读取的 {column} 与合法阶段不一致")
    if not np.isfinite(scoring["target_tvt"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("评分目标含 NaN/Inf")


def _read_old_paths(well_id: str, expected_row_index: np.ndarray) -> pd.DataFrame:
    """读取旧五温度路径，并与 strict P2 的自然隐藏行逐位对齐。"""

    path = OLD_PATH_CACHE_DIR / f"{well_id}.parquet"
    columns = ["row_index", "last_visible_tvt", *OLD_PATH_COLUMNS.values()]
    old = pd.read_parquet(path, columns=columns).sort_values(
        "row_index",
        kind="mergesort",
    )
    validate_row_alignment(
        f"井 {well_id} 旧五温度路径",
        expected_row_index,
        old["row_index"].to_numpy(dtype=np.int64),
    )
    return old.reset_index(drop=True)


def _rmse_from_sse(sse: float, rows: int) -> float:
    """从误差平方和和行数计算 RMSE。"""

    return float(math.sqrt(float(sse) / int(rows)))


def score_after_legal(
    selected: pd.DataFrame,
    base_legal: pd.DataFrame,
    legal_per_well: pd.DataFrame,
    artifact_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """合法区完整后读取目标，评价方向信号和三模式/五温度候选覆盖。"""

    scoring = guarded_target_read(
        selected,
        artifact_dir,
        lambda: read_targets_after_legal(selected),
    )
    _validate_second_read(base_legal, scoring)
    legal_by_well = legal_per_well.set_index("well_id")
    oracle_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    for selected_row in selected.itertuples(index=False):
        well_id = str(selected_row.well_id)
        base = base_legal.loc[base_legal["well_id"].eq(well_id)].sort_values("row_index")
        target = scoring.loc[scoring["well_id"].eq(well_id)].sort_values("row_index")
        row_index = base["row_index"].to_numpy(dtype=np.int64)
        validate_row_alignment(
            f"井 {well_id} 目标",
            row_index,
            target["row_index"].to_numpy(dtype=np.int64),
        )
        legal_path, _ = _legal_cache_paths(artifact_dir, well_id)
        mode = pd.read_parquet(legal_path).sort_values("row_index")
        validate_row_alignment(
            f"井 {well_id} 三模式",
            row_index,
            mode["row_index"].to_numpy(dtype=np.int64),
        )
        old = _read_old_paths(well_id, row_index)
        truth = target["target_tvt"].to_numpy(dtype=np.float64)
        p2_prediction = base["pred_tvt"].to_numpy(dtype=np.float64)
        last_tvt = mode["last_visible_tvt"].to_numpy(dtype=np.float64)
        old_last_tvt = old["last_visible_tvt"].to_numpy(dtype=np.float64)
        predictions: dict[str, np.ndarray] = {
            name: last_tvt + mode[f"pf_mode_{name}_delta"].to_numpy(dtype=np.float64)
            for name in MODE_NAMES
        }
        predictions.update(
            {
                name: old_last_tvt + old[column].to_numpy(dtype=np.float64)
                for name, column in OLD_PATH_COLUMNS.items()
            }
        )
        squared_sums = {
            name: float(np.sum(np.square(prediction - truth)))
            for name, prediction in predictions.items()
        }
        hidden_rows = len(truth)
        best_mode_name = min(MODE_NAMES, key=lambda name: squared_sums[name])
        best_old_name = min(OLD_TEMPERATURE_NAMES, key=lambda name: squared_sums[name])
        legal_values = legal_by_well.loc[well_id]
        row_output: dict[str, Any] = {
            "well_id": well_id,
            "fold": int(selected_row.fold),
            "hidden_rows": hidden_rows,
            "mean_residual": float(np.mean(truth - p2_prediction)),
            "base_sse": float(np.sum(np.square(p2_prediction - truth))),
            **{f"{name}_sse": squared_sums[name] for name in (*MODE_NAMES, *OLD_TEMPERATURE_NAMES)},
            "best_mode_name": best_mode_name,
            "best_old_name": best_old_name,
            "best_mode_rmse": _rmse_from_sse(squared_sums[best_mode_name], hidden_rows),
            "best_old_rmse": _rmse_from_sse(squared_sums[best_old_name], hidden_rows),
            "direction_score": float(legal_values["direction_score"]),
            "rotated_direction_score": float(legal_values["rotated_direction_score"]),
            "high_minus_low_separation": float(legal_values["high_minus_low_separation"]),
            "gr_observed_fraction": float(legal_values["gr_observed_fraction"]),
            "minimum_mode_seed_count": int(legal_values["minimum_mode_seed_count"]),
        }
        oracle_rows.append(row_output)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "fold": int(selected_row.fold),
                    "row_index": row_index.astype(np.int32, copy=False),
                    "target_tvt": truth,
                    "p2_pred_tvt": p2_prediction,
                    **{f"{name}_tvt": prediction for name, prediction in predictions.items()},
                }
            )
        )

    oracle_per_well = pd.DataFrame(oracle_rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if len(oracle_per_well) >= 2:
        donor_indices = build_deranged_indices(len(oracle_per_well), DERANGEMENT_SEED)
        cross_well_control_available = True
    else:
        # 单井 smoke 无法构造无固定点跨井置换；只验证其余数据流，正式 131 井不走此分支。
        donor_indices = np.asarray([0], dtype=np.int64)
        cross_well_control_available = False
    oracle_per_well["cross_well_direction_score"] = oracle_per_well[
        "direction_score"
    ].to_numpy(dtype=np.float64)[donor_indices]
    oracle_per_well["cross_well_donor_id"] = oracle_per_well["well_id"].to_numpy()[donor_indices]
    if cross_well_control_available and oracle_per_well["well_id"].eq(
        oracle_per_well["cross_well_donor_id"]
    ).any():
        raise RuntimeError("跨井负对照仍出现自身 donor")

    mean_residual = oracle_per_well["mean_residual"].to_numpy(dtype=np.float64)
    direction_metrics = compute_direction_metrics(
        oracle_per_well["direction_score"].to_numpy(dtype=np.float64),
        mean_residual,
    )
    rotated_metrics = compute_direction_metrics(
        oracle_per_well["rotated_direction_score"].to_numpy(dtype=np.float64),
        mean_residual,
    )
    cross_metrics = compute_direction_metrics(
        oracle_per_well["cross_well_direction_score"].to_numpy(dtype=np.float64),
        mean_residual,
    )
    coverage_metrics = compute_three_vs_five_oracle(oracle_per_well)
    direction_slices = {
        "direction_quintiles": build_direction_quintiles(oracle_per_well),
        "high_low_separation": compute_binary_slice_metrics(
            oracle_per_well,
            "high_minus_low_separation",
        ),
        "gr_observed_fraction": compute_binary_slice_metrics(
            oracle_per_well,
            "gr_observed_fraction",
        ),
        "minimum_mode_seed_count": compute_binary_slice_metrics(
            oracle_per_well,
            "minimum_mode_seed_count",
        ),
    }
    auc = float(direction_metrics["roc_auc"])
    rotated_auc = float(rotated_metrics["roc_auc"])
    cross_auc = float(cross_metrics["roc_auc"])
    direction_gates = {
        "auc_ge_0_60": bool(np.isfinite(auc) and auc >= 0.60),
        "balanced_accuracy_ge_0_60": bool(
            np.isfinite(float(direction_metrics["balanced_accuracy"]))
            and float(direction_metrics["balanced_accuracy"]) >= 0.60
        ),
        "auc_beats_rotated_by_0_10": bool(
            np.isfinite(auc) and np.isfinite(rotated_auc) and auc - rotated_auc >= 0.10
        ),
        "auc_beats_cross_by_0_10": bool(
            np.isfinite(auc) and np.isfinite(cross_auc) and auc - cross_auc >= 0.10
        ),
        "spearman_positive": bool(float(direction_metrics["spearman"]) > 0.0),
    }
    coverage_gates = {
        "oracle_improvement_ge_0_10": bool(
            coverage_metrics["five_temperature_oracle_pooled_rmse"]
            - coverage_metrics["three_mode_oracle_pooled_rmse"]
            >= 0.10
        ),
        "three_better_fraction_ge_0_55": bool(
            coverage_metrics["three_better_well_fraction"] >= 0.55
        ),
        "strongest_5pct_share_le_0_60": bool(
            coverage_metrics["strongest_5pct_positive_gain_share"] <= 0.60
        ),
    }
    oracle_metrics = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(oracle_per_well)),
        "hidden_rows": int(oracle_per_well["hidden_rows"].sum()),
        "direction": direction_metrics,
        "rotated_ll_control": rotated_metrics,
        "cross_well_control": cross_metrics,
        "auc_minus_rotated": auc - rotated_auc,
        "auc_minus_cross_well": auc - cross_auc,
        "direction_slices": direction_slices,
        "candidate_coverage": coverage_metrics,
        "direction_gates": direction_gates,
        "candidate_coverage_gates": coverage_gates,
        "direction_group_passed": bool(all(direction_gates.values())),
        "candidate_coverage_group_passed": bool(all(coverage_gates.values())),
        "fixed_positive_direction": "direction_score>0 means support for higher TVT and m>0",
        "derangement_seed": DERANGEMENT_SEED,
        "cross_well_control_available": cross_well_control_available,
        "shadow_target_access": False,
    }
    oracle_predictions = pd.concat(prediction_parts, ignore_index=True)
    per_fold_rows: list[dict[str, Any]] = []
    for path_name in ("base", *MODE_NAMES, *OLD_TEMPERATURE_NAMES):
        if path_name == "base":
            sse = float(oracle_per_well["base_sse"].sum())
        else:
            sse = float(oracle_per_well[f"{path_name}_sse"].sum())
        per_fold_rows.append(
            {
                "fold": 0,
                "path": path_name,
                "wells": int(len(oracle_per_well)),
                "hidden_rows": int(oracle_per_well["hidden_rows"].sum()),
                "rmse": _rmse_from_sse(sse, int(oracle_per_well["hidden_rows"].sum())),
            }
        )
    return oracle_metrics, oracle_per_well, oracle_predictions, pd.DataFrame(per_fold_rows)


def build_legal_metrics(legal_per_well: pd.DataFrame) -> dict[str, Any]:
    """只用无标签信息评价三个模式是否具备最低可用形态。"""

    each_mode_ge3 = legal_per_well["minimum_mode_seed_count"].ge(3)
    separation_ge1 = legal_per_well["high_minus_low_separation"].ge(1.0)
    gates = {
        "all_seed_caches_complete": bool(len(legal_per_well) == EXPECTED_OUTER_WELLS),
        "each_mode_ge3_fraction_ge_0_80": bool(each_mode_ge3.mean() >= 0.80),
        "separation_ge1_fraction_ge_0_50": bool(separation_ge1.mean() >= 0.50),
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(legal_per_well)),
        "hidden_rows": int(legal_per_well["hidden_rows"].sum()),
        "each_mode_ge3_fraction": float(each_mode_ge3.mean()),
        "separation_ge1_fraction": float(separation_ge1.mean()),
        "mode_usability_gates": gates,
        "mode_usability_group_passed": bool(all(gates.values())),
        "shadow_overlap_wells": 0,
        "hidden_tvt_read_during_legal_stage": False,
    }


def build_config(experiment_fingerprint: str, formal: bool) -> dict[str, Any]:
    """保存本次固定输入、参数、语义和文件血缘。"""

    return {
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": "P3B00_group5_p2p02_v1",
        "outer_fold": 0,
        "formal": bool(formal),
        "source_p2p02": str(OUTER_SOURCE),
        "source_p2p02_sha256": OUTER_SOURCE_SHA256,
        "source_shared_cache_dir": str(SHARED_CACHE_DIR),
        "source_shared_cache_fingerprint": SHARED_CACHE_FINGERPRINT,
        "source_old_five_path_dir": str(OLD_PATH_CACHE_DIR),
        "number_of_seeds": FORMAL_NUMBER_OF_SEEDS,
        "number_of_modes": FORMAL_NUMBER_OF_MODES,
        "clustering_input": ["full_path_mean_delta", "endpoint_delta"],
        "clustering": "ward_k3_no_zscore_optimal_ordering_false",
        "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
        "mode_order": "center_mean_then_endpoint_then_min_seed_ascending",
        "ll_negative_control_rotation": FORMAL_ROTATION,
        "cross_well_derangement_seed": DERANGEMENT_SEED,
        "degenerate_separation_ft": FORMAL_DEGENERATE_SEPARATION_FT,
        "experiment_fingerprint": experiment_fingerprint,
        "model_training": False,
        "shadow_target_access": False,
    }


def _write_conclusion(
    path: Path,
    legal_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
) -> None:
    """用固定五段格式写当前实现的证据边界。"""

    mode_passed = bool(legal_metrics["mode_usability_group_passed"])
    direction_passed = bool(oracle_metrics["direction_group_passed"])
    coverage_passed = bool(oracle_metrics["candidate_coverage_group_passed"])
    overall = mode_passed and direction_passed and coverage_passed
    coverage = oracle_metrics["candidate_coverage"]
    direction = oracle_metrics["direction"]
    text = f"""# {EXPERIMENT_ID} 结论

## 数据直接证明的事实

- 三模式最低可用性门槛：{'通过' if mode_passed else '未通过'}。
- 预注册正向 ROC-AUC：{direction['roc_auc']}；balanced accuracy：{direction['balanced_accuracy']}。
- 三模式 oracle pooled RMSE：{coverage['three_mode_oracle_pooled_rmse']:.6f} ft。
- 五温度 oracle pooled RMSE：{coverage['five_temperature_oracle_pooled_rmse']:.6f} ft。
- 全部门槛：{'通过' if overall else '未通过'}。

## 基于事实的合理推断

当前固定的整井均值加末端位置 Ward 三模式，{'具备' if overall else '尚不具备'}进入三分类修正的完整证据。

## 仍然没有验证的猜测

其他聚类表示、其他模式数量或沿井深变化的模式是否更可辨识，本实验没有回答。

## 当前实验只能否定的具体实现

若未通过，只能否定“整井均值 + 末端位置、Ward K=3、scale8 mass”这一固定实现，不能否定全部 PF 多分支信息。

## 下一步最便宜的验证

{'按路线进入偏低/中性/偏高三分类。' if overall else '停止当前实现，保留合法模式与 oracle 诊断，不在 outer0 上改参数。'}
"""
    path.write_text(text, encoding="utf-8")


def run_experiment(max_wells: int | None, output_dir: Path) -> dict[str, Any]:
    """执行合法生成、物理门闩和事后诊断；smoke 与正式目录完全分离。"""

    started = time.perf_counter()
    development, shadow_ids = load_development_registry(FOLD_REGISTRY, SHADOW_REGISTRY)
    selected = select_outer_registry(development, max_wells)
    formal = max_wells is None
    artifact_dir = resolve_run_artifact_dir(output_dir.resolve(), max_wells)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if set(selected["well_id"].astype(str)).intersection(shadow_ids):
        raise RuntimeError("PFM01 选择阶段出现影子井")

    base_legal = load_strict_outer_legal(selected)
    validate_outer_contract(base_legal, selected, shadow_ids, formal=formal)
    core_hash = file_sha256(CLEAN_ROOT / "src/p3_pfm01_ordered_pf_modes.py")
    experiment_fingerprint = stable_json_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "strict_source_sha256": OUTER_SOURCE_SHA256,
            "shared_cache_fingerprint": SHARED_CACHE_FINGERPRINT,
            "core_sha256": core_hash,
            "selected_wells": selected["well_id"].astype(str).tolist(),
            "contract": "ordered_mean_endpoint_ward3_scale8_rotate64_v1",
        }
    )
    config = build_config(experiment_fingerprint, formal=formal)
    feature_list = {
        "row_level": [
            "pf_mode_low_delta",
            "pf_mode_middle_delta",
            "pf_mode_high_delta",
        ],
        "well_level": [
            "low_mode_mass",
            "middle_mode_mass",
            "high_mode_mass",
            "direction_score",
            "high_minus_low_separation",
            "high_minus_low_endpoint_separation",
            "p2_position_raw",
            "p2_position_within_mode_envelope",
            "low_seed_count",
            "middle_seed_count",
            "high_seed_count",
            "gr_observed_fraction",
        ],
        "model_feature_use": False,
    }
    parameter_list = {
        "number_of_seeds": FORMAL_NUMBER_OF_SEEDS,
        "ward_clusters": FORMAL_NUMBER_OF_MODES,
        "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
        "ll_rotation": FORMAL_ROTATION,
        "derangement_seed": DERANGEMENT_SEED,
        "degenerate_separation_ft": FORMAL_DEGENERATE_SEPARATION_FT,
    }
    write_json_atomic(artifact_dir / "config.json", config)
    write_json_atomic(artifact_dir / "feature_list.json", feature_list)
    write_json_atomic(artifact_dir / "parameter_list.json", parameter_list)

    print(
        f"{EXPERIMENT_ID}：{len(selected)} 口开发井，"
        f"{int(selected['hidden_rows'].sum()):,} 行；输出 {artifact_dir}",
        flush=True,
    )
    print("主要耗时是逐井读取 128 条路径；单井原子缓存，重新运行可续跑。", flush=True)
    runtimes: list[dict[str, Any]] = []
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        well_id = str(row.well_id)
        base_well = base_legal.loc[base_legal["well_id"].eq(well_id)]
        runtime = generate_legal_well(
            well_id=well_id,
            fold=int(row.fold),
            base_well=base_well,
            artifact_dir=artifact_dir,
            experiment_fingerprint=experiment_fingerprint,
        )
        runtimes.append(runtime)
        state = "缓存" if runtime.get("cache_hit") else "新算"
        summary = runtime["legal_summary"]
        print(
            f"PFM01 {number}/{len(selected)}：{well_id}（{state}，"
            f"mass L/M/H={summary['low_mode_mass']:.3f}/"
            f"{summary['middle_mode_mass']:.3f}/{summary['high_mode_mass']:.3f}）",
            flush=True,
        )

    legal_per_well = pd.DataFrame(
        [runtime["legal_summary"] for runtime in runtimes]
    ).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    assert_legal_columns_safe(legal_per_well.columns)
    write_csv_atomic(artifact_dir / "legal/per_well.csv", legal_per_well)
    legal_path_parts = [
        pd.read_parquet(_legal_cache_paths(artifact_dir, str(row.well_id))[0])
        for row in selected.itertuples(index=False)
    ]
    legal_predictions = pd.concat(legal_path_parts, ignore_index=True).sort_values(
        ["well_id", "row_index"],
        kind="mergesort",
    )
    assert_legal_columns_safe(legal_predictions.columns)
    write_parquet_atomic(artifact_dir / "predictions.parquet", legal_predictions)
    write_csv_atomic(
        artifact_dir / "per_fold.csv",
        pd.DataFrame(
            {
                "fold": [0],
                "wells": [len(selected)],
                "hidden_rows": [int(selected["hidden_rows"].sum())],
            }
        ),
    )
    legal_metrics = build_legal_metrics(legal_per_well)
    if not formal:
        # smoke 不应用 131 口门槛；只证明数据流和物理隔离可以执行。
        legal_metrics["mode_usability_gates"]["all_seed_caches_complete"] = True
        legal_metrics["mode_usability_group_passed"] = bool(
            all(legal_metrics["mode_usability_gates"].values())
        )
    write_json_atomic(artifact_dir / "metrics.json", legal_metrics)

    # 到此所有合法文件均已物理落盘；以下代码才允许读取 target_tvt。
    oracle_metrics, oracle_per_well, oracle_predictions, oracle_per_fold = score_after_legal(
        selected=selected,
        base_legal=base_legal,
        legal_per_well=legal_per_well,
        artifact_dir=artifact_dir,
    )
    write_json_atomic(artifact_dir / "oracle/metrics.json", oracle_metrics)
    write_csv_atomic(artifact_dir / "oracle/per_well.csv", oracle_per_well)
    write_parquet_atomic(artifact_dir / "oracle/predictions.parquet", oracle_predictions)
    write_csv_atomic(artifact_dir / "oracle/per_fold.csv", oracle_per_fold)
    _write_conclusion(artifact_dir / "conclusion.md", legal_metrics, oracle_metrics)
    runtime_summary = {
        "experiment_id": EXPERIMENT_ID,
        "formal": formal,
        "wells": int(len(selected)),
        "hidden_rows": int(selected["hidden_rows"].sum()),
        "cache_hits": int(sum(bool(runtime.get("cache_hit")) for runtime in runtimes)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "experiment_fingerprint": experiment_fingerprint,
        "strict_source_sha256": OUTER_SOURCE_SHA256,
        "shared_cache_fingerprint": SHARED_CACHE_FINGERPRINT,
        "legal_completed_before_target_read": True,
        "shadow_overlap_wells": 0,
        "shadow_target_access": False,
    }
    write_json_atomic(artifact_dir / "runtime.json", runtime_summary)
    print(json.dumps(_json_safe(oracle_metrics), ensure_ascii=False, indent=2), flush=True)
    return {
        "legal_metrics": legal_metrics,
        "oracle_metrics": oracle_metrics,
        "runtime": runtime_summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-wells",
        type=int,
        default=None,
        help="只允许 1～3，用于独立 smoke；不传则运行正式 outer0 131 井。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(max_wells=args.max_wells, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
