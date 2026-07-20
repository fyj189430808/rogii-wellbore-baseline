"""生成 P3-MDP01 的无标签动态路径、两条安全路径和两组负对照缓存。"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_mdp01_dynamic_mode_path import (  # noqa: E402
    CANDIDATE_NAMES,
    DEFAULT_PARAMETERS,
    DynamicModePathResult,
    build_dynamic_mode_paths,
)


EXPERIMENT_ID = "P3_MDP01_dynamic_mode_path_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_mdp01_dynamic_mode_path_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
RUN_MODES = ("smoke3", "folds12", "all")
LEGAL_HORIZONTAL_COLUMNS = ["MD", "Z", "GR", "TVT_input"]
LEGAL_TYPEWELL_COLUMNS = ["TVT", "GR"]
P2_LEGAL_COLUMNS = ["well_id", "fold", "row_index", "pred_tvt"]
MODE_DELTA_COLUMNS = [
    "pf_mode_low_delta",
    "pf_mode_middle_delta",
    "pf_mode_high_delta",
]
LEGAL_CACHE_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "md",
    "last_visible_tvt",
    "mdp_dynamic_delta",
    "mdp_selected_state",
    "mdp_margin",
    "mdp_safe_10_delta",
    "mdp_safe_25_delta",
    "mdp_gr_shift_dynamic_delta",
    "mdp_gr_shift_selected_state",
    "mdp_gr_shift_safe_10_delta",
    "mdp_gr_shift_safe_25_delta",
    "mdp_cost_permutation_dynamic_delta",
    "mdp_cost_permutation_selected_state",
    "mdp_cost_permutation_safe_10_delta",
    "mdp_cost_permutation_safe_25_delta",
    "_cache_fingerprint",
]
CORE_ROW_COLUMNS = [
    "row_index",
    "md",
    "last_visible_tvt",
    "mdp_dynamic_delta",
    "mdp_selected_state",
    "mdp_margin",
    "mdp_safe_10_delta",
    "mdp_safe_25_delta",
    "mdp_gr_shift_dynamic_delta",
    "mdp_gr_shift_selected_state",
    "mdp_gr_shift_safe_10_delta",
    "mdp_gr_shift_safe_25_delta",
    "mdp_cost_permutation_dynamic_delta",
    "mdp_cost_permutation_selected_state",
    "mdp_cost_permutation_safe_10_delta",
    "mdp_cost_permutation_safe_25_delta",
]
FINITE_CACHE_COLUMNS = [
    "md",
    "last_visible_tvt",
    "mdp_dynamic_delta",
    "mdp_safe_10_delta",
    "mdp_safe_25_delta",
    "mdp_gr_shift_dynamic_delta",
    "mdp_gr_shift_safe_10_delta",
    "mdp_gr_shift_safe_25_delta",
    "mdp_cost_permutation_dynamic_delta",
    "mdp_cost_permutation_safe_10_delta",
    "mdp_cost_permutation_safe_25_delta",
]
STATE_COLUMNS = [
    "mdp_selected_state",
    "mdp_gr_shift_selected_state",
    "mdp_cost_permutation_selected_state",
]


def file_sha256(path: Path) -> str:
    """分块计算文件内容指纹。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    """把合同字典稳定序列化后计算指纹。"""

    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """原子保存 JSON，避免中断留下半个文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子保存逐行缓存或逐块诊断。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shadow_mask(frame: pd.DataFrame) -> pd.Series:
    if "is_shadow" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    values = frame["is_shadow"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_development_registry(
    folds_path: Path, shadow_path: Path
) -> tuple[pd.DataFrame, set[str]]:
    """仅用井号、fold 和冻结行数移除影子井。"""

    folds = pd.read_csv(
        folds_path,
        usecols=["well_id", "fold", "hidden_rows"],
        dtype={"well_id": str},
    )
    shadow_header = pd.read_csv(shadow_path, nrows=0).columns.tolist()
    shadow_columns = ["well_id"]
    if "is_shadow" in shadow_header:
        shadow_columns.append("is_shadow")
    shadow = pd.read_csv(
        shadow_path, usecols=shadow_columns, dtype={"well_id": str}
    )
    shadow_ids = set(shadow.loc[_shadow_mask(shadow), "well_id"].astype(str))
    folds["well_id"] = folds["well_id"].astype(str)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(np.int64)
    folds["hidden_rows"] = pd.to_numeric(
        folds["hidden_rows"], errors="raise"
    ).astype(np.int64)
    if folds["well_id"].isna().any() or folds["well_id"].duplicated().any():
        raise ValueError("fold registry contains invalid well_id")
    if not set(folds["fold"]).issubset({0, 1, 2, 3, 4}):
        raise ValueError("fold must be in 0..4")
    if (folds["hidden_rows"] <= 0).any():
        raise ValueError("hidden_rows must be positive")
    if not shadow_ids.issubset(set(folds["well_id"])):
        raise ValueError("shadow registry contains unknown wells")
    development = folds.loc[~folds["well_id"].isin(shadow_ids)].copy()
    development = development.sort_values("well_id", kind="stable").reset_index(
        drop=True
    )
    if set(development["well_id"]).intersection(shadow_ids):
        raise RuntimeError("development and shadow wells overlap")
    return development, shadow_ids


def select_registry(development: pd.DataFrame, run_mode: str) -> pd.DataFrame:
    """smoke 固定三井；正式首轮固定 folds 1、2；all 使用 657 井。"""

    if run_mode not in RUN_MODES:
        raise ValueError(f"unsupported run mode: {run_mode}")
    ordered = development.sort_values("well_id", kind="stable")
    if run_mode == "smoke3":
        selected = ordered.head(3)
    elif run_mode == "folds12":
        selected = ordered.loc[ordered["fold"].isin([1, 2])]
    else:
        selected = ordered
    if selected.empty:
        raise ValueError("selected registry is empty")
    return selected.reset_index(drop=True)


def resolve_run_dir(artifact_dir: Path, run_mode: str) -> Path:
    """smoke 与正式缓存物理分开；folds12 与 all 可逐井复用。"""

    if run_mode == "smoke3":
        return Path(artifact_dir) / "smoke_3"
    return Path(artifact_dir)


def read_legal_inputs(
    raw_train_dir: Path, well_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """显式投影合法原始列，水平井真值列不会进入内存。"""

    horizontal_path = Path(raw_train_dir) / f"{well_id}__horizontal_well.csv"
    typewell_path = Path(raw_train_dir) / f"{well_id}__typewell.csv"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        raise FileNotFoundError(f"missing raw inputs for well {well_id}")
    horizontal = pd.read_csv(horizontal_path, usecols=LEGAL_HORIZONTAL_COLUMNS)
    typewell = pd.read_csv(typewell_path, usecols=LEGAL_TYPEWELL_COLUMNS)
    return horizontal[LEGAL_HORIZONTAL_COLUMNS], typewell[LEGAL_TYPEWELL_COLUMNS]


def load_p2_legal_rows(
    predictions_path: Path, selected_wells: list[str]
) -> pd.DataFrame:
    """在 Arrow 层先按井过滤，并且只投影预测与自然键。"""

    dataset = ds.dataset(predictions_path, format="parquet")
    table = dataset.to_table(
        columns=P2_LEGAL_COLUMNS,
        filter=ds.field("well_id").isin(list(map(str, selected_wells))),
    )
    frame = table.to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    return frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(
        drop=True
    )


def load_mode_config(mode_config_path: Path) -> tuple[dict[str, Any], str]:
    """读取 PFM02 已完成配置，并返回其冻结路径指纹。"""

    config = json.loads(Path(mode_config_path).read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "experiment_fingerprint",
        "selected_wells",
        "selected_rows",
        "shadow_wells",
        "hidden_tvt_read",
    }
    if missing := required.difference(config):
        raise ValueError(f"mode config missing fields: {sorted(missing)}")
    if config["experiment_id"] != "P3_PFM02_mode_paths_v1":
        raise ValueError("wrong PFM02 mode experiment")
    if config["hidden_tvt_read"] is not False or int(config["shadow_wells"]) != 0:
        raise ValueError("PFM02 mode config violates legal boundary")
    return config, str(config["experiment_fingerprint"])


def load_mode_cache(path: Path) -> pd.DataFrame:
    """读取一井 PFM02 三模式合法缓存。"""

    required = [
        "well_id",
        "fold",
        "row_index",
        "last_visible_tvt",
        *MODE_DELTA_COLUMNS,
        "_cache_fingerprint",
    ]
    return pd.read_parquet(path, columns=required)


def assemble_candidate_tvt(
    p2: pd.DataFrame,
    mode: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    hidden_rows: int,
    expected_mode_fingerprint: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    """按自然键拼出 [P2, low, middle, high] 四条绝对 TVT 路径。"""

    if len(p2) != hidden_rows or len(mode) != hidden_rows:
        raise ValueError("hidden row count mismatch")
    for frame, name in ((p2, "P2"), (mode, "mode")):
        if not frame["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"{name} well_id mismatch")
        if not pd.to_numeric(frame["fold"], errors="raise").eq(fold).all():
            raise ValueError(f"{name} fold mismatch")
    p2_sorted = p2.sort_values("row_index", kind="stable").reset_index(drop=True)
    mode_sorted = mode.sort_values("row_index", kind="stable").reset_index(drop=True)
    p2_keys = pd.to_numeric(p2_sorted["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    mode_keys = pd.to_numeric(mode_sorted["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if len(np.unique(p2_keys)) != hidden_rows or not np.array_equal(p2_keys, mode_keys):
        raise ValueError("P2 and mode row_index do not align")
    if hidden_rows > 1 and np.any(np.diff(p2_keys) <= 0):
        raise ValueError("row_index must be strictly increasing")
    fingerprints = mode_sorted["_cache_fingerprint"].astype(str).unique().tolist()
    if fingerprints != [expected_mode_fingerprint]:
        raise ValueError("mode cache fingerprint mismatch")
    last_values = pd.to_numeric(
        mode_sorted["last_visible_tvt"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(last_values).all() or not np.allclose(
        last_values, last_values[0], rtol=0.0, atol=1e-8
    ):
        raise ValueError("last_visible_tvt must be finite and constant within a well")
    last_visible_tvt = float(last_values[0])
    p2_path = pd.to_numeric(p2_sorted["pred_tvt"], errors="raise").to_numpy(
        dtype=np.float64
    )
    mode_delta = mode_sorted[MODE_DELTA_COLUMNS].to_numpy(dtype=np.float64)
    candidate_tvt = np.column_stack(
        [p2_path, last_visible_tvt + mode_delta]
    ).astype(np.float64, copy=False)
    if candidate_tvt.shape != (hidden_rows, len(CANDIDATE_NAMES)):
        raise ValueError("candidate path shape mismatch")
    if not np.isfinite(candidate_tvt).all():
        raise ValueError("candidate paths contain NaN or Inf")
    return p2_keys, last_visible_tvt, candidate_tvt


def _source_signature(path: Path) -> dict[str, Any]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(Path(path)),
    }


def build_experiment_fingerprint(
    *,
    config: Mapping[str, Any],
    folds_path: Path,
    shadow_path: Path,
    p2_predictions_path: Path,
    mode_config_path: Path,
) -> tuple[str, dict[str, Any]]:
    """把配置、依赖代码和四个冻结来源都绑进缓存指纹。"""

    signatures = {
        "fold_registry": _source_signature(folds_path),
        "shadow_registry": _source_signature(shadow_path),
        "p2_predictions": _source_signature(p2_predictions_path),
        "mode_config": _source_signature(mode_config_path),
        "core": _source_signature(CLEAN_ROOT / "src" / "p3_mdp01_dynamic_mode_path.py"),
        "runner": _source_signature(Path(__file__).resolve()),
    }
    fingerprint = stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": dict(config),
            "parameters": asdict(DEFAULT_PARAMETERS),
            "candidate_names": list(CANDIDATE_NAMES),
            "source_sha256": {
                name: value["sha256"] for name, value in signatures.items()
            },
        }
    )
    return fingerprint, signatures


def _validate_legal_frame(
    frame: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    hidden_rows: int,
    fingerprint: str,
    row_index: np.ndarray,
) -> None:
    if frame.columns.tolist() != LEGAL_CACHE_COLUMNS:
        raise ValueError("legal cache columns do not match the frozen schema")
    if len(frame) != hidden_rows:
        raise ValueError("legal cache row count mismatch")
    if not frame["well_id"].astype(str).eq(well_id).all():
        raise ValueError("legal cache well_id mismatch")
    if not pd.to_numeric(frame["fold"], errors="raise").eq(fold).all():
        raise ValueError("legal cache fold mismatch")
    if not np.array_equal(
        frame["row_index"].to_numpy(dtype=np.int64), np.asarray(row_index, dtype=np.int64)
    ):
        raise ValueError("legal cache row_index mismatch")
    if frame["_cache_fingerprint"].astype(str).unique().tolist() != [fingerprint]:
        raise ValueError("legal cache fingerprint mismatch")
    if not np.isfinite(frame[FINITE_CACHE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("legal cache path values contain NaN or Inf")
    for name in STATE_COLUMNS:
        states = frame[name].to_numpy(dtype=np.int64)
        if not np.isin(states, np.arange(len(CANDIDATE_NAMES))).all():
            raise ValueError(f"invalid state values in {name}")
    margins = frame["mdp_margin"].to_numpy(dtype=np.float64)
    if not (np.isnan(margins) | np.isfinite(margins)).all():
        raise ValueError("invalid margin values")


def validate_cache_hit(
    cache_path: Path,
    runtime_path: Path,
    *,
    well_id: str,
    fold: int,
    hidden_rows: int,
    fingerprint: str,
    row_index: np.ndarray,
    source_signatures: Mapping[str, Any],
) -> dict[str, Any] | None:
    """仅复用指纹、自然键、来源内容和缓存内容全部一致的一井。"""

    if not Path(cache_path).is_file() or not Path(runtime_path).is_file():
        return None
    try:
        runtime = json.loads(Path(runtime_path).read_text(encoding="utf-8"))
        required = {
            "experiment_id",
            "experiment_fingerprint",
            "well_id",
            "fold",
            "hidden_rows",
            "hidden_tvt_read",
            "source_signatures",
            "cache_sha256",
            "block_cache_sha256",
            "block_cache_path",
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
        if runtime["hidden_tvt_read"] is not False:
            return None
        if runtime["source_signatures"] != dict(source_signatures):
            return None
        if runtime["cache_sha256"] != file_sha256(Path(cache_path)):
            return None
        block_path = Path(runtime["block_cache_path"])
        if not block_path.is_file():
            return None
        if runtime["block_cache_sha256"] != file_sha256(block_path):
            return None
        frame = pd.read_parquet(cache_path)
        _validate_legal_frame(
            frame,
            well_id=well_id,
            fold=fold,
            hidden_rows=hidden_rows,
            fingerprint=fingerprint,
            row_index=row_index,
        )
        runtime["cache_hit"] = True
        return runtime
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _build_legal_frame(
    result: DynamicModePathResult,
    *,
    well_id: str,
    fold: int,
    hidden_rows: int,
    fingerprint: str,
    row_index: np.ndarray,
) -> pd.DataFrame:
    missing = set(CORE_ROW_COLUMNS).difference(result.row_output.columns)
    if missing:
        raise ValueError(f"core row output missing fields: {sorted(missing)}")
    frame = result.row_output[CORE_ROW_COLUMNS].copy()
    frame.insert(0, "fold", np.full(hidden_rows, fold, dtype=np.int64))
    frame.insert(0, "well_id", np.repeat(well_id, hidden_rows))
    frame["_cache_fingerprint"] = fingerprint
    frame = frame[LEGAL_CACHE_COLUMNS]
    _validate_legal_frame(
        frame,
        well_id=well_id,
        fold=fold,
        hidden_rows=hidden_rows,
        fingerprint=fingerprint,
        row_index=row_index,
    )
    return frame


def _build_block_frame(
    result: DynamicModePathResult, *, well_id: str, fold: int, fingerprint: str
) -> pd.DataFrame:
    frame = result.block_output.copy()
    frame.insert(0, "fold", np.full(len(frame), fold, dtype=np.int64))
    frame.insert(0, "well_id", np.repeat(well_id, len(frame)))
    frame["_cache_fingerprint"] = fingerprint
    return frame


def generate_legal_paths(
    *,
    folds_path: Path,
    shadow_path: Path,
    raw_train_dir: Path,
    p2_predictions_path: Path,
    mode_cache_dir: Path,
    mode_config_path: Path,
    config: Mapping[str, Any],
    artifact_dir: Path,
    run_mode: str,
    workers: int,
) -> dict[str, Any]:
    """完成所选井全部合法缓存；该函数没有任何隐藏目标入口。"""

    started = time.perf_counter()
    if workers <= 0:
        raise ValueError("workers must be positive")
    development, shadow_ids = load_development_registry(folds_path, shadow_path)
    selected = select_registry(development, run_mode)
    mode_config, mode_fingerprint = load_mode_config(mode_config_path)
    if set(selected["well_id"]).intersection(shadow_ids):
        raise RuntimeError("selected wells overlap shadow holdout")
    fingerprint, global_signatures = build_experiment_fingerprint(
        config=config,
        folds_path=folds_path,
        shadow_path=shadow_path,
        p2_predictions_path=p2_predictions_path,
        mode_config_path=mode_config_path,
    )
    run_dir = resolve_run_dir(artifact_dir, run_mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    p2_all = load_p2_legal_rows(
        p2_predictions_path, selected["well_id"].astype(str).tolist()
    )
    p2_groups = {
        str(well_id): group.copy()
        for well_id, group in p2_all.groupby("well_id", sort=False)
    }
    selected_ids = selected["well_id"].astype(str).tolist()
    if set(p2_groups) != set(selected_ids):
        raise ValueError("P2 legal rows do not cover exactly the selected wells")
    missing_modes = [
        well_id
        for well_id in selected_ids
        if not (Path(mode_cache_dir) / f"{well_id}.parquet").is_file()
    ]
    if missing_modes:
        raise FileNotFoundError(f"missing mode caches: {missing_modes[:5]}")
    write_json_atomic(
        run_dir / "config.json",
        {
            **dict(config),
            "run_mode": run_mode,
            "workers": workers,
            "selected_wells": int(len(selected)),
            "selected_rows": int(selected["hidden_rows"].sum()),
            "shadow_overlap": 0,
            "mode_experiment_fingerprint": mode_fingerprint,
            "experiment_fingerprint": fingerprint,
            "hidden_tvt_read": False,
            "model_training": False,
            "parameters": asdict(DEFAULT_PARAMETERS),
            "candidate_names": list(CANDIDATE_NAMES),
            "global_source_signatures": global_signatures,
            "mode_config_contract": mode_config,
        },
    )

    def build_one(row: Any) -> dict[str, Any]:
        well_started = time.perf_counter()
        well_id = str(row.well_id)
        fold = int(row.fold)
        hidden_rows = int(row.hidden_rows)
        horizontal_path = Path(raw_train_dir) / f"{well_id}__horizontal_well.csv"
        typewell_path = Path(raw_train_dir) / f"{well_id}__typewell.csv"
        mode_path = Path(mode_cache_dir) / f"{well_id}.parquet"
        source_signatures = {
            "horizontal": _source_signature(horizontal_path),
            "typewell": _source_signature(typewell_path),
            "mode_cache": _source_signature(mode_path),
        }
        p2 = p2_groups[well_id]
        mode = load_mode_cache(mode_path)
        row_index, last_visible_tvt, candidate_tvt = assemble_candidate_tvt(
            p2,
            mode,
            well_id=well_id,
            fold=fold,
            hidden_rows=hidden_rows,
            expected_mode_fingerprint=mode_fingerprint,
        )
        cache_path = run_dir / "legal_cache" / f"{well_id}.parquet"
        block_path = run_dir / "legal_block_cache" / f"{well_id}.parquet"
        runtime_path = run_dir / "legal_runtime" / f"{well_id}.json"
        hit = validate_cache_hit(
            cache_path,
            runtime_path,
            well_id=well_id,
            fold=fold,
            hidden_rows=hidden_rows,
            fingerprint=fingerprint,
            row_index=row_index,
            source_signatures=source_signatures,
        )
        if hit is not None:
            hit["cache_hit"] = True
            hit["elapsed_seconds"] = float(time.perf_counter() - well_started)
            write_json_atomic(runtime_path, hit)
            print(f"{well_id}: cache hit", flush=True)
            return hit
        horizontal, typewell = read_legal_inputs(raw_train_dir, well_id)
        hidden_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        if not np.array_equal(hidden_index.astype(np.int64), row_index):
            raise ValueError(f"{well_id}: natural hidden row_index mismatch")
        result = build_dynamic_mode_paths(
            horizontal=horizontal,
            typewell=typewell,
            row_index=row_index,
            candidate_tvt=candidate_tvt,
            last_visible_tvt=last_visible_tvt,
            well_id=well_id,
        )
        if result.audit.get("hidden_target_read") is not False:
            raise ValueError(f"{well_id}: core audit violates target-free contract")
        legal_frame = _build_legal_frame(
            result,
            well_id=well_id,
            fold=fold,
            hidden_rows=hidden_rows,
            fingerprint=fingerprint,
            row_index=row_index,
        )
        block_frame = _build_block_frame(
            result, well_id=well_id, fold=fold, fingerprint=fingerprint
        )
        write_parquet_atomic(cache_path, legal_frame)
        write_parquet_atomic(block_path, block_frame)
        runtime = {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "well_id": well_id,
            "fold": fold,
            "hidden_rows": hidden_rows,
            "blocks": int(len(block_frame)),
            "hidden_tvt_read": False,
            "cache_hit": False,
            "source_signatures": source_signatures,
            "cache_sha256": file_sha256(cache_path),
            "block_cache_path": str(block_path.resolve()),
            "block_cache_sha256": file_sha256(block_path),
            "core_audit": result.audit,
            "elapsed_seconds": float(time.perf_counter() - well_started),
        }
        write_json_atomic(runtime_path, runtime)
        if validate_cache_hit(
            cache_path,
            runtime_path,
            well_id=well_id,
            fold=fold,
            hidden_rows=hidden_rows,
            fingerprint=fingerprint,
            row_index=row_index,
            source_signatures=source_signatures,
        ) is None:
            raise RuntimeError(f"{well_id}: newly written cache failed validation")
        print(f"{well_id}: generated", flush=True)
        return runtime

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(build_one, row): str(row.well_id)
            for row in selected.itertuples(index=False)
        }
        for future in concurrent.futures.as_completed(future_map):
            well_id = future_map[future]
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001
                errors.append(
                    {
                        "well_id": well_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(f"{well_id}: FAILED {type(error).__name__}: {error}", flush=True)
    completed_ids = {str(record["well_id"]) for record in records}
    complete = not errors and completed_ids == set(selected_ids)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "run_mode": run_mode,
        "completed": complete,
        "experiment_fingerprint": fingerprint,
        "selected_wells": int(len(selected)),
        "selected_rows": int(selected["hidden_rows"].sum()),
        "completed_wells": int(len(records)),
        "completed_rows": int(sum(int(record["hidden_rows"]) for record in records)),
        "cache_hits": int(sum(bool(record.get("cache_hit")) for record in records)),
        "shadow_overlap": 0,
        "hidden_tvt_read": False,
        "errors": errors,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(run_dir / f"runtime_{run_mode}.json", summary)
    if not complete:
        raise RuntimeError(f"legal generation failed for {len(errors)} wells")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=RUN_MODES, default="smoke3")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--workers", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_args(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong MDP01 experiment_id")
    workers = int(arguments.workers or config["workers"])
    paths = {
        name: (CLEAN_ROOT / str(config[name])).resolve()
        for name in (
            "fold_registry",
            "shadow_registry",
            "raw_train_dir",
            "p2_predictions",
            "mode_cache_dir",
            "mode_config",
        )
    }
    development, _ = load_development_registry(
        paths["fold_registry"], paths["shadow_registry"]
    )
    selected = select_registry(development, arguments.mode)
    print(
        f"P3-MDP01 {arguments.mode}: {len(selected)} wells, "
        f"{int(selected['hidden_rows'].sum())} hidden rows, {workers} workers",
        flush=True,
    )
    print(f"output: {resolve_run_dir(arguments.artifact_dir, arguments.mode)}", flush=True)
    print("rerun the same command to resume from valid per-well caches", flush=True)
    summary = generate_legal_paths(
        folds_path=paths["fold_registry"],
        shadow_path=paths["shadow_registry"],
        raw_train_dir=paths["raw_train_dir"],
        p2_predictions_path=paths["p2_predictions"],
        mode_cache_dir=paths["mode_cache_dir"],
        mode_config_path=paths["mode_config"],
        config=config,
        artifact_dir=arguments.artifact_dir.resolve(),
        run_mode=arguments.mode,
        workers=workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

