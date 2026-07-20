"""逐井流式运行 P3-PFM-D02 A–F 只读 representation oracle 阶梯。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_pfmd02_representation_oracle import (  # noqa: E402
    CANDIDATE_NAMES,
    MODE_NAMES,
    build_ordered_representatives,
    compute_segment_sse,
    pointwise_envelope_sse,
    pooled_micro_rmse,
    resolve_run_artifact_dir,
    segment_ids_from_md,
    solve_b25_simplex,
    solve_simplex_least_squares,
    solve_switching_dp,
    validate_development_selection,
    validate_well_alignment,
)


EXPERIMENT_ID = "P3_PFM_D02_representation_oracle_ladder_v1"
FORMAL_WELLS = 657
FORMAL_ROWS = 3_211_872
FORMAL_P2_DEVELOPMENT_RMSE = 10.272146267501086
FORMAL_P2_FULL_OLD_RMSE = 10.30570499
FORMAL_SHARED_FINGERPRINT = "91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0"
FORMAL_SHARED_FORMAT_VERSION = 1
FORMAL_MODE_EXPERIMENT_FINGERPRINT = "0e37e94cfe0e7c7a9d9be5b3d456a6b29e6738a1f7df5c2c8c3a3fdf19e2a8d4"
FORMAL_MODE_CONFIG_SHA256 = "5edd378e8da38a07ecf719750731a8849fad6363633058cdea19c182206213af"
FORMAL_PFM01_CORE_SHA256 = "2794c417dc33c8887135224d119553f168c446230844e1ad8db74831c7e510a0"
FORMAL_REFERENCE_RMSE = {
    "A3": 8.245752,
    "A4": 7.538825,
    "D128": 6.930948,
    "F_mode3": 7.794340,
    "F_seed128": 5.600719,
}
WINDOWS_FT = (250, 500, 1000)
METHODS = (
    "A4",
    "A3",
    "B0",
    "B25",
    "C250_independent",
    "C250_dp",
    "C500_independent",
    "C500_dp",
    "C1000_independent",
    "C1000_dp",
    "D128",
    "D128_plus_P2",
    "E_center3",
    "E_rowmedian3",
    "E_medoid3",
    "F_mode3",
    "F_seed128",
)
DEFAULT_OUTPUT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
DEFAULT_FOLDS_PATH = CLEAN_ROOT / "artifacts" / "folds" / "balanced_well_5fold_v1.csv"
DEFAULT_SHADOW_PATH = CLEAN_ROOT / "artifacts" / "P3_shadow_holdout_v1" / "shadow_holdout.csv"
DEFAULT_P2_PATH = CLEAN_ROOT / "artifacts" / "P2_P02_multiscale_pf_paths_v1" / "predictions.parquet"
DEFAULT_MODE_DIR = CLEAN_ROOT / "artifacts" / "P3_PFM02_mode_paths_v1" / "legal_cache"
DEFAULT_SEED_DIR = CLEAN_ROOT / "artifacts" / "P3_shared_pf_seed_paths_v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(payload: Any, path: Path) -> None:
    """先写同目录临时文件，再用 os.replace 原子替换。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """以原子替换保存 CSV，避免中断留下半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_segment_checkpoint(path: Path) -> pd.DataFrame:
    """读取逐段 checkpoint，并在解析入口保留全数字井号的前导 0。"""

    frame = pd.read_csv(path, dtype={"well_id": str})
    if "well_id" not in frame.columns or frame["well_id"].isna().any():
        raise ValueError("逐段 checkpoint 缺少有效 well_id")
    return frame


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_ready(dict(payload)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _input_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(path),
    }


def build_code_hash_contract() -> dict[str, str]:
    """返回会影响 oracle 数值或输入解释的全部代码内容指纹。"""

    return {
        "runner": _file_sha256(Path(__file__)),
        "oracle_core": _file_sha256(CLEAN_ROOT / "src" / "p3_pfmd02_representation_oracle.py"),
        "pfm01_mode_core": _file_sha256(CLEAN_ROOT / "src" / "p3_pfm01_ordered_pf_modes.py"),
    }


def load_and_validate_mode_config(path: Path) -> dict[str, Any]:
    """把 PFM02 mode config 的内容哈希和冻结来源指纹作为正式合同。"""

    observed_sha256 = _file_sha256(path)
    if observed_sha256 != FORMAL_MODE_CONFIG_SHA256:
        raise ValueError(
            f"PFM02 mode config SHA256 不匹配：{observed_sha256}"
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("PFM02 mode config 无法解析") from exc
    expected = {
        "experiment_id": "P3_PFM02_mode_paths_v1",
        "experiment_fingerprint": FORMAL_MODE_EXPERIMENT_FINGERPRINT,
        "shared_fingerprint": FORMAL_SHARED_FINGERPRINT,
        "pfm01_core_sha256": FORMAL_PFM01_CORE_SHA256,
        "selected_wells": FORMAL_WELLS,
        "selected_rows": FORMAL_ROWS,
        "shadow_wells": 0,
        "hidden_tvt_read": False,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"PFM02 mode config 的 {key} 不满足冻结合同")
    current_pfm01_sha256 = _file_sha256(CLEAN_ROOT / "src" / "p3_pfm01_ordered_pf_modes.py")
    if current_pfm01_sha256 != FORMAL_PFM01_CORE_SHA256:
        raise ValueError("PFM01 mode 依赖核心 SHA256 与冻结 config 不一致")
    return config


def _has_content_hash(signature: Mapping[str, Any]) -> bool:
    value = signature.get("sha256")
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def checkpoint_matches(
    checkpoint: Mapping[str, Any],
    *,
    fingerprint: str,
    well_id: str,
    fold: int,
    rows: int,
    mode_signature: Mapping[str, Any],
    seed_signature: Mapping[str, Any],
) -> bool:
    """只有实验合同、自然键和两份逐井源文件指纹全匹配才复用。"""

    return bool(
        _has_content_hash(mode_signature)
        and _has_content_hash(seed_signature)
        and _has_content_hash(checkpoint.get("mode_source_signature", {}))
        and _has_content_hash(checkpoint.get("seed_source_signature", {}))
        and checkpoint.get("experiment_fingerprint") == fingerprint
        and checkpoint.get("well_id") == well_id
        and int(checkpoint.get("fold", -1)) == fold
        and int(checkpoint.get("hidden_rows", -1)) == rows
        and checkpoint.get("mode_source_signature") == dict(mode_signature)
        and checkpoint.get("seed_source_signature") == dict(seed_signature)
    )


def load_development_registry(
    folds_path: Path,
    shadow_path: Path,
) -> tuple[pd.DataFrame, set[str], dict[str, Any]]:
    """只用井号和冻结 hidden_rows 去掉影子井，不读取影子目标。"""

    folds = pd.read_csv(folds_path)
    shadow = pd.read_csv(shadow_path)
    if "well_id" not in shadow:
        raise ValueError("影子井表缺少 well_id")
    shadow_wells = set(shadow["well_id"].astype(str))
    if not {"well_id", "fold", "hidden_rows"}.issubset(folds.columns):
        raise ValueError("fold 表缺少 well_id/fold/hidden_rows")
    folds = folds[["well_id", "fold", "hidden_rows"]].copy()
    folds["well_id"] = folds["well_id"].astype(str)
    selected = folds.loc[~folds["well_id"].isin(shadow_wells)].copy()
    selected["fold"] = selected["fold"].astype(np.int64)
    selected["hidden_rows"] = selected["hidden_rows"].astype(np.int64)
    selected = selected.sort_values("well_id", kind="stable").reset_index(drop=True)
    validate_development_selection(selected, shadow_wells)
    audit = {
        "fold_registry_wells": int(len(folds)),
        "shadow_registry_wells": int(len(shadow_wells)),
        "development_wells": int(len(selected)),
        "development_rows": int(selected["hidden_rows"].sum()),
        "shadow_intersection": int(len(set(selected["well_id"]).intersection(shadow_wells))),
    }
    return selected, shadow_wells, audit


def load_p2_arrow_filtered(p2_path: Path, selected_wells: Sequence[str]) -> pd.DataFrame:
    """在 Arrow 层按开发井过滤后才转 pandas。"""

    columns = ["well_id", "fold", "row_index", "md", "target_tvt", "pred_tvt"]
    source = ds.dataset(p2_path, format="parquet")
    table = source.to_table(
        columns=columns,
        filter=ds.field("well_id").isin(list(map(str, selected_wells))),
    )
    frame = table.to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    return frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)


def load_seed_npz(path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    required = {
        "seed_delta",
        "final_ll",
        "seed_ids",
        "row_index",
        "hidden_md",
        "last_tvt",
        "_cache_fingerprint",
        "_format_version",
    }
    with np.load(path, allow_pickle=False) as source:
        if missing := required.difference(source.files):
            raise ValueError(f"seed NPZ 缺字段：{sorted(missing)}")
        data = {name: np.asarray(source[name]) for name in required}
    paths = np.asarray(data["seed_delta"], dtype=np.float64)
    likelihoods = np.asarray(data["final_ll"], dtype=np.float64)
    seed_ids = np.asarray(data["seed_ids"], dtype=np.int64)
    if paths.shape != (128, expected_rows):
        raise ValueError("seed_delta 必须为 [128, expected_rows]")
    if likelihoods.shape != (128,) or seed_ids.shape != (128,) or not np.array_equal(seed_ids, np.arange(128)):
        raise ValueError("final_ll/seed_ids 不满足冻结 128 seed 合同")
    if np.asarray(data["last_tvt"]).shape != (1,):
        raise ValueError("last_tvt 必须为 shape [1]")
    cache_fingerprint = np.asarray(data["_cache_fingerprint"])
    if cache_fingerprint.shape != (1,) or str(cache_fingerprint[0]) != FORMAL_SHARED_FINGERPRINT:
        raise ValueError("seed NPZ fingerprint 不满足冻结 shared cache 合同")
    format_version = np.asarray(data["_format_version"])
    if format_version.shape != (1,) or int(format_version[0]) != FORMAL_SHARED_FORMAT_VERSION:
        raise ValueError("seed NPZ format version 不满足冻结合同")
    if not np.isfinite(np.concatenate([paths.ravel(), likelihoods, np.asarray(data["last_tvt"], dtype=np.float64)])).all():
        raise ValueError("seed NPZ 含 NaN/Inf")
    return data


def load_mode_cache(path: Path, *, expected_fingerprint: str) -> pd.DataFrame:
    """读取一井 mode parquet，并绑定到已验证 config 的统一指纹。"""

    frame = pq.read_table(path).to_pandas()
    if "_cache_fingerprint" not in frame.columns:
        raise ValueError("mode cache 缺少 fingerprint")
    fingerprints = frame["_cache_fingerprint"].astype(str).unique().tolist()
    if fingerprints != [expected_fingerprint]:
        raise ValueError(
            f"mode cache fingerprint 与 PFM02 config 不一致：{fingerprints}"
        )
    return frame


def _hard_select(sse: np.ndarray) -> tuple[int, float]:
    values = np.asarray(sse, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("硬选 SSE 非法")
    index = int(np.argmin(values))
    return index, float(values[index])


def _store_method(record: dict[str, Any], method: str, sse: float, rows: int) -> None:
    record[f"{method}_sse"] = float(sse)
    record[f"{method}_rmse"] = float(math.sqrt(float(sse) / rows))


def evaluate_one_well(
    p2: pd.DataFrame,
    mode: pd.DataFrame,
    seed: Mapping[str, np.ndarray],
    *,
    well_id: str,
    fold: int,
    expected_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """在一口井内一次完成 A–F，任何数组都不跨井保留。"""

    validate_well_alignment(p2, mode, seed, well_id=well_id, fold=fold, expected_rows=expected_rows)
    required_p2 = {"target_tvt", "pred_tvt"}
    required_mode = {"last_visible_tvt", "pf_mode_low_delta", "pf_mode_middle_delta", "pf_mode_high_delta"}
    if missing := required_p2.difference(p2.columns):
        raise ValueError(f"P2 oracle 缺列：{sorted(missing)}")
    if missing := required_mode.difference(mode.columns):
        raise ValueError(f"mode cache 缺列：{sorted(missing)}")

    target = p2["target_tvt"].to_numpy(dtype=np.float64)
    p2_path = p2["pred_tvt"].to_numpy(dtype=np.float64)
    md = p2["md"].to_numpy(dtype=np.float64)
    last_visible_values = mode["last_visible_tvt"].to_numpy(dtype=np.float64)
    if not np.allclose(last_visible_values, last_visible_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("mode cache 的 last_visible_tvt 井内不恒定")
    last_visible_tvt = float(last_visible_values[0])
    seed_last_tvt = float(np.asarray(seed["last_tvt"], dtype=np.float64)[0])
    if not math.isclose(last_visible_tvt, seed_last_tvt, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("mode 与 seed 的 last_visible_tvt 不一致")

    mode_delta = mode[[f"pf_mode_{name}_delta" for name in MODE_NAMES]].to_numpy(dtype=np.float64)
    mode_paths = mode_delta + last_visible_tvt
    seed_delta = np.asarray(seed["seed_delta"], dtype=np.float64)
    seed_paths = seed_delta + last_visible_tvt
    representatives = build_ordered_representatives(
        seed_delta,
        np.asarray(seed["final_ll"], dtype=np.float64),
        np.asarray(seed["seed_ids"], dtype=np.int64),
    )
    center_max_abs_error = float(np.max(np.abs(representatives.center_paths.T - mode_delta)))
    if not np.allclose(representatives.center_paths.T, mode_delta, rtol=1e-7, atol=1e-6):
        raise ValueError(f"重算 scale8 模式中心与 legal cache 不一致：{center_max_abs_error}")

    four = np.column_stack([p2_path, mode_paths])
    four_sse = np.sum(np.square(four - target[:, None]), axis=0)
    mode_sse = four_sse[1:]
    record: dict[str, Any] = {
        "well_id": well_id,
        "fold": int(fold),
        "hidden_rows": int(expected_rows),
        "center_max_abs_error": center_max_abs_error,
    }

    a4_index, a4_sse = _hard_select(four_sse)
    a3_index, a3_sse = _hard_select(mode_sse)
    _store_method(record, "A4", a4_sse, expected_rows)
    _store_method(record, "A3", a3_sse, expected_rows)
    record["A4_candidate"] = CANDIDATE_NAMES[a4_index]
    record["A3_candidate"] = MODE_NAMES[a3_index]

    b0 = solve_simplex_least_squares(four, target)
    b25 = solve_b25_simplex(four, target)
    _store_method(record, "B0", b0.sse, expected_rows)
    _store_method(record, "B25", b25.sse, expected_rows)
    for index, name in enumerate(CANDIDATE_NAMES):
        record[f"B0_q_{name}"] = float(b0.weights[index])
        record[f"B25_q_{name}"] = float(b25.weights[index])
    record["B0_active_count"] = int(b0.active_count)
    record["B25_active_count"] = int(b25.active_count)

    positive_steps = np.diff(md)
    positive_steps = positive_steps[positive_steps > 0.0]
    if positive_steps.size == 0:
        raise ValueError("无法从该井得到 median_positive_md_step")
    median_positive_step = float(np.median(positive_steps))
    record["median_positive_md_step"] = median_positive_step
    segment_records: list[dict[str, Any]] = []
    for window_ft in WINDOWS_FT:
        segment_ids = segment_ids_from_md(md, float(window_ft))
        segment_sse = compute_segment_sse(four, target, segment_ids)
        independent_states = np.argmin(segment_sse, axis=1).astype(np.int64)
        independent_sse = float(np.sum(segment_sse[np.arange(len(segment_sse)), independent_states]))
        switch_penalty = float(max(1, int(np.rint(window_ft / median_positive_step))))
        dp = solve_switching_dp(segment_sse, switch_penalty)
        _store_method(record, f"C{window_ft}_independent", independent_sse, expected_rows)
        _store_method(record, f"C{window_ft}_dp", dp.raw_sse, expected_rows)
        record[f"C{window_ft}_dp_penalized_objective"] = float(dp.penalized_objective)
        record[f"C{window_ft}_dp_switch_count"] = int(dp.switch_count)
        record[f"C{window_ft}_switch_penalty"] = switch_penalty
        for segment_id in range(segment_sse.shape[0]):
            mask = segment_ids == segment_id
            ordered_sse = np.sort(segment_sse[segment_id])
            margin = float(ordered_sse[1] - ordered_sse[0])
            row: dict[str, Any] = {
                "well_id": well_id,
                "fold": int(fold),
                "window_ft": int(window_ft),
                "segment_index": int(segment_id),
                "segment_start_md": float(md[0] + segment_id * window_ft),
                "segment_end_md": float(md[0] + (segment_id + 1) * window_ft),
                "observed_first_md": float(md[mask][0]),
                "observed_last_md": float(md[mask][-1]),
                "rows": int(np.sum(mask)),
                "independent_state": CANDIDATE_NAMES[int(independent_states[segment_id])],
                "dp_state": CANDIDATE_NAMES[int(dp.states[segment_id])],
                "margin": margin,
                "switch_penalty": switch_penalty,
            }
            for state_index, name in enumerate(CANDIDATE_NAMES):
                row[f"{name}_sse"] = float(segment_sse[segment_id, state_index])
            segment_records.append(row)

    seed_sse = np.sum(np.square(seed_paths - target[None, :]), axis=1)
    d128_index, d128_sse = _hard_select(seed_sse)
    d128_plus_index, d128_plus_sse = _hard_select(np.concatenate([[four_sse[0]], seed_sse]))
    _store_method(record, "D128", d128_sse, expected_rows)
    _store_method(record, "D128_plus_P2", d128_plus_sse, expected_rows)
    record["D128_seed_id"] = int(np.asarray(seed["seed_ids"], dtype=np.int64)[d128_index])
    record["D128_plus_P2_candidate"] = "P2" if d128_plus_index == 0 else f"seed_{d128_plus_index - 1}"

    representative_paths = {
        "E_center3": representatives.center_paths.T + last_visible_tvt,
        "E_rowmedian3": representatives.rowmedian_paths.T + last_visible_tvt,
        "E_medoid3": representatives.medoid_paths.T + last_visible_tvt,
    }
    for method, paths in representative_paths.items():
        per_candidate_sse = np.sum(np.square(paths - target[:, None]), axis=0)
        index, sse = _hard_select(per_candidate_sse)
        _store_method(record, method, sse, expected_rows)
        record[f"{method}_candidate"] = MODE_NAMES[index]
    if not math.isclose(record["E_center3_sse"], record["A3_sse"], rel_tol=1e-10, abs_tol=1e-6):
        raise RuntimeError("E-center3 必须与 A3 数值一致")
    for index, name in enumerate(MODE_NAMES):
        record[f"{name}_medoid_seed_id"] = int(representatives.medoid_seed_ids[index])

    _store_method(record, "F_mode3", pointwise_envelope_sse(mode_paths, target), expected_rows)
    _store_method(record, "F_seed128", pointwise_envelope_sse(seed_paths.T, target), expected_rows)
    return record, pd.DataFrame(segment_records)


def build_metrics_tables(
    per_well: pd.DataFrame,
    *,
    methods: Iterable[str] = METHODS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """同时生成每法总体指标、固定比较和逐折 pooled micro。"""

    methods = tuple(methods)
    if per_well.empty:
        raise ValueError("per_well 不能为空")
    method_metrics: dict[str, Any] = {}
    fold_records: list[dict[str, Any]] = []
    for method in methods:
        sse_column = f"{method}_sse"
        rmse_column = f"{method}_rmse"
        if sse_column not in per_well or rmse_column not in per_well:
            raise ValueError(f"per_well 缺少 {method} 指标")
        pooled = pooled_micro_rmse(per_well[sse_column], per_well["hidden_rows"])
        rmses = per_well[rmse_column].to_numpy(dtype=np.float64)
        worst_index = int(np.argmax(rmses))
        method_metrics[method] = {
            "pooled_micro_rmse": pooled,
            "macro_well_rmse": float(np.mean(rmses)),
            "median_well_rmse": float(np.median(rmses)),
            "p90_well_rmse": float(np.quantile(rmses, 0.9)),
            "worst_well_id": str(per_well.iloc[worst_index]["well_id"]),
            "worst_well_rmse": float(rmses[worst_index]),
            "total_sse": float(per_well[sse_column].sum()),
            "total_rows": int(per_well["hidden_rows"].sum()),
        }
        for fold, group in per_well.groupby("fold", sort=True):
            fold_records.append(
                {
                    "method": method,
                    "fold": int(fold),
                    "wells": int(len(group)),
                    "rows": int(group["hidden_rows"].sum()),
                    "sse": float(group[sse_column].sum()),
                    "micro_rmse": pooled_micro_rmse(group[sse_column], group["hidden_rows"]),
                }
            )

    comparisons: dict[str, float] = {}
    references = {
        "B0": "A4",
        "B25": "A4",
        "C250_independent": "A4",
        "C250_dp": "A4",
        "C500_independent": "A4",
        "C500_dp": "A4",
        "C1000_independent": "A4",
        "C1000_dp": "A4",
        "D128": "A3",
        "E_rowmedian3": "E_center3",
        "E_medoid3": "E_center3",
    }
    for candidate, reference in references.items():
        if candidate in method_metrics and reference in method_metrics:
            comparisons[f"{candidate}_vs_{reference}_improvement_ft"] = float(
                method_metrics[reference]["pooled_micro_rmse"]
                - method_metrics[candidate]["pooled_micro_rmse"]
            )
    signals = {
        "continuous_mixture": bool(
            comparisons.get("B0_vs_A4_improvement_ft", -math.inf) >= 0.30
            and comparisons.get("B25_vs_A4_improvement_ft", -math.inf) > 0.0
        ),
        "segmented_c250_dp": bool(comparisons.get("C250_dp_vs_A4_improvement_ft", -math.inf) >= 0.30),
        "shape_clustering_d128": bool(comparisons.get("D128_vs_A3_improvement_ft", -math.inf) >= 0.30),
        "medoid_improvement_positive": bool(comparisons.get("E_medoid3_vs_E_center3_improvement_ft", -math.inf) > 0.0),
        "medoid_improvement_at_least_0_10": bool(comparisons.get("E_medoid3_vs_E_center3_improvement_ft", -math.inf) >= 0.10),
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(per_well)),
        "rows": int(per_well["hidden_rows"].sum()),
        "methods": method_metrics,
        "fixed_comparisons": comparisons,
        "decision_signals": signals,
        "pointwise_methods_are_undeployable": ["F_mode3", "F_seed128"],
    }, pd.DataFrame(fold_records)


def _build_conclusion(metrics: Mapping[str, Any]) -> str:
    comparisons = metrics["fixed_comparisons"]
    signals = metrics["decision_signals"]
    return "\n".join(
        [
            "# P3-PFM-D02 结论",
            "",
            "## 数据直接证明的事实",
            "",
            f"- B0 相对 A4 改善 {comparisons.get('B0_vs_A4_improvement_ft', float('nan')):.6f} ft。",
            f"- B25 相对 A4 改善 {comparisons.get('B25_vs_A4_improvement_ft', float('nan')):.6f} ft。",
            f"- C250-DP 相对 A4 改善 {comparisons.get('C250_dp_vs_A4_improvement_ft', float('nan')):.6f} ft。",
            f"- D128 相对 A3 改善 {comparisons.get('D128_vs_A3_improvement_ft', float('nan')):.6f} ft。",
            f"- medoid3 相对 center3 改善 {comparisons.get('E_medoid3_vs_E_center3_improvement_ft', float('nan')):.6f} ft；>0={signals.get('medoid_improvement_positive')}，>=0.10={signals.get('medoid_improvement_at_least_0_10')}。",
            "- F-mode3 与 F-seed128 是不可部署的逐行 coverage，只作上限展示。",
            "",
            "## 基于事实的合理推断",
            "",
            f"- 连续混合信号={signals.get('continuous_mixture')}；分段信号={signals.get('segmented_c250_dp')}；形状聚类信号={signals.get('shape_clustering_d128')}。",
            "",
            "## 仍然没有验证的猜测",
            "",
            "- oracle 增量能否被只用合法可见信息的选择器或动态成本稳定恢复，尚未验证。",
            "",
            "## 当前实验只能否定的具体实现",
            "",
            "- 只能判断冻结候选表示、固定连续混合和固定窗口分段各自的 oracle 上限，不能否定整个 PF 信息源。",
            "",
            "## 下一步最便宜的验证",
            "",
            "- 严格按预注册门槛选择一个最强分支，先做合法信号的小规模 smoke；不得把本目录 oracle 输出接入正式特征。",
            "",
        ]
    )


def _validate_formal_regression(metrics: Mapping[str, Any]) -> dict[str, float]:
    observed: dict[str, float] = {}
    for method, expected in FORMAL_REFERENCE_RMSE.items():
        actual = float(metrics["methods"][method]["pooled_micro_rmse"])
        observed[method] = actual
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-5):
            raise RuntimeError(f"{method} 正式回归值 {actual} 与独立预览 {expected} 不一致")
    return observed


def run_oracle_ladder(
    *,
    folds_path: Path = DEFAULT_FOLDS_PATH,
    shadow_path: Path = DEFAULT_SHADOW_PATH,
    p2_path: Path = DEFAULT_P2_PATH,
    mode_dir: Path = DEFAULT_MODE_DIR,
    seed_dir: Path = DEFAULT_SEED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_wells: int | None = None,
) -> dict[str, Any]:
    """运行 smoke 或正式诊断；正式运行可读取逐井原子 checkpoint 续跑。"""

    started = time.perf_counter()
    development, shadow_wells, selection_audit = load_development_registry(folds_path, shadow_path)
    if len(development) != FORMAL_WELLS or int(development["hidden_rows"].sum()) != FORMAL_ROWS:
        raise ValueError("冻结开发范围必须恰为 657 井、3,211,872 行")
    if max_wells is not None:
        if max_wells not in (1, 2, 3):
            raise ValueError("--max-wells 只允许 1、2、3")
        selected = development.head(max_wells).copy()
    else:
        selected = development
    validate_development_selection(selected, shadow_wells)
    run_dir = resolve_run_artifact_dir(output_dir, max_wells)
    mode_config_path = mode_dir.parent / "config.json"
    mode_config = load_and_validate_mode_config(mode_config_path)

    missing_modes = [well for well in selected["well_id"] if not (mode_dir / f"{well}.parquet").is_file()]
    missing_seeds = [well for well in selected["well_id"] if not (seed_dir / f"{well}.npz").is_file()]
    if missing_modes or missing_seeds:
        raise FileNotFoundError(f"缺少 mode={missing_modes[:3]} seed={missing_seeds[:3]}")

    config_base = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_type": "read_only_oracle",
        "max_wells": max_wells,
        "selected_wells": selected["well_id"].tolist(),
        "selected_rows": int(selected["hidden_rows"].sum()),
        "windows_ft": list(WINDOWS_FT),
        "candidate_order": list(CANDIDATE_NAMES),
        "ward_k": 3,
        "descriptor": ["full_path_mean_delta", "endpoint_delta"],
        "descriptor_standardized": False,
        "likelihood_scale": 8.0,
        "b25_definition": "q=0.75*e_P2+0.25*v, v in simplex4",
        "dp_penalty": "max(1, round(window_ft/median_positive_md_step))",
        "inputs": {
            "folds": _input_signature(folds_path),
            "shadow": _input_signature(shadow_path),
            "p2": _input_signature(p2_path),
            "mode_dir": str(mode_dir.resolve()),
            "seed_dir": str(seed_dir.resolve()),
            "mode_config": _input_signature(mode_config_path),
        },
        "source_fingerprint_contract": {
            "shared_seed_fingerprint": FORMAL_SHARED_FINGERPRINT,
            "shared_seed_format_version": FORMAL_SHARED_FORMAT_VERSION,
            "mode_experiment_fingerprint": mode_config["experiment_fingerprint"],
            "mode_config_sha256": FORMAL_MODE_CONFIG_SHA256,
        },
        "code_sha256": build_code_hash_contract(),
        "hidden_target_usage": "oracle_diagnostic_only",
        "formal_feature_or_cache_output": False,
    }
    fingerprint = _canonical_hash(config_base)
    config = {**config_base, "experiment_fingerprint": fingerprint}
    write_json_atomic(config, run_dir / "config.json")

    p2 = load_p2_arrow_filtered(p2_path, selected["well_id"].tolist())
    if set(p2["well_id"]).intersection(shadow_wells):
        raise RuntimeError("Arrow 过滤后的 P2 含影子井")
    if len(p2) != int(selected["hidden_rows"].sum()):
        raise ValueError("Arrow 过滤后的 P2 行数与冻结注册表不一致")
    p2_rmse = pooled_micro_rmse(
        np.square(p2["pred_tvt"].to_numpy(dtype=np.float64) - p2["target_tvt"].to_numpy(dtype=np.float64)),
        np.ones(len(p2), dtype=np.int64),
    )
    if max_wells is None and not math.isclose(p2_rmse, FORMAL_P2_DEVELOPMENT_RMSE, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(f"开发集 P2 复算分数错误：{p2_rmse}")

    checkpoint_dir = run_dir / "checkpoint" / "per_well"
    segment_checkpoint_dir = run_dir / "checkpoint" / "per_segment"
    per_well_records: list[dict[str, Any]] = []
    per_segment_frames: list[pd.DataFrame] = []
    cache_hits = 0
    maximum_center_error = 0.0
    p2_groups = p2.groupby("well_id", sort=False)
    for ordinal, registry_row in enumerate(selected.itertuples(index=False), start=1):
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        rows = int(registry_row.hidden_rows)
        checkpoint_path = checkpoint_dir / f"{well_id}.json"
        segment_checkpoint_path = segment_checkpoint_dir / f"{well_id}.csv"
        mode_path = mode_dir / f"{well_id}.parquet"
        seed_path = seed_dir / f"{well_id}.npz"
        mode_signature = _input_signature(mode_path)
        seed_signature = _input_signature(seed_path)
        if checkpoint_path.is_file() and segment_checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_matches(
                checkpoint,
                fingerprint=fingerprint,
                well_id=well_id,
                fold=fold,
                rows=rows,
                mode_signature=mode_signature,
                seed_signature=seed_signature,
            ):
                record = dict(checkpoint["result"])
                segments = read_segment_checkpoint(segment_checkpoint_path)
                cache_hits += 1
                print(f"[{ordinal}/{len(selected)}] {well_id}: checkpoint hit", flush=True)
            else:
                raise ValueError(f"{well_id} checkpoint 指纹或自然键不匹配，拒绝复用")
        else:
            well_p2 = p2_groups.get_group(well_id).sort_values("row_index", kind="stable").reset_index(drop=True)
            well_mode = load_mode_cache(
                mode_path,
                expected_fingerprint=str(mode_config["experiment_fingerprint"]),
            )
            well_mode = well_mode.sort_values("row_index", kind="stable").reset_index(drop=True)
            well_seed = load_seed_npz(seed_path, rows)
            record, segments = evaluate_one_well(
                well_p2,
                well_mode,
                well_seed,
                well_id=well_id,
                fold=fold,
                expected_rows=rows,
            )
            write_json_atomic(
                {
                    "experiment_fingerprint": fingerprint,
                    "well_id": well_id,
                    "fold": fold,
                    "hidden_rows": rows,
                    "mode_source_signature": mode_signature,
                    "seed_source_signature": seed_signature,
                    "result": record,
                },
                checkpoint_path,
            )
            write_csv_atomic(segments, segment_checkpoint_path)
            print(f"[{ordinal}/{len(selected)}] {well_id}: completed", flush=True)
        maximum_center_error = max(maximum_center_error, float(record["center_max_abs_error"]))
        per_well_records.append(record)
        per_segment_frames.append(segments)

    per_well = pd.DataFrame(per_well_records).sort_values("well_id", kind="stable").reset_index(drop=True)
    per_segment = pd.concat(per_segment_frames, ignore_index=True).sort_values(
        ["well_id", "window_ft", "segment_index"], kind="stable"
    )
    metrics, per_fold = build_metrics_tables(per_well)
    metrics["experiment_fingerprint"] = fingerprint
    formal_regression = _validate_formal_regression(metrics) if max_wells is None else None
    input_audit = {
        **selection_audit,
        "run_wells": int(len(selected)),
        "run_rows": int(selected["hidden_rows"].sum()),
        "run_shadow_intersection": 0,
        "p2_arrow_filter_before_pandas": True,
        "p2_development_or_smoke_rmse": p2_rmse,
        "formal_p2_development_expected_rmse": FORMAL_P2_DEVELOPMENT_RMSE,
        "frozen_p2_full_old_rmse": FORMAL_P2_FULL_OLD_RMSE,
        "natural_key_alignment_wells": int(len(selected)),
        "mode_center_recomputed_wells": int(len(selected)),
        "maximum_mode_center_abs_error": maximum_center_error,
        "hidden_target_read": True,
        "hidden_target_usage": "oracle_diagnostic_only",
        "oracle_output_is_legal_feature_or_cache": False,
        "formal_reference_regression": formal_regression,
    }
    runtime = {
        "experiment_fingerprint": fingerprint,
        "wells": int(len(selected)),
        "rows": int(selected["hidden_rows"].sum()),
        "checkpoint_hits": int(cache_hits),
        "checkpoint_misses": int(len(selected) - cache_hits),
        "total_elapsed_seconds": float(time.perf_counter() - started),
        "streaming_policy": "one well at a time; no global 128 x all-row array",
        "resume_policy": "validate per-well checkpoint fingerprint and natural key",
    }
    write_csv_atomic(per_well, run_dir / "per_well.csv")
    write_csv_atomic(per_fold, run_dir / "per_fold.csv")
    write_csv_atomic(per_segment, run_dir / "per_segment.csv")
    write_json_atomic(input_audit, run_dir / "input_audit.json")
    write_json_atomic(metrics, run_dir / "metrics.json")
    write_json_atomic(runtime, run_dir / "runtime.json")
    write_text_atomic(_build_conclusion(metrics), run_dir / "conclusion.md")
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wells", type=int, choices=(1, 2, 3))
    parser.add_argument("--folds-path", type=Path, default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--shadow-path", type=Path, default=DEFAULT_SHADOW_PATH)
    parser.add_argument("--p2-path", type=Path, default=DEFAULT_P2_PATH)
    parser.add_argument("--mode-dir", type=Path, default=DEFAULT_MODE_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = resolve_run_artifact_dir(args.output_dir, args.max_wells)
    selected_text = "657 口正式开发井" if args.max_wells is None else f"{args.max_wells} 口 smoke 井"
    print(
        f"P3-PFM-D02 将处理 {selected_text}；主要耗时为逐井 Ward/medoid/分段 oracle；"
        f"输出={run_dir}；中断后按逐井 checkpoint 指纹续跑。",
        flush=True,
    )
    run_oracle_ladder(
        folds_path=args.folds_path,
        shadow_path=args.shadow_path,
        p2_path=args.p2_path,
        mode_dir=args.mode_dir,
        seed_dir=args.seed_dir,
        output_dir=args.output_dir,
        max_wells=args.max_wells,
    )


if __name__ == "__main__":
    main()
