"""P3_FINAL_CANDIDATE_v1 的一次性影子集验证入口。

候选阶段只读取测试期可获得的列；影子真值只能在候选完整落盘并通过
SHA-256 校验之后读取。这个脚本不训练模型，也不搜索任何参数。
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
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_up01_robust_u_projection import robust_polynomial_projection  # noqa: E402
from src.p3_up03_up01_plus_pfs_correction import build_up03_candidate  # noqa: E402
from scripts.run_p3_pfs01_fixed_lag_particle_smoothing import generate_one_well  # noqa: E402


EXPERIMENT_ID = "P4_FINAL00_UP03_shadow_v1"
CANDIDATE_ID = "P4_FINAL00_UP03_v1"
EXPECTED_SHADOW_WELLS = 116
EXPECTED_SHADOW_FOLD_COUNTS = {0: 24, 1: 23, 2: 23, 3: 23, 4: 23}
UP01_DEGREE = 2
UP01_BLEND = 0.50
PFS_LAG_FT = 1000.0
PFS_CORRECTION = 0.25
GATE = {
    "minimum_pooled_improvement_ft": 0.15,
    "minimum_improved_folds": 3,
    "minimum_well_win_rate": 0.52,
    "maximum_bootstrap_ci_upper_ft": 0.0,
    "maximum_p90_degradation_ft": 0.20,
}
LEGAL_P2_COLUMNS = ("well_id", "fold", "row_index", "md", "pred_tvt")
KEY_COLUMNS = ["well_id", "fold", "row_index"]
FOLD_REGISTRY = CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv"
SHADOW_REGISTRY = CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv"
PREDICTION_PATH = CLEAN_ROOT / "artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet"
B00_FEATURE_CACHE = CLEAN_ROOT / "artifacts/B00_simple_lgbm_v1/feature_cache.parquet"
RAW_TRAIN_DIR = (CLEAN_ROOT / "../input/data/raw/train").resolve()
SOURCE_PF_CONFIG = CLEAN_ROOT / "configs/p2_p01_multiseed_pf_mean_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts/P4_FINAL00_UP03_shadow_v1"
PFS_FINGERPRINT = "4bf62f495ef530bb3ca53d65d15475f7bc3784fe5abaebbbb1053584704c4480"


def _shadow_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def load_shadow_registry(fold_path: Path, shadow_path: Path) -> pd.DataFrame:
    """从冻结登记表选择且只选择 116 口影子井。"""

    folds = pd.read_csv(fold_path, dtype={"well_id": str})
    shadow = pd.read_csv(shadow_path, dtype={"well_id": str})
    required_folds = {"well_id", "fold", "hidden_rows"}
    required_shadow = {"well_id", "fold", "hidden_row_count", "is_shadow"}
    if missing := required_folds.difference(folds.columns):
        raise ValueError(f"冻结 fold 登记表缺列：{sorted(missing)}")
    if missing := required_shadow.difference(shadow.columns):
        raise ValueError(f"冻结影子登记表缺列：{sorted(missing)}")
    shadow = shadow.loc[_shadow_mask(shadow["is_shadow"])].copy()
    if folds["well_id"].duplicated().any() or shadow["well_id"].duplicated().any():
        raise ValueError("冻结登记表包含重复井号")
    if len(shadow) != EXPECTED_SHADOW_WELLS:
        raise ValueError(f"冻结影子井必须恰好为 {EXPECTED_SHADOW_WELLS} 口")

    selected = shadow[["well_id", "fold", "hidden_row_count"]].merge(
        folds[["well_id", "fold", "hidden_rows"]],
        on="well_id",
        how="left",
        suffixes=("_shadow", "_fold"),
        validate="one_to_one",
    )
    if selected[["fold_fold", "hidden_rows"]].isna().any().any():
        raise ValueError("冻结影子井未被 fold 登记表完整覆盖")
    if not np.array_equal(
        selected["fold_shadow"].to_numpy(dtype=np.int64),
        selected["fold_fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("冻结影子登记表与 fold 登记表的 fold 不一致")
    if not np.array_equal(
        selected["hidden_row_count"].to_numpy(dtype=np.int64),
        selected["hidden_rows"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("冻结影子登记表的隐藏行数发生漂移")
    selected = selected.rename(columns={"fold_fold": "fold"})[
        ["well_id", "fold", "hidden_rows"]
    ]
    counts = {
        int(key): int(value)
        for key, value in selected.groupby("fold").size().to_dict().items()
    }
    if counts != EXPECTED_SHADOW_FOLD_COUNTS:
        raise ValueError(f"冻结影子 fold 分布发生漂移：{counts}")
    return selected.sort_values("well_id", kind="stable").reset_index(drop=True)


def load_legal_p2_rows(prediction_path: Path, registry: pd.DataFrame) -> pd.DataFrame:
    """Arrow 层物理只读取 P2-P02 的非目标列。"""

    dataset = arrow_dataset.dataset(prediction_path, format="parquet")
    if missing := set(LEGAL_P2_COLUMNS).difference(dataset.schema.names):
        raise ValueError(f"P2-P02 预测缺少合法列：{sorted(missing)}")
    well_ids = sorted(registry["well_id"].astype(str).tolist())
    frame = dataset.to_table(
        columns=list(LEGAL_P2_COLUMNS),
        filter=arrow_dataset.field("well_id").isin(well_ids),
    ).to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_index"]).any():
        raise ValueError("P2-P02 影子候选存在重复行键")
    if set(frame["well_id"]) != set(well_ids):
        raise ValueError("P2-P02 没有完整覆盖冻结影子井")
    expected = registry.set_index("well_id")
    actual_rows = frame.groupby("well_id").size()
    if not actual_rows.equals(expected.loc[actual_rows.index, "hidden_rows"]):
        raise ValueError("P2-P02 影子候选隐藏行数不匹配")
    actual_folds = frame.groupby("well_id")["fold"].first().astype(np.int64)
    if not actual_folds.equals(expected.loc[actual_folds.index, "fold"].astype(np.int64)):
        raise ValueError("P2-P02 影子候选 fold 不匹配")
    numeric = frame[["fold", "row_index", "md", "pred_tvt"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("P2-P02 合法列含 NaN/Inf")
    return frame.loc[:, list(LEGAL_P2_COLUMNS)]


def build_legal_candidate(legal_p2: pd.DataFrame, pfs_lag1000: pd.DataFrame) -> pd.DataFrame:
    """按冻结公式构造 UP01 和最终 UP03；函数不接受目标列。"""

    if "target_tvt" in legal_p2.columns or "target_tvt" in pfs_lag1000.columns:
        raise ValueError("候选生成阶段禁止接收 target_tvt")
    required_p2 = {*LEGAL_P2_COLUMNS, "z_current"}
    required_pfs = {*KEY_COLUMNS, "last_visible_tvt", "pfs_lag1000_delta"}
    if missing := required_p2.difference(legal_p2.columns):
        raise ValueError(f"合法 P2 表缺列：{sorted(missing)}")
    if missing := required_pfs.difference(pfs_lag1000.columns):
        raise ValueError(f"PFS lag1000 表缺列：{sorted(missing)}")
    merged = legal_p2.merge(
        pfs_lag1000[list(required_pfs)],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(legal_p2) or len(merged) != len(pfs_lag1000):
        raise ValueError("P2-P02 与 PFS lag1000 行键不完全一致")
    merged = merged.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    up01 = np.empty(len(merged), dtype=np.float64)
    for _, positions in merged.groupby("well_id", sort=False).groups.items():
        pos = np.asarray(list(positions), dtype=np.int64)
        frame = merged.iloc[pos]
        md = frame["md"].to_numpy(dtype=np.float64)
        z = frame["z_current"].to_numpy(dtype=np.float64)
        base_tvt = frame["pred_tvt"].to_numpy(dtype=np.float64)
        base_u = base_tvt + z
        projected_u = robust_polynomial_projection(md, base_u, degree=UP01_DEGREE)
        up01[pos] = base_u + UP01_BLEND * (projected_u - base_u) - z
    p3b00 = merged["pred_tvt"].to_numpy(dtype=np.float64)
    pfs_absolute = (
        merged["last_visible_tvt"].to_numpy(dtype=np.float64)
        + merged["pfs_lag1000_delta"].to_numpy(dtype=np.float64)
    )
    up03 = build_up03_candidate(up01, p3b00, pfs_absolute, PFS_CORRECTION)
    result = merged[KEY_COLUMNS + ["md"]].copy()
    result["p3b00_pred_tvt"] = p3b00
    result["up01_degree2_blend50_tvt"] = up01
    result["pfs_lag1000_abs_tvt"] = pfs_absolute
    result["up03_pred_tvt"] = up03
    if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("最终合法候选含 NaN/Inf")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_z_context(cache_path: Path, legal_p2: pd.DataFrame) -> pd.DataFrame:
    well_ids = sorted(legal_p2["well_id"].astype(str).unique())
    dataset = arrow_dataset.dataset(cache_path, format="parquet")
    context = dataset.to_table(
        columns=["well_id", "row_index", "z_current"],
        filter=arrow_dataset.field("well_id").isin(well_ids),
    ).to_pandas()
    context["well_id"] = context["well_id"].astype(str)
    merged = legal_p2.merge(
        context,
        on=["well_id", "row_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(legal_p2):
        raise RuntimeError("B00 的 Z 上下文没有完整覆盖影子候选行")
    return merged


def generate_shadow_pfs(
    registry: pd.DataFrame,
    output_dir: Path,
    *,
    workers: int,
    force: bool = False,
) -> dict[str, Any]:
    """复用冻结 PFS 内核，只给影子井生成无目标路径并支持逐井续跑。"""

    if workers <= 0:
        raise ValueError("workers 必须为正整数")
    source_config = json.loads(SOURCE_PF_CONFIG.read_text(encoding="utf-8"))
    artifact_dir = output_dir / "pfs"
    tasks = [
        {
            "well_id": str(row.well_id),
            "fold": int(row.fold),
            "hidden_rows": int(row.hidden_rows),
            "raw_train_dir": str(RAW_TRAIN_DIR),
            "artifact_dir": str(artifact_dir),
            "particle_filter": source_config["particle_filter"],
            "fingerprint": PFS_FINGERPRINT,
            "force": bool(force),
        }
        for row in registry.itertuples(index=False)
    ]
    started = time.perf_counter()
    runtimes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_one_well, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            runtime = future.result()
            runtimes.append(runtime)
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(tasks) - completed)
            print(
                f"PFS [{completed}/{len(tasks)}] {runtime['well_id']} "
                f"rows={runtime['hidden_rows']} cache={runtime.get('cache_hit', False)} "
                f"ETA={eta / 60:.1f}min",
                flush=True,
            )
    runtimes.sort(key=lambda item: str(item["well_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runtimes).to_csv(output_dir / "pfs_runtime_per_well.csv", index=False)
    summary = {
        "wells": len(runtimes),
        "rows": int(sum(int(item["hidden_rows"]) for item in runtimes)),
        "cache_hits": int(sum(bool(item.get("cache_hit")) for item in runtimes)),
        "cache_misses": int(sum(not bool(item.get("cache_hit")) for item in runtimes)),
        "hidden_target_read": False,
        "pfs_fingerprint": PFS_FINGERPRINT,
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "pfs_generation_runtime.json", summary)
    return summary


def load_shadow_pfs(registry: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        cache_path = output_dir / "pfs/legal_cache" / f"{well_id}.parquet"
        runtime_path = output_dir / "pfs/legal_runtime" / f"{well_id}.json"
        if not cache_path.is_file() or not runtime_path.is_file():
            raise FileNotFoundError(f"影子 PFS 缓存不完整：{well_id}")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("hidden_tvt_read") is not False:
            raise RuntimeError(f"PFS runtime 声明读取过隐藏真值：{well_id}")
        if runtime.get("experiment_fingerprint") != PFS_FINGERPRINT:
            raise RuntimeError(f"PFS fingerprint 不匹配：{well_id}")
        if runtime.get("cache_sha256") != file_sha256(cache_path):
            raise RuntimeError(f"PFS 缓存哈希不匹配：{well_id}")
        frame = pd.read_parquet(
            cache_path,
            columns=[
                "well_id",
                "fold",
                "row_index",
                "last_visible_tvt",
                "pfs_lag1000_delta",
                "_cache_fingerprint",
            ],
        )
        if set(frame["_cache_fingerprint"].astype(str)) != {PFS_FINGERPRINT}:
            raise RuntimeError(f"PFS parquet fingerprint 不匹配：{well_id}")
        if len(frame) != int(row.hidden_rows) or set(frame["fold"].astype(int)) != {int(row.fold)}:
            raise RuntimeError(f"PFS 行数或 fold 不匹配：{well_id}")
        parts.append(frame.drop(columns="_cache_fingerprint"))
    result = pd.concat(parts, ignore_index=True)
    result["well_id"] = result["well_id"].astype(str)
    if result.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("影子 PFS 存在重复行键")
    return result


def land_legal_candidate(
    output_dir: Path,
    registry: pd.DataFrame,
    legal: pd.DataFrame,
    pfs_summary: dict[str, Any],
) -> tuple[Path, Path]:
    candidate_path = output_dir / "legal_candidates.parquet"
    manifest_path = output_dir / "legal_generation_manifest.json"
    _write_parquet(candidate_path, legal)
    full_shadow = (
        len(registry) == EXPECTED_SHADOW_WELLS
        and registry.groupby("fold").size().to_dict() == EXPECTED_SHADOW_FOLD_COUNTS
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID,
        "generation_stage": "deterministic_pipeline_development_pending_shadow",
        "candidate_formula": "UP01_degree2_blend50 + 0.25*(PFS_lag1000_abs-P3B00)",
        "candidate_generation_complete": True,
        "formal_shadow_complete": bool(full_shadow),
        "hidden_target_read": False,
        "wells": int(legal["well_id"].nunique()),
        "rows": int(len(legal)),
        "fold_counts": {str(k): int(v) for k, v in registry.groupby("fold").size().to_dict().items()},
        "pfs_generation": pfs_summary,
        "source_sha256": {
            "fold_registry": file_sha256(FOLD_REGISTRY),
            "shadow_registry": file_sha256(SHADOW_REGISTRY),
            "p2_predictions": file_sha256(PREDICTION_PATH),
            "b00_feature_cache": file_sha256(B00_FEATURE_CACHE),
            "source_pf_config": file_sha256(SOURCE_PF_CONFIG),
        },
        "legal_candidates_sha256": file_sha256(candidate_path),
        "scored_paths": ["P3B00", "P4_FINAL00_UP03_v1"],
        "forbidden_selection_candidates": ["UP01", "UP02", "UP08"],
    }
    _write_json(manifest_path, manifest)
    return candidate_path, manifest_path


def load_targets_after_candidate_landed(
    candidate_path: Path,
    manifest_path: Path,
    prediction_path: Path,
) -> pd.DataFrame:
    """只有候选文件与无目标清单完全匹配后，才物理读取 target_tvt。"""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_generation_complete") is not True:
        raise RuntimeError("合法候选尚未完整落盘")
    if manifest.get("hidden_target_read") is not False:
        raise RuntimeError("候选生成清单的 hidden_target_read 不合格")
    if manifest.get("legal_candidates_sha256") != file_sha256(candidate_path):
        raise RuntimeError("合法候选 SHA-256 哈希不匹配")
    candidate_keys = pd.read_parquet(candidate_path, columns=KEY_COLUMNS)
    candidate_keys["well_id"] = candidate_keys["well_id"].astype(str)
    if len(candidate_keys) != int(manifest["rows"]):
        raise RuntimeError("候选行数与清单不一致")
    well_ids = sorted(candidate_keys["well_id"].unique())
    dataset = arrow_dataset.dataset(prediction_path, format="parquet")
    target = dataset.to_table(
        columns=KEY_COLUMNS + ["target_tvt"],
        filter=arrow_dataset.field("well_id").isin(well_ids),
    ).to_pandas()
    target["well_id"] = target["well_id"].astype(str)
    if set(map(tuple, target[KEY_COLUMNS].to_numpy())) != set(
        map(tuple, candidate_keys[KEY_COLUMNS].to_numpy())
    ):
        raise RuntimeError("影子真值与已落盘候选行键不一致")
    if not np.isfinite(target["target_tvt"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("影子 target_tvt 含 NaN/Inf")
    return target[KEY_COLUMNS + ["target_tvt"]]


def evaluate_shadow_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "pooled_improvement_ge_015": float(metrics["pooled_improvement_ft"])
        >= GATE["minimum_pooled_improvement_ft"],
        "improved_folds_ge_3": int(metrics["improved_folds"])
        >= GATE["minimum_improved_folds"],
        "well_win_rate_ge_052": float(metrics["well_win_rate"])
        >= GATE["minimum_well_win_rate"],
        "bootstrap_ci_upper_lt_0": float(metrics["bootstrap_ci95_upper_ft"])
        < GATE["maximum_bootstrap_ci_upper_ft"],
        "p90_degradation_le_020": float(metrics["p90_degradation_ft"])
        <= GATE["maximum_p90_degradation_ft"],
    }
    return {**checks, "passed": bool(all(checks.values()))}


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def score_shadow(
    legal: pd.DataFrame,
    target: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = legal.merge(target, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(scored) != len(legal) or len(scored) != len(target):
        raise RuntimeError("合法候选与影子真值没有一一对齐")
    truth = scored["target_tvt"].to_numpy(dtype=np.float64)
    base = scored["p3b00_pred_tvt"].to_numpy(dtype=np.float64)
    candidate = scored["up03_pred_tvt"].to_numpy(dtype=np.float64)
    base_rmse = _rmse(truth, base)
    candidate_rmse = _rmse(truth, candidate)

    fold_rows: list[dict[str, Any]] = []
    for fold, frame in scored.groupby("fold", sort=True):
        fold_target = frame["target_tvt"].to_numpy(dtype=np.float64)
        fold_base = _rmse(fold_target, frame["p3b00_pred_tvt"].to_numpy(dtype=np.float64))
        fold_candidate = _rmse(fold_target, frame["up03_pred_tvt"].to_numpy(dtype=np.float64))
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(frame["well_id"].nunique()),
                "rows": int(len(frame)),
                "p3b00_rmse": fold_base,
                "up03_rmse": fold_candidate,
                "improvement_ft": fold_base - fold_candidate,
            }
        )
    per_fold = pd.DataFrame(fold_rows)

    well_rows: list[dict[str, Any]] = []
    for (well_id, fold), frame in scored.groupby(["well_id", "fold"], sort=True):
        well_target = frame["target_tvt"].to_numpy(dtype=np.float64)
        base_error = frame["p3b00_pred_tvt"].to_numpy(dtype=np.float64) - well_target
        candidate_error = frame["up03_pred_tvt"].to_numpy(dtype=np.float64) - well_target
        well_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "rows": int(len(frame)),
                "p3b00_sse": float(np.dot(base_error, base_error)),
                "up03_sse": float(np.dot(candidate_error, candidate_error)),
                "p3b00_rmse": float(np.sqrt(np.mean(base_error * base_error))),
                "up03_rmse": float(np.sqrt(np.mean(candidate_error * candidate_error))),
            }
        )
    per_well = pd.DataFrame(well_rows)
    n = len(per_well)
    rng = np.random.default_rng(bootstrap_seed)
    deltas = np.empty(bootstrap_resamples, dtype=np.float64)
    rows = per_well["rows"].to_numpy(dtype=np.float64)
    base_sse = per_well["p3b00_sse"].to_numpy(dtype=np.float64)
    candidate_sse = per_well["up03_sse"].to_numpy(dtype=np.float64)
    for index in range(bootstrap_resamples):
        sample = rng.integers(0, n, size=n)
        sample_rows = float(rows[sample].sum())
        deltas[index] = np.sqrt(candidate_sse[sample].sum() / sample_rows) - np.sqrt(
            base_sse[sample].sum() / sample_rows
        )
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    base_p90 = float(per_well["p3b00_rmse"].quantile(0.90))
    candidate_p90 = float(per_well["up03_rmse"].quantile(0.90))
    metrics: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID,
        "result_class": "deterministic_pipeline_development",
        "wells": int(n),
        "rows": int(len(scored)),
        "p3b00_pooled_rmse": base_rmse,
        "up03_pooled_rmse": candidate_rmse,
        "pooled_improvement_ft": base_rmse - candidate_rmse,
        "fold_improvements_ft": per_fold["improvement_ft"].astype(float).tolist(),
        "improved_folds": int((per_fold["improvement_ft"] > 0).sum()),
        "well_win_rate": float((per_well["up03_rmse"] < per_well["p3b00_rmse"]).mean()),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_delta_definition": "UP03_RMSE_minus_P3B00_RMSE",
        "bootstrap_ci95_low_ft": float(ci_low),
        "bootstrap_ci95_upper_ft": float(ci_high),
        "p3b00_p90_well_rmse": base_p90,
        "up03_p90_well_rmse": candidate_p90,
        "p90_degradation_ft": candidate_p90 - base_p90,
        "shadow_target_read": True,
        "parameters_searched_on_shadow": False,
        "paths_scored": ["P3B00", CANDIDATE_ID],
    }
    metrics["gate"] = evaluate_shadow_gate(metrics)
    metrics["result_class"] = (
        "shadow_confirmed_pipeline"
        if metrics["gate"]["passed"]
        else "deterministic_pipeline_development"
    )
    return metrics, per_fold, per_well, scored


def run_generate(output_dir: Path, *, workers: int, smoke_wells: int, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_shadow_registry(FOLD_REGISTRY, SHADOW_REGISTRY)
    if smoke_wells:
        registry = registry.head(int(smoke_wells)).copy()
    print(
        f"影子候选生成：{len(registry)}口井，{int(registry['hidden_rows'].sum()):,}行，"
        f"workers={workers}，target_tvt=false，输出={output_dir}",
        flush=True,
    )
    pfs_summary = generate_shadow_pfs(registry, output_dir, workers=workers, force=force)
    legal_p2 = load_legal_p2_rows(PREDICTION_PATH, registry)
    legal_with_z = load_z_context(B00_FEATURE_CACHE, legal_p2)
    pfs = load_shadow_pfs(registry, output_dir)
    legal = build_legal_candidate(legal_with_z, pfs)
    candidate_path, manifest_path = land_legal_candidate(output_dir, registry, legal, pfs_summary)
    _write_json(
        output_dir / "config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_id": CANDIDATE_ID,
            "generation_stage": "deterministic_pipeline_development_pending_shadow",
            "learned_models": 1,
            "base_model_id": "P3B00_group5_p2p02_v1",
            "up01_degree": UP01_DEGREE,
            "up01_blend": UP01_BLEND,
            "pfs_lag_ft": PFS_LAG_FT,
            "pfs_correction": PFS_CORRECTION,
            "gate": GATE,
            "shadow_target_read": False,
            "model_training_in_postprocess": False,
            "test_reproducible": True,
        },
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate_path),
                "manifest": str(manifest_path),
                "wells": int(legal["well_id"].nunique()),
                "rows": int(len(legal)),
                "hidden_target_read": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def run_score(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return metrics
    candidate_path = output_dir / "legal_candidates.parquet"
    manifest_path = output_dir / "legal_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("formal_shadow_complete") is not True:
        raise RuntimeError("只允许对完整冻结的116口影子候选评分")
    target = load_targets_after_candidate_landed(candidate_path, manifest_path, PREDICTION_PATH)
    legal = pd.read_parquet(candidate_path)
    metrics, per_fold, per_well, scored = score_shadow(legal, target)
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    _write_parquet(output_dir / "scored_predictions.parquet", scored)
    _write_json(metrics_path, metrics)
    _write_json(
        output_dir / "score_runtime.json",
        {
            "shadow_target_read": True,
            "parameters_searched_on_shadow": False,
            "paths_scored": ["P3B00", CANDIDATE_ID],
            "legal_candidates_sha256": file_sha256(candidate_path),
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "score"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke-wells", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.stage == "generate":
        run_generate(
            output_dir,
            workers=int(args.workers),
            smoke_wells=int(args.smoke_wells),
            force=bool(args.force),
        )
    else:
        if args.smoke_wells or args.force:
            raise ValueError("影子评分不允许 smoke 或 force")
        run_score(output_dir)


if __name__ == "__main__":
    main()
