"""从 PF128 和严格折外 P2 路径生成 PFM04 三个合法井级摘要。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

# 复用已经经过 PFM03 测试的井筛选、NPZ 验证和原子写文件代码，避免另写一套血缘逻辑。
from scripts import generate_p3_pfm03_shape_mode_cache as common_generator  # noqa: E402
from src.p3_pfm01_ordered_pf_modes import (  # noqa: E402
    FORMAL_LIKELIHOOD_SCALE,
    FORMAL_NUMBER_OF_SEEDS,
    MODE_NAMES,
    cluster_seed_descriptors,
    compute_p2_position,
)


EXPERIMENT_ID = "P3_PFM04_mode_direction_summary_v1"
SHARED_FINGERPRINT = common_generator.SHARED_FINGERPRINT
NEW_FEATURES = [
    "direction_score",
    "p2_position_raw",
    "high_minus_low_separation",
]
P2_LEGAL_COLUMNS = ["well_id", "fold", "row_index", "pred_tvt"]
LEGAL_COLUMNS = [
    "well_id",
    "fold",
    *NEW_FEATURES,
    "source_seed_sha256",
    "source_p2_sha256",
    "_cache_fingerprint",
]


def stable_mode_direction_summaries(
    seed_delta: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    p2_delta: np.ndarray,
) -> dict[str, float]:
    """稳定计算三个摘要，避免远离最佳似然的整个模式在中心计算前下溢。"""

    paths = np.asarray(seed_delta, dtype=np.float64)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    p2_path = np.asarray(p2_delta, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[0] != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("PFM04 必须输入 128 条二维 seed 路径")
    if likelihoods.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("PFM04 final_ll 数量必须与 128 条路径一致")
    if ids.shape != (FORMAL_NUMBER_OF_SEEDS,) or len(np.unique(ids)) != len(ids):
        raise ValueError("PFM04 seed_ids 必须与路径一一对应且不能重复")
    if p2_path.shape != (paths.shape[1],):
        raise ValueError("PFM04 P2 delta 必须与 seed 路径等长")
    if not np.isfinite(paths).all() or not np.isfinite(likelihoods).all():
        raise ValueError("PFM04 seed 路径或 final_ll 含 NaN/Inf")
    if not np.isfinite(p2_path).all():
        raise ValueError("PFM04 P2 delta 含 NaN/Inf")

    raw_labels = cluster_seed_descriptors(paths)
    # 总模式质量使用 logsumexp；即使最终 exp 下溢为 0，也不影响模式内部中心路径。
    scaled_ll = likelihoods / FORMAL_LIKELIHOOD_SCALE
    all_log_total = float(logsumexp(scaled_ll))
    records: list[dict[str, Any]] = []
    for raw_label in sorted(np.unique(raw_labels).tolist()):
        member_indices = np.flatnonzero(raw_labels == raw_label)
        member_ll = likelihoods[member_indices]
        # 中心权重只在当前模式内部平移，最优成员恒有 exp(0)=1，不会整体下溢。
        shifted = (member_ll - float(np.max(member_ll))) / FORMAL_LIKELIHOOD_SCALE
        unnormalized = np.exp(shifted)
        member_weights = unnormalized / float(np.sum(unnormalized))
        center_path = np.sum(
            paths[member_indices] * member_weights[:, None],
            axis=0,
            dtype=np.float64,
        )
        member_log_total = float(logsumexp(scaled_ll[member_indices]))
        # 数学上小于 float64 表示范围的质量允许记为 0；方向分数仍是有限数。
        mode_mass = float(np.exp(member_log_total - all_log_total))
        mode_mass = float(np.clip(mode_mass, 0.0, 1.0))
        records.append(
            {
                "center_path": center_path,
                "mass": mode_mass,
                "mean_delta": float(np.mean(center_path)),
                "endpoint_delta": float(center_path[-1]),
                "minimum_seed_id": int(np.min(ids[member_indices])),
            }
        )
    records.sort(
        key=lambda record: (
            float(record["mean_delta"]),
            float(record["endpoint_delta"]),
            int(record["minimum_seed_id"]),
        )
    )
    modes = {name: record for name, record in zip(MODE_NAMES, records, strict=True)}
    low_path = np.asarray(modes["low"]["center_path"], dtype=np.float64)
    high_path = np.asarray(modes["high"]["center_path"], dtype=np.float64)
    p2_position_raw, _, _ = compute_p2_position(p2_path, low_path, high_path)
    return {
        "direction_score": float(modes["high"]["mass"] - modes["low"]["mass"]),
        "p2_position_raw": float(p2_position_raw),
        "high_minus_low_separation": float(np.mean(high_path - low_path)),
    }


def _legal_frame_sha256(frame: pd.DataFrame) -> str:
    """只对明确读取的合法列计算内容指纹，不读取预测文件中的隐藏目标列。"""

    ordered = frame[P2_LEGAL_COLUMNS].sort_values(
        ["well_id", "row_index"], kind="stable"
    )
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """原子保存 PFM04 自己的一井一行 schema，不借用 PFM03 的逐行 schema。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _load_legal_p2_predictions(
    path: Path,
    selected_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """仅读取 P2 OOF 的自然键、fold 和预测值，并核对每口开发井完整覆盖。"""

    if not path.is_file():
        raise FileNotFoundError(f"P2 预测不存在：{path}")
    # columns 参数是关键边界：含隐藏真值的同一 parquet 中，目标列不会被载入内存。
    predictions = pd.read_parquet(path, columns=P2_LEGAL_COLUMNS)
    predictions["well_id"] = predictions["well_id"].astype(str)
    predictions["fold"] = predictions["fold"].astype(np.int64)
    predictions["row_index"] = predictions["row_index"].astype(np.int64)
    selected_ids = set(selected_registry["well_id"].astype(str))
    predictions = predictions.loc[predictions["well_id"].isin(selected_ids)].copy()
    if predictions.duplicated(["well_id", "row_index"]).any():
        raise ValueError("P2 合法列含重复 well_id+row_index")
    expected_rows = int(selected_registry["hidden_rows"].sum())
    if len(predictions) != expected_rows:
        raise ValueError(f"P2 合法行数应为 {expected_rows}，实际为 {len(predictions)}")
    observed = predictions.groupby("well_id", sort=False).agg(
        fold=("fold", "first"),
        hidden_rows=("row_index", "size"),
        fold_count=("fold", "nunique"),
    )
    expected = selected_registry.set_index("well_id")[["fold", "hidden_rows"]].copy()
    expected.index = expected.index.astype(str)
    observed = observed.reindex(expected.index)
    if observed.isna().any().any():
        raise ValueError("P2 合法列缺少开发井")
    if not np.array_equal(observed["fold"].to_numpy(dtype=np.int64), expected["fold"]):
        raise ValueError("P2 fold 与冻结 registry 不一致")
    if not np.array_equal(
        observed["hidden_rows"].to_numpy(dtype=np.int64), expected["hidden_rows"]
    ):
        raise ValueError("P2 每井隐藏行数与冻结 registry 不一致")
    if not observed["fold_count"].eq(1).all():
        raise ValueError("同一口井在 P2 合法列中出现多个 fold")
    values = predictions["pred_tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("P2 路径含 NaN/Inf")
    return predictions, _legal_frame_sha256(predictions)


def _checkpoint_paths(run_dir: Path, well_id: str) -> tuple[Path, Path]:
    """返回单井摘要和单井运行血缘文件，支持中断后继续。"""

    return (
        run_dir / "legal" / "checkpoints" / f"{well_id}.parquet",
        run_dir / "legal" / "runtime" / f"{well_id}.json",
    )


def _load_checkpoint(
    cache_path: Path,
    runtime_path: Path,
    *,
    well_id: str,
    fold: int,
    experiment_fingerprint: str,
    seed_sha256: str,
    p2_sha256: str,
) -> pd.DataFrame | None:
    """只有代码、配置和两项逐井输入都完全一致时才接受旧摘要。"""

    if not cache_path.is_file() or not runtime_path.is_file():
        return None
    try:
        cache = pd.read_parquet(cache_path)
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if cache.columns.tolist() != LEGAL_COLUMNS or len(cache) != 1:
            return None
        expected_runtime = {
            "well_id": well_id,
            "fold": fold,
            "experiment_fingerprint": experiment_fingerprint,
            "source_seed_sha256": seed_sha256,
            "source_p2_sha256": p2_sha256,
            "hidden_tvt_read": False,
        }
        if any(runtime.get(key) != value for key, value in expected_runtime.items()):
            return None
        if str(cache.at[0, "well_id"]) != well_id or int(cache.at[0, "fold"]) != fold:
            return None
        if cache.at[0, "_cache_fingerprint"] != experiment_fingerprint:
            return None
        if cache.at[0, "source_seed_sha256"] != seed_sha256:
            return None
        if cache.at[0, "source_p2_sha256"] != p2_sha256:
            return None
        if not np.isfinite(cache[NEW_FEATURES].to_numpy(dtype=np.float64)).all():
            return None
        return cache
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def generate_mode_direction_cache(
    folds_path: Path,
    shadow_path: Path,
    shared_dir: Path,
    p2_predictions_path: Path,
    output_dir: Path,
    *,
    max_wells: int | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """生成一井一行的三个合法摘要；函数全过程不读取隐藏 TVT。"""

    started = time.perf_counter()
    if workers <= 0:
        raise ValueError("workers 必须为正整数")
    folds, selected, shadow_ids = common_generator._load_development_selection(
        folds_path, shadow_path
    )
    if max_wells is None:
        common_generator._validate_formal_selection(folds, selected, shadow_ids)
    else:
        if max_wells not in (1, 2, 3):
            raise ValueError("--max-wells 只允许 1、2、3")
        selected = selected.head(max_wells).copy()
    selected["well_id"] = selected["well_id"].astype(str)

    p2_predictions, p2_legal_sha256 = _load_legal_p2_predictions(
        p2_predictions_path, selected
    )
    p2_by_well = {
        str(well_id): group.sort_values("row_index", kind="stable").reset_index(drop=True)
        for well_id, group in p2_predictions.groupby("well_id", sort=False)
    }
    generator_sha256 = common_generator.file_sha256(Path(__file__).resolve())
    ordered_mode_core = CLEAN_ROOT / "src" / "p3_pfm01_ordered_pf_modes.py"
    ordered_mode_core_sha256 = common_generator.file_sha256(ordered_mode_core)
    experiment_fingerprint = common_generator._stable_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "shared_fingerprint": SHARED_FINGERPRINT,
            "p2_legal_sha256": p2_legal_sha256,
            "generator_sha256": generator_sha256,
            "ordered_mode_core_sha256": ordered_mode_core_sha256,
            "features": NEW_FEATURES,
            "mode_builder": "stable_mode_direction_summaries_v1",
            "likelihood_scale": 8.0,
        }
    )
    run_dir = output_dir if max_wells is None else output_dir / f"smoke_{max_wells}"
    common_generator._write_json_atomic(
        {
            "experiment_id": EXPERIMENT_ID,
            "selected_wells": int(len(selected)),
            "selected_rows": int(selected["hidden_rows"].sum()),
            "max_wells": max_wells,
            "workers": workers,
            "input_columns": {
                "p2_predictions": P2_LEGAL_COLUMNS,
                "pf128": [
                    "seed_delta",
                    "row_index",
                    "last_tvt",
                    "final_ll",
                    "seed_ids",
                    "_cache_fingerprint",
                    "_format_version",
                ],
            },
            "experiment_fingerprint": experiment_fingerprint,
            "p2_legal_sha256": p2_legal_sha256,
            "hidden_tvt_read": False,
        },
        run_dir / "legal_generation_config.json",
    )
    common_generator._write_json_atomic(NEW_FEATURES, run_dir / "legal_feature_list.json")

    def build_one(registry_row: Any) -> tuple[pd.DataFrame, bool]:
        """读取一口井的两条合法来源，计算三个标量并原子写 checkpoint。"""

        well_started = time.perf_counter()
        well_id = str(registry_row.well_id)
        fold = int(registry_row.fold)
        expected_rows = int(registry_row.hidden_rows)
        if well_id in shadow_ids:
            raise RuntimeError("影子井不能进入 PFM04 摘要生成")
        seed_path = shared_dir / f"{well_id}.npz"
        seed_sha256 = common_generator.file_sha256(seed_path)
        p2_well = p2_by_well[well_id]
        p2_sha256 = _legal_frame_sha256(p2_well)
        cache_path, runtime_path = _checkpoint_paths(run_dir, well_id)
        cached = _load_checkpoint(
            cache_path,
            runtime_path,
            well_id=well_id,
            fold=fold,
            experiment_fingerprint=experiment_fingerprint,
            seed_sha256=seed_sha256,
            p2_sha256=p2_sha256,
        )
        if cached is not None:
            print(f"{well_id}: cache hit", flush=True)
            return cached, True

        seed = common_generator._read_seed_cache(seed_path, expected_rows)
        expected_index = p2_well["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(seed["row_index"], expected_index):
            raise ValueError(f"PFM04 {well_id} 的 PF128 与 P2 row_index 不一致")
        last_visible_tvt = float(seed["last_tvt"][0])
        p2_delta = p2_well["pred_tvt"].to_numpy(dtype=np.float64) - last_visible_tvt
        legal_summary = stable_mode_direction_summaries(
            seed_delta=seed["seed_delta"],
            final_ll=seed["final_ll"],
            seed_ids=seed["seed_ids"],
            p2_delta=p2_delta,
        )
        row = {
            "well_id": well_id,
            "fold": fold,
            **{name: float(legal_summary[name]) for name in NEW_FEATURES},
            "source_seed_sha256": seed_sha256,
            "source_p2_sha256": p2_sha256,
            "_cache_fingerprint": experiment_fingerprint,
        }
        cache = pd.DataFrame([row], columns=LEGAL_COLUMNS)
        if not np.isfinite(cache[NEW_FEATURES].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"PFM04 {well_id} 摘要含 NaN/Inf")
        _write_parquet_atomic(cache, cache_path)
        common_generator._write_json_atomic(
            {
                "experiment_id": EXPERIMENT_ID,
                "well_id": well_id,
                "fold": fold,
                "rows": expected_rows,
                "experiment_fingerprint": experiment_fingerprint,
                "source_seed_sha256": seed_sha256,
                "source_p2_sha256": p2_sha256,
                "shared_fingerprint": SHARED_FINGERPRINT,
                "hidden_tvt_read": False,
                "cache_hit": False,
                "elapsed_seconds": float(time.perf_counter() - well_started),
            },
            runtime_path,
        )
        print(f"{well_id}: generated", flush=True)
        return cache, False

    frames: list[pd.DataFrame] = []
    cache_hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for frame, was_hit in executor.map(build_one, selected.itertuples(index=False)):
            frames.append(frame)
            cache_hits += int(was_hit)
    legal = pd.concat(frames, ignore_index=True).sort_values("well_id", kind="stable")
    if len(legal) != len(selected) or legal["well_id"].nunique() != len(selected):
        raise RuntimeError("PFM04 合并摘要不是严格一井一行")
    _write_parquet_atomic(legal[LEGAL_COLUMNS], run_dir / "legal" / "per_well.parquet")
    runtime = {
        "wells": int(len(legal)),
        "rows": int(selected["hidden_rows"].sum()),
        "cache_hits": int(cache_hits),
        "cache_misses": int(len(legal) - cache_hits),
        "experiment_fingerprint": experiment_fingerprint,
        "p2_legal_sha256": p2_legal_sha256,
        "hidden_tvt_read": False,
        "total_elapsed_seconds": float(time.perf_counter() - started),
    }
    common_generator._write_json_atomic(runtime, run_dir / "legal_runtime.json")
    return runtime


def main() -> None:
    """命令行默认生成 657 井正式缓存；--max-wells 3 写入独立 smoke 目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wells", type=int, choices=(1, 2, 3))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / EXPERIMENT_ID,
    )
    args = parser.parse_args()
    summary = generate_mode_direction_cache(
        CLEAN_ROOT / "artifacts" / "folds" / "balanced_well_5fold_v1.csv",
        CLEAN_ROOT / "artifacts" / "P3_shadow_holdout_v1" / "shadow_holdout.csv",
        CLEAN_ROOT / "artifacts" / "P3_shared_pf_seed_paths_v1",
        CLEAN_ROOT / "artifacts" / "P2_P02_multiscale_pf_paths_v1" / "predictions.parquet",
        args.output_dir,
        max_wells=args.max_wells,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
