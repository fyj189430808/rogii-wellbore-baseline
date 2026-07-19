"""运行 P3-N01a：用 outer-train 邻井四控制点修正 PF scale8 整段路径。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_d00_residual_structure import normalized_progress  # noqa: E402
from src.p3_n01a_neighbor_residual_audit import (  # noqa: E402
    MAX_AZIMUTH_DIFFERENCE_DEG,
    MAX_DISTANCE_FT,
    MAX_NEIGHBORS,
    MIN_PARALLEL_OVERLAP_FT,
    aggregate_control_profiles,
    build_fixed_shuffle,
    compute_trajectory_geometry,
    fit_source_control4,
    geometry_is_eligible,
    interpolate_control4,
    neighbor_weight,
)


EXPERIMENT_ID = "P3_N01a_pf_scale8_neighbor_residual_audit_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_n01a_pf_scale8_neighbor_residual_audit_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
LEGAL_COLUMNS = [
    "well_id",
    "fold",
    "row_index",
    "md",
    "baseline_scale8_tvt",
    "neighbor_corrected_tvt",
    "shuffled_corrected_tvt",
    "support_well_count",
    "support_total_weight",
    "eta",
    "source_well_ids_json",
    "_cache_fingerprint",
]


@dataclass(frozen=True)
class SourceRecord:
    well_id: str
    hidden_xy: np.ndarray
    bbox: tuple[float, float, float, float]
    controls: np.ndarray
    typewell_tail_fingerprint: str


def _workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shadow_ids(path: Path) -> set[str]:
    shadow = pd.read_csv(path)
    if "is_shadow" in shadow.columns:
        flags = shadow["is_shadow"]
        mask = flags if flags.dtype == bool else flags.astype(str).str.lower().isin({"1", "true", "yes"})
        shadow = shadow.loc[mask]
    return set(shadow["well_id"].astype(str))


def _development_registry(config: dict[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    registry = pd.read_csv(_workspace_path(str(config["fold_registry"])))
    shadow = _shadow_ids(_workspace_path(str(config["shadow_registry"])))
    registry["well_id"] = registry["well_id"].astype(str)
    development = registry.loc[~registry["well_id"].isin(shadow)].copy()
    development = development.sort_values("well_id").reset_index(drop=True)
    if len(development) != int(config["development_wells"]):
        raise ValueError(f"开发井数量不是 {config['development_wells']}：{len(development)}")
    if set(development["well_id"]).intersection(shadow):
        raise RuntimeError("开发井与 shadow 重叠")
    return development, shadow


def _hidden_rows(frame: pd.DataFrame) -> pd.DataFrame:
    hidden = frame.loc[frame["TVT_input"].isna()].copy()
    if len(hidden) < 2:
        raise ValueError("自然隐藏段少于两行")
    return hidden


def _bbox(xy: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(xy[:, 0].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].min()),
        float(xy[:, 1].max()),
    )


def _bbox_distance(
    target_bbox: tuple[float, float, float, float],
    source_bbox: tuple[float, float, float, float],
) -> float:
    target_min_x, target_max_x, target_min_y, target_max_y = target_bbox
    source_min_x, source_max_x, source_min_y, source_max_y = source_bbox
    dx = max(target_min_x - source_max_x, source_min_x - target_max_x, 0.0)
    dy = max(target_min_y - source_max_y, source_min_y - target_max_y, 0.0)
    return float(np.hypot(dx, dy))


def _typewell_tail_fingerprint(path: Path, span_ft: float) -> str:
    typewell = pd.read_csv(path, usecols=["TVT", "GR"])
    numeric = typewell.apply(pd.to_numeric, errors="coerce").dropna().sort_values("TVT")
    if len(numeric) == 0:
        raise ValueError(f"Typewell 没有有限 TVT/GR：{path.name}")
    maximum_tvt = float(numeric["TVT"].max())
    tail = numeric.loc[numeric["TVT"] >= maximum_tvt - float(span_ft)]
    digest = hashlib.sha256()
    for tvt, gr in tail.itertuples(index=False, name=None):
        relative_tvt = float(tvt) - maximum_tvt
        digest.update(f"{relative_tvt:.12g},{float(gr):.12g}\n".encode("ascii"))
    return digest.hexdigest()


def _tail_fingerprints(
    well_ids: list[str],
    raw_dir: Path,
    span_ft: float,
) -> dict[str, str]:
    return {
        well_id: _typewell_tail_fingerprint(
            raw_dir / f"{well_id}__typewell.csv",
            span_ft,
        )
        for well_id in well_ids
    }


def _load_pf_cache(cache_dir: Path, well_id: str) -> pd.DataFrame:
    path = cache_dir / f"{well_id}.parquet"
    frame = pd.read_parquet(
        path,
        columns=["well_id", "row_index", "last_visible_tvt", "pf128_scale_8_delta"],
    )
    if not frame["well_id"].astype(str).eq(well_id).all():
        raise ValueError(f"PF 缓存井号错误：{well_id}")
    return frame.sort_values("row_index").reset_index(drop=True)


def _direct_path(cache: pd.DataFrame) -> np.ndarray:
    return (
        cache["last_visible_tvt"].to_numpy(dtype=np.float64)
        + cache["pf128_scale_8_delta"].to_numpy(dtype=np.float64)
    )


def _load_source_records(
    source_ids: list[str],
    raw_dir: Path,
    pf_cache_dir: Path,
    tail_fingerprints: dict[str, str],
) -> dict[str, SourceRecord]:
    records: dict[str, SourceRecord] = {}
    for number, well_id in enumerate(source_ids, start=1):
        horizontal = pd.read_csv(
            raw_dir / f"{well_id}__horizontal_well.csv",
            usecols=["MD", "X", "Y", "TVT", "TVT_input"],
        )
        hidden = _hidden_rows(horizontal)
        pf_cache = _load_pf_cache(pf_cache_dir, well_id)
        row_index = hidden.index.to_numpy(dtype=np.int64)
        if not np.array_equal(row_index, pf_cache["row_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"{well_id} source 隐藏行与 PF 缓存不一致")
        hidden_xy = hidden[["X", "Y"]].to_numpy(dtype=np.float64)
        controls = fit_source_control4(
            hidden["MD"].to_numpy(dtype=np.float64),
            hidden["TVT"].to_numpy(dtype=np.float64),
            _direct_path(pf_cache),
        )
        records[well_id] = SourceRecord(
            well_id=well_id,
            hidden_xy=hidden_xy,
            bbox=_bbox(hidden_xy),
            controls=controls,
            typewell_tail_fingerprint=tail_fingerprints[well_id],
        )
        if number % 100 == 0 or number == len(source_ids):
            print(f"  source profile {number}/{len(source_ids)}", flush=True)
    return records


def _select_neighbors(
    target_xy: np.ndarray,
    target_tail_fingerprint: str,
    sources: dict[str, SourceRecord],
) -> list[tuple[SourceRecord, Any, float]]:
    target_bbox = _bbox(target_xy)
    bbox_candidates = [
        source
        for source in sources.values()
        if _bbox_distance(target_bbox, source.bbox) <= MAX_DISTANCE_FT
    ]
    eligible: list[tuple[SourceRecord, Any, float]] = []
    for source in bbox_candidates:
        geometry = compute_trajectory_geometry(target_xy, source.hidden_xy)
        if not geometry_is_eligible(geometry):
            continue
        weight = neighbor_weight(
            geometry.minimum_distance_ft,
            geometry.azimuth_difference_deg,
            geometry.parallel_overlap_ft,
            source.typewell_tail_fingerprint == target_tail_fingerprint,
        )
        eligible.append((source, geometry, weight))
    eligible.sort(key=lambda item: (-item[2], item[1].minimum_distance_ft, item[0].well_id))
    return eligible[:MAX_NEIGHBORS]


def _generate_one_target(
    well_id: str,
    fold: int,
    raw_dir: Path,
    pf_cache_dir: Path,
    sources: dict[str, SourceRecord],
    shuffled_profile_ids: dict[str, str],
    tail_fingerprints: dict[str, str],
    fingerprint: str,
) -> pd.DataFrame:
    # 验证井这里不读取 TVT；真值只在全部合法缓存落盘后由评分阶段读取。
    horizontal = pd.read_csv(
        raw_dir / f"{well_id}__horizontal_well.csv",
        usecols=["MD", "X", "Y", "TVT_input"],
    )
    hidden = _hidden_rows(horizontal)
    pf_cache = _load_pf_cache(pf_cache_dir, well_id)
    row_index = hidden.index.to_numpy(dtype=np.int64)
    if not np.array_equal(row_index, pf_cache["row_index"].to_numpy(dtype=np.int64)):
        raise ValueError(f"{well_id} 验证隐藏行与 PF 缓存不一致")
    target_xy = hidden[["X", "Y"]].to_numpy(dtype=np.float64)
    selected = _select_neighbors(
        target_xy,
        tail_fingerprints[well_id],
        sources,
    )
    weights = np.asarray([item[2] for item in selected], dtype=np.float64)
    reverse_flags = np.asarray(
        [item[1].reverse_source_controls for item in selected],
        dtype=bool,
    )
    real_controls = [item[0].controls for item in selected]
    shuffled_controls = [
        sources[shuffled_profile_ids[item[0].well_id]].controls
        for item in selected
    ]
    real_aggregate = aggregate_control_profiles(real_controls, weights, reverse_flags)
    shuffled_aggregate = aggregate_control_profiles(shuffled_controls, weights, reverse_flags)
    progress = normalized_progress(hidden["MD"].to_numpy(dtype=np.float64))
    baseline = _direct_path(pf_cache)
    real_correction = interpolate_control4(real_aggregate.shrunk_controls, progress)
    shuffled_correction = interpolate_control4(shuffled_aggregate.shrunk_controls, progress)
    source_ids_json = json.dumps(
        [item[0].well_id for item in selected],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return pd.DataFrame(
        {
            "well_id": well_id,
            "fold": int(fold),
            "row_index": row_index,
            "md": hidden["MD"].to_numpy(dtype=np.float64),
            "baseline_scale8_tvt": baseline,
            "neighbor_corrected_tvt": baseline + real_correction,
            "shuffled_corrected_tvt": baseline + shuffled_correction,
            "support_well_count": len(selected),
            "support_total_weight": real_aggregate.total_weight,
            "eta": real_aggregate.eta,
            "source_well_ids_json": source_ids_json,
            "_cache_fingerprint": fingerprint,
        },
        columns=LEGAL_COLUMNS,
    )


def _valid_cache(path: Path, well_id: str, fold: int, fingerprint: str, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
        return bool(
            list(frame.columns) == LEGAL_COLUMNS
            and len(frame) == int(expected_rows)
            and frame["well_id"].astype(str).eq(well_id).all()
            and frame["fold"].astype(int).eq(int(fold)).all()
            and frame["_cache_fingerprint"].astype(str).eq(fingerprint).all()
        )
    except Exception:
        return False


def _cache_fingerprint(config_path: Path, config: dict[str, Any]) -> str:
    payload = {
        "config_sha256": _file_sha256(config_path),
        "core_sha256": _file_sha256(CLEAN_ROOT / "src" / "p3_n01a_neighbor_residual_audit.py"),
        "experiment_id": config["experiment_id"],
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _generate_fold(
    outer_fold: int,
    target_rows: pd.DataFrame,
    development: pd.DataFrame,
    raw_dir: Path,
    pf_cache_dir: Path,
    tail_fingerprints: dict[str, str],
    output_dir: Path,
    fingerprint: str,
    shuffle_seed: int,
) -> None:
    source_ids = sorted(
        development.loc[development["fold"].astype(int) != int(outer_fold), "well_id"].astype(str)
    )
    target_ids = target_rows["well_id"].astype(str).tolist()
    if set(source_ids).intersection(target_ids):
        raise RuntimeError(f"fold {outer_fold} source 泄漏验证井")
    print(
        f"fold {outer_fold}: validation={len(target_ids)}, outer_train_source={len(source_ids)}",
        flush=True,
    )
    sources = _load_source_records(
        source_ids,
        raw_dir,
        pf_cache_dir,
        tail_fingerprints,
    )
    # 两折都使用同一个预注册 seed；source 集合不同会自然产生各自的固定排列。
    shuffled_profile_ids = build_fixed_shuffle(source_ids, seed=shuffle_seed)
    cache_dir = output_dir / "legal_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(target_rows.itertuples(index=False), start=1):
        well_id = str(row.well_id)
        cache_path = cache_dir / f"{well_id}.parquet"
        if _valid_cache(cache_path, well_id, outer_fold, fingerprint, int(row.hidden_rows)):
            status = "reuse"
        else:
            output = _generate_one_target(
                well_id=well_id,
                fold=outer_fold,
                raw_dir=raw_dir,
                pf_cache_dir=pf_cache_dir,
                sources=sources,
                shuffled_profile_ids=shuffled_profile_ids,
                tail_fingerprints=tail_fingerprints,
                fingerprint=fingerprint,
            )
            output.to_parquet(cache_path, index=False, compression="zstd")
            status = "generated"
        if number % 10 == 0 or number == len(target_rows):
            print(f"  fold {outer_fold} validation {number}/{len(target_rows)} ({status})", flush=True)


def _rmse_from_sse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows)))


def _score_targets(
    target_rows: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    per_well_rows: list[dict[str, Any]] = []
    oracle_dir = output_dir / "oracle_cache"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for row in target_rows.itertuples(index=False):
        well_id = str(row.well_id)
        legal = pd.read_parquet(output_dir / "legal_cache" / f"{well_id}.parquet")
        horizontal = pd.read_csv(
            raw_dir / f"{well_id}__horizontal_well.csv",
            usecols=["MD", "TVT", "TVT_input"],
        )
        hidden = _hidden_rows(horizontal)
        row_index = hidden.index.to_numpy(dtype=np.int64)
        if not np.array_equal(row_index, legal["row_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"{well_id} 评分行与合法缓存不一致")
        target = hidden["TVT"].to_numpy(dtype=np.float64)
        baseline = legal["baseline_scale8_tvt"].to_numpy(dtype=np.float64)
        corrected = legal["neighbor_corrected_tvt"].to_numpy(dtype=np.float64)
        shuffled = legal["shuffled_corrected_tvt"].to_numpy(dtype=np.float64)
        own_controls = fit_source_control4(
            hidden["MD"].to_numpy(dtype=np.float64),
            target,
            baseline,
        )
        progress = normalized_progress(hidden["MD"].to_numpy(dtype=np.float64))
        oracle = baseline + interpolate_control4(own_controls, progress)
        pd.DataFrame(
            {
                "well_id": well_id,
                "row_index": row_index,
                "target_tvt": target,
                "own_control4_oracle_tvt": oracle,
            }
        ).to_parquet(oracle_dir / f"{well_id}.parquet", index=False, compression="zstd")
        errors = {
            "baseline": target - baseline,
            "neighbor": target - corrected,
            "shuffled": target - shuffled,
            "oracle": target - oracle,
        }
        values: dict[str, Any] = {
            "well_id": well_id,
            "fold": int(row.fold),
            "rows": int(len(target)),
            "support_well_count": int(legal["support_well_count"].iloc[0]),
            "support_total_weight": float(legal["support_total_weight"].iloc[0]),
            "eta": float(legal["eta"].iloc[0]),
        }
        for name, error in errors.items():
            sse = float(error @ error)
            values[f"{name}_sse"] = sse
            values[f"{name}_rmse"] = _rmse_from_sse(sse, len(target))
        per_well_rows.append(values)
    per_well = pd.DataFrame(per_well_rows).sort_values(["fold", "well_id"]).reset_index(drop=True)

    fold_rows: list[dict[str, Any]] = []
    for fold, subset in per_well.groupby("fold", sort=True):
        rows = int(subset["rows"].sum())
        metrics = {name: _rmse_from_sse(float(subset[f"{name}_sse"].sum()), rows) for name in ["baseline", "neighbor", "shuffled", "oracle"]}
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(len(subset)),
                "rows": rows,
                "supported_wells": int((subset["support_well_count"] > 0).sum()),
                "supported_well_fraction": float((subset["support_well_count"] > 0).mean()),
                **{f"{name}_rmse": value for name, value in metrics.items()},
                "improvement_vs_baseline_ft": metrics["baseline"] - metrics["neighbor"],
                "improvement_vs_shuffled_ft": metrics["shuffled"] - metrics["neighbor"],
            }
        )
    per_fold = pd.DataFrame(fold_rows)
    total_rows = int(per_well["rows"].sum())
    pooled = {
        name: _rmse_from_sse(float(per_well[f"{name}_sse"].sum()), total_rows)
        for name in ["baseline", "neighbor", "shuffled", "oracle"]
    }
    supported_fraction = float((per_well["support_well_count"] > 0).mean())
    gates = {
        "support_coverage": bool(supported_fraction >= 0.50),
        "real_vs_shuffled_pooled": bool(pooled["shuffled"] - pooled["neighbor"] >= 0.20),
        "real_vs_shuffled_each_fold": bool((per_fold["improvement_vs_shuffled_ft"] > 0.0).all()),
        "real_vs_baseline_pooled": bool(pooled["baseline"] - pooled["neighbor"] >= 0.10),
        "single_fold_degradation": bool((per_fold["improvement_vs_baseline_ft"] >= -0.25).all()),
    }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "development_wells_tested": int(len(per_well)),
        "development_rows_tested": total_rows,
        "shadow_overlap_wells": 0,
        "supported_wells": int((per_well["support_well_count"] > 0).sum()),
        "supported_well_fraction": supported_fraction,
        "pooled_rmse": pooled,
        "improvement_vs_baseline_ft": float(pooled["baseline"] - pooled["neighbor"]),
        "improvement_vs_shuffled_ft": float(pooled["shuffled"] - pooled["neighbor"]),
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "model_training": False,
    }
    return per_well, per_fold, summary


def _conclusion(summary: dict[str, Any]) -> str:
    pooled = summary["pooled_rmse"]
    passed = bool(summary["all_gates_passed"])
    return (
        f"数据直接证明的事实：{summary['development_wells_tested']} 口开发井中，"
        f"{summary['supported_well_fraction']:.1%} 有至少一口合格邻井。原 scale8、真实邻井修正、"
        f"打乱 profile 和自身 control4 oracle 的 pooled RMSE 分别为 "
        f"{pooled['baseline']:.6f}、{pooled['neighbor']:.6f}、{pooled['shuffled']:.6f}、{pooled['oracle']:.6f} ft。\n\n"
        f"基于事实的合理推断：当前固定实现{'通过' if passed else '没有通过'}预注册筛查门槛。\n\n"
        "仍然没有验证的猜测：其他距离尺度、控制点数量、局部位置权重或学习型 OOF 权重是否有效。\n\n"
        "当前实验只能否定的具体实现：2500 ft/45 度/500 ft 门槛、最多 8 井、当前固定权重和 eta 收缩的直接四控制点迁移。\n\n"
        "下一步最便宜的验证：按路线图门槛决定停止 N01a 或再进入正式 OOF 残差学习；自身 oracle 不用于晋级。\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P3-N01a 邻井残差迁移诊断")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["smoke", "fold01"], default="fold01")
    parser.add_argument("--smoke-wells", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    config = _read_json(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("实验 ID 与 runner 不一致")
    development, shadow = _development_registry(config)
    raw_dir = _workspace_path(str(config["raw_train_dir"]))
    pf_cache_dir = _workspace_path(str(config["pf_legal_cache_dir"]))
    fingerprint = _cache_fingerprint(args.config, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "config.json", config)

    fold_targets = development.loc[development["fold"].astype(int).isin([0, 1])].copy()
    if len(fold_targets) != int(config["fold01_validation_wells"]):
        raise ValueError(f"folds 0-1 开发井数量错误：{len(fold_targets)}")
    if args.mode == "smoke":
        fold_targets = fold_targets.sort_values(["fold", "well_id"]).head(int(args.smoke_wells)).copy()
    if set(fold_targets["well_id"]).intersection(shadow):
        raise RuntimeError("目标列表含 shadow 井")

    print(
        f"N01a mode={args.mode}, validation={len(fold_targets)}, dev={len(development)}, shadow_excluded={len(shadow)}",
        flush=True,
    )
    all_development_ids = development["well_id"].astype(str).tolist()
    tail_fingerprints = _tail_fingerprints(
        all_development_ids,
        raw_dir,
        float(config["typewell_tail_span_ft"]),
    )
    for outer_fold, target_rows in fold_targets.groupby("fold", sort=True):
        _generate_fold(
            outer_fold=int(outer_fold),
            target_rows=target_rows,
            development=development,
            raw_dir=raw_dir,
            pf_cache_dir=pf_cache_dir,
            tail_fingerprints=tail_fingerprints,
            output_dir=args.output_dir,
            fingerprint=fingerprint,
            shuffle_seed=int(config["shuffle_seed"]),
        )

    per_well, per_fold, summary = _score_targets(
        fold_targets,
        raw_dir,
        args.output_dir,
        args.mode,
    )
    suffix = args.mode
    per_well.to_csv(args.output_dir / f"per_well_{suffix}.csv", index=False, lineterminator="\n")
    per_fold.to_csv(args.output_dir / f"per_fold_{suffix}.csv", index=False, lineterminator="\n")
    _write_json(args.output_dir / f"metrics_{suffix}.json", summary)
    runtime = {
        "mode": args.mode,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_wells": int(len(fold_targets)),
        "cache_fingerprint": fingerprint,
        "resume": "rerun same command; matching per-well legal caches are reused",
    }
    _write_json(args.output_dir / f"runtime_{suffix}.json", runtime)
    (args.output_dir / f"conclusion_{suffix}.md").write_text(
        _conclusion(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
