"""按井生成 F05b 的 10 个归一化 PF 内部统计，并合并为只读 CV 缓存。"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.build_f05a_deterministic_candidate_cache import (  # noqa: E402
    _json_fingerprint,
    _load_existing_runtime,
    _write_json_atomic,
    _write_runtime_table,
    build_base_lineage,
    build_well_cache,
    build_well_lineage,
    file_sha256,
    is_reusable_well_cache,
    load_registry,
    merge_per_well_caches,
    per_well_cache_path,
)
from src.f05b_internal_stats import (  # noqa: E402
    F05B_FEATURE_COLUMNS,
    build_internal_stats_features,
)


EXPERIMENT_ID = "F05b_internal_stats_cache_v1"
F05B_COLUMNS = list(F05B_FEATURE_COLUMNS)
FIXED_SEED = 42
DEFAULT_WORKERS = 4
EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
EXPECTED_REGISTRY_SHA256 = (
    "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab"
)

DEFAULT_RAW_TRAIN_DIR = PROJECT_ROOT / "input" / "data" / "raw" / "train"
DEFAULT_REGISTRY_PATH = (
    CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv"
)
DEFAULT_GENERATOR_PATH = CLEAN_ROOT / "src" / "f05b_internal_stats.py"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "f05b_internal_stats_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID


def build_f05b_well_cache(
    well_id: str,
    horizontal_path: Path,
    typewell_path: Path,
    cache_path: Path,
    cache_fingerprint: str,
    expected_hidden_rows: int,
    rebuild: bool = False,
) -> dict:
    """生成一口井；底层固定只读取 MD/Z/GR/TVT_input 与 Typewell TVT/GR。"""

    return build_well_cache(
        well_id=well_id,
        horizontal_path=horizontal_path,
        typewell_path=typewell_path,
        cache_path=cache_path,
        cache_fingerprint=cache_fingerprint,
        expected_hidden_rows=expected_hidden_rows,
        feature_columns=F05B_COLUMNS,
        seed=FIXED_SEED,
        rebuild=rebuild,
        generator=build_internal_stats_features,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 F05b 归一化 PF 内部统计缓存")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_RAW_TRAIN_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def _report_progress(done: int, total: int, generated: int, reused: int) -> None:
    if done % 25 == 0 or done == total:
        print(
            f"F05b 缓存进度：{done}/{total} 井，生成={generated}，复用={reused}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers 必须至少为 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须至少为 1")
    if len(F05B_COLUMNS) != 10:
        raise ValueError("F05b 必须恰好输出路线图规定的 10 个特征")

    raw_train_dir = args.train_dir.resolve()
    registry_path = args.registry.resolve()
    config_path = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    per_well_dir = artifact_dir / "per_well"
    final_cache_path = artifact_dir / "internal_stats_cache.parquet"
    metadata_path = artifact_dir / "meta.json"
    runtime_path = artifact_dir / "per_well_runtime.csv"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    per_well_dir.mkdir(parents=True, exist_ok=True)

    registry_hash = file_sha256(registry_path)
    if registry_hash != EXPECTED_REGISTRY_SHA256:
        raise ValueError("固定 fold 注册表 SHA-256 不一致")
    registry = load_registry(registry_path)
    if len(registry) != EXPECTED_WELLS:
        raise ValueError(f"固定注册表井数错误：{len(registry)} != {EXPECTED_WELLS}")
    if int(registry["hidden_rows"].sum()) != EXPECTED_ROWS:
        raise ValueError("固定注册表隐藏行数错误")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("internal_stat_feature_columns") != F05B_COLUMNS:
        raise ValueError("配置中的 F05b 特征列或顺序与生成器不一致")

    base_lineage = build_base_lineage(
        DEFAULT_GENERATOR_PATH,
        config_path,
        registry_path,
        raw_train_dir,
        seed=FIXED_SEED,
    )
    expected_fingerprints: dict[str, str] = {}
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        lineage = build_well_lineage(
            base_lineage,
            raw_train_dir / f"{well_id}__horizontal_well.csv",
            raw_train_dir / f"{well_id}__typewell.csv",
        )
        expected_fingerprints[well_id] = str(lineage["fingerprint"])

    selected = registry
    if args.limit is not None:
        selected = registry.iloc[: min(int(args.limit), len(registry))]
    registry_order = registry["well_id"].astype(str).tolist()
    runtime_by_well = _load_existing_runtime(runtime_path)

    print(
        f"开始 {EXPERIMENT_ID}：{len(selected)}/{len(registry)} 井，"
        f"workers={args.workers}，rebuild={bool(args.rebuild)}",
        flush=True,
    )
    pending: list[tuple[str, Path, Path, Path, str, int]] = []
    done = generated_count = reused_count = 0
    for row in selected.itertuples(index=False):
        well_id = str(row.well_id)
        cache_path = per_well_cache_path(per_well_dir, well_id)
        fingerprint = expected_fingerprints[well_id]
        hidden_rows = int(row.hidden_rows)
        if not args.rebuild and is_reusable_well_cache(
            cache_path,
            well_id,
            fingerprint,
            hidden_rows,
            F05B_COLUMNS,
        ):
            runtime_by_well[well_id] = {
                "well_id": well_id,
                "status": "reused",
                "rows": hidden_rows,
                "seconds": 0.0,
                "cache_path": str(cache_path.resolve()),
                "cache_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            done += 1
            reused_count += 1
            _report_progress(done, len(selected), generated_count, reused_count)
        else:
            pending.append(
                (
                    well_id,
                    raw_train_dir / f"{well_id}__horizontal_well.csv",
                    raw_train_dir / f"{well_id}__typewell.csv",
                    cache_path,
                    fingerprint,
                    hidden_rows,
                )
            )

    if pending:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    build_f05b_well_cache,
                    well_id,
                    horizontal_path,
                    typewell_path,
                    cache_path,
                    fingerprint,
                    hidden_rows,
                    bool(args.rebuild),
                ): well_id
                for (
                    well_id,
                    horizontal_path,
                    typewell_path,
                    cache_path,
                    fingerprint,
                    hidden_rows,
                ) in pending
            }
            for future in as_completed(futures):
                well_id = futures[future]
                runtime = future.result()
                runtime_by_well[well_id] = runtime
                done += 1
                if runtime["status"] == "generated":
                    generated_count += 1
                else:
                    reused_count += 1
                _write_runtime_table(runtime_path, runtime_by_well, registry_order)
                _report_progress(done, len(selected), generated_count, reused_count)

    _write_runtime_table(runtime_path, runtime_by_well, registry_order)
    missing_or_stale = []
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        if not is_reusable_well_cache(
            per_well_cache_path(per_well_dir, well_id),
            well_id,
            expected_fingerprints[well_id],
            int(row.hidden_rows),
            F05B_COLUMNS,
        ):
            missing_or_stale.append(well_id)
    if missing_or_stale:
        print(
            f"仍缺少或过期 {len(missing_or_stale)} 口井；已保留单井缓存，暂不合并。",
            flush=True,
        )
        return

    summary = merge_per_well_caches(
        registry=registry,
        cache_dir=per_well_dir,
        expected_fingerprints=expected_fingerprints,
        output_path=final_cache_path,
        feature_columns=F05B_COLUMNS,
        expected_wells=EXPECTED_WELLS,
        expected_rows=EXPECTED_ROWS,
    )
    ordered_fingerprints = [
        {"well_id": well_id, "fingerprint": expected_fingerprints[well_id]}
        for well_id in registry_order
    ]
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "wells": int(summary["wells"]),
        "rows": int(summary["rows"]),
        "keys_unique": bool(summary["keys_unique"]),
        "registry_order_preserved": True,
        "seed": FIXED_SEED,
        "workers": int(args.workers),
        "feature_columns": F05B_COLUMNS,
        "feature_count": len(F05B_COLUMNS),
        "cache_path": str(final_cache_path),
        "cache_sha256": file_sha256(final_cache_path),
        "per_well_cache_dir": str(per_well_dir),
        "base_lineage": base_lineage,
        "cache_fingerprint": _json_fingerprint(
            {
                "base_lineage": base_lineage,
                "ordered_well_fingerprints": ordered_fingerprints,
            }
        ),
    }
    _write_json_atomic(metadata_path, metadata)
    print(
        f"F05b 完整缓存完成：{summary['wells']} 井，{summary['rows']:,} 行",
        flush=True,
    )


if __name__ == "__main__":
    main()
