"""运行 P2-D01：PF 中心分块多尺度 GR 可辨识性审计。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import read_json, write_json  # noqa: E402
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.p2_d01_pf_centered_gr import (  # noqa: E402
    attach_oracle_diagnostics,
    build_block_score_landscape,
)


EXPERIMENT_ID = "P2_D01_pf_centered_multiscale_gr_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p2_d01_pf_centered_multiscale_gr_v1.json"
SUPPORTED_MODES = ["smoke", "all"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析三井 smoke 或全量诊断模式。"""

    parser = argparse.ArgumentParser(description="运行 P2-D01 PF 中心 GR 审计")
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def restore_pf_center(
    tvt_input: np.ndarray,
    candidate_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """按精确 row_index 用末个可见 TVT 加 pf_ancc_delta 还原隐藏 PF 路径。"""

    tvt_input = np.asarray(tvt_input, dtype=np.float64)
    visible_positions = np.flatnonzero(np.isfinite(tvt_input))
    hidden_positions = np.flatnonzero(~np.isfinite(tvt_input))
    if len(visible_positions) == 0 or len(hidden_positions) == 0:
        raise ValueError("水平井必须同时包含可见前缀和隐藏后缀")
    visible_end = int(visible_positions[-1])
    if not np.array_equal(visible_positions, np.arange(visible_end + 1)):
        raise ValueError("TVT_input 可见区不是连续前缀")
    if not np.array_equal(
        hidden_positions,
        np.arange(visible_end + 1, len(tvt_input)),
    ):
        raise ValueError("TVT_input 隐藏区不是连续后缀")
    required_columns = {"row_index", "pf_ancc_delta"}
    missing_columns = required_columns - set(candidate_rows.columns)
    if missing_columns:
        raise ValueError(f"PF 候选缓存缺少列：{sorted(missing_columns)}")
    cached_positions = candidate_rows["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(cached_positions, hidden_positions):
        raise ValueError("PF 候选缓存 row_index 与自然隐藏行不一致")
    pf_delta = candidate_rows["pf_ancc_delta"].to_numpy(dtype=np.float64)
    if not np.isfinite(pf_delta).all():
        raise ValueError("pf_ancc_delta 含 NaN 或 Inf")
    last_visible_tvt = float(tvt_input[visible_end])
    return hidden_positions, last_visible_tvt + pf_delta


def _validate_config(config: dict[str, object]) -> None:
    """拒绝与预注册 D01 网格、尺度、控制或真值隔离不一致的配置。"""

    expected_offsets = np.arange(-40.0, 42.0, 2.0).tolist()
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("P2-D01 experiment_id 不正确")
    if config.get("center_path") != "last_visible_tvt_plus_pf_ancc_delta":
        raise ValueError("P2-D01 中心路径发生变化")
    if config.get("offset_grid_ft") != expected_offsets:
        raise ValueError("P2-D01 offset 网格发生变化")
    if config.get("block_widths_ft") != [50.0, 100.0]:
        raise ValueError("P2-D01 block width 发生变化")
    if config.get("smoothing_widths_ft") != [5.0, 11.0, 21.0, 51.0, 101.0]:
        raise ValueError("P2-D01 GR 平滑尺度发生变化")
    if int(config.get("minimum_valid_pairs", -1)) != 30:
        raise ValueError("P2-D01 minimum_valid_pairs 发生变化")
    if config.get("hidden_tvt_access") != "oracle_attachment_only":
        raise ValueError("隐藏 TVT 没有被限制到 oracle 附加步骤")
    if config.get("supported_modes") != SUPPORTED_MODES:
        raise ValueError("P2-D01 supported_modes 与 runner 不一致")


def _fingerprint(
    config: dict[str, object],
    config_path: Path,
    registry_path: Path,
    candidate_metadata_path: Path,
) -> str:
    """用配置、fold、冻结 PF 来源和两份执行代码构造可恢复指纹。"""

    payload = {
        "config": config,
        "config_sha256": file_sha256(config_path),
        "registry_sha256": file_sha256(registry_path),
        "candidate_metadata_sha256": file_sha256(candidate_metadata_path),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "scorer_sha256": file_sha256(
            CLEAN_ROOT / "src" / "p2_d01_pf_centered_gr.py"
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_control(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    true_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: dict[str, object],
    control: str,
) -> pd.DataFrame:
    """对一个 hidden/prefix 控制按两个固定 block width 生成 legal+oracle 表。"""

    output_tables: list[pd.DataFrame] = []
    for block_width_ft in config["block_widths_ft"]:
        landscape = build_block_score_landscape(
            md=md,
            horizontal_gr=horizontal_gr,
            center_tvt=center_tvt,
            typewell_tvt=typewell_tvt,
            typewell_gr=typewell_gr,
            offsets_ft=np.asarray(config["offset_grid_ft"], dtype=np.float64),
            block_width_ft=float(block_width_ft),
            smoothing_widths_ft=[
                float(width) for width in config["smoothing_widths_ft"]
            ],
            minimum_valid_pairs=int(config["minimum_valid_pairs"]),
            second_peak_minimum_distance_ft=float(
                config["second_peak_minimum_distance_ft"]
            ),
        )
        diagnostic = attach_oracle_diagnostics(
            landscape,
            true_tvt=true_tvt,
            center_tvt=center_tvt,
        )
        diagnostic.insert(0, "control", control)
        output_tables.append(diagnostic)
    return pd.concat(output_tables, ignore_index=True)


def process_one_well(task: dict[str, object]) -> dict[str, object]:
    """读取一口井的合法输入和独立 oracle TVT，保存可恢复逐井诊断。"""

    well_id = str(task["well_id"])
    fold_id = int(task["fold"])
    config = task["config"]
    if not isinstance(config, dict):
        raise ValueError("worker config 必须是字典")
    raw_train_dir = Path(str(task["raw_train_dir"]))
    candidate_per_well_dir = Path(str(task["candidate_per_well_dir"]))
    output_path = Path(str(task["output_path"]))
    runtime_path = Path(str(task["runtime_path"]))
    fingerprint = str(task["fingerprint"])
    if output_path.is_file() and runtime_path.is_file():
        previous_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if previous_runtime.get("fingerprint") == fingerprint:
            return previous_runtime

    start = time.perf_counter()
    horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
    candidate_path = candidate_per_well_dir / f"{well_id}.parquet"
    legal_horizontal = pd.read_csv(
        horizontal_path,
        usecols=["MD", "GR", "TVT_input"],
    )
    candidate_rows = pd.read_parquet(candidate_path)
    hidden_positions, pf_center_tvt = restore_pf_center(
        legal_horizontal["TVT_input"].to_numpy(dtype=np.float64),
        candidate_rows,
    )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])

    # 隐藏真值只在 legal score 输入都准备完以后单独读取并用于 attach oracle。
    oracle_tvt = pd.read_csv(horizontal_path, usecols=["TVT"])["TVT"].to_numpy(
        dtype=np.float64
    )
    md = legal_horizontal["MD"].to_numpy(dtype=np.float64)
    gr = legal_horizontal["GR"].to_numpy(dtype=np.float64)
    tvt_input = legal_horizontal["TVT_input"].to_numpy(dtype=np.float64)
    visible_positions = np.flatnonzero(np.isfinite(tvt_input))
    hidden_md = md[hidden_positions]
    hidden_gr = gr[hidden_positions]
    hidden_true_tvt = oracle_tvt[hidden_positions]
    typewell_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    typewell_gr = typewell["GR"].to_numpy(dtype=np.float64)

    real_table = _score_control(
        hidden_md,
        hidden_gr,
        pf_center_tvt,
        hidden_true_tvt,
        typewell_tvt,
        typewell_gr,
        config,
        control="hidden_real",
    )
    shifted_gr = np.roll(hidden_gr, max(len(hidden_gr) // 2, 1))
    negative_table = _score_control(
        hidden_md,
        shifted_gr,
        pf_center_tvt,
        hidden_true_tvt,
        typewell_tvt,
        typewell_gr,
        config,
        control="hidden_circular_shift",
    )
    prefix_table = _score_control(
        md[visible_positions],
        gr[visible_positions],
        tvt_input[visible_positions],
        tvt_input[visible_positions],
        typewell_tvt,
        typewell_gr,
        config,
        control="visible_prefix_zero",
    )
    combined = pd.concat(
        [real_table, negative_table, prefix_table],
        ignore_index=True,
    )
    combined.insert(0, "fold", fold_id)
    combined.insert(0, "well_id", well_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False, compression="zstd")

    runtime = {
        "well_id": well_id,
        "fold": fold_id,
        "fingerprint": fingerprint,
        "hidden_rows": int(len(hidden_positions)),
        "diagnostic_rows": int(len(combined)),
        "seconds": float(time.perf_counter() - start),
        "output_path": str(output_path.resolve()),
    }
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime


def _control_metrics(table: pd.DataFrame) -> dict[str, object]:
    """汇总一个 control、一个 block width 的 ensemble 覆盖和排名。"""

    ensemble = table.loc[table["scale_label"].eq("ensemble")].copy()
    valid_rank = ensemble["true_offset_rank"].notna()
    valid_error = ensemble["best_offset_abs_error_ft"].notna()
    return {
        "blocks": int(len(ensemble)),
        "valid_rank_blocks": int(valid_rank.sum()),
        "valid_rank_fraction": float(valid_rank.mean()) if len(ensemble) else np.nan,
        "true_offset_grid_coverage": float(ensemble["true_offset_in_grid"].mean())
        if len(ensemble)
        else np.nan,
        "true_offset_top5_rate": float(
            ensemble.loc[valid_rank, "true_offset_top5"].mean()
        )
        if valid_rank.any()
        else np.nan,
        "median_best_offset_abs_error_ft": float(
            ensemble.loc[valid_error, "best_offset_abs_error_ft"].median()
        )
        if valid_error.any()
        else np.nan,
        "median_best_ncc": float(ensemble["best_ncc"].median()),
        "median_peak_gap": float(ensemble["peak_gap"].median()),
        "median_scale_best_offset_iqr_ft": float(
            ensemble["scale_best_offset_iqr_ft"].median()
        ),
    }


def build_summary(
    all_diagnostics: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    """构造总体/逐折指标和五条预注册证据的判断。"""

    summaries: list[dict[str, object]] = []
    for (control, block_width), group in all_diagnostics.groupby(
        ["control", "block_width_ft"],
        sort=True,
    ):
        summaries.append(
            {
                "control": str(control),
                "block_width_ft": float(block_width),
                **_control_metrics(group),
            }
        )
    fold_summaries: list[dict[str, object]] = []
    for (fold_id, control, block_width), group in all_diagnostics.groupby(
        ["fold", "control", "block_width_ft"],
        sort=True,
    ):
        fold_summaries.append(
            {
                "fold": int(fold_id),
                "control": str(control),
                "block_width_ft": float(block_width),
                **_control_metrics(group),
            }
        )

    evidence_by_width: list[dict[str, object]] = []
    conditions = config["success_conditions"]
    for block_width in config["block_widths_ft"]:
        real = all_diagnostics.loc[
            all_diagnostics["control"].eq("hidden_real")
            & all_diagnostics["block_width_ft"].eq(float(block_width))
            & all_diagnostics["scale_label"].eq("ensemble")
        ].copy()
        negative = all_diagnostics.loc[
            all_diagnostics["control"].eq("hidden_circular_shift")
            & all_diagnostics["block_width_ft"].eq(float(block_width))
            & all_diagnostics["scale_label"].eq("ensemble")
        ].copy()
        prefix = all_diagnostics.loc[
            all_diagnostics["control"].eq("visible_prefix_zero")
            & all_diagnostics["block_width_ft"].eq(float(block_width))
            & all_diagnostics["scale_label"].eq("ensemble")
        ].copy()
        real_valid = real["true_offset_rank"].notna()
        negative_valid = negative["true_offset_rank"].notna()
        prefix_valid = prefix["true_offset_rank"].notna()
        real_top5 = float(real.loc[real_valid, "true_offset_top5"].mean())
        negative_top5 = float(
            negative.loc[negative_valid, "true_offset_top5"].mean()
        )
        prefix_top5 = float(prefix.loc[prefix_valid, "true_offset_top5"].mean())
        real_error = float(real["best_offset_abs_error_ft"].median())
        negative_error = float(negative["best_offset_abs_error_ft"].median())
        coverage = float(real["true_offset_in_grid"].mean())

        confidence_source = real.loc[
            real["peak_gap"].notna()
            & real["scale_best_offset_iqr_ft"].notna()
            & real["best_offset_abs_error_ft"].notna()
        ].copy()
        peak_q25 = float(confidence_source["peak_gap"].quantile(0.25))
        peak_q75 = float(confidence_source["peak_gap"].quantile(0.75))
        scale_q50 = float(
            confidence_source["scale_best_offset_iqr_ft"].quantile(0.50)
        )
        scale_q75 = float(
            confidence_source["scale_best_offset_iqr_ft"].quantile(0.75)
        )
        high_confidence = confidence_source.loc[
            confidence_source["peak_gap"].ge(peak_q75)
            & confidence_source["scale_best_offset_iqr_ft"].le(scale_q50)
        ]
        low_confidence = confidence_source.loc[
            confidence_source["peak_gap"].le(peak_q25)
            | confidence_source["scale_best_offset_iqr_ft"].ge(scale_q75)
        ]
        high_error = float(high_confidence["best_offset_abs_error_ft"].median())
        low_error = float(low_confidence["best_offset_abs_error_ft"].median())

        checks = {
            "grid_coverage_pass": coverage
            >= float(conditions["minimum_true_offset_grid_coverage"]),
            "prefix_top5_pass": prefix_top5
            >= float(conditions["minimum_prefix_top5_rate"]),
            "top5_gain_pass": (real_top5 - negative_top5)
            >= float(conditions["minimum_top5_gain_vs_negative"]),
            "median_error_gain_pass": (negative_error - real_error)
            >= float(conditions["minimum_median_error_gain_vs_negative_ft"]),
            "confidence_order_pass": high_error < low_error,
        }
        evidence_by_width.append(
            {
                "block_width_ft": float(block_width),
                "coverage": coverage,
                "prefix_top5_rate": prefix_top5,
                "real_top5_rate": real_top5,
                "negative_top5_rate": negative_top5,
                "top5_gain": real_top5 - negative_top5,
                "real_median_abs_error_ft": real_error,
                "negative_median_abs_error_ft": negative_error,
                "median_error_gain_ft": negative_error - real_error,
                "high_confidence_blocks": int(len(high_confidence)),
                "low_confidence_blocks": int(len(low_confidence)),
                "high_confidence_median_error_ft": high_error,
                "low_confidence_median_error_ft": low_error,
                "checks": checks,
                "all_checks_pass": bool(all(checks.values())),
            }
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "overall": summaries,
        "folds": fold_summaries,
        "evidence_by_block_width": evidence_by_width,
        "gr_scorer_supported_for_future_path_work": bool(
            any(item["all_checks_pass"] for item in evidence_by_width)
        ),
    }


def _build_per_well(
    diagnostics: pd.DataFrame,
    baseline_per_well_path: Path,
) -> pd.DataFrame:
    """汇总 ensemble 的逐井/块宽/control 指标并连接 P2B00 井误差。"""

    ensemble = diagnostics.loc[diagnostics["scale_label"].eq("ensemble")].copy()
    rows: list[dict[str, object]] = []
    for (well_id, fold_id, block_width, control), group in ensemble.groupby(
        ["well_id", "fold", "block_width_ft", "control"],
        sort=True,
    ):
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold_id),
                "block_width_ft": float(block_width),
                "control": str(control),
                **_control_metrics(group),
            }
        )
    per_well = pd.DataFrame(rows)
    baseline = pd.read_csv(baseline_per_well_path, dtype={"well_id": str})[
        ["well_id", "prediction_rmse", "baseline_rmse"]
    ]
    return per_well.merge(baseline, on="well_id", how="left", validate="many_to_one")


def _aggregate_outputs(
    selected_registry: pd.DataFrame,
    artifact_dir: Path,
    config: dict[str, object],
) -> dict[str, object]:
    """合并逐井文件，写三类控制、逐井表和最终 summary。"""

    tables = [
        pd.read_parquet(artifact_dir / "per_well" / f"{well_id}.parquet")
        for well_id in selected_registry["well_id"].astype(str)
    ]
    diagnostics = pd.concat(tables, ignore_index=True)
    real = diagnostics.loc[diagnostics["control"].eq("hidden_real")]
    negative = diagnostics.loc[
        diagnostics["control"].eq("hidden_circular_shift")
    ]
    prefix = diagnostics.loc[diagnostics["control"].eq("visible_prefix_zero")]
    real.to_parquet(
        artifact_dir / "block_diagnostics.parquet",
        index=False,
        compression="zstd",
    )
    negative.to_parquet(
        artifact_dir / "negative_block_diagnostics.parquet",
        index=False,
        compression="zstd",
    )
    prefix.to_parquet(
        artifact_dir / "prefix_positive_control.parquet",
        index=False,
        compression="zstd",
    )
    baseline_path = CLEAN_ROOT / str(config["baseline_per_well_metrics"])
    per_well = _build_per_well(diagnostics, baseline_path)
    per_well.to_csv(artifact_dir / "per_well.csv", index=False)
    summary = build_summary(diagnostics, config)
    write_json(artifact_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """执行三井 smoke 或可恢复的 773 井全量可辨识性审计。"""

    args = parse_args(argv)
    config_path = args.config.resolve()
    config = read_json(config_path)
    _validate_config(config)
    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    if file_sha256(registry_path) != config["fold_registry_sha256"]:
        raise ValueError("P2-D01 fold 注册表 SHA-256 不匹配")
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(config["expected_wells"]),
        expected_rows=int(config["expected_hidden_rows"]),
    )
    candidate_metadata_path = CLEAN_ROOT / str(config["candidate_cache_metadata"])
    candidate_metadata = read_json(candidate_metadata_path)
    if candidate_metadata.get("candidate_cache_sha256") != config[
        "candidate_cache_sha256"
    ]:
        raise ValueError("冻结 PF 候选缓存 SHA-256 不匹配")
    fingerprint = _fingerprint(
        config,
        config_path,
        registry_path,
        candidate_metadata_path,
    )
    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "config.json", config)
    write_json(
        artifact_dir / "lineage.json",
        {
            "fingerprint": fingerprint,
            "legal_horizontal_columns": ["MD", "GR", "TVT_input"],
            "typewell_columns": ["TVT", "GR"],
            "center_source": "frozen pf_ancc_delta",
            "hidden_tvt_access": "separate oracle attachment after legal score",
            "candidate_cache_metadata": candidate_metadata,
            "fold_registry_sha256": config["fold_registry_sha256"],
        },
    )

    selected_registry = (
        registry.iloc[: int(config["smoke_wells"])].copy()
        if args.mode == "smoke"
        else registry.copy()
    )
    raw_train_dir = (CLEAN_ROOT / str(config["raw_train_dir"])).resolve()
    candidate_per_well_dir = (
        CLEAN_ROOT / str(config["candidate_per_well_dir"])
    ).resolve()
    tasks: list[dict[str, object]] = []
    for registry_row in selected_registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        tasks.append(
            {
                "well_id": well_id,
                "fold": int(registry_row.fold),
                "config": config,
                "raw_train_dir": str(raw_train_dir),
                "candidate_per_well_dir": str(candidate_per_well_dir),
                "output_path": str(artifact_dir / "per_well" / f"{well_id}.parquet"),
                "runtime_path": str(
                    artifact_dir / "per_well_runtime" / f"{well_id}.json"
                ),
                "fingerprint": fingerprint,
            }
        )

    run_start = time.perf_counter()
    runtimes: list[dict[str, object]] = []
    workers = 1 if args.mode == "smoke" else int(config["workers"])
    if workers == 1:
        for task_number, task in enumerate(tasks, start=1):
            runtimes.append(process_one_well(task))
            print(
                f"P2-D01 进度：{task_number}/{len(tasks)} 井，"
                f"当前={task['well_id']}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_well = {
                pool.submit(process_one_well, task): str(task["well_id"])
                for task in tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_well):
                runtimes.append(future.result())
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(
                        f"P2-D01 进度：{completed}/{len(tasks)} 井",
                        flush=True,
                    )

    summary = _aggregate_outputs(selected_registry, artifact_dir, config)
    runtime = {
        "mode": args.mode,
        "fingerprint": fingerprint,
        "wells": int(len(selected_registry)),
        "workers": int(workers),
        "wall_seconds": float(time.perf_counter() - run_start),
        "sum_well_seconds": float(sum(float(row["seconds"]) for row in runtimes)),
        "per_well": sorted(runtimes, key=lambda row: str(row["well_id"])),
    }
    write_json(artifact_dir / f"runtime_{args.mode}.json", runtime)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

