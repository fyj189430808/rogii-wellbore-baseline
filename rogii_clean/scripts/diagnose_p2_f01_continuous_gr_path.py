"""运行 P2-F01：PF 中心上的整段连续 GR offset 路径诊断。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# clean 项目根目录，用于脚本直接运行时导入 src 和 scripts。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p2_d01_pf_centered_gr import restore_pf_center  # noqa: E402
from scripts.diagnose_p2_s01_outer_fold_surface import (  # noqa: E402
    file_sha256,
    load_baseline_predictions,
    load_registry,
    rmse_from_sse,
    stable_json_hash,
    write_json_atomic,
    write_parquet_atomic,
)
from src.p2_f01_continuous_gr_path import (  # noqa: E402
    LegalContinuousPathResult,
    build_legal_continuous_gr_path,
)


# 冻结实验身份和默认输入输出位置。
EXPERIMENT_ID = "P2_F01_continuous_gr_offset_path_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_f01_continuous_gr_offset_path_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析三井 smoke 或全量 773 井模式，以及可覆盖的产物目录。"""

    parser = argparse.ArgumentParser(description="运行 P2-F01 连续 GR offset 路径诊断")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def validate_config(config: dict[str, Any]) -> None:
    """拒绝任何与预注册中心、网格、尺度或硬约束不一致的运行配置。"""

    expected_values = {
        "experiment_id": EXPERIMENT_ID,
        "center_path": "last_visible_tvt_plus_pf_ancc_delta",
        "block_width_ft": 50.0,
        "smoothing_widths_ft": [5.0, 11.0, 21.0, 51.0, 101.0],
        "minimum_valid_pairs": 30,
        "minimum_valid_scales": 3,
        "allowed_offset_changes_ft": [-4.0, -2.0, 0.0, 2.0, 4.0],
        "maximum_change_acceleration_ft": 2.0,
        "initial_offset_ft": 0.0,
        "initial_change_ft": 0.0,
        "negative_control": "hidden_gr_circular_shift_half_segment",
        "prefix_injection_amplitude_ft": 12.0,
        "prefix_maximum_length_ft": 1000.0,
        "hidden_tvt_access": "oracle_scoring_only_after_legal_path_saved",
    }
    for key, expected_value in expected_values.items():
        if config.get(key) != expected_value:
            raise ValueError(f"P2-F01 配置 {key} 不等于冻结值 {expected_value}")
    expected_offsets = np.arange(-40.0, 42.0, 2.0).tolist()
    if config.get("offset_grid_ft") != expected_offsets:
        raise ValueError("P2-F01 offset 网格被修改")
    if config.get("supported_modes") != list(SUPPORTED_MODES):
        raise ValueError("P2-F01 supported_modes 被修改")


def validate_external_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """核对 fold、冻结 PF 血缘和 P2B00 评价表的完整文件指纹。"""

    fold_path = CLEAN_ROOT / str(config["fold_registry"])
    if file_sha256(fold_path) != str(config["fold_registry_sha256"]):
        raise ValueError("P2-F01 fold 注册表 SHA-256 不一致")

    candidate_metadata_path = CLEAN_ROOT / str(config["candidate_cache_metadata"])
    if file_sha256(candidate_metadata_path) != str(
        config["candidate_cache_metadata_sha256"]
    ):
        raise ValueError("P2-F01 候选缓存 metadata SHA-256 不一致")
    candidate_metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
    if candidate_metadata.get("candidate_cache_sha256") != str(
        config["candidate_cache_sha256"]
    ):
        raise ValueError("P2-F01 冻结候选缓存 SHA 血缘不一致")
    if not bool(candidate_metadata.get("completed")):
        raise ValueError("P2-F01 冻结候选缓存没有完成标记")
    if int(candidate_metadata.get("wells", -1)) != int(config["expected_wells"]):
        raise ValueError("P2-F01 冻结候选缓存井数不一致")
    if int(candidate_metadata.get("rows", -1)) != int(config["expected_hidden_rows"]):
        raise ValueError("P2-F01 冻结候选缓存行数不一致")

    baseline_path = CLEAN_ROOT / str(config["baseline_predictions"])
    if file_sha256(baseline_path) != str(config["baseline_predictions_sha256"]):
        raise ValueError("P2-F01 P2B00 predictions SHA-256 不一致")
    return candidate_metadata


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """将配置、两份执行代码和所有冻结外部输入组合成恢复指纹。"""

    return stable_json_hash(
        {
            "config": config,
            "core_sha256": file_sha256(
                CLEAN_ROOT / "src" / "p2_f01_continuous_gr_path.py"
            ),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "fold_registry_sha256": config["fold_registry_sha256"],
            "candidate_cache_metadata_sha256": config[
                "candidate_cache_metadata_sha256"
            ],
            "candidate_cache_sha256": config["candidate_cache_sha256"],
            "baseline_predictions_sha256": config["baseline_predictions_sha256"],
        }
    )


def _build_path(
    row_index: np.ndarray,
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: dict[str, Any],
) -> LegalContinuousPathResult:
    """把冻结配置逐项传给纯合法核心，避免 worker 内出现隐藏默认值。"""

    return build_legal_continuous_gr_path(
        row_index=np.asarray(row_index, dtype=np.int64),
        md=np.asarray(md, dtype=np.float64),
        horizontal_gr=np.asarray(horizontal_gr, dtype=np.float64),
        center_tvt=np.asarray(center_tvt, dtype=np.float64),
        typewell_tvt=np.asarray(typewell_tvt, dtype=np.float64),
        typewell_gr=np.asarray(typewell_gr, dtype=np.float64),
        offsets_ft=np.asarray(config["offset_grid_ft"], dtype=np.float64),
        block_width_ft=float(config["block_width_ft"]),
        smoothing_widths_ft=[float(value) for value in config["smoothing_widths_ft"]],
        minimum_valid_pairs=int(config["minimum_valid_pairs"]),
        minimum_valid_scales=int(config["minimum_valid_scales"]),
        allowed_changes_ft=np.asarray(
            config["allowed_offset_changes_ft"],
            dtype=np.float64,
        ),
        maximum_change_acceleration_ft=float(
            config["maximum_change_acceleration_ft"]
        ),
        initial_offset_ft=float(config["initial_offset_ft"]),
        initial_change_ft=float(config["initial_change_ft"]),
    )


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """用文件句柄写压缩 NPZ，再原子替换，防止中断留下半个 score 缓存。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_path.replace(path)


def run_program_controls(config: dict[str, Any]) -> dict[str, float]:
    """运行不接触真实隐藏标签的合成路径和全缺失回零程序正对照。"""

    # 1000 个 1 ft 行构成 20 个 50 ft 控制块。
    md = np.arange(1000, dtype=np.float64)
    block_md_mid = np.arange(24.5, 1000.0, 50.0)

    # 该控制点路径严格满足 4 ft 一阶和 2 ft 二阶硬约束。
    known_block_offset = np.array(
        [0, 2, 4, 6, 8, 10, 12, 12, 10, 8, 6, 4, 2, 0, 0, 2, 4, 4, 2, 0],
        dtype=np.float64,
    )
    known_row_offset = np.interp(md, block_md_mid, known_block_offset)
    center_tvt = 100.0 + 0.2 * md
    true_tvt = center_tvt + known_row_offset

    # 非重复的 chirp 加局部波形让合成 Typewell 同时包含低频和高频形态。
    typewell_tvt = np.arange(0.0, 400.01, 0.2)
    typewell_gr = (
        80.0
        + 20.0 * np.sin(0.002 * np.square(typewell_tvt))
        + 10.0 * np.sin(typewell_tvt / 3.7)
        + 0.05 * typewell_tvt
    )
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)
    row_index = np.arange(len(md), dtype=np.int64)

    synthetic = _build_path(
        row_index=row_index,
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    recovered_offset = synthetic.row_path["f01_offset_path"].to_numpy(
        dtype=np.float64
    )
    synthetic_rmse = float(
        np.sqrt(np.mean(np.square(recovered_offset - known_row_offset)))
    )

    # 同一问题把 GR 全部设为缺失，输出必须完全由回退项决定并逐点等于零。
    all_missing = _build_path(
        row_index=row_index,
        md=md,
        horizontal_gr=np.full(len(md), np.nan, dtype=np.float64),
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    missing_offset = all_missing.row_path["f01_offset_path"].to_numpy(
        dtype=np.float64
    )
    return {
        "synthetic_path_rmse_ft": synthetic_rmse,
        "all_missing_max_abs_offset_ft": float(np.max(np.abs(missing_offset))),
    }


def _prefix_positive_control(
    legal_horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """在可见前缀注入已知半正弦偏移，比较真实 GR 和循环平移 GR 的恢复。"""

    md_all = legal_horizontal["MD"].to_numpy(dtype=np.float64)
    gr_all = legal_horizontal["GR"].to_numpy(dtype=np.float64)
    tvt_input = legal_horizontal["TVT_input"].to_numpy(dtype=np.float64)
    visible_positions = np.flatnonzero(np.isfinite(tvt_input))
    if len(visible_positions) == 0:
        raise ValueError("可见前缀正对照找不到 TVT_input")

    visible_end_md = float(md_all[visible_positions[-1]])
    prefix_start_md = visible_end_md - float(config["prefix_maximum_length_ft"])
    selected_positions = visible_positions[md_all[visible_positions] >= prefix_start_md]
    if len(selected_positions) < int(config["minimum_valid_pairs"]):
        selected_positions = visible_positions
    if len(selected_positions) < 3:
        raise ValueError("可见前缀正对照不足 3 行")

    prefix_md = md_all[selected_positions]
    prefix_gr = gr_all[selected_positions]
    prefix_truth = tvt_input[selected_positions]
    md_span = float(prefix_md[-1] - prefix_md[0])
    if md_span <= 0.0:
        progress = np.zeros(len(prefix_md), dtype=np.float64)
    else:
        progress = (prefix_md - prefix_md[0]) / md_span
    injection = float(config["prefix_injection_amplitude_ft"]) * np.sin(
        np.pi * progress
    )
    injected_center = prefix_truth + injection

    real = _build_path(
        row_index=selected_positions,
        md=prefix_md,
        horizontal_gr=prefix_gr,
        center_tvt=injected_center,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    shifted_gr = np.roll(prefix_gr, max(len(prefix_gr) // 2, 1))
    shifted = _build_path(
        row_index=selected_positions,
        md=prefix_md,
        horizontal_gr=shifted_gr,
        center_tvt=injected_center,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )

    recovered = real.row_path["f01_corrected_tvt"].to_numpy(dtype=np.float64)
    shifted_recovered = shifted.row_path["f01_corrected_tvt"].to_numpy(
        dtype=np.float64
    )
    return {
        "rows": int(len(prefix_truth)),
        "sse_injected_center": float(
            np.sum(np.square(injected_center - prefix_truth))
        ),
        "sse_recovered": float(np.sum(np.square(recovered - prefix_truth))),
        "sse_shifted": float(np.sum(np.square(shifted_recovered - prefix_truth))),
    }


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """第一阶段只生成并落盘合法路径；该函数从不读取隐藏 TVT。"""

    well_id = str(task["well_id"])
    fold_id = int(task["fold"])
    config = dict(task["config"])
    raw_train_dir = Path(str(task["raw_train_dir"]))
    candidate_per_well_dir = Path(str(task["candidate_per_well_dir"]))
    artifact_dir = Path(str(task["artifact_dir"]))
    fingerprint = str(task["fingerprint"])

    real_path_output = artifact_dir / "legal_paths" / f"{well_id}.parquet"
    negative_path_output = artifact_dir / "negative_paths" / f"{well_id}.parquet"
    block_output = artifact_dir / "legal_blocks" / f"{well_id}.parquet"
    score_output = artifact_dir / "legal_scores" / f"{well_id}.npz"
    runtime_output = artifact_dir / "legal_runtime" / f"{well_id}.json"
    required_outputs = [
        real_path_output,
        negative_path_output,
        block_output,
        score_output,
        runtime_output,
    ]
    if all(path.is_file() for path in required_outputs):
        previous = json.loads(runtime_output.read_text(encoding="utf-8"))
        if previous.get("experiment_fingerprint") == fingerprint:
            return previous

    started = time.perf_counter()
    horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
    candidate_path = candidate_per_well_dir / f"{well_id}.parquet"

    # 第一次物理读取只包含比赛测试期可获得的当前井列。
    legal_horizontal = pd.read_csv(
        horizontal_path,
        usecols=["MD", "GR", "TVT_input"],
    )
    candidate_rows = pd.read_parquet(
        candidate_path,
        columns=["row_index", "pf_ancc_delta", "_cache_fingerprint"],
    )
    hidden_positions, pf_center_tvt = restore_pf_center(
        legal_horizontal["TVT_input"].to_numpy(dtype=np.float64),
        candidate_rows,
    )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    typewell_gr = typewell["GR"].to_numpy(dtype=np.float64)
    md_all = legal_horizontal["MD"].to_numpy(dtype=np.float64)
    gr_all = legal_horizontal["GR"].to_numpy(dtype=np.float64)
    hidden_md = md_all[hidden_positions]
    hidden_gr = gr_all[hidden_positions]

    real_result = _build_path(
        row_index=hidden_positions,
        md=hidden_md,
        horizontal_gr=hidden_gr,
        center_tvt=pf_center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    shifted_hidden_gr = np.roll(hidden_gr, max(len(hidden_gr) // 2, 1))
    negative_result = _build_path(
        row_index=hidden_positions,
        md=hidden_md,
        horizontal_gr=shifted_hidden_gr,
        center_tvt=pf_center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )

    visible_tvt = legal_horizontal["TVT_input"].dropna().to_numpy(dtype=np.float64)
    last_visible_tvt = float(visible_tvt[-1])
    real_path = real_result.row_path.copy()
    real_path.insert(0, "well_id", well_id)
    real_path.insert(1, "fold", fold_id)
    real_path.insert(3, "f01_pf_center_tvt", pf_center_tvt)
    real_path["f01_corrected_delta"] = (
        real_path["f01_corrected_tvt"].to_numpy(dtype=np.float64)
        - last_visible_tvt
    )
    real_path["f01_independent_delta"] = (
        real_path["f01_independent_tvt"].to_numpy(dtype=np.float64)
        - last_visible_tvt
    )

    negative_path = negative_result.row_path.copy()
    negative_path.insert(0, "well_id", well_id)
    negative_path.insert(1, "fold", fold_id)
    negative_path.insert(3, "f01_pf_center_tvt", pf_center_tvt)

    block_path = real_result.block_path.copy()
    block_path.insert(0, "well_id", well_id)
    block_path.insert(1, "fold", fold_id)

    # 先保存所有合法产物；这些列和数组中没有隐藏 TVT 或 oracle 误差。
    write_parquet_atomic(real_path_output, real_path)
    write_parquet_atomic(negative_path_output, negative_path)
    write_parquet_atomic(block_output, block_path)
    _write_npz_atomic(
        score_output,
        {
            "row_index": hidden_positions.astype(np.int32),
            "row_to_block": real_result.row_to_block.astype(np.int32),
            "offsets_ft": real_result.offsets_ft.astype(np.float32),
            "scale_scores": real_result.scale_scores.astype(np.float32),
            "scale_pair_counts": np.clip(
                real_result.scale_pair_counts,
                0.0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16),
            "block_md_mid": real_result.block_path["block_md_mid"].to_numpy(
                dtype=np.float32
            ),
        },
    )

    prefix_metrics = _prefix_positive_control(
        legal_horizontal=legal_horizontal,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "well_id": well_id,
        "fold": fold_id,
        "hidden_rows": int(len(hidden_positions)),
        "prefix_control": prefix_metrics,
        "candidate_cache_fingerprints": sorted(
            candidate_rows["_cache_fingerprint"].astype(str).unique().tolist()
        ),
        "hidden_tvt_read": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(runtime_output, runtime)
    return runtime


def score_generated_paths(
    selected_registry: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_positions: dict[str, np.ndarray],
    artifact_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """第二阶段在全部合法路径落盘后，才读取冻结真值列并计算逐井指标。"""

    rows: list[dict[str, Any]] = []
    prefix_totals: dict[str, float | int] = {
        "rows": 0,
        "sse_injected_center": 0.0,
        "sse_recovered": 0.0,
        "sse_shifted": 0.0,
    }
    maximum_offset = float(np.max(np.abs(config["offset_grid_ft"])))

    for completed_number, registry_row in enumerate(
        selected_registry.itertuples(index=False),
        start=1,
    ):
        well_id = str(registry_row.well_id)
        fold_id = int(registry_row.fold)
        real_path = pd.read_parquet(artifact_dir / "legal_paths" / f"{well_id}.parquet")
        negative_path = pd.read_parquet(
            artifact_dir / "negative_paths" / f"{well_id}.parquet"
        )
        positions = baseline_positions.get(well_id)
        if positions is None:
            raise ValueError(f"P2B00 predictions 缺少井 {well_id}")
        expected = baseline.iloc[positions].sort_values("row_index", kind="stable")
        expected = expected.reset_index(drop=True)
        real_path = real_path.sort_values("row_index", kind="stable").reset_index(drop=True)
        negative_path = negative_path.sort_values("row_index", kind="stable").reset_index(
            drop=True
        )
        expected_rows = expected["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            real_path["row_index"].to_numpy(dtype=np.int64),
            expected_rows,
        ):
            raise ValueError(f"{well_id} F01 路径与 P2B00 行键不一致")
        if not np.array_equal(
            negative_path["row_index"].to_numpy(dtype=np.int64),
            expected_rows,
        ):
            raise ValueError(f"{well_id} F01 负对照与 P2B00 行键不一致")
        if not expected["fold"].astype(int).eq(fold_id).all():
            raise ValueError(f"{well_id} F01 fold 与 P2B00 不一致")

        truth = expected["target_tvt"].to_numpy(dtype=np.float64)
        predictions = {
            "continuous": real_path["f01_corrected_tvt"].to_numpy(dtype=np.float64),
            "negative": negative_path["f01_corrected_tvt"].to_numpy(dtype=np.float64),
            "independent": real_path["f01_independent_tvt"].to_numpy(dtype=np.float64),
            "pf": real_path["f01_pf_center_tvt"].to_numpy(dtype=np.float64),
            "carry": expected["carry_tvt"].to_numpy(dtype=np.float64),
            "p2b00": expected["pred_tvt"].to_numpy(dtype=np.float64),
        }
        squared_error = {
            name: np.square(prediction - truth)
            for name, prediction in predictions.items()
        }
        hidden_rows = int(len(truth))
        metric_row: dict[str, Any] = {
            "well_id": well_id,
            "fold": fold_id,
            "hidden_rows": hidden_rows,
            "true_offset_in_grid_rows": int(
                np.sum(np.abs(truth - predictions["pf"]) <= maximum_offset)
            ),
            "median_support_fraction": float(
                real_path["f01_support_fraction"].median()
            ),
            "missing_fallback_fraction": float(
                real_path["f01_missing_fallback"].mean()
            ),
            "median_forward_backward_margin": float(
                real_path["f01_forward_backward_margin"].median()
            ),
        }
        for name, values in squared_error.items():
            sse = float(np.sum(values))
            metric_row[f"sse_{name}"] = sse
            metric_row[f"{name}_rmse"] = rmse_from_sse(sse, hidden_rows)
        rows.append(metric_row)

        runtime = json.loads(
            (artifact_dir / "legal_runtime" / f"{well_id}.json").read_text(
                encoding="utf-8"
            )
        )
        prefix = runtime["prefix_control"]
        for key in prefix_totals:
            prefix_totals[key] = prefix_totals[key] + prefix[key]
        if completed_number % 100 == 0 or completed_number == len(selected_registry):
            print(
                f"P2-F01 oracle 评分：{completed_number}/{len(selected_registry)}",
                flush=True,
            )

    per_well = pd.DataFrame(rows)
    per_well = per_well.sort_values("well_id", kind="stable").reset_index(drop=True)
    return per_well, prefix_totals


def build_summary(
    per_well_df: pd.DataFrame,
    prefix_totals: dict[str, float | int],
    program_controls: dict[str, float],
    config: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """汇总路径、正负对照和所有预注册诊断门槛。"""

    if per_well_df.empty:
        raise ValueError("P2-F01 没有逐井指标")
    total_rows = int(per_well_df["hidden_rows"].sum())
    method_names = ("continuous", "negative", "independent", "pf", "carry", "p2b00")
    micro_rmse = {
        name: rmse_from_sse(float(per_well_df[f"sse_{name}"].sum()), total_rows)
        for name in method_names
    }

    fold_rows: list[dict[str, float | int]] = []
    for fold_id, fold_group in per_well_df.groupby("fold", sort=True):
        fold_hidden_rows = int(fold_group["hidden_rows"].sum())
        fold_row: dict[str, float | int] = {
            "fold": int(fold_id),
            "wells": int(len(fold_group)),
            "hidden_rows": fold_hidden_rows,
        }
        for name in method_names:
            fold_row[f"{name}_rmse"] = rmse_from_sse(
                float(fold_group[f"sse_{name}"].sum()),
                fold_hidden_rows,
            )
        fold_row["improvement_vs_pf_ft"] = float(
            fold_row["pf_rmse"] - fold_row["continuous_rmse"]
        )
        fold_rows.append(fold_row)

    prefix_rows = int(prefix_totals["rows"])
    if prefix_rows <= 0:
        raise ValueError("P2-F01 可见前缀正对照没有评价行")
    prefix_rmse = {
        "injected_center": rmse_from_sse(
            float(prefix_totals["sse_injected_center"]),
            prefix_rows,
        ),
        "recovered": rmse_from_sse(
            float(prefix_totals["sse_recovered"]),
            prefix_rows,
        ),
        "shifted": rmse_from_sse(
            float(prefix_totals["sse_shifted"]),
            prefix_rows,
        ),
    }
    prefix_improvement_vs_center = float(
        prefix_rmse["injected_center"] - prefix_rmse["recovered"]
    )
    prefix_improvement_vs_shifted = float(
        prefix_rmse["shifted"] - prefix_rmse["recovered"]
    )
    improvement_vs_pf = float(micro_rmse["pf"] - micro_rmse["continuous"])
    improvement_vs_negative = float(
        micro_rmse["negative"] - micro_rmse["continuous"]
    )
    improvement_vs_independent = float(
        micro_rmse["independent"] - micro_rmse["continuous"]
    )
    folds_better_than_pf = int(
        sum(row["continuous_rmse"] < row["pf_rmse"] for row in fold_rows)
    )
    maximum_fold_degradation = float(
        max(row["continuous_rmse"] - row["pf_rmse"] for row in fold_rows)
    )
    well_win_rate = float(
        per_well_df["continuous_rmse"].lt(per_well_df["pf_rmse"]).mean()
    )
    p90_continuous = float(per_well_df["continuous_rmse"].quantile(0.9))
    p90_pf = float(per_well_df["pf_rmse"].quantile(0.9))
    p90_degradation = p90_continuous - p90_pf
    grid_coverage = float(
        per_well_df["true_offset_in_grid_rows"].sum() / total_rows
    )

    conditions = config["success_conditions"]
    checks = {
        "synthetic_path_pass": bool(
            program_controls["synthetic_path_rmse_ft"]
            <= float(conditions["maximum_synthetic_path_rmse_ft"])
        ),
        "all_missing_zero_pass": bool(
            program_controls["all_missing_max_abs_offset_ft"] <= 1e-12
        ),
        "prefix_injection_pass": bool(
            prefix_improvement_vs_center
            >= float(conditions["minimum_prefix_improvement_vs_injected_center_ft"])
        ),
        "prefix_shift_control_pass": bool(
            prefix_improvement_vs_shifted
            >= float(conditions["minimum_prefix_improvement_vs_shifted_gr_ft"])
        ),
        "grid_coverage_pass": bool(
            grid_coverage >= float(conditions["minimum_true_offset_grid_coverage"])
        ),
        "pf_improvement_pass": bool(
            improvement_vs_pf
            >= float(conditions["minimum_improvement_vs_pf_center_ft"])
        ),
        "fold_consistency_pass": bool(
            folds_better_than_pf >= int(conditions["minimum_folds_better_than_pf"])
        ),
        "single_fold_guard_pass": bool(
            maximum_fold_degradation
            <= float(conditions["maximum_single_fold_degradation_vs_pf_ft"])
        ),
        "shifted_hidden_control_pass": bool(
            improvement_vs_negative
            >= float(conditions["minimum_improvement_vs_shifted_gr_ft"])
        ),
        "continuity_control_pass": bool(
            improvement_vs_independent
            >= float(conditions["minimum_improvement_vs_independent_blocks_ft"])
        ),
        "well_win_rate_pass": bool(
            well_win_rate >= float(conditions["minimum_well_win_rate_vs_pf"])
        ),
        "p90_guard_pass": bool(
            p90_degradation
            <= float(conditions["maximum_p90_degradation_vs_pf_ft"])
        ),
    }

    return {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(per_well_df)),
        "hidden_rows": total_rows,
        "micro_rmse": micro_rmse,
        "macro_well_rmse": {
            name: float(per_well_df[f"{name}_rmse"].mean())
            for name in method_names
        },
        "median_well_rmse": {
            name: float(per_well_df[f"{name}_rmse"].median())
            for name in ("continuous", "pf", "p2b00")
        },
        "p90_well_rmse": {
            "continuous": p90_continuous,
            "pf": p90_pf,
            "p2b00": float(per_well_df["p2b00_rmse"].quantile(0.9)),
        },
        "worst_well_rmse": {
            name: float(per_well_df[f"{name}_rmse"].max())
            for name in ("continuous", "pf", "p2b00")
        },
        "prefix_control_rows": prefix_rows,
        "prefix_control_rmse": prefix_rmse,
        "prefix_improvement_vs_injected_center_ft": prefix_improvement_vs_center,
        "prefix_improvement_vs_shifted_gr_ft": prefix_improvement_vs_shifted,
        "true_offset_grid_coverage": grid_coverage,
        "improvement_vs_pf_ft": improvement_vs_pf,
        "improvement_vs_shifted_gr_ft": improvement_vs_negative,
        "improvement_vs_independent_blocks_ft": improvement_vs_independent,
        "folds_better_than_pf": folds_better_than_pf,
        "maximum_single_fold_degradation_vs_pf_ft": maximum_fold_degradation,
        "well_win_rate_vs_pf": well_win_rate,
        "p90_degradation_vs_pf_ft": p90_degradation,
        "program_controls": program_controls,
        "folds": fold_rows,
        "checks": checks,
        "continuous_gr_path_supported": bool(all(checks.values())),
        "wall_seconds": float(wall_seconds),
    }


def main(argv: list[str] | None = None) -> None:
    """先完成所有合法路径，再隔离执行 oracle 评分并写总体结论。"""

    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    candidate_metadata = validate_external_inputs(config)
    registry = load_registry(config)
    fingerprint = experiment_fingerprint(config)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifact_dir / "config.json", config)

    program_controls = run_program_controls(config)
    write_json_atomic(artifact_dir / "program_controls.json", program_controls)
    conditions = config["success_conditions"]
    if program_controls["synthetic_path_rmse_ft"] > float(
        conditions["maximum_synthetic_path_rmse_ft"]
    ):
        raise RuntimeError("P2-F01 合成路径程序正对照失败")
    if program_controls["all_missing_max_abs_offset_ft"] > 1e-12:
        raise RuntimeError("P2-F01 全 GR 缺失没有严格回到零 offset")

    write_json_atomic(
        artifact_dir / "lineage.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "legal_horizontal_columns": ["MD", "GR", "TVT_input"],
            "typewell_columns": ["TVT", "GR"],
            "candidate_columns": ["row_index", "pf_ancc_delta", "_cache_fingerprint"],
            "center_source": "frozen pf_ancc_delta",
            "candidate_cache_metadata": candidate_metadata,
            "fold_registry_sha256": config["fold_registry_sha256"],
            "baseline_predictions_sha256": config["baseline_predictions_sha256"],
            "legal_score_forbidden_arrays": [
                "TVT",
                "true_offset",
                "true_rank",
                "oracle_error",
            ],
            "hidden_tvt_read_after_all_legal_paths": True,
        },
    )

    if args.mode == "smoke":
        selected_registry = registry.head(int(config["smoke_wells"])).copy()
    else:
        selected_registry = registry.copy()
    raw_train_dir = (CLEAN_ROOT / str(config["raw_train_dir"])).resolve()
    candidate_per_well_dir = (
        CLEAN_ROOT / str(config["candidate_per_well_dir"])
    ).resolve()

    print(
        f"P2-F01 {args.mode}：{len(selected_registry)} 口井；"
        f"每井 real/循环平移 hidden + real/循环平移 prefix；输出 {artifact_dir}",
        flush=True,
    )
    started = time.perf_counter()
    tasks = [
        {
            "well_id": str(row.well_id),
            "fold": int(row.fold),
            "config": config,
            "raw_train_dir": str(raw_train_dir),
            "candidate_per_well_dir": str(candidate_per_well_dir),
            "artifact_dir": str(artifact_dir),
            "fingerprint": fingerprint,
        }
        for row in selected_registry.itertuples(index=False)
    ]
    runtimes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    workers = 1 if args.mode == "smoke" else max(1, int(config["workers"]))

    if workers == 1:
        for completed_number, task in enumerate(tasks, start=1):
            try:
                runtimes.append(generate_one_well(task))
            except Exception as error:  # noqa: BLE001 - 保存逐井错误后统一中止。
                errors.append({"well_id": str(task["well_id"]), "error": repr(error)})
            print(
                f"P2-F01 legal：{completed_number}/{len(tasks)}，失败 {len(errors)}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_by_well = {
                executor.submit(generate_one_well, task): str(task["well_id"])
                for task in tasks
            }
            for completed_number, future in enumerate(
                concurrent.futures.as_completed(future_by_well),
                start=1,
            ):
                well_id = future_by_well[future]
                try:
                    runtimes.append(future.result())
                except Exception as error:  # noqa: BLE001 - 保存全部错误后统一中止。
                    errors.append({"well_id": well_id, "error": repr(error)})
                if completed_number % 10 == 0 or completed_number == len(tasks):
                    print(
                        f"P2-F01 legal：{completed_number}/{len(tasks)}，失败 {len(errors)}",
                        flush=True,
                    )

    if errors:
        write_json_atomic(
            artifact_dir / f"errors_{args.mode}.json",
            {"experiment_fingerprint": fingerprint, "errors": errors},
        )
        raise RuntimeError(f"P2-F01 有 {len(errors)} 口井合法路径失败")

    # 只有当选中井的全部合法路径都已成功落盘，才加载含 target_tvt 的冻结评价表。
    baseline, baseline_positions = load_baseline_predictions(config)
    per_well, prefix_totals = score_generated_paths(
        selected_registry=selected_registry,
        baseline=baseline,
        baseline_positions=baseline_positions,
        artifact_dir=artifact_dir,
        config=config,
    )
    wall_seconds = float(time.perf_counter() - started)
    summary = build_summary(
        per_well_df=per_well,
        prefix_totals=prefix_totals,
        program_controls=program_controls,
        config=config,
        wall_seconds=wall_seconds,
    )
    summary["mode"] = str(args.mode)
    summary["experiment_fingerprint"] = fingerprint
    suffix = "smoke" if args.mode == "smoke" else "all"
    per_well.to_csv(artifact_dir / f"per_well_{suffix}.csv", index=False)
    write_json_atomic(artifact_dir / f"summary_{suffix}.json", summary)
    write_json_atomic(
        artifact_dir / f"runtime_{suffix}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "mode": str(args.mode),
            "wells": int(len(per_well)),
            "wall_seconds": wall_seconds,
            "workers": workers,
            "legal_generation_seconds_sum": float(
                sum(float(item["elapsed_seconds"]) for item in runtimes)
            ),
        },
    )
    if args.mode == "all":
        per_well.to_csv(artifact_dir / "per_well.csv", index=False)
        write_json_atomic(artifact_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
