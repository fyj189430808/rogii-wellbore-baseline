"""运行 P2-F01b：在同一 MD 轴平滑两条 GR 序列的连续路径诊断。"""

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

# PF 中心只由自然可见前缀和冻结的 pf_ancc_delta 恢复，不读取隐藏 TVT。
from scripts.diagnose_p2_d01_pf_centered_gr import restore_pf_center  # noqa: E402
from scripts.diagnose_p2_f01_continuous_gr_path import (  # noqa: E402
    _write_npz_atomic,
    validate_external_inputs as validate_frozen_external_inputs,
)
from scripts.diagnose_p2_s01_outer_fold_surface import (  # noqa: E402
    file_sha256,
    load_baseline_predictions,
    load_registry,
    rmse_from_sse,
    stable_json_hash,
    write_json_atomic,
    write_parquet_atomic,
)
from src.p2_f01b_path_domain_smoothing import (  # noqa: E402
    build_legal_path_domain_gr_path,
    build_path_domain_score_landscape,
)


# 这些常量唯一确定本次预注册实验和默认产物位置。
EXPERIMENT_ID = "P2_F01b_path_domain_smoothing_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_f01b_path_domain_smoothing_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "fold0", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析运行阶段、冻结配置和独立产物目录。"""

    parser = argparse.ArgumentParser(description="运行 P2-F01b 同一 MD 轴 GR 路径诊断")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def validate_config(config: dict[str, Any]) -> None:
    """拒绝对唯一修改、路径中心、网格和动态规划约束的静默变更。"""

    # 每个值都来自已经写死的实验卡，runner 不允许运行时另选版本。
    expected_values: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "center_path": "last_visible_tvt_plus_pf_ancc_delta",
        "score_domain": "sample_raw_typewell_then_smooth_both_sequences_on_md",
        "block_width_ft": 50.0,
        "smoothing_widths_md_ft": [5.0, 11.0, 21.0, 51.0, 101.0],
        "minimum_valid_pairs": 30,
        "minimum_valid_scales": 3,
        "allowed_offset_changes_ft": [-4.0, -2.0, 0.0, 2.0, 4.0],
        "maximum_change_acceleration_ft": 2.0,
        "initial_offset_ft": 0.0,
        "initial_change_ft": 0.0,
        "emission_loss": "same_as_p2_f01_v1",
        "negative_control": "hidden_gr_circular_shift_half_segment",
        "independent_control": "minimum_emission_cost_per_block",
        "prefix_signal_control": "last_1000ft_visible_half_sine_real_gr_vs_half_roll",
        "prefix_injection_amplitude_ft": 12.0,
        "prefix_maximum_length_ft": 1000.0,
        "translation_equivariance_shift_ft": 12.0,
        "hidden_tvt_access": "oracle_scoring_only_after_all_selected_legal_paths_saved",
        "workers": 4,
        "supported_modes": list(SUPPORTED_MODES),
    }
    for key, expected_value in expected_values.items():
        if config.get(key) != expected_value:
            raise ValueError(f"P2-F01b 配置 {key} 不等于冻结值 {expected_value}")

    # offset 网格必须正好是 -40 到 40 ft、步长 2 ft。
    expected_offsets = np.arange(-40.0, 42.0, 2.0).tolist()
    if config.get("offset_grid_ft") != expected_offsets:
        raise ValueError("P2-F01b offset_grid_ft 被修改")

    # smoke 井的顺序也是预注册的一部分，不能根据结果挑井。
    expected_smoke_wells = ["000d7d20", "00bbac68", "00e12e8b"]
    if config.get("smoke_well_ids") != expected_smoke_wells:
        raise ValueError("P2-F01b smoke_well_ids 被修改")
    if int(config.get("fold0_id", -1)) != 0:
        raise ValueError("P2-F01b fold0_id 被修改")


def validate_external_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """复用 F01 的冻结输入审计，核对 fold、候选缓存和 P2B00 哈希。"""

    # F01b 沿用同一批冻结外部输入，因此直接复用已经验证过的完整审计函数。
    return validate_frozen_external_inputs(config)


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """组合配置、F01b 核心、runner 和冻结输入，生成逐井恢复指纹。"""

    fingerprint_parts = {
        "config": config,
        "core_sha256": file_sha256(
            CLEAN_ROOT / "src" / "p2_f01b_path_domain_smoothing.py"
        ),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "fold_registry_sha256": config["fold_registry_sha256"],
        "candidate_cache_metadata_sha256": config[
            "candidate_cache_metadata_sha256"
        ],
        "candidate_cache_sha256": config["candidate_cache_sha256"],
        "baseline_predictions_sha256": config["baseline_predictions_sha256"],
    }
    return stable_json_hash(fingerprint_parts)


def select_registry(
    registry: pd.DataFrame,
    mode: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """按预注册规则选择 smoke、fold 0 或全部井，且保持固定顺序。"""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"P2-F01b 不支持运行模式 {mode}")
    if mode == "all":
        return registry.copy().reset_index(drop=True)
    if mode == "fold0":
        fold_id = int(config["fold0_id"])
        selected = registry.loc[registry["fold"].astype(int).eq(fold_id)].copy()
        return selected.reset_index(drop=True)

    # smoke 必须严格按配置中的顺序取井，而不是依赖注册表当前排序。
    registry_by_well = registry.copy()
    registry_by_well["well_id"] = registry_by_well["well_id"].astype(str)
    registry_by_well = registry_by_well.set_index("well_id", drop=False)
    selected_rows: list[pd.Series] = []
    for well_id in config["smoke_well_ids"]:
        if str(well_id) not in registry_by_well.index:
            raise ValueError(f"P2-F01b smoke 井 {well_id} 不在 fold 注册表")
        selected_rows.append(registry_by_well.loc[str(well_id)])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _build_path(
    row_index: np.ndarray,
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: dict[str, Any],
):
    """把冻结配置逐项传入纯合法核心，不在 worker 中留下隐藏默认值。"""

    return build_legal_path_domain_gr_path(
        row_index=np.asarray(row_index, dtype=np.int64),
        md=np.asarray(md, dtype=np.float64),
        horizontal_gr=np.asarray(horizontal_gr, dtype=np.float64),
        center_tvt=np.asarray(center_tvt, dtype=np.float64),
        typewell_tvt=np.asarray(typewell_tvt, dtype=np.float64),
        typewell_gr=np.asarray(typewell_gr, dtype=np.float64),
        offsets_ft=np.asarray(config["offset_grid_ft"], dtype=np.float64),
        block_width_ft=float(config["block_width_ft"]),
        smoothing_widths_md_ft=[
            float(value) for value in config["smoothing_widths_md_ft"]
        ],
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


def _make_nonperiodic_typewell() -> tuple[np.ndarray, np.ndarray]:
    """构造含不对称标志层的合成 Typewell，供程序正对照使用。"""

    # 0.25 ft 网格足以覆盖下面合成井的全部候选 TVT。
    typewell_tvt = np.arange(0.0, 260.25, 0.25, dtype=np.float64)
    background = 65.0 + 0.018 * typewell_tvt
    marker_one = 30.0 * np.exp(-0.5 * np.square((typewell_tvt - 86.0) / 2.3))
    marker_two = -21.0 * np.exp(-0.5 * np.square((typewell_tvt - 118.0) / 4.7))
    marker_three = 17.0 * np.exp(-0.5 * np.square((typewell_tvt - 151.0) / 1.6))
    texture = 5.0 * np.sin(typewell_tvt / 5.3) + 2.0 * np.cos(typewell_tvt / 2.7)
    typewell_gr = background + marker_one + marker_two + marker_three + texture
    return typewell_tvt, typewell_gr


def run_program_controls(config: dict[str, Any]) -> dict[str, Any]:
    """运行合成路径、平移等变和全 GR 缺失三项无真值程序检查。"""

    typewell_tvt, typewell_gr = _make_nonperiodic_typewell()

    # 合成井的 MD 与 TVT 比例不是 1，专门检查 F01b 的 MD 轴平滑实现。
    md = np.arange(0.0, 1001.0, 1.0, dtype=np.float64)
    center_tvt = 75.0 + 0.075 * md
    block_md_mid = np.arange(25.0, 1026.0, 50.0, dtype=np.float64)
    known_block_offset = np.array(
        [
            0.0,
            -2.0,
            -4.0,
            -6.0,
            -8.0,
            -8.0,
            -6.0,
            -4.0,
            -2.0,
            0.0,
            2.0,
            4.0,
            6.0,
            8.0,
            8.0,
            6.0,
            4.0,
            2.0,
            0.0,
            -2.0,
            -4.0,
        ],
        dtype=np.float64,
    )
    known_row_offset = np.interp(md, block_md_mid, known_block_offset)
    horizontal_gr = np.interp(
        center_tvt + known_row_offset,
        typewell_tvt,
        typewell_gr,
    )
    synthetic = _build_path(
        row_index=np.arange(len(md), dtype=np.int64),
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    recovered_offset = synthetic.row_path["f01b_offset_path"].to_numpy(
        dtype=np.float64
    )
    synthetic_rmse = float(
        np.sqrt(np.mean(np.square(recovered_offset - known_row_offset)))
    )

    # 平移等变检查：中心加 12 ft、offset 减 12 ft 后仍是同一绝对 TVT。
    translation_md = np.arange(0.0, 401.0, 1.0, dtype=np.float64)
    translation_center = 82.0 + 0.08 * translation_md
    translation_gr = np.interp(
        translation_center + 6.0,
        typewell_tvt,
        typewell_gr,
    )
    offsets = np.asarray(config["offset_grid_ft"], dtype=np.float64)
    original_landscape = build_path_domain_score_landscape(
        md=translation_md,
        horizontal_gr=translation_gr,
        center_tvt=translation_center,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=offsets,
        block_width_ft=float(config["block_width_ft"]),
        smoothing_widths_md_ft=config["smoothing_widths_md_ft"],
        minimum_valid_pairs=int(config["minimum_valid_pairs"]),
    )
    translation_shift = float(config["translation_equivariance_shift_ft"])
    shifted_landscape = build_path_domain_score_landscape(
        md=translation_md,
        horizontal_gr=translation_gr,
        center_tvt=translation_center + translation_shift,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=offsets,
        block_width_ft=float(config["block_width_ft"]),
        smoothing_widths_md_ft=config["smoothing_widths_md_ft"],
        minimum_valid_pairs=int(config["minimum_valid_pairs"]),
    )
    original_positions = np.flatnonzero(
        (offsets - translation_shift >= offsets[0])
        & (offsets - translation_shift <= offsets[-1])
    )
    shifted_positions = np.array(
        [
            int(np.flatnonzero(np.isclose(offsets, offset - translation_shift))[0])
            for offset in offsets[original_positions]
        ],
        dtype=np.int64,
    )
    original_scores = original_landscape.scores[:, :, original_positions]
    shifted_scores = shifted_landscape.scores[:, :, shifted_positions]
    original_counts = original_landscape.pair_counts[:, :, original_positions]
    shifted_counts = shifted_landscape.pair_counts[:, :, shifted_positions]
    original_finite = np.isfinite(original_scores)
    shifted_finite = np.isfinite(shifted_scores)
    finite_mask_equal = bool(np.array_equal(original_finite, shifted_finite))
    pair_counts_equal = bool(np.array_equal(original_counts, shifted_counts))
    if finite_mask_equal and bool(original_finite.any()):
        maximum_ncc_difference = float(
            np.max(
                np.abs(
                    original_scores[original_finite]
                    - shifted_scores[shifted_finite]
                )
            )
        )
    elif finite_mask_equal:
        maximum_ncc_difference = 0.0
    else:
        maximum_ncc_difference = float("inf")

    # 没有任何水平井 GR 时，路径必须逐行保留 PF 中心，即 offset 为零。
    all_missing = _build_path(
        row_index=np.arange(len(md), dtype=np.int64),
        md=md,
        horizontal_gr=np.full(len(md), np.nan, dtype=np.float64),
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    missing_offset = all_missing.row_path["f01b_offset_path"].to_numpy(
        dtype=np.float64
    )

    return {
        "synthetic_path_rmse_ft": synthetic_rmse,
        "translation_max_ncc_difference": maximum_ncc_difference,
        "translation_finite_mask_equal": finite_mask_equal,
        "translation_pair_counts_equal": pair_counts_equal,
        "all_missing_max_abs_offset_ft": float(np.max(np.abs(missing_offset))),
    }


def evaluate_stage_checks(
    mode: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    """只应用当前阶段能够回答的预注册门槛，避免 smoke 被五折条件误杀。"""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"P2-F01b 不支持阶段 {mode}")
    conditions = config["success_conditions"]

    # 三个阶段都必须通过程序检查和真实可见前缀信号检查。
    checks: dict[str, bool] = {
        "synthetic_path_pass": bool(
            metrics["synthetic_path_rmse_ft"]
            <= float(conditions["maximum_synthetic_path_rmse_ft"])
        ),
        "translation_ncc_pass": bool(
            metrics["translation_max_ncc_difference"]
            <= float(conditions["maximum_translation_ncc_difference"])
        ),
        "translation_finite_mask_pass": bool(
            metrics["translation_finite_mask_equal"]
        ),
        "translation_pair_counts_pass": bool(
            metrics["translation_pair_counts_equal"]
        ),
        "all_missing_zero_pass": bool(
            metrics["all_missing_max_abs_offset_ft"]
            <= float(conditions["maximum_all_missing_offset_ft"])
        ),
        "prefix_signal_pass": bool(
            metrics["prefix_improvement_vs_shifted_gr_ft"]
            >= float(
                conditions["minimum_smoke_prefix_real_vs_shifted_improvement_ft"]
            )
        ),
    }

    # smoke 只判断程序是否正确、三井真实前缀是否含比循环平移更强的信号。
    if mode == "smoke":
        checks["stage_supported"] = bool(all(checks.values()))
        return checks

    # fold 0 与全量共同使用的路径、负对照和风险切片门槛。
    checks.update(
        {
            "shifted_hidden_control_pass": bool(
                metrics["improvement_vs_shifted_gr_ft"]
                >= float(conditions["minimum_improvement_vs_shifted_gr_ft"])
            ),
            "continuity_control_pass": bool(
                metrics["improvement_vs_independent_blocks_ft"]
                >= float(conditions["minimum_improvement_vs_independent_blocks_ft"])
            ),
            "well_win_rate_pass": bool(
                metrics["well_win_rate_vs_pf"]
                >= float(conditions["minimum_well_win_rate_vs_pf"])
            ),
            "p90_guard_pass": bool(
                metrics["p90_degradation_vs_pf_ft"]
                <= float(conditions["maximum_p90_degradation_vs_pf_ft"])
            ),
            "grid_coverage_pass": bool(
                metrics["true_offset_grid_coverage"]
                >= float(conditions["minimum_true_offset_grid_coverage"])
            ),
        }
    )
    if mode == "fold0":
        checks["fold0_pf_improvement_pass"] = bool(
            metrics["improvement_vs_pf_ft"]
            >= float(conditions["minimum_fold0_improvement_vs_pf_ft"])
        )
    else:
        checks["full_pf_improvement_pass"] = bool(
            metrics["improvement_vs_pf_ft"]
            >= float(conditions["minimum_full_improvement_vs_pf_ft"])
        )
        checks["full_fold_consistency_pass"] = bool(
            metrics["folds_better_than_pf"]
            >= int(conditions["minimum_full_folds_better_than_pf"])
        )
        checks["full_single_fold_guard_pass"] = bool(
            metrics["maximum_single_fold_degradation_vs_pf_ft"]
            <= float(conditions["maximum_full_single_fold_degradation_vs_pf_ft"])
        )
    checks["stage_supported"] = bool(all(checks.values()))
    return checks


def _prefix_positive_control(
    legal_horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """给真实可见前缀中心注入半正弦，比较真实 GR 与半段循环平移 GR。"""

    md_all = legal_horizontal["MD"].to_numpy(dtype=np.float64)
    gr_all = legal_horizontal["GR"].to_numpy(dtype=np.float64)
    tvt_input = legal_horizontal["TVT_input"].to_numpy(dtype=np.float64)
    visible_positions = np.flatnonzero(np.isfinite(tvt_input))
    if len(visible_positions) == 0:
        raise ValueError("P2-F01b 可见前缀找不到 TVT_input")

    visible_end_md = float(md_all[visible_positions[-1]])
    prefix_start_md = visible_end_md - float(config["prefix_maximum_length_ft"])
    selected_positions = visible_positions[md_all[visible_positions] >= prefix_start_md]
    if len(selected_positions) < int(config["minimum_valid_pairs"]):
        selected_positions = visible_positions
    if len(selected_positions) < 3:
        raise ValueError("P2-F01b 可见前缀不足 3 行")

    prefix_md = md_all[selected_positions]
    prefix_gr = gr_all[selected_positions]
    prefix_truth = tvt_input[selected_positions]
    md_span = float(prefix_md[-1] - prefix_md[0])
    if md_span <= 0.0:
        progress = np.zeros(len(prefix_md), dtype=np.float64)
    else:
        progress = (prefix_md - prefix_md[0]) / md_span
    injected_center = prefix_truth + float(
        config["prefix_injection_amplitude_ft"]
    ) * np.sin(np.pi * progress)

    real_result = _build_path(
        row_index=selected_positions,
        md=prefix_md,
        horizontal_gr=prefix_gr,
        center_tvt=injected_center,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    shifted_prefix_gr = np.roll(prefix_gr, max(len(prefix_gr) // 2, 1))
    shifted_result = _build_path(
        row_index=selected_positions,
        md=prefix_md,
        horizontal_gr=shifted_prefix_gr,
        center_tvt=injected_center,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        config=config,
    )
    recovered_tvt = real_result.row_path["f01b_corrected_tvt"].to_numpy(
        dtype=np.float64
    )
    shifted_tvt = shifted_result.row_path["f01b_corrected_tvt"].to_numpy(
        dtype=np.float64
    )
    return {
        "rows": int(len(prefix_truth)),
        "sse_injected_center": float(
            np.sum(np.square(injected_center - prefix_truth))
        ),
        "sse_recovered": float(np.sum(np.square(recovered_tvt - prefix_truth))),
        "sse_shifted": float(np.sum(np.square(shifted_tvt - prefix_truth))),
    }


def generate_one_well(task: dict[str, Any]) -> dict[str, Any]:
    """只用合法列生成一口井的真实、负对照路径和原始得分缓存。"""

    well_id = str(task["well_id"])
    fold_id = int(task["fold"])
    config = dict(task["config"])
    raw_train_dir = Path(str(task["raw_train_dir"]))
    candidate_per_well_dir = Path(str(task["candidate_per_well_dir"]))
    artifact_dir = Path(str(task["artifact_dir"]))
    fingerprint = str(task["fingerprint"])

    # 每口井分别落盘，runtime 中的完整指纹决定能否安全续跑。
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
        previous_runtime = json.loads(runtime_output.read_text(encoding="utf-8"))
        if previous_runtime.get("experiment_fingerprint") == fingerprint:
            return previous_runtime

    started = time.perf_counter()
    horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
    candidate_path = candidate_per_well_dir / f"{well_id}.parquet"

    # 合法阶段物理读取的水平井列只有 MD、GR 和自然可见 TVT_input。
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

    # 循环平移连同缺失掩码一起移动，作为与真实 GR 严格配对的负对照。
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
    real_path.insert(3, "f01b_pf_center_tvt", pf_center_tvt)
    real_path["f01b_corrected_delta"] = (
        real_path["f01b_corrected_tvt"].to_numpy(dtype=np.float64)
        - last_visible_tvt
    )
    real_path["f01b_independent_delta"] = (
        real_path["f01b_independent_tvt"].to_numpy(dtype=np.float64)
        - last_visible_tvt
    )

    negative_path = negative_result.row_path.copy()
    negative_path.insert(0, "well_id", well_id)
    negative_path.insert(1, "fold", fold_id)
    negative_path.insert(3, "f01b_pf_center_tvt", pf_center_tvt)
    block_path = real_result.block_path.copy()
    block_path.insert(0, "well_id", well_id)
    block_path.insert(1, "fold", fold_id)

    # NPZ 只保存合法原始得分；数组名中禁止出现 TVT 真值、oracle 和真 offset。
    score_arrays = {
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
    }
    forbidden_fragments = ("tvt", "true", "oracle", "target", "error", "rank")
    for array_name in score_arrays:
        lowered_name = array_name.lower()
        if any(fragment in lowered_name for fragment in forbidden_fragments):
            raise ValueError(f"P2-F01b legal score 出现禁止数组名 {array_name}")

    # 先保存所有隐藏段合法产物，然后再做同样合法的可见前缀信号检查。
    write_parquet_atomic(real_path_output, real_path)
    write_parquet_atomic(negative_path_output, negative_path)
    write_parquet_atomic(block_output, block_path)
    _write_npz_atomic(score_output, score_arrays)
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
    """所有选中井合法路径落盘后，才用冻结隐藏真值计算诊断指标。"""

    metric_rows: list[dict[str, Any]] = []
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
        baseline_row_positions = baseline_positions.get(well_id)
        if baseline_row_positions is None:
            raise ValueError(f"P2B00 predictions 缺少井 {well_id}")
        expected = baseline.iloc[baseline_row_positions]
        expected = expected.sort_values("row_index", kind="stable").reset_index(drop=True)
        real_path = real_path.sort_values("row_index", kind="stable").reset_index(drop=True)
        negative_path = negative_path.sort_values(
            "row_index", kind="stable"
        ).reset_index(drop=True)
        expected_rows = expected["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            real_path["row_index"].to_numpy(dtype=np.int64),
            expected_rows,
        ):
            raise ValueError(f"{well_id} F01b 路径与 P2B00 行键不一致")
        if not np.array_equal(
            negative_path["row_index"].to_numpy(dtype=np.int64),
            expected_rows,
        ):
            raise ValueError(f"{well_id} F01b 负对照与 P2B00 行键不一致")
        if not expected["fold"].astype(int).eq(fold_id).all():
            raise ValueError(f"{well_id} F01b fold 与 P2B00 不一致")

        truth = expected["target_tvt"].to_numpy(dtype=np.float64)
        predictions = {
            "continuous": real_path["f01b_corrected_tvt"].to_numpy(dtype=np.float64),
            "negative": negative_path["f01b_corrected_tvt"].to_numpy(dtype=np.float64),
            "independent": real_path["f01b_independent_tvt"].to_numpy(dtype=np.float64),
            "pf": real_path["f01b_pf_center_tvt"].to_numpy(dtype=np.float64),
            "carry": expected["carry_tvt"].to_numpy(dtype=np.float64),
            "p2b00": expected["pred_tvt"].to_numpy(dtype=np.float64),
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
                real_path["f01b_support_fraction"].median()
            ),
            "missing_fallback_fraction": float(
                real_path["f01b_missing_fallback"].mean()
            ),
            "median_forward_backward_margin": float(
                real_path["f01b_forward_backward_margin"].median()
            ),
        }
        for method_name, prediction in predictions.items():
            squared_error = np.square(prediction - truth)
            sum_squared_error = float(np.sum(squared_error))
            metric_row[f"sse_{method_name}"] = sum_squared_error
            metric_row[f"{method_name}_rmse"] = rmse_from_sse(
                sum_squared_error,
                hidden_rows,
            )
        metric_rows.append(metric_row)

        runtime = json.loads(
            (artifact_dir / "legal_runtime" / f"{well_id}.json").read_text(
                encoding="utf-8"
            )
        )
        prefix_metrics = runtime["prefix_control"]
        for key in prefix_totals:
            prefix_totals[key] = prefix_totals[key] + prefix_metrics[key]
        if completed_number % 100 == 0 or completed_number == len(selected_registry):
            print(
                f"P2-F01b oracle 评分：{completed_number}/{len(selected_registry)}",
                flush=True,
            )

    per_well = pd.DataFrame(metric_rows)
    per_well = per_well.sort_values("well_id", kind="stable").reset_index(drop=True)
    return per_well, prefix_totals


def build_summary(
    mode: str,
    per_well_df: pd.DataFrame,
    prefix_totals: dict[str, float | int],
    program_controls: dict[str, Any],
    config: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """汇总 pooled、逐井、逐折指标，并应用当前阶段预注册门槛。"""

    if per_well_df.empty:
        raise ValueError("P2-F01b 没有逐井指标")
    total_rows = int(per_well_df["hidden_rows"].sum())
    method_names = ("continuous", "negative", "independent", "pf", "carry", "p2b00")
    micro_rmse = {
        method_name: rmse_from_sse(
            float(per_well_df[f"sse_{method_name}"].sum()),
            total_rows,
        )
        for method_name in method_names
    }

    fold_rows: list[dict[str, float | int]] = []
    for fold_id, fold_group in per_well_df.groupby("fold", sort=True):
        fold_hidden_rows = int(fold_group["hidden_rows"].sum())
        fold_row: dict[str, float | int] = {
            "fold": int(fold_id),
            "wells": int(len(fold_group)),
            "hidden_rows": fold_hidden_rows,
        }
        for method_name in method_names:
            fold_row[f"{method_name}_rmse"] = rmse_from_sse(
                float(fold_group[f"sse_{method_name}"].sum()),
                fold_hidden_rows,
            )
        fold_row["improvement_vs_pf_ft"] = float(
            fold_row["pf_rmse"] - fold_row["continuous_rmse"]
        )
        fold_rows.append(fold_row)

    prefix_rows = int(prefix_totals["rows"])
    if prefix_rows <= 0:
        raise ValueError("P2-F01b 可见前缀正对照没有评价行")
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
    p90_continuous = float(per_well_df["continuous_rmse"].quantile(0.9))
    p90_pf = float(per_well_df["pf_rmse"].quantile(0.9))
    improvements = {
        "prefix_improvement_vs_shifted_gr_ft": float(
            prefix_rmse["shifted"] - prefix_rmse["recovered"]
        ),
        "improvement_vs_pf_ft": float(
            micro_rmse["pf"] - micro_rmse["continuous"]
        ),
        "improvement_vs_shifted_gr_ft": float(
            micro_rmse["negative"] - micro_rmse["continuous"]
        ),
        "improvement_vs_independent_blocks_ft": float(
            micro_rmse["independent"] - micro_rmse["continuous"]
        ),
    }
    folds_better_than_pf = int(
        sum(row["continuous_rmse"] < row["pf_rmse"] for row in fold_rows)
    )
    maximum_fold_degradation = float(
        max(row["continuous_rmse"] - row["pf_rmse"] for row in fold_rows)
    )
    stage_metrics: dict[str, Any] = {
        **program_controls,
        **improvements,
        "well_win_rate_vs_pf": float(
            per_well_df["continuous_rmse"].lt(per_well_df["pf_rmse"]).mean()
        ),
        "p90_degradation_vs_pf_ft": float(p90_continuous - p90_pf),
        "true_offset_grid_coverage": float(
            per_well_df["true_offset_in_grid_rows"].sum() / total_rows
        ),
        "folds_better_than_pf": folds_better_than_pf,
        "maximum_single_fold_degradation_vs_pf_ft": maximum_fold_degradation,
    }
    checks = evaluate_stage_checks(mode, stage_metrics, config)

    return {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "wells": int(len(per_well_df)),
        "hidden_rows": total_rows,
        "micro_rmse": micro_rmse,
        "macro_well_rmse": {
            method_name: float(per_well_df[f"{method_name}_rmse"].mean())
            for method_name in method_names
        },
        "median_well_rmse": {
            method_name: float(per_well_df[f"{method_name}_rmse"].median())
            for method_name in ("continuous", "pf", "p2b00")
        },
        "p90_well_rmse": {
            "continuous": p90_continuous,
            "pf": p90_pf,
            "p2b00": float(per_well_df["p2b00_rmse"].quantile(0.9)),
        },
        "worst_well_rmse": {
            method_name: float(per_well_df[f"{method_name}_rmse"].max())
            for method_name in ("continuous", "pf", "p2b00")
        },
        "prefix_control_rows": prefix_rows,
        "prefix_control_rmse": prefix_rmse,
        **stage_metrics,
        "folds": fold_rows,
        "checks": checks,
        "stage_supported": bool(checks["stage_supported"]),
        "wall_seconds": float(wall_seconds),
    }


def _run_legal_generation(
    tasks: list[dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """按指定进程数生成合法路径，收集所有逐井错误后统一中止。"""

    runtimes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if workers == 1:
        for completed_number, task in enumerate(tasks, start=1):
            try:
                runtimes.append(generate_one_well(task))
            except Exception as error:  # noqa: BLE001 - 错误必须带井号完整保存。
                errors.append({"well_id": str(task["well_id"]), "error": repr(error)})
            print(
                f"P2-F01b legal：{completed_number}/{len(tasks)}，失败 {len(errors)}",
                flush=True,
            )
        return runtimes, errors

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
            except Exception as error:  # noqa: BLE001 - 错误必须带井号完整保存。
                errors.append({"well_id": well_id, "error": repr(error)})
            if completed_number % 10 == 0 or completed_number == len(tasks):
                print(
                    f"P2-F01b legal：{completed_number}/{len(tasks)}，失败 {len(errors)}",
                    flush=True,
                )
    return runtimes, errors


def main(argv: list[str] | None = None) -> None:
    """先生成全部合法路径，再隔离读取 target 评分并写阶段结论。"""

    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    candidate_metadata = validate_external_inputs(config)
    registry = load_registry(config)
    selected_registry = select_registry(registry, str(args.mode), config)
    fingerprint = experiment_fingerprint(config)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(artifact_dir / "config.json", config)

    # 程序正对照在读取任何真实井隐藏标签前运行，失败时立即停止。
    program_controls = run_program_controls(config)
    write_json_atomic(artifact_dir / "program_controls.json", program_controls)
    conditions = config["success_conditions"]
    technical_failures = {
        "synthetic_path": program_controls["synthetic_path_rmse_ft"]
        > float(conditions["maximum_synthetic_path_rmse_ft"]),
        "translation_ncc": program_controls["translation_max_ncc_difference"]
        > float(conditions["maximum_translation_ncc_difference"]),
        "translation_mask": not bool(
            program_controls["translation_finite_mask_equal"]
        ),
        "translation_pairs": not bool(
            program_controls["translation_pair_counts_equal"]
        ),
        "all_missing": program_controls["all_missing_max_abs_offset_ft"]
        > float(conditions["maximum_all_missing_offset_ft"]),
    }
    failed_controls = [name for name, failed in technical_failures.items() if failed]
    if failed_controls:
        raise RuntimeError(f"P2-F01b 程序正对照失败：{failed_controls}")

    # lineage 明确记录合法输入边界和隐藏真值的延迟读取顺序。
    write_json_atomic(
        artifact_dir / "lineage.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "legal_horizontal_columns": ["MD", "GR", "TVT_input"],
            "typewell_columns": ["TVT", "GR"],
            "candidate_columns": ["row_index", "pf_ancc_delta", "_cache_fingerprint"],
            "center_source": "frozen pf_ancc_delta",
            "score_domain": config["score_domain"],
            "candidate_cache_metadata": candidate_metadata,
            "fold_registry_sha256": config["fold_registry_sha256"],
            "baseline_predictions_sha256": config["baseline_predictions_sha256"],
            "legal_score_forbidden_arrays": [
                "TVT",
                "target",
                "true_offset",
                "true_rank",
                "oracle_error",
            ],
            "hidden_tvt_read_after_all_selected_legal_paths": True,
        },
    )

    raw_train_dir = (CLEAN_ROOT / str(config["raw_train_dir"])).resolve()
    candidate_per_well_dir = (
        CLEAN_ROOT / str(config["candidate_per_well_dir"])
    ).resolve()
    workers = 1 if args.mode == "smoke" else int(config["workers"])
    print(
        f"P2-F01b {args.mode}：{len(selected_registry)} 口井；"
        f"每井 hidden/prefix 各跑真实与半段循环平移 GR；"
        f"{workers} 进程；输出 {artifact_dir}",
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
    runtimes, errors = _run_legal_generation(tasks, workers)
    if errors:
        write_json_atomic(
            artifact_dir / f"errors_{args.mode}.json",
            {"experiment_fingerprint": fingerprint, "errors": errors},
        )
        raise RuntimeError(f"P2-F01b 有 {len(errors)} 口井合法路径失败")

    # 这是唯一加载 target_tvt 的位置；执行到这里时，选中井的合法路径已全部落盘。
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
        mode=str(args.mode),
        per_well_df=per_well,
        prefix_totals=prefix_totals,
        program_controls=program_controls,
        config=config,
        wall_seconds=wall_seconds,
    )
    summary["experiment_fingerprint"] = fingerprint

    # 每个阶段使用独立文件名；all 另写无后缀正式副本，便于后续登记。
    stage_name = str(args.mode)
    per_well.to_csv(artifact_dir / f"per_well_{stage_name}.csv", index=False)
    write_json_atomic(artifact_dir / f"summary_{stage_name}.json", summary)
    write_json_atomic(
        artifact_dir / f"runtime_{stage_name}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_fingerprint": fingerprint,
            "mode": stage_name,
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
