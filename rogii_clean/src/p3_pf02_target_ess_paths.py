"""P3-PF02：用二分温度把 128 条冻结 PF 种子路径聚合到固定目标 ESS。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.p3_d01_pf_observation_audit import seed_weights, summarize_weight_scale
from src.p2_p01_multiseed_pf import (
    _kernel_arguments,
    particle_filter_all_seeds_numba,
    prepare_particle_filter_inputs,
)


TARGET_ESS_VALUES = (2, 8, 32, 96)
TARGET_ESS_PATH_COLUMNS = [f"pf128_ess{target}_delta" for target in TARGET_ESS_VALUES]


def solve_temperature_for_target_ess(
    seed_log_likelihoods: np.ndarray,
    target_ess: float,
    maximum_iterations: int = 100,
) -> dict[str, float]:
    """二分求温度，使 softmax 权重的有效路径数尽量等于固定目标。"""
    log_likelihoods = np.asarray(seed_log_likelihoods, dtype=np.float64)
    if log_likelihoods.ndim != 1 or len(log_likelihoods) < 2:
        raise ValueError("seed_log_likelihoods 必须是一维且至少含两条路径")
    if not np.isfinite(log_likelihoods).all():
        raise ValueError("seed_log_likelihoods 必须全部有限")
    if not np.isfinite(target_ess) or not 1.0 < target_ess < len(log_likelihoods):
        raise ValueError("target_ess 必须严格位于 1 与路径数量之间")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations 必须为正数")

    # 温度趋近零时，最大似然并列路径决定可达到的最小 ESS。
    maximum_likelihood_count = int(np.count_nonzero(log_likelihoods == log_likelihoods.max()))
    if target_ess < maximum_likelihood_count:
        raise ValueError("最大似然存在并列，所给 target_ess 无法达到")

    lower_temperature = 0.0
    upper_temperature = 1.0
    upper_ess = summarize_weight_scale(log_likelihoods, upper_temperature)[
        "effective_sample_size"
    ]
    while upper_ess < target_ess:
        upper_temperature *= 2.0
        if not np.isfinite(upper_temperature):
            raise RuntimeError("无法为目标 ESS 找到有限温度上界")
        upper_ess = summarize_weight_scale(log_likelihoods, upper_temperature)[
            "effective_sample_size"
        ]

    selected_temperature = upper_temperature
    selected_ess = upper_ess
    absolute_tolerance = max(1e-8, float(target_ess) * 1e-9)
    for _ in range(maximum_iterations):
        middle_temperature = (lower_temperature + upper_temperature) / 2.0
        if middle_temperature <= 0.0:
            middle_temperature = np.nextafter(0.0, 1.0)
        middle_ess = summarize_weight_scale(log_likelihoods, middle_temperature)[
            "effective_sample_size"
        ]
        selected_temperature = middle_temperature
        selected_ess = middle_ess
        if abs(middle_ess - target_ess) <= absolute_tolerance:
            break
        if middle_ess < target_ess:
            lower_temperature = middle_temperature
        else:
            upper_temperature = middle_temperature

    return {
        "temperature": float(selected_temperature),
        "achieved_ess": float(selected_ess),
        "target_ess": float(target_ess),
        "absolute_ess_error": float(abs(selected_ess - target_ess)),
    }


def aggregate_target_ess_paths(
    seed_predictions: np.ndarray,
    seed_log_likelihoods: np.ndarray,
    last_visible_tvt: float,
    target_effective_sample_sizes: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """在内存中聚合目标 ESS 路径，只返回每行 delta 和井级求解报告。"""
    predictions = np.asarray(seed_predictions, dtype=np.float64)
    likelihoods = np.asarray(seed_log_likelihoods, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[0] != len(likelihoods):
        raise ValueError("seed_predictions 必须为 [路径数, 隐藏行数] 并与似然数量一致")
    if predictions.shape[1] == 0 or not np.isfinite(predictions).all():
        raise ValueError("seed_predictions 不能为空且必须全部有限")
    if not np.isfinite(last_visible_tvt):
        raise ValueError("last_visible_tvt 必须有限")
    targets = tuple(int(target) for target in target_effective_sample_sizes)
    if len(targets) == 0 or len(set(targets)) != len(targets):
        raise ValueError("目标 ESS 必须是非空且不重复的整数序列")

    paths: dict[str, np.ndarray] = {}
    report: dict[str, float] = {}
    for target_ess in targets:
        solution = solve_temperature_for_target_ess(likelihoods, float(target_ess))
        temperature = solution["temperature"]
        weights = seed_weights(likelihoods, temperature)
        absolute_path = weights @ predictions
        path_name = f"pf128_ess{target_ess}_delta"
        # 冻结 P2-P01 的数值约定：绝对 TVT 先压成 float32，再减 float32 末值。
        absolute_path_float32 = absolute_path.astype(np.float32)
        last_visible_tvt_float32 = np.float32(last_visible_tvt)
        paths[path_name] = absolute_path_float32 - last_visible_tvt_float32
        weight_summary = summarize_weight_scale(likelihoods, temperature)
        report[f"ess{target_ess}_temperature"] = float(temperature)
        report[f"ess{target_ess}_achieved_ess"] = float(
            weight_summary["effective_sample_size"]
        )
        report[f"ess{target_ess}_absolute_error"] = float(
            abs(weight_summary["effective_sample_size"] - target_ess)
        )
        report[f"ess{target_ess}_max_weight"] = float(weight_summary["max_weight"])
    return paths, report


def require_exact_log_likelihood_match(
    replayed_log_likelihoods: np.ndarray,
    d01_log_likelihoods: np.ndarray,
) -> None:
    """要求本次冻结回放的 128 个 LL 与 D01 缓存逐位完全相同。"""
    replayed = np.ascontiguousarray(replayed_log_likelihoods, dtype=np.float64)
    expected = np.ascontiguousarray(d01_log_likelihoods, dtype=np.float64)
    exact_match = replayed.shape == expected.shape and np.array_equal(
        replayed.view(np.uint64), expected.view(np.uint64)
    )
    if not exact_match:
        maximum_difference = (
            float(np.max(np.abs(replayed - expected)))
            if replayed.shape == expected.shape and replayed.size > 0
            else float("inf")
        )
        raise RuntimeError(
            "本次 PF 的种子对数似然与 D01 缓存没有逐位一致；"
            f"max_abs_diff={maximum_difference}"
        )


def build_target_ess_pf_features(
    horizontal_well: pd.DataFrame,
    typewell: pd.DataFrame,
    parameters: dict[str, object],
    d01_log_likelihoods: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """冻结旧 PF 完整回放一次，核对 LL 后生成四条预登记目标 ESS 路径。"""
    prepared = prepare_particle_filter_inputs(horizontal_well, typewell, parameters)
    seed_predictions, replayed_log_likelihoods = particle_filter_all_seeds_numba(
        **_kernel_arguments(prepared)
    )
    require_exact_log_likelihood_match(replayed_log_likelihoods, d01_log_likelihoods)
    delta_paths, target_report = aggregate_target_ess_paths(
        seed_predictions=seed_predictions,
        seed_log_likelihoods=replayed_log_likelihoods,
        last_visible_tvt=float(prepared["last_visible_tvt"]),
        target_effective_sample_sizes=TARGET_ESS_VALUES,
    )
    hidden_rows = len(prepared["row_index"])
    output = pd.DataFrame(
        {
            "row_index": prepared["row_index"],
            "last_visible_tvt": np.full(
                hidden_rows,
                np.float32(prepared["last_visible_tvt"]),
                dtype=np.float32,
            ),
            **delta_paths,
        }
    )
    if list(output.columns) != ["row_index", "last_visible_tvt", *TARGET_ESS_PATH_COLUMNS]:
        raise RuntimeError("目标 ESS 路径 schema 不正确")
    if not np.isfinite(output.drop(columns="row_index").to_numpy(dtype=np.float64)).all():
        raise RuntimeError("目标 ESS 路径含 NaN 或无穷值")
    quality: dict[str, float | bool] = {
        "d01_log_likelihood_bitwise_match": True,
        "legacy_gr_sigma": float(prepared["gr_sigma"]),
        **target_report,
    }
    return output, quality
