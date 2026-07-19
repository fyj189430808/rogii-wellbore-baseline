"""P3-MODE01：把 128 条 PF seed 路径固定压缩为三个路径模式。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src.p3_pf03_segmented_likelihood import read_pf03_shared_cache


FORMAL_NUMBER_OF_SEEDS = 128
FORMAL_NUMBER_OF_MODES = 3
FORMAL_SAMPLE_POINTS = 64
FORMAL_LIKELIHOOD_SCALE = 8.0
FORMAL_STABILITY_REPEATS = 8
FORMAL_STABILITY_SAMPLE_SIZE = 64
FORMAL_STABILITY_RANDOM_SEED = 20260719
SHARED_CACHE_FIELDS = {
    "seed_delta",
    "row_index",
    "hidden_md",
    "last_tvt",
    "final_ll",
    "seed_ids",
}
FORMAL_FEATURE_COLUMNS = [
    "pf_mode1_delta",
    "pf_mode2_delta",
    "pf_mode3_delta",
    "pf_mode1_mass",
    "pf_mode2_mass",
    "pf_mode12_margin",
    "pf_mode12_separation",
    "pf_mode_split_fraction",
]


def _sample_on_normalized_md(
    seed_delta: np.ndarray,
    hidden_md: np.ndarray,
    sample_points: int = FORMAL_SAMPLE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """把每条完整路径插值到相同的 0～1 井深进度网格。"""

    paths = np.asarray(seed_delta, dtype=np.float64)
    md = np.asarray(hidden_md, dtype=np.float64)
    if paths.ndim != 2 or md.ndim != 1 or paths.shape[1] != len(md):
        raise ValueError("seed_delta 与 hidden_md 的形状不一致")
    if paths.shape[1] == 0 or sample_points < 3:
        raise ValueError("隐藏路径为空或采样点过少")
    if not np.isfinite(paths).all() or not np.isfinite(md).all():
        raise ValueError("seed_delta 与 hidden_md 必须全部有限")
    if np.any(np.diff(md) < 0):
        raise ValueError("hidden_md 必须单调不减")

    target_progress = np.linspace(0.0, 1.0, sample_points, dtype=np.float64)
    md_span = float(md[-1] - md[0])
    if md_span <= 0.0:
        sampled = np.repeat(paths[:, :1], sample_points, axis=1)
        return sampled, target_progress

    source_progress = (md - md[0]) / md_span
    sampled = np.empty((paths.shape[0], sample_points), dtype=np.float64)
    for seed_index in range(paths.shape[0]):
        sampled[seed_index] = np.interp(
            target_progress,
            source_progress,
            paths[seed_index],
        )
    return sampled, target_progress


def _scale8_weights(final_ll: np.ndarray) -> np.ndarray:
    """按冻结 scale=8 softmax，把每条 seed 的最终 LL 变成概率质量。"""

    likelihoods = np.asarray(final_ll, dtype=np.float64)
    if likelihoods.ndim != 1 or likelihoods.size == 0:
        raise ValueError("final_ll 必须是一维非空数组")
    if not np.isfinite(likelihoods).all():
        raise ValueError("final_ll 必须全部有限")
    shifted = (likelihoods - float(np.max(likelihoods))) / FORMAL_LIKELIHOOD_SCALE
    unnormalized = np.exp(shifted)
    return unnormalized / float(np.sum(unnormalized))


def validate_shared_cache_arrays(arrays: dict[str, np.ndarray]) -> None:
    """严格核对 PF03→MODE01 共享缓存的字段、dtype、shape 和行键。"""

    if set(arrays) != SHARED_CACHE_FIELDS:
        raise ValueError(
            f"MODE01 共享缓存字段不一致：{sorted(set(arrays))}"
        )
    expected_dtypes = {
        "seed_delta": np.dtype(np.float32),
        "row_index": np.dtype(np.int32),
        "hidden_md": np.dtype(np.float32),
        "last_tvt": np.dtype(np.float64),
        "final_ll": np.dtype(np.float64),
        "seed_ids": np.dtype(np.int32),
    }
    for field_name, expected_dtype in expected_dtypes.items():
        if np.asarray(arrays[field_name]).dtype != expected_dtype:
            raise ValueError(
                f"MODE01 共享缓存 {field_name} dtype 必须为 {expected_dtype}"
            )

    seed_delta = np.asarray(arrays["seed_delta"])
    row_index = np.asarray(arrays["row_index"])
    hidden_md = np.asarray(arrays["hidden_md"])
    last_tvt = np.asarray(arrays["last_tvt"])
    final_ll = np.asarray(arrays["final_ll"])
    seed_ids = np.asarray(arrays["seed_ids"])
    if seed_delta.ndim != 2 or seed_delta.shape[0] != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("MODE01 seed_delta shape 必须为 [128, N]")
    hidden_rows = seed_delta.shape[1]
    if row_index.shape != (hidden_rows,) or hidden_md.shape != (hidden_rows,):
        raise ValueError("MODE01 row_index/hidden_md shape 与 seed_delta 不一致")
    if last_tvt.shape != (1,) or final_ll.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("MODE01 last_tvt 或 final_ll shape 不一致")
    if seed_ids.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("MODE01 seed_ids shape 必须为 [128]")
    if not np.array_equal(seed_ids, np.arange(FORMAL_NUMBER_OF_SEEDS, dtype=np.int32)):
        raise ValueError("MODE01 seed_ids 必须严格为 0～127")
    if hidden_rows <= 0 or np.any(np.diff(row_index.astype(np.int64)) <= 0):
        raise ValueError("MODE01 row_index 必须严格递增且非空")
    if np.any(np.diff(hidden_md.astype(np.float64)) < 0):
        raise ValueError("MODE01 hidden_md 必须单调不减")
    if not np.isfinite(seed_delta).all() or not np.isfinite(hidden_md).all():
        raise ValueError("MODE01 seed_delta 与 hidden_md 必须全部有限")
    if not np.isfinite(last_tvt).all() or not np.isfinite(final_ll).all():
        raise ValueError("MODE01 last_tvt 与 final_ll 必须全部有限")


def read_mode01_shared_cache(
    path: str,
    expected_fingerprint: str,
) -> dict[str, np.ndarray]:
    """先核对 PF03 格式和指纹，再执行 MODE01 的更严格 schema 审计。"""

    arrays = read_pf03_shared_cache(path, expected_fingerprint)
    validate_shared_cache_arrays(arrays)
    return arrays


def _cluster_and_order_modes(
    paths: np.ndarray,
    hidden_md: np.ndarray,
    likelihoods: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """执行无 zscore 的 Ward K=3，并按冻结平局规则排列三个模式。"""

    sampled, _ = _sample_on_normalized_md(paths, hidden_md)
    mean_sampled_path = np.mean(sampled, axis=0, keepdims=True)
    centered_for_clustering = sampled - mean_sampled_path
    hierarchy = linkage(centered_for_clustering, method="ward", optimal_ordering=False)
    raw_labels = cut_tree(hierarchy, n_clusters=[FORMAL_NUMBER_OF_MODES]).reshape(-1)
    if len(np.unique(raw_labels)) != FORMAL_NUMBER_OF_MODES:
        raise RuntimeError("Ward 聚类没有得到固定的三个模式")

    weights = _scale8_weights(likelihoods)
    unordered_modes: list[dict[str, Any]] = []
    for raw_label in sorted(np.unique(raw_labels).tolist()):
        member_mask = raw_labels == raw_label
        member_indices = np.flatnonzero(member_mask)
        center_full = np.mean(paths[member_mask], axis=0, dtype=np.float64)
        center_sampled = np.mean(sampled[member_mask], axis=0, dtype=np.float64)
        centered_distance = sampled[member_mask] - center_sampled
        path_rmse = np.sqrt(np.mean(np.square(centered_distance), axis=1))
        unordered_modes.append(
            {
                "raw_label": int(raw_label),
                "member_indices": member_indices,
                "center_full": center_full,
                "center_sampled": center_sampled,
                "mass": float(np.sum(weights[member_mask])),
                "seed_count": int(np.sum(member_mask)),
                "endpoint_delta": float(center_full[-1]),
                "minimum_seed_id": int(np.min(ids[member_mask])),
                "within_rmse_mean": float(np.mean(path_rmse)),
                "within_rmse_max": float(np.max(path_rmse)),
            }
        )

    ordered_modes = sorted(
        unordered_modes,
        key=lambda mode: (
            -float(mode["mass"]),
            -int(mode["seed_count"]),
            float(mode["endpoint_delta"]),
            int(mode["minimum_seed_id"]),
        ),
    )
    return sampled, raw_labels, ordered_modes


def find_mode_split_fraction(
    mode1_sampled: np.ndarray,
    mode2_sampled: np.ndarray,
) -> float:
    """返回两模式首次连续三点达到固定分离阈值的归一化进度。"""

    separation = np.abs(
        np.asarray(mode1_sampled, dtype=np.float64)
        - np.asarray(mode2_sampled, dtype=np.float64)
    )
    if separation.ndim != 1 or separation.size < 3 or not np.isfinite(separation).all():
        raise ValueError("模式采样路径必须是等长、有限的一维数组")
    threshold = max(1.0, 0.25 * float(np.max(separation)))
    qualifies = separation >= threshold
    for start_index in range(separation.size - 2):
        if bool(np.all(qualifies[start_index : start_index + 3])):
            return float(start_index / (separation.size - 1))
    return 1.0


def build_three_mode_features(
    seed_delta: np.ndarray,
    hidden_md: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """固定 Ward K=3，输出三条中心路径及首批八项正式特征。"""

    paths = np.asarray(seed_delta, dtype=np.float64)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    if paths.shape[0] != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("MODE01 正式输入必须恰好包含 128 条 seed 路径")
    if likelihoods.shape != (FORMAL_NUMBER_OF_SEEDS,):
        raise ValueError("final_ll 必须与 128 条 seed 路径一一对应")
    if ids.shape != (FORMAL_NUMBER_OF_SEEDS,) or len(np.unique(ids)) != len(ids):
        raise ValueError("seed_ids 必须是 128 个不重复编号")

    _, _, ordered_modes = _cluster_and_order_modes(
        paths,
        np.asarray(hidden_md, dtype=np.float64),
        likelihoods,
        ids,
    )
    mode1 = ordered_modes[0]
    mode2 = ordered_modes[1]
    split_fraction = find_mode_split_fraction(
        np.asarray(mode1["center_sampled"]),
        np.asarray(mode2["center_sampled"]),
    )

    number_of_rows = paths.shape[1]
    mode1_mass = float(mode1["mass"])
    mode2_mass = float(mode2["mass"])
    feature_frame = pd.DataFrame(
        {
            "pf_mode1_delta": np.asarray(ordered_modes[0]["center_full"]),
            "pf_mode2_delta": np.asarray(ordered_modes[1]["center_full"]),
            "pf_mode3_delta": np.asarray(ordered_modes[2]["center_full"]),
            "pf_mode1_mass": np.full(number_of_rows, mode1_mass),
            "pf_mode2_mass": np.full(number_of_rows, mode2_mass),
            "pf_mode12_margin": np.full(number_of_rows, mode1_mass - mode2_mass),
            "pf_mode12_separation": (
                np.asarray(mode1["center_full"])
                - np.asarray(mode2["center_full"])
            ),
            "pf_mode_split_fraction": np.full(number_of_rows, split_fraction),
        },
        columns=FORMAL_FEATURE_COLUMNS,
    )

    diagnostics: dict[str, Any] = {
        "number_of_seeds": FORMAL_NUMBER_OF_SEEDS,
        "number_of_modes": FORMAL_NUMBER_OF_MODES,
        "sample_points": FORMAL_SAMPLE_POINTS,
        "clustering": "ward_k3_centered_mean_path_no_zscore",
        "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
        "mode12_split_fraction": split_fraction,
    }
    for mode_number, mode in enumerate(ordered_modes, start=1):
        diagnostics[f"mode{mode_number}_mass"] = float(mode["mass"])
        diagnostics[f"mode{mode_number}_seed_count"] = int(mode["seed_count"])
        diagnostics[f"mode{mode_number}_seed_fraction"] = float(
            int(mode["seed_count"]) / FORMAL_NUMBER_OF_SEEDS
        )
        diagnostics[f"mode{mode_number}_endpoint_delta"] = float(
            mode["endpoint_delta"]
        )
        diagnostics[f"mode{mode_number}_minimum_seed_id"] = int(
            mode["minimum_seed_id"]
        )
        diagnostics[f"mode{mode_number}_within_rmse_mean"] = float(
            mode["within_rmse_mean"]
        )
        diagnostics[f"mode{mode_number}_within_rmse_max"] = float(
            mode["within_rmse_max"]
        )
    return feature_frame, diagnostics


def compute_half_seed_stability(
    seed_delta: np.ndarray,
    hidden_md: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    repeats: int = FORMAL_STABILITY_REPEATS,
    random_seed: int = FORMAL_STABILITY_RANDOM_SEED,
) -> dict[str, Any]:
    """用固定随机半样本检查三模式中心和聚类成员关系是否稳定。"""

    paths = np.asarray(seed_delta, dtype=np.float64)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    if paths.shape[0] != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("稳定性诊断必须输入 128 条 seed 路径")
    if repeats <= 0:
        raise ValueError("稳定性重复次数必须为正")
    full_sampled, full_labels, full_modes = _cluster_and_order_modes(
        paths,
        np.asarray(hidden_md, dtype=np.float64),
        likelihoods,
        ids,
    )
    full_centers = np.stack(
        [np.asarray(mode["center_sampled"], dtype=np.float64) for mode in full_modes]
    )

    generator = np.random.default_rng(random_seed)
    all_matched_rmse: list[np.ndarray] = []
    all_ari: list[float] = []
    for _ in range(repeats):
        selected_indices = np.sort(
            generator.choice(
                FORMAL_NUMBER_OF_SEEDS,
                size=FORMAL_STABILITY_SAMPLE_SIZE,
                replace=False,
            )
        )
        _, half_labels, half_modes = _cluster_and_order_modes(
            paths[selected_indices],
            np.asarray(hidden_md, dtype=np.float64),
            likelihoods[selected_indices],
            ids[selected_indices],
        )
        half_centers = np.stack(
            [
                np.asarray(mode["center_sampled"], dtype=np.float64)
                for mode in half_modes
            ]
        )
        cost = np.sqrt(
            np.mean(
                np.square(full_centers[:, None, :] - half_centers[None, :, :]),
                axis=2,
            )
        )
        full_assignment, half_assignment = linear_sum_assignment(cost)
        matched = np.empty(FORMAL_NUMBER_OF_MODES, dtype=np.float64)
        matched[full_assignment] = cost[full_assignment, half_assignment]
        all_matched_rmse.append(matched)
        all_ari.append(
            float(adjusted_rand_score(full_labels[selected_indices], half_labels))
        )

    matched_matrix = np.stack(all_matched_rmse)
    output: dict[str, Any] = {
        "stability_repeats": int(repeats),
        "stability_half_seed_count": FORMAL_STABILITY_SAMPLE_SIZE,
        "stability_random_seed": int(random_seed),
        "stability_mean_ari": float(np.mean(all_ari)),
        "stability_min_ari": float(np.min(all_ari)),
        "stability_center_rmse_mean": float(np.mean(matched_matrix)),
        "stability_center_rmse_max": float(np.max(matched_matrix)),
    }
    for mode_index in range(FORMAL_NUMBER_OF_MODES):
        output[f"stability_mode{mode_index + 1}_center_rmse_mean"] = float(
            np.mean(matched_matrix[:, mode_index])
        )
        output[f"stability_mode{mode_index + 1}_center_rmse_max"] = float(
            np.max(matched_matrix[:, mode_index])
        )
    return output
