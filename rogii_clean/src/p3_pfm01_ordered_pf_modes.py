"""P3-PFM01：把 128 条 PF 路径压缩成具有统一高低方向的三个模式。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


FORMAL_NUMBER_OF_SEEDS = 128
FORMAL_NUMBER_OF_MODES = 3
FORMAL_LIKELIHOOD_SCALE = 8.0
FORMAL_ROTATION = 64
FORMAL_DEGENERATE_SEPARATION_FT = 1e-6
MODE_NAMES = ("low", "middle", "high")
OLD_TEMPERATURE_NAMES = ("mean", "scale3", "scale5", "scale8", "scale12")


def _as_finite_paths(seed_delta: np.ndarray) -> np.ndarray:
    """把 seed 路径转成二维浮点数组，并拒绝空路径或非有限数。"""

    paths = np.asarray(seed_delta, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[0] <= 0:
        raise ValueError("seed_delta 必须是非空二维数组")
    if paths.shape[1] <= 0 or not np.isfinite(paths).all():
        raise ValueError("seed_delta 不能为空，也不能含 NaN/Inf")
    return paths


def build_seed_descriptors(seed_delta: np.ndarray) -> np.ndarray:
    """每条路径只保留整段均值和末端值，输出 shape=[seed 数, 2]。"""

    paths = _as_finite_paths(seed_delta)
    full_path_mean = np.mean(paths, axis=1, dtype=np.float64)
    endpoint_delta = paths[:, -1]
    # 这里故意不做 z-score：两列都以 ft 为单位，冻结实验要求保留原始尺度。
    return np.column_stack([full_path_mean, endpoint_delta])


def cluster_seed_descriptors(seed_delta: np.ndarray) -> np.ndarray:
    """对 [整段均值, 末端值] 做固定 Ward K=3，返回每条 seed 的原始簇号。"""

    descriptors = build_seed_descriptors(seed_delta)
    if descriptors.shape[0] < FORMAL_NUMBER_OF_MODES:
        raise ValueError("Ward K=3 至少需要三条 seed 路径")
    hierarchy = linkage(descriptors, method="ward", optimal_ordering=False)
    labels = cut_tree(hierarchy, n_clusters=[FORMAL_NUMBER_OF_MODES]).reshape(-1)
    labels = labels.astype(np.int64, copy=False)
    if len(np.unique(labels)) != FORMAL_NUMBER_OF_MODES:
        raise RuntimeError("Ward 聚类没有得到冻结的三个模式")
    return labels


def scale8_weights(final_ll: np.ndarray) -> np.ndarray:
    """用固定温度 8 的 softmax，把最终对数似然转换成总和为 1 的质量。"""

    likelihoods = np.asarray(final_ll, dtype=np.float64)
    if likelihoods.ndim != 1 or likelihoods.size == 0:
        raise ValueError("final_ll 必须是一维非空数组")
    if not np.isfinite(likelihoods).all():
        raise ValueError("final_ll 不能含 NaN/Inf")
    shifted = (likelihoods - float(np.max(likelihoods))) / FORMAL_LIKELIHOOD_SCALE
    unnormalized = np.exp(shifted)
    denominator = float(np.sum(unnormalized))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("scale8 softmax 的分母非法")
    return unnormalized / denominator


def order_mode_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按中心整段均值、末端值、最小 seed_id 升序排列为低、中、高。"""

    if len(records) != FORMAL_NUMBER_OF_MODES:
        raise ValueError("必须恰好输入三个模式记录")
    return sorted(
        records,
        key=lambda record: (
            float(record["mean_delta"]),
            float(record["endpoint_delta"]),
            int(record["minimum_seed_id"]),
        ),
    )


def _summarize_one_membership(
    paths: np.ndarray,
    weights: np.ndarray,
    seed_ids: np.ndarray,
    member_mask: np.ndarray,
    raw_label: int,
) -> dict[str, Any]:
    """在一个固定成员集合内，用 LL 权重计算中心路径和该模式的总质量。"""

    member_indices = np.flatnonzero(member_mask)
    if member_indices.size == 0:
        raise ValueError("模式成员不能为空")
    mode_mass = float(np.sum(weights[member_mask]))
    if mode_mass <= 0.0 or not np.isfinite(mode_mass):
        raise RuntimeError("模式质量必须是有限正数")
    within_mode_weights = weights[member_mask] / mode_mass
    center_path = np.sum(
        paths[member_mask] * within_mode_weights[:, None],
        axis=0,
        dtype=np.float64,
    )
    member_seed_ids = seed_ids[member_mask].astype(np.int64, copy=True)
    return {
        "raw_label": int(raw_label),
        "member_indices": member_indices.astype(np.int64, copy=False),
        "member_seed_ids": member_seed_ids,
        "center_path": center_path,
        "mass": mode_mass,
        "seed_count": int(member_indices.size),
        "mean_delta": float(np.mean(center_path)),
        "endpoint_delta": float(center_path[-1]),
        "minimum_seed_id": int(np.min(member_seed_ids)),
    }


def summarize_ordered_modes(
    seed_delta: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    raw_labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """计算三个 LL 加权中心，再以绝对路径位置命名 low/middle/high。"""

    paths = _as_finite_paths(seed_delta)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    labels = np.asarray(raw_labels, dtype=np.int64)
    number_of_seeds = paths.shape[0]
    if likelihoods.shape != (number_of_seeds,):
        raise ValueError("final_ll 与 seed 路径数量不一致")
    if ids.shape != (number_of_seeds,) or len(np.unique(ids)) != number_of_seeds:
        raise ValueError("seed_ids 必须与路径一一对应且不能重复")
    if labels.shape != (number_of_seeds,) or len(np.unique(labels)) != 3:
        raise ValueError("raw_labels 必须给每条 seed 分配到恰好三个模式")

    weights = scale8_weights(likelihoods)
    records: list[dict[str, Any]] = []
    for raw_label in sorted(np.unique(labels).tolist()):
        records.append(
            _summarize_one_membership(
                paths=paths,
                weights=weights,
                seed_ids=ids,
                member_mask=labels == raw_label,
                raw_label=int(raw_label),
            )
        )
    ordered = order_mode_records(records)
    return {name: record for name, record in zip(MODE_NAMES, ordered, strict=True)}


def fixed_mode_reweight(
    seed_delta: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    named_member_seed_ids: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """只更换 LL 权重，不重新聚类，也不改变 low/middle/high 的成员集合。"""

    paths = _as_finite_paths(seed_delta)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    ids = np.asarray(seed_ids, dtype=np.int64)
    if likelihoods.shape != (paths.shape[0],) or ids.shape != (paths.shape[0],):
        raise ValueError("fixed_mode_reweight 的路径、LL 和 seed_id 数量不一致")
    if set(named_member_seed_ids) != set(MODE_NAMES):
        raise ValueError("固定成员必须同时提供 low/middle/high")
    weights = scale8_weights(likelihoods)
    output: dict[str, dict[str, Any]] = {}
    covered_ids: list[int] = []
    for raw_label, name in enumerate(MODE_NAMES):
        member_ids = np.asarray(named_member_seed_ids[name], dtype=np.int64)
        member_mask = np.isin(ids, member_ids)
        if int(np.sum(member_mask)) != len(member_ids):
            raise ValueError(f"{name} 模式成员与当前 seed_ids 不一致")
        record = _summarize_one_membership(
            paths=paths,
            weights=weights,
            seed_ids=ids,
            member_mask=member_mask,
            raw_label=raw_label,
        )
        output[name] = record
        covered_ids.extend(record["member_seed_ids"].tolist())
    if sorted(covered_ids) != sorted(ids.tolist()):
        raise ValueError("固定模式成员没有恰好覆盖全部 seed")
    return output


def compute_p2_position(
    p2_delta: np.ndarray,
    low_path: np.ndarray,
    high_path: np.ndarray,
    degenerate_threshold_ft: float = FORMAL_DEGENERATE_SEPARATION_FT,
) -> tuple[float, float, bool]:
    """计算 P2 路径在低高模式均值之间的位置，并另存未裁剪与 [0,1] 版本。"""

    p2 = np.asarray(p2_delta, dtype=np.float64)
    low = np.asarray(low_path, dtype=np.float64)
    high = np.asarray(high_path, dtype=np.float64)
    if p2.ndim != 1 or p2.shape != low.shape or p2.shape != high.shape:
        raise ValueError("P2、low 和 high 必须是等长一维路径")
    if not np.isfinite(np.concatenate([p2, low, high])).all():
        raise ValueError("P2 或模式路径含 NaN/Inf")
    denominator = float(np.mean(high) - np.mean(low))
    if abs(denominator) < float(degenerate_threshold_ft):
        return 0.5, 0.5, True
    raw_position = float((np.mean(p2) - np.mean(low)) / denominator)
    clipped_position = float(np.clip(raw_position, 0.0, 1.0))
    return raw_position, clipped_position, False


def build_ordered_mode_features(
    seed_delta: np.ndarray,
    final_ll: np.ndarray,
    seed_ids: np.ndarray,
    p2_pred_tvt: np.ndarray,
    last_visible_tvt: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """完成固定聚类，输出三条模式路径、合法井级特征和成员诊断。"""

    paths = _as_finite_paths(seed_delta)
    if paths.shape[0] != FORMAL_NUMBER_OF_SEEDS:
        raise ValueError("正式 PFM01 必须恰好输入 128 条 seed 路径")
    ids = np.asarray(seed_ids, dtype=np.int64)
    likelihoods = np.asarray(final_ll, dtype=np.float64)
    p2_prediction = np.asarray(p2_pred_tvt, dtype=np.float64)
    if p2_prediction.shape != (paths.shape[1],):
        raise ValueError("P2 路径长度与 seed 路径长度不一致")
    if ids.shape != (FORMAL_NUMBER_OF_SEEDS,) or not np.array_equal(
        ids,
        np.arange(FORMAL_NUMBER_OF_SEEDS, dtype=np.int64),
    ):
        raise ValueError("正式 seed_ids 必须严格为 0～127")

    raw_labels = cluster_seed_descriptors(paths)
    modes = summarize_ordered_modes(paths, likelihoods, ids, raw_labels)
    memberships = {
        name: np.asarray(mode["member_seed_ids"], dtype=np.int64)
        for name, mode in modes.items()
    }
    rotated_likelihoods = np.roll(likelihoods, FORMAL_ROTATION)
    rotated_modes = fixed_mode_reweight(paths, rotated_likelihoods, ids, memberships)

    low_path = np.asarray(modes["low"]["center_path"], dtype=np.float64)
    middle_path = np.asarray(modes["middle"]["center_path"], dtype=np.float64)
    high_path = np.asarray(modes["high"]["center_path"], dtype=np.float64)
    p2_delta = p2_prediction - float(last_visible_tvt)
    raw_position, clipped_position, degenerate = compute_p2_position(
        p2_delta,
        low_path,
        high_path,
    )

    row_features = pd.DataFrame(
        {
            "pf_mode_low_delta": low_path,
            "pf_mode_middle_delta": middle_path,
            "pf_mode_high_delta": high_path,
        }
    )
    legal_summary: dict[str, Any] = {
        "low_mode_mass": float(modes["low"]["mass"]),
        "middle_mode_mass": float(modes["middle"]["mass"]),
        "high_mode_mass": float(modes["high"]["mass"]),
        "direction_score": float(modes["high"]["mass"] - modes["low"]["mass"]),
        "rotated_low_mode_mass": float(rotated_modes["low"]["mass"]),
        "rotated_middle_mode_mass": float(rotated_modes["middle"]["mass"]),
        "rotated_high_mode_mass": float(rotated_modes["high"]["mass"]),
        "rotated_direction_score": float(
            rotated_modes["high"]["mass"] - rotated_modes["low"]["mass"]
        ),
        "high_minus_low_separation": float(np.mean(high_path - low_path)),
        "high_minus_low_endpoint_separation": float(high_path[-1] - low_path[-1]),
        "p2_position_raw": raw_position,
        "p2_position_within_mode_envelope": clipped_position,
        "mode_envelope_degenerate": bool(degenerate),
        "low_seed_count": int(modes["low"]["seed_count"]),
        "middle_seed_count": int(modes["middle"]["seed_count"]),
        "high_seed_count": int(modes["high"]["seed_count"]),
        "minimum_mode_seed_count": int(
            min(modes[name]["seed_count"] for name in MODE_NAMES)
        ),
    }
    diagnostics = {
        "clustering_input": "full_path_mean_and_endpoint_delta_no_zscore",
        "ward_k": FORMAL_NUMBER_OF_MODES,
        "likelihood_scale": FORMAL_LIKELIHOOD_SCALE,
        "rotation": FORMAL_ROTATION,
        "memberships": {
            name: modes[name]["member_seed_ids"].astype(int).tolist()
            for name in MODE_NAMES
        },
        "raw_labels": raw_labels.astype(int).tolist(),
    }
    return row_features, legal_summary, diagnostics


def _safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    method: str,
) -> float:
    """方差为零时返回 NaN，否则计算固定方向的 Pearson 或 Spearman。"""

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"未知相关系数 {method}")


def compute_direction_metrics(
    direction_score: np.ndarray,
    mean_residual: np.ndarray,
    minimum_abs_residual: float = 2.0,
) -> dict[str, float | int]:
    """按预注册正向直接评价方向分数，绝不根据结果翻转符号。"""

    score = np.asarray(direction_score, dtype=np.float64)
    residual = np.asarray(mean_residual, dtype=np.float64)
    if score.ndim != 1 or score.shape != residual.shape or score.size == 0:
        raise ValueError("direction_score 与 mean_residual 必须是等长非空一维数组")
    if not np.isfinite(score).all() or not np.isfinite(residual).all():
        raise ValueError("方向分数或平均残差含 NaN/Inf")

    strong_mask = np.abs(residual) >= float(minimum_abs_residual)
    strong_score = score[strong_mask]
    strong_label = residual[strong_mask] > 0.0
    if strong_score.size == 0 or len(np.unique(strong_label)) < 2:
        auc = float("nan")
        balanced = float("nan")
        accuracy = float("nan")
    else:
        # 正类固定为 m>0，direction_score 越大越支持高 TVT；这里没有自动翻转。
        auc = float(roc_auc_score(strong_label, strong_score))
        predicted_label = strong_score > 0.0
        accuracy = float(accuracy_score(strong_label, predicted_label))
        balanced = float(balanced_accuracy_score(strong_label, predicted_label))
    return {
        "pearson": _safe_correlation(score, residual, "pearson"),
        "spearman": _safe_correlation(score, residual, "spearman"),
        "strong_wells": int(np.sum(strong_mask)),
        "strong_positive_wells": int(np.sum(strong_label)),
        "roc_auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }


def build_direction_quintiles(per_well: pd.DataFrame) -> list[dict[str, Any]]:
    """按方向分数五分位汇总真实平均残差，便于看信号是否单调。"""

    required = {"direction_score", "mean_residual"}
    if missing := required.difference(per_well.columns):
        raise ValueError(f"方向五分位缺列：{sorted(missing)}")
    ordered_rank = per_well["direction_score"].rank(method="first")
    # smoke 只有 1～3 口井，此时用能成立的最多分箱数；正式 131 井仍固定五分位。
    number_of_bins = min(5, len(per_well))
    quintile = pd.qcut(
        ordered_rank,
        q=number_of_bins,
        labels=False,
        duplicates="drop",
    )
    working = per_well.copy()
    working["direction_quintile"] = quintile.astype(int) + 1
    output: list[dict[str, Any]] = []
    for value, group in working.groupby("direction_quintile", sort=True):
        output.append(
            {
                "direction_quintile": int(value),
                "wells": int(len(group)),
                "direction_score_mean": float(group["direction_score"].mean()),
                "mean_residual_mean": float(group["mean_residual"].mean()),
                "positive_residual_fraction": float(group["mean_residual"].gt(0).mean()),
            }
        )
    return output


def compute_binary_slice_metrics(
    per_well: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    """按指定合法量的中位数分成低/高两组，并分别报告固定方向指标。"""

    if column not in per_well:
        raise ValueError(f"切片列不存在：{column}")
    threshold = float(per_well[column].median())
    output: list[dict[str, Any]] = []
    for name, mask in (
        ("low", per_well[column] <= threshold),
        ("high", per_well[column] > threshold),
    ):
        group = per_well.loc[mask]
        if group.empty:
            metrics: dict[str, Any] = {
                "pearson": float("nan"),
                "spearman": float("nan"),
                "strong_wells": 0,
                "strong_positive_wells": 0,
                "roc_auc": float("nan"),
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
            }
        else:
            metrics = compute_direction_metrics(
                group["direction_score"].to_numpy(),
                group["mean_residual"].to_numpy(),
            )
        output.append(
            {
                "slice_column": column,
                "slice": name,
                "threshold": threshold,
                "wells": int(len(group)),
                **metrics,
            }
        )
    return output


def compute_three_vs_five_oracle(per_well_sse: pd.DataFrame) -> dict[str, Any]:
    """逐井挑选三模式或五温度中的最小 SSE，再按全部评价行汇总 micro RMSE。"""

    mode_columns = [f"{name}_sse" for name in MODE_NAMES]
    old_columns = [f"{name}_sse" for name in OLD_TEMPERATURE_NAMES]
    required = {"hidden_rows", *mode_columns, *old_columns}
    if missing := required.difference(per_well_sse.columns):
        raise ValueError(f"oracle SSE 表缺列：{sorted(missing)}")
    if per_well_sse.empty:
        raise ValueError("oracle SSE 表不能为空")
    values = per_well_sse[[*mode_columns, *old_columns]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("oracle SSE 必须是非负有限数")
    hidden_rows = per_well_sse["hidden_rows"].to_numpy(dtype=np.int64)
    if np.any(hidden_rows <= 0):
        raise ValueError("每井 hidden_rows 必须为正")

    mode_sse = per_well_sse[mode_columns].to_numpy(dtype=np.float64)
    old_sse = per_well_sse[old_columns].to_numpy(dtype=np.float64)
    best_mode_index = np.argmin(mode_sse, axis=1)
    best_old_index = np.argmin(old_sse, axis=1)
    row_number = np.arange(len(per_well_sse))
    best_mode_sse = mode_sse[row_number, best_mode_index]
    best_old_sse = old_sse[row_number, best_old_index]
    total_rows = int(np.sum(hidden_rows))
    three_rmse = float(math.sqrt(float(np.sum(best_mode_sse)) / total_rows))
    five_rmse = float(math.sqrt(float(np.sum(best_old_sse)) / total_rows))

    positive_gain = np.maximum(best_old_sse - best_mode_sse, 0.0)
    strongest_count = max(1, int(math.ceil(0.05 * len(per_well_sse))))
    strongest_gain = float(np.sum(np.sort(positive_gain)[::-1][:strongest_count]))
    total_positive_gain = float(np.sum(positive_gain))
    strongest_share = strongest_gain / total_positive_gain if total_positive_gain > 0 else 0.0
    best_mode_names = np.asarray(MODE_NAMES, dtype=object)[best_mode_index]
    return {
        "three_mode_oracle_pooled_rmse": three_rmse,
        "five_temperature_oracle_pooled_rmse": five_rmse,
        "three_minus_five_oracle_rmse": three_rmse - five_rmse,
        "three_better_well_fraction": float(np.mean(best_mode_sse < best_old_sse)),
        "low_best_fraction": float(np.mean(best_mode_names == "low")),
        "middle_best_fraction": float(np.mean(best_mode_names == "middle")),
        "high_best_fraction": float(np.mean(best_mode_names == "high")),
        "strongest_5pct_positive_gain_share": strongest_share,
        "strongest_5pct_wells": strongest_count,
        "positive_gain_wells": int(np.sum(positive_gain > 0.0)),
    }
