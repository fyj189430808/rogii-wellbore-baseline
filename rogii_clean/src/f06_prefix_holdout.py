"""F06：用当前井可见前缀制造六个伪隐藏段，衡量五类候选的历史可靠性。"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.f05a_candidate_reproduction import build_candidate_features


F06_CUT_NAMES = (
    "ratio50",
    "ratio65",
    "ratio75",
    "horizon250",
    "horizon500",
    "horizon1000",
)

F06_FAMILY_NAMES = ("carry", "geometry", "pf", "beam", "typewell")

# 每个候选族与 F05a 固定候选列的映射。carry 不读取候选列，始终预测增量为 0。
F06_FAMILY_SOURCE_COLUMNS = {
    "carry": None,
    "geometry": "pf_z_delta",
    "pf": "pf_ancc_delta",
    "beam": "beam_mean_d",
    "typewell": "sc_ens_d",
}

_CUT_METRICS = ("rmse", "gain_vs_carry", "gain_vs_pf", "rank", "coverage")
_AGGREGATE_METRICS = (
    "median_gain",
    "worst_gain",
    "gain_std",
    "gain_slope_vs_actual_horizon",
)
_GLOBAL_FEATURES = (
    "f06_rank_spearman_mean",
    "f06_rank_kendall_mean",
    "f06_winner_stability",
    "f06_modal_top2_set_rate",
)

F06_FEATURE_COLUMNS = tuple(
    f"f06_{cut_name}_{family_name}_{metric_name}"
    for cut_name in F06_CUT_NAMES
    for family_name in F06_FAMILY_NAMES
    for metric_name in _CUT_METRICS
) + tuple(
    f"f06_{family_name}_{metric_name}"
    for family_name in F06_FAMILY_NAMES
    for metric_name in _AGGREGATE_METRICS
) + _GLOBAL_FEATURES


def _validate_and_prepare(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    """只保留测试期合法列，并检查 TVT_input 是连续的可见前缀。"""

    required_horizontal = ("MD", "Z", "GR", "TVT_input")
    required_typewell = ("TVT", "GR")
    missing_horizontal = set(required_horizontal) - set(horizontal_df.columns)
    missing_typewell = set(required_typewell) - set(typewell_df.columns)
    if missing_horizontal:
        raise ValueError(f"水平井缺少列：{sorted(missing_horizontal)}")
    if missing_typewell:
        raise ValueError(f"Typewell 缺少列：{sorted(missing_typewell)}")

    normalized_seed = int(seed)
    if normalized_seed < 0 or normalized_seed >= 2**31 - 1:
        raise ValueError("seed 必须位于 [0, 2**31-1) 内")

    # 主动丢弃 TVT、surface 等列，确保后续函数不可能读取自然隐藏真值。
    horizontal = horizontal_df.loc[:, required_horizontal].copy()
    for column in required_horizontal:
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    typewell = typewell_df.loc[:, required_typewell].copy()

    visible_mask = horizontal["TVT_input"].notna().to_numpy()
    visible_count = int(visible_mask.sum())
    if visible_count < 20:
        raise ValueError("F06 至少需要 20 行可见前缀，才能保留 50% 前缀重建候选")
    if visible_mask[visible_count:].any() or not visible_mask[:visible_count].all():
        raise ValueError("TVT_input 必须是连续可见前缀，后面只能是自然隐藏段")

    visible_md = horizontal.loc[visible_mask, "MD"].to_numpy(dtype=np.float64)
    visible_tvt = horizontal.loc[visible_mask, "TVT_input"].to_numpy(dtype=np.float64)
    if not np.isfinite(visible_md).all() or not np.isfinite(visible_tvt).all():
        raise ValueError("可见前缀的 MD 或 TVT_input 含非有限值")
    if np.any(np.diff(visible_md) <= 0.0):
        raise ValueError("可见前缀 MD 必须严格递增")
    return horizontal, typewell, visible_md, visible_tvt, normalized_seed


def _cut_visible_counts(visible_md: np.ndarray) -> dict[str, int | None]:
    """返回每个 cut 保留的可见行数；物理跨度不足时返回 None。"""

    visible_count = len(visible_md)
    cuts: dict[str, int | None] = {
        "ratio50": int(np.floor(visible_count * 0.50)),
        "ratio65": int(np.floor(visible_count * 0.65)),
        "ratio75": int(np.floor(visible_count * 0.75)),
    }
    total_span = float(visible_md[-1] - visible_md[0])
    for horizon in (250.0, 500.0, 1000.0):
        cut_name = f"horizon{int(horizon)}"
        if total_span < horizon:
            cuts[cut_name] = None
            continue
        target_md = visible_md[-1] - horizon
        anchor_position = int(np.argmin(np.abs(visible_md - target_md)))
        cut_visible_count = anchor_position + 1
        if cut_visible_count < 10 or cut_visible_count >= visible_count:
            cuts[cut_name] = None
        else:
            cuts[cut_name] = cut_visible_count
    return cuts


def _empty_cut_metrics() -> dict[str, dict[str, float]]:
    """为一个不支持的 cut 生成固定的 NaN/零覆盖结构。"""

    return {
        family_name: {
            "rmse": np.nan,
            "gain_vs_carry": np.nan,
            "gain_vs_pf": np.nan,
            "rank": np.nan,
            "coverage": 0.0,
        }
        for family_name in F06_FAMILY_NAMES
    }


def _score_one_cut(
    original_horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    visible_count: int,
    original_visible_tvt: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """隐藏 cut 后的 TVT_input、重建候选，并只在原可见伪 holdout 行上评分。"""

    rebuilt_horizontal = original_horizontal.copy()
    rebuilt_horizontal.iloc[visible_count:, rebuilt_horizontal.columns.get_loc("TVT_input")] = np.nan
    candidates = build_candidate_features(rebuilt_horizontal, typewell, seed=seed)
    if "row_index" not in candidates.columns:
        raise ValueError("候选重建结果缺少 row_index")
    if candidates["row_index"].duplicated().any():
        raise ValueError("候选重建结果的 row_index 重复")

    candidate_by_row = candidates.set_index("row_index")
    pseudo_indices = original_horizontal.index[visible_count : len(original_visible_tvt)]
    anchor_tvt = float(original_visible_tvt[visible_count - 1])
    true_delta = original_visible_tvt[visible_count:] - anchor_tvt
    pseudo_count = len(true_delta)
    if pseudo_count == 0:
        return _empty_cut_metrics()

    predictions: dict[str, np.ndarray] = {
        "carry": np.zeros(pseudo_count, dtype=np.float64)
    }
    for family_name in F06_FAMILY_NAMES[1:]:
        source_column = F06_FAMILY_SOURCE_COLUMNS[family_name]
        if source_column not in candidate_by_row.columns:
            raise ValueError(f"候选重建结果缺少 {source_column}")
        predictions[family_name] = candidate_by_row[source_column].reindex(
            pseudo_indices
        ).to_numpy(dtype=np.float64)

    metrics = _empty_cut_metrics()
    for family_name, predicted_delta in predictions.items():
        valid_mask = np.isfinite(predicted_delta) & np.isfinite(true_delta)
        coverage = float(valid_mask.mean())
        metrics[family_name]["coverage"] = coverage
        if coverage >= 0.95:
            squared_error = (predicted_delta[valid_mask] - true_delta[valid_mask]) ** 2
            metrics[family_name]["rmse"] = float(np.sqrt(np.mean(squared_error)))

    carry_rmse = metrics["carry"]["rmse"]
    pf_rmse = metrics["pf"]["rmse"]
    valid_families = [
        family_name
        for family_name in F06_FAMILY_NAMES
        if np.isfinite(metrics[family_name]["rmse"])
    ]
    if valid_families:
        rmse_series = pd.Series(
            {family_name: metrics[family_name]["rmse"] for family_name in valid_families}
        )
        ranks = rmse_series.rank(method="average", ascending=True)
        for family_name in valid_families:
            family_rmse = metrics[family_name]["rmse"]
            metrics[family_name]["rank"] = float(ranks[family_name])
            if np.isfinite(carry_rmse):
                metrics[family_name]["gain_vs_carry"] = float(carry_rmse - family_rmse)
            if np.isfinite(pf_rmse):
                metrics[family_name]["gain_vs_pf"] = float(pf_rmse - family_rmse)
    return metrics


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """排名本身已是秩，故 Spearman 等于两个排名向量的 Pearson 相关。"""

    valid = np.isfinite(first) & np.isfinite(second)
    first_valid = first[valid]
    second_valid = second[valid]
    if len(first_valid) < 2:
        return np.nan
    if np.std(first_valid) == 0.0 or np.std(second_valid) == 0.0:
        return np.nan
    return float(np.corrcoef(first_valid, second_valid)[0, 1])


def _kendall_tau_b(first: np.ndarray, second: np.ndarray) -> float:
    """计算含并列排名时的 Kendall tau-b，不额外依赖 scipy。"""

    valid = np.isfinite(first) & np.isfinite(second)
    x = first[valid]
    y = second[valid]
    if len(x) < 2:
        return np.nan
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for left in range(len(x) - 1):
        for right in range(left + 1, len(x)):
            difference_x = x[left] - x[right]
            difference_y = y[left] - y[right]
            if difference_x == 0.0 and difference_y == 0.0:
                continue
            if difference_x == 0.0:
                ties_x += 1
            elif difference_y == 0.0:
                ties_y += 1
            elif difference_x * difference_y > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denominator == 0.0:
        return np.nan
    return float((concordant - discordant) / denominator)


def _global_consistency_features(
    all_cut_metrics: dict[str, dict[str, dict[str, float]]]
) -> dict[str, float]:
    """汇总 cut 两两排名相关、赢家稳定率和众数 top2 集合占比。"""

    rank_vectors: list[np.ndarray] = []
    winners: list[str] = []
    top2_sets: list[frozenset[str]] = []
    for cut_name in F06_CUT_NAMES:
        cut_metrics = all_cut_metrics[cut_name]
        ranks = np.asarray(
            [cut_metrics[family]["rank"] for family in F06_FAMILY_NAMES],
            dtype=np.float64,
        )
        rank_vectors.append(ranks)
        valid_positions = np.flatnonzero(np.isfinite(ranks))
        if len(valid_positions) > 0:
            winner_position = min(valid_positions, key=lambda position: ranks[position])
            winners.append(F06_FAMILY_NAMES[winner_position])
        if len(valid_positions) >= 2:
            ordered_positions = sorted(valid_positions, key=lambda position: ranks[position])
            top2_sets.append(
                frozenset(F06_FAMILY_NAMES[position] for position in ordered_positions[:2])
            )

    spearman_values: list[float] = []
    kendall_values: list[float] = []
    for left in range(len(rank_vectors) - 1):
        for right in range(left + 1, len(rank_vectors)):
            spearman = _rank_correlation(rank_vectors[left], rank_vectors[right])
            kendall = _kendall_tau_b(rank_vectors[left], rank_vectors[right])
            if np.isfinite(spearman):
                spearman_values.append(spearman)
            if np.isfinite(kendall):
                kendall_values.append(kendall)

    winner_stability = np.nan
    if winners:
        winner_stability = max(Counter(winners).values()) / len(winners)
    top2_stability = np.nan
    if top2_sets:
        top2_stability = max(Counter(top2_sets).values()) / len(top2_sets)
    return {
        "f06_rank_spearman_mean": (
            float(np.mean(spearman_values)) if spearman_values else np.nan
        ),
        "f06_rank_kendall_mean": (
            float(np.mean(kendall_values)) if kendall_values else np.nan
        ),
        "f06_winner_stability": float(winner_stability),
        "f06_modal_top2_set_rate": float(top2_stability),
    }


def build_prefix_holdout_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """返回一行井级 F06 特征；自然隐藏 TVT 和 surface 不会被读取或传入候选器。"""

    horizontal, typewell, visible_md, visible_tvt, normalized_seed = _validate_and_prepare(
        horizontal_df,
        typewell_df,
        seed,
    )
    cut_counts = _cut_visible_counts(visible_md)
    all_cut_metrics: dict[str, dict[str, dict[str, float]]] = {}
    actual_horizons: dict[str, float] = {}
    for cut_name in F06_CUT_NAMES:
        cut_visible_count = cut_counts[cut_name]
        if cut_visible_count is None:
            all_cut_metrics[cut_name] = _empty_cut_metrics()
            actual_horizons[cut_name] = np.nan
            continue
        all_cut_metrics[cut_name] = _score_one_cut(
            horizontal,
            typewell,
            cut_visible_count,
            visible_tvt,
            normalized_seed,
        )
        actual_horizons[cut_name] = float(
            visible_md[-1] - visible_md[cut_visible_count - 1]
        )

    features: dict[str, float] = {}
    for cut_name in F06_CUT_NAMES:
        for family_name in F06_FAMILY_NAMES:
            for metric_name in _CUT_METRICS:
                feature_name = f"f06_{cut_name}_{family_name}_{metric_name}"
                features[feature_name] = all_cut_metrics[cut_name][family_name][metric_name]

    for family_name in F06_FAMILY_NAMES:
        gains = np.asarray(
            [
                all_cut_metrics[cut_name][family_name]["gain_vs_carry"]
                for cut_name in F06_CUT_NAMES
            ],
            dtype=np.float64,
        )
        horizons = np.asarray(
            [actual_horizons[cut_name] for cut_name in F06_CUT_NAMES],
            dtype=np.float64,
        )
        valid = np.isfinite(gains) & np.isfinite(horizons)
        if valid.any():
            features[f"f06_{family_name}_median_gain"] = float(np.median(gains[valid]))
            features[f"f06_{family_name}_worst_gain"] = float(np.min(gains[valid]))
            features[f"f06_{family_name}_gain_std"] = float(np.std(gains[valid]))
        else:
            features[f"f06_{family_name}_median_gain"] = np.nan
            features[f"f06_{family_name}_worst_gain"] = np.nan
            features[f"f06_{family_name}_gain_std"] = np.nan

        slope = np.nan
        if int(valid.sum()) >= 2:
            valid_horizons = horizons[valid]
            valid_gains = gains[valid]
            centered_horizons = valid_horizons - valid_horizons.mean()
            denominator = float(np.sum(centered_horizons**2))
            if denominator > 0.0:
                slope = float(
                    np.sum(centered_horizons * (valid_gains - valid_gains.mean()))
                    / denominator
                )
        features[f"f06_{family_name}_gain_slope_vs_actual_horizon"] = slope

    features.update(_global_consistency_features(all_cut_metrics))
    return pd.DataFrame([{column: features[column] for column in F06_FEATURE_COLUMNS}])

