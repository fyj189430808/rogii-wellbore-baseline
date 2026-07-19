"""P3-D01 的单井路径评分与井级统计分析纯函数。

本模块只接收内存中的表格，不读取文件、不训练模型，也不生成正式特征。
oracle 字段只在这里的评分结果中出现，用于判断固定温度路径的理论改进空间。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


TEMPERATURES = (3, 5, 8, 12)
PATH_NAMES = (
    "p2p02",
    "pf128_mean",
    "scale_3",
    "scale_5",
    "scale_8",
    "scale_12",
    "oracle_scale",
)
BIN_COLUMNS = (
    "hidden_rows",
    "missing_gr_fraction",
    "longest_gr_gap_md_ft",
    "p2p02_rmse",
)


def _require_columns(frame: pd.DataFrame, required_columns: Sequence[str], frame_name: str) -> None:
    """确认表格包含本次计算必需的列。"""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} 缺少列：{missing_columns}")


def _finite_values(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    """拒绝无法转换为数值或包含 NaN、正负无穷的输入。"""
    for column in columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"{frame_name}.{column} 含非有限值")


def _json_scalar(value: Any) -> Any:
    """把 numpy 标量转换成标准 json 能直接写出的 Python 标量。"""
    if isinstance(value, np.generic):
        return value.item()
    return value


def score_well_paths(target_rows: pd.DataFrame, legal_path_cache: pd.DataFrame) -> dict[str, Any]:
    """按 row_index 对齐一口井的真值和合法路径，并计算每条路径的 SSE/RMSE。

    输入中的 ``target_rows`` 只允许来自已经解锁的开发井评分区；
    ``legal_path_cache`` 只包含不看隐藏真值生成的路径。函数本身不接触文件系统。
    """
    target_columns = ("well_id", "fold", "row_index", "target_tvt", "pred_tvt")
    legal_columns = (
        "row_index",
        "last_visible_tvt",
        "pf128_mean_tvt",
        "pf128_scale_3_delta",
        "pf128_scale_5_delta",
        "pf128_scale_8_delta",
        "pf128_scale_12_delta",
    )
    _require_columns(target_rows, target_columns, "target_rows")
    _require_columns(legal_path_cache, legal_columns, "legal_path_cache")
    if target_rows.empty:
        raise ValueError("target_rows 不能为空")
    if target_rows["well_id"].isna().any() or target_rows["well_id"].nunique(dropna=False) != 1:
        raise ValueError("target_rows 必须只包含一口井")
    if target_rows["fold"].isna().any() or target_rows["fold"].nunique(dropna=False) != 1:
        raise ValueError("target_rows 必须只包含一个 fold")
    if target_rows["row_index"].isna().any() or target_rows["row_index"].duplicated().any():
        raise ValueError("target_rows.row_index 必须非空且唯一")
    if legal_path_cache["row_index"].isna().any() or legal_path_cache["row_index"].duplicated().any():
        raise ValueError("legal_path_cache.row_index 必须非空且唯一")
    _finite_values(target_rows, ("row_index", "target_tvt", "pred_tvt"), "target_rows")
    _finite_values(legal_path_cache, legal_columns, "legal_path_cache")

    # 外连接只用于审计键集合；任何只出现在一侧的行都会使评分失去一对一含义。
    key_audit = target_rows[["row_index"]].merge(
        legal_path_cache[["row_index"]],
        on="row_index",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not key_audit["_merge"].eq("both").all():
        raise ValueError("target_rows 与 legal_path_cache 的 row_index 集合不完全相同")

    joined = target_rows.loc[:, target_columns].merge(
        legal_path_cache.loc[:, legal_columns],
        on="row_index",
        how="inner",
        validate="one_to_one",
    )
    target_tvt = joined["target_tvt"].to_numpy(dtype=np.float64)
    last_visible_tvt = joined["last_visible_tvt"].to_numpy(dtype=np.float64)
    path_predictions = {
        "p2p02": joined["pred_tvt"].to_numpy(dtype=np.float64),
        "pf128_mean": joined["pf128_mean_tvt"].to_numpy(dtype=np.float64),
    }
    for temperature in TEMPERATURES:
        scale_delta = joined[f"pf128_scale_{temperature}_delta"].to_numpy(dtype=np.float64)
        # 这里必须保持旧 P2-P01 的 float64(last_visible_tvt) + float64(delta) 评分语义。
        path_predictions[f"scale_{temperature}"] = last_visible_tvt + scale_delta

    hidden_rows = int(len(joined))
    result: dict[str, Any] = {
        "well_id": _json_scalar(target_rows["well_id"].iloc[0]),
        "fold": _json_scalar(target_rows["fold"].iloc[0]),
        "hidden_rows": hidden_rows,
    }
    for path_name, prediction in path_predictions.items():
        residual = prediction - target_tvt
        sum_squared_error = float(np.dot(residual, residual))
        result[f"{path_name}_sse"] = sum_squared_error
        result[f"{path_name}_rmse"] = float(np.sqrt(sum_squared_error / hidden_rows))

    # np.argmin 会在平手时返回第一个位置，因此自然执行 3、5、8、12 的固定优先级。
    scale_errors = np.asarray([result[f"scale_{temperature}_sse"] for temperature in TEMPERATURES])
    best_position = int(np.argmin(scale_errors))
    best_temperature = int(TEMPERATURES[best_position])
    result["oracle_best_scale"] = best_temperature
    result["oracle_scale_sse"] = float(result[f"scale_{best_temperature}_sse"])
    result["oracle_scale_rmse"] = float(result[f"scale_{best_temperature}_rmse"])
    result["best_scale_is_oracle"] = True
    return result


def aggregate_path_metrics(per_well: pd.DataFrame) -> pd.DataFrame:
    """汇总各路径的 pooled 和逐井指标；oracle_scale 只代表事后上限。"""
    required_columns = ["well_id", "hidden_rows"]
    for path_name in PATH_NAMES:
        required_columns.extend((f"{path_name}_sse", f"{path_name}_rmse"))
    _require_columns(per_well, required_columns, "per_well")
    if per_well.empty:
        raise ValueError("per_well 不能为空")
    if per_well["well_id"].isna().any() or per_well["well_id"].duplicated().any():
        raise ValueError("per_well 每口井必须恰好一行")
    numeric_columns = ["hidden_rows"]
    for path_name in PATH_NAMES:
        numeric_columns.extend((f"{path_name}_sse", f"{path_name}_rmse"))
    _finite_values(per_well, numeric_columns, "per_well")
    hidden_rows = per_well["hidden_rows"].to_numpy(dtype=np.float64)
    if np.any(hidden_rows <= 0):
        raise ValueError("hidden_rows 必须大于 0")

    metric_rows: list[dict[str, Any]] = []
    total_rows = int(np.sum(hidden_rows))
    for path_name in PATH_NAMES:
        sse_values = per_well[f"{path_name}_sse"].to_numpy(dtype=np.float64)
        rmse_values = per_well[f"{path_name}_rmse"].to_numpy(dtype=np.float64)
        if np.any(sse_values < 0) or np.any(rmse_values < 0):
            raise ValueError(f"{path_name} 的 SSE/RMSE 不能为负数")
        total_sse = float(np.sum(sse_values))
        metric_rows.append(
            {
                "path_name": path_name,
                "well_count": int(len(per_well)),
                "row_count": total_rows,
                "sse": total_sse,
                "pooled_rmse": float(np.sqrt(total_sse / total_rows)),
                "macro_well_rmse": float(np.mean(rmse_values)),
                "median_well_rmse": float(np.median(rmse_values)),
                "p90_well_rmse": float(np.quantile(rmse_values, 0.90)),
                "worst_well_rmse": float(np.max(rmse_values)),
                "is_oracle_upper_bound": path_name == "oracle_scale",
            }
        )
    return pd.DataFrame(metric_rows)


def _rank_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """计算平均秩后的 Pearson 相关，即带并列值处理的 Spearman rho。"""
    x_rank = pd.Series(x_values).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_values).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def spearman_report(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    permutations: int = 1000,
    seed: int = 29,
) -> dict[str, Any]:
    """计算井级 Spearman 相关和可复现的双侧置换比例。"""
    _require_columns(frame, (x_column, y_column), "frame")
    if isinstance(permutations, bool) or not isinstance(permutations, (int, np.integer)):
        raise ValueError("permutations 必须是非负整数")
    if int(permutations) < 0:
        raise ValueError("permutations 不能小于 0")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed 必须是整数")

    x_numeric = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=np.float64)
    y_numeric = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(x_numeric) & np.isfinite(y_numeric)
    x_values = x_numeric[finite_mask]
    y_values = y_numeric[finite_mask]
    number_of_pairs = int(finite_mask.sum())
    base_result: dict[str, Any] = {
        "x_column": x_column,
        "y_column": y_column,
        "n_pairs": number_of_pairs,
        "valid": False,
        "rho": None,
        "permutation_p": None,
        "permutations": int(permutations),
        "seed": int(seed),
    }
    if number_of_pairs < 3:
        return base_result
    if np.all(x_values == x_values[0]) or np.all(y_values == y_values[0]):
        return base_result

    observed_rho = _rank_correlation(x_values, y_values)
    x_rank = pd.Series(x_values).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_values).rank(method="average").to_numpy(dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    number_of_permutations = int(permutations)
    if number_of_permutations == 0:
        exceedances = 0
    else:
        # 随机排列的生成顺序仍与旧循环完全一致；这里只把 1000 次
        # np.corrcoef 合并成一次矩阵乘法，避免全量分析耗费数分钟。
        repeated_y_ranks = np.broadcast_to(
            y_rank,
            (number_of_permutations, number_of_pairs),
        ).copy()
        # Generator.permuted 会独立打乱每一行；同一 seed 下的结果与逐行
        # 调用 generator.permutation 完全相同，但循环在 NumPy 内部完成。
        permuted_y_ranks = generator.permuted(repeated_y_ranks, axis=1)
        centered_x_rank = x_rank - np.mean(x_rank)
        centered_y_rank = y_rank - np.mean(y_rank)
        denominator = float(
            np.sqrt(
                np.dot(centered_x_rank, centered_x_rank)
                * np.dot(centered_y_rank, centered_y_rank)
            )
        )
        permutation_correlations = (
            (permuted_y_ranks - np.mean(y_rank)) @ centered_x_rank
        ) / denominator
        exceedances = int(
            np.count_nonzero(
                np.abs(permutation_correlations) >= abs(observed_rho)
            )
        )
    permutation_p = (1 + exceedances) / (number_of_permutations + 1)
    base_result.update(
        {
            "valid": True,
            "rho": float(observed_rho),
            "permutation_p": float(permutation_p),
        }
    )
    return base_result


def _stable_seed(base_seed: int, *parts: Any) -> int:
    """通过 SHA-256 派生随机种子，避免 Python hash 在不同进程变化。"""
    payload = json.dumps([int(base_seed), *parts], ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def correlation_tables(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    permutations: int = 1000,
    seed: int = 29,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别计算全开发集和每个 fold 的相关，随后统计折方向一致数。"""
    _require_columns(frame, ("fold",), "frame")
    if frame["fold"].isna().any():
        raise ValueError("frame.fold 不能缺失")
    if not pairs:
        raise ValueError("pairs 不能为空")
    fold_values = sorted(frame["fold"].drop_duplicates().tolist(), key=lambda value: str(value))
    overall_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for pair_index, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError("pairs 的每一项必须恰好包含 x 和 y 两个列名")
        x_column, y_column = pair
        overall_seed = _stable_seed(seed, "overall", pair_index, x_column, y_column)
        overall_report = spearman_report(frame, x_column, y_column, permutations, overall_seed)
        pair_fold_reports: list[dict[str, Any]] = []
        for fold_value in fold_values:
            fold_frame = frame.loc[frame["fold"] == fold_value]
            fold_seed = _stable_seed(seed, "fold", pair_index, x_column, y_column, fold_value)
            fold_report = spearman_report(fold_frame, x_column, y_column, permutations, fold_seed)
            fold_report["fold"] = _json_scalar(fold_value)
            pair_fold_reports.append(fold_report)
            fold_rows.append(fold_report)

        valid_fold_reports = [report for report in pair_fold_reports if report["valid"]]
        same_direction_folds = 0
        overall_rho = overall_report["rho"]
        if overall_report["valid"] and overall_rho != 0.0:
            for report in valid_fold_reports:
                fold_rho = report["rho"]
                if fold_rho != 0.0 and np.sign(fold_rho) == np.sign(overall_rho):
                    same_direction_folds += 1
        overall_report["valid_folds"] = int(len(valid_fold_reports))
        overall_report["same_direction_folds"] = int(same_direction_folds)
        overall_rows.append(overall_report)

    overall_table = pd.DataFrame(overall_rows)
    fold_table = pd.DataFrame(fold_rows)
    # 表格允许用 NaN 表示无法估计；汇总 dict 则由 spearman_report 使用 None 保证 JSON 安全。
    for table in (overall_table, fold_table):
        table["rho"] = pd.to_numeric(table["rho"], errors="coerce")
        table["permutation_p"] = pd.to_numeric(table["permutation_p"], errors="coerce")
    return overall_table, fold_table


def add_diagnostic_columns(per_well: pd.DataFrame) -> pd.DataFrame:
    """按预登记公式添加缺失率、sigma 差异、权重塌缩和过尖锐标记。"""
    required_columns = [
        "observed_gr_fraction",
        "gr_sigma",
        "observed_only_gr_sigma",
        "scale_3_rmse",
        "scale_12_rmse",
        "pf128_mean_rmse",
    ]
    required_columns.extend(f"scale_{temperature}_effective_sample_size" for temperature in TEMPERATURES)
    _require_columns(per_well, required_columns, "per_well")
    _finite_values(per_well, required_columns, "per_well")
    result = per_well.copy()
    observed_fraction = result["observed_gr_fraction"].to_numpy(dtype=np.float64)
    if np.any((observed_fraction < 0.0) | (observed_fraction > 1.0)):
        raise ValueError("observed_gr_fraction 必须位于 [0, 1]")
    result["missing_gr_fraction"] = 1.0 - observed_fraction
    observed_sigma = result["observed_only_gr_sigma"].to_numpy(dtype=np.float64)
    all_sigma = result["gr_sigma"].to_numpy(dtype=np.float64)
    denominator = np.maximum(np.abs(observed_sigma), 1.0e-12)
    result["sigma_absolute_relative_difference"] = np.abs(all_sigma - observed_sigma) / denominator
    for temperature in TEMPERATURES:
        ess_column = f"scale_{temperature}_effective_sample_size"
        ess_values = result[ess_column].to_numpy(dtype=np.float64)
        if np.any((ess_values < 1.0) | (ess_values > 128.0)):
            raise ValueError(f"{ess_column} 必须位于 128 粒子允许的 [1, 128] 区间")
        result[f"scale_{temperature}_collapsed_severe"] = ess_values <= 2.0
        result[f"scale_{temperature}_collapsed_medium"] = ess_values <= 8.0
    better_smooth_rmse = np.minimum(
        result["scale_12_rmse"].to_numpy(dtype=np.float64),
        result["pf128_mean_rmse"].to_numpy(dtype=np.float64),
    )
    result["scale_3_easy_oversharp"] = (
        result["scale_3_effective_sample_size"].to_numpy(dtype=np.float64) <= 2.0
    ) & (result["scale_3_rmse"].to_numpy(dtype=np.float64) - better_smooth_rmse >= 0.5)
    return result


def _assign_stable_quartiles(frame: pd.DataFrame, value_column: str) -> pd.Series:
    """按数值和井号稳定排序，再平衡分到四组；输入换行顺序不会改变结果。"""
    if frame.empty:
        raise ValueError("无法对空表分组")
    # DataFrame 的 index 只是外部标签，可能重复，不能拿它当作一行的唯一身份。
    reset_frame = frame.reset_index(drop=True)
    sorting_frame = pd.DataFrame(
        {
            "value": pd.to_numeric(reset_frame[value_column], errors="coerce"),
            "well_sort_key": reset_frame["well_id"].astype(str),
            "row_position": np.arange(len(reset_frame), dtype=np.int64),
        }
    )
    if not np.isfinite(sorting_frame["value"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{value_column} 含非有限值")
    ordered_indices = sorting_frame.sort_values(
        ["value", "well_sort_key"], kind="mergesort"
    )["row_position"].to_numpy(dtype=np.int64)
    number_of_rows = len(ordered_indices)
    quartiles = np.empty(number_of_rows, dtype=np.int64)
    for position, row_position in enumerate(ordered_indices):
        bin_number = min(4, (4 * position) // number_of_rows + 1)
        quartiles[row_position] = bin_number
    return pd.Series(quartiles, index=reset_frame.index, dtype=np.int64)


def build_binned_metrics(per_well: pd.DataFrame) -> pd.DataFrame:
    """按四个预登记变量稳定分四组，计算路径误差、ESS 和塌缩比例。"""
    if "missing_gr_fraction" not in per_well.columns:
        work = add_diagnostic_columns(per_well)
    else:
        work = per_well.copy()
    required_columns = [
        "well_id",
        "hidden_rows",
        "pf128_mean_sse",
        "p2p02_sse",
        *BIN_COLUMNS,
    ]
    for temperature in TEMPERATURES:
        required_columns.extend(
            (
                f"scale_{temperature}_effective_sample_size",
                f"scale_{temperature}_collapsed_severe",
                f"scale_{temperature}_collapsed_medium",
            )
        )
    _require_columns(work, required_columns, "per_well")
    if work.empty:
        raise ValueError("per_well 不能为空")
    if work["well_id"].isna().any() or work["well_id"].duplicated().any():
        raise ValueError("per_well 每口井必须恰好一行")
    numeric_columns = ["hidden_rows", "pf128_mean_sse", "p2p02_sse", *BIN_COLUMNS]
    numeric_columns.extend(f"scale_{temperature}_effective_sample_size" for temperature in TEMPERATURES)
    _finite_values(work, numeric_columns, "per_well")
    if np.any(work["hidden_rows"].to_numpy(dtype=np.float64) <= 0):
        raise ValueError("hidden_rows 必须大于 0")
    if np.any(work[["pf128_mean_sse", "p2p02_sse"]].to_numpy(dtype=np.float64) < 0):
        raise ValueError("SSE 不能为负数")

    binned_rows: list[dict[str, Any]] = []
    for bin_column in BIN_COLUMNS:
        bin_numbers = _assign_stable_quartiles(work, bin_column)
        for bin_number in range(1, 5):
            selected_positions = np.flatnonzero(bin_numbers.to_numpy() == bin_number)
            selected = work.iloc[selected_positions]
            if selected.empty:
                continue
            row_count = int(selected["hidden_rows"].sum())
            row: dict[str, Any] = {
                "bin_column": bin_column,
                "bin_number": bin_number,
                "well_count": int(len(selected)),
                "row_count": row_count,
                "bin_min": float(selected[bin_column].min()),
                "bin_max": float(selected[bin_column].max()),
                "pf128_mean_pooled_rmse": float(
                    np.sqrt(selected["pf128_mean_sse"].sum() / row_count)
                ),
                "p2p02_pooled_rmse": float(np.sqrt(selected["p2p02_sse"].sum() / row_count)),
            }
            for temperature in TEMPERATURES:
                row[f"scale_{temperature}_ess_median"] = float(
                    selected[f"scale_{temperature}_effective_sample_size"].median()
                )
                row[f"scale_{temperature}_severe_collapse_rate"] = float(
                    selected[f"scale_{temperature}_collapsed_severe"].mean()
                )
                row[f"scale_{temperature}_medium_collapse_rate"] = float(
                    selected[f"scale_{temperature}_collapsed_medium"].mean()
                )
            binned_rows.append(row)
    return pd.DataFrame(binned_rows)


def _correlation_candidates(
    overall_correlations: pd.DataFrame,
    allowed_x: set[str],
    allowed_y: set[str],
    rho_threshold: float,
    require_significant: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """从 overall 表提取符合指定变量组合的相关证据。"""
    required_columns = (
        "x_column",
        "y_column",
        "valid",
        "rho",
        "permutation_p",
        "same_direction_folds",
        "valid_folds",
    )
    _require_columns(overall_correlations, required_columns, "overall_correlations")
    candidates: list[dict[str, Any]] = []
    any_passed = False
    for _, source_row in overall_correlations.iterrows():
        if source_row["x_column"] not in allowed_x or source_row["y_column"] not in allowed_y:
            continue
        valid_value = source_row["valid"]
        if not isinstance(valid_value, (bool, np.bool_)):
            raise ValueError("overall_correlations.valid 必须是布尔值")
        valid = bool(valid_value)

        direction_value = source_row["same_direction_folds"]
        valid_folds_value = source_row["valid_folds"]
        try:
            direction_number = float(direction_value)
            valid_folds_number = float(valid_folds_value)
        except (TypeError, ValueError) as error:
            raise ValueError("折方向计数必须是整数") from error
        if (
            not np.isfinite(direction_number)
            or direction_number != np.floor(direction_number)
            or direction_number < 0.0
            or direction_number > 5.0
        ):
            raise ValueError("same_direction_folds 必须是 [0, 5] 内的整数")
        if (
            not np.isfinite(valid_folds_number)
            or valid_folds_number != np.floor(valid_folds_number)
            or valid_folds_number < 0.0
            or valid_folds_number > 5.0
        ):
            raise ValueError("valid_folds 必须是 [0, 5] 内的整数")
        same_direction_folds = int(direction_number)
        valid_folds = int(valid_folds_number)
        if same_direction_folds > valid_folds:
            raise ValueError("same_direction_folds 不能大于 valid_folds")

        rho: float | None = None
        permutation_p: float | None = None
        if valid:
            try:
                rho = float(source_row["rho"])
                permutation_p = float(source_row["permutation_p"])
            except (TypeError, ValueError) as error:
                raise ValueError("有效相关行的 rho 和 permutation_p 必须是数值") from error
            if not np.isfinite(rho) or rho < -1.0 or rho > 1.0:
                raise ValueError("有效相关行的 rho 必须是 [-1, 1] 内的有限值")
            if not np.isfinite(permutation_p) or permutation_p < 0.0 or permutation_p > 1.0:
                raise ValueError("有效相关行的 permutation_p 必须是 [0, 1] 内的有限值")
        passed = bool(
            valid
            and rho is not None
            and abs(rho) >= rho_threshold
            and same_direction_folds >= 4
            and (not require_significant or (permutation_p is not None and permutation_p <= 0.05))
        )
        candidates.append(
            {
                "x_column": str(source_row["x_column"]),
                "y_column": str(source_row["y_column"]),
                "valid": valid,
                "rho": rho,
                "permutation_p": permutation_p,
                "same_direction_folds": same_direction_folds,
                "valid_folds": valid_folds,
                "passed": passed,
            }
        )
        any_passed = any_passed or passed
    return candidates, bool(any_passed)


def _pooled_rmse(frame: pd.DataFrame, sse_column: str) -> float:
    """用总 SSE 除以总评价行计算 pooled RMSE。"""
    return float(np.sqrt(frame[sse_column].sum() / frame["hidden_rows"].sum()))


def _missing_quartile_evidence(per_well: pd.DataFrame) -> dict[str, Any]:
    """计算全体及各折最高缺失组相对最低缺失组的 PF mean 误差差。"""
    _require_columns(
        per_well,
        ("fold", "well_id", "hidden_rows", "missing_gr_fraction", "pf128_mean_sse"),
        "per_well",
    )

    def group_gap(frame: pd.DataFrame) -> float | None:
        if len(frame) < 4:
            return None
        bins = _assign_stable_quartiles(frame, "missing_gr_fraction").to_numpy()
        low_group = frame.iloc[np.flatnonzero(bins == 1)]
        high_group = frame.iloc[np.flatnonzero(bins == 4)]
        if low_group.empty or high_group.empty:
            return None
        return _pooled_rmse(high_group, "pf128_mean_sse") - _pooled_rmse(low_group, "pf128_mean_sse")

    overall_gap = group_gap(per_well)
    fold_gaps: list[dict[str, Any]] = []
    positive_folds = 0
    for fold_value in sorted(per_well["fold"].drop_duplicates().tolist(), key=lambda value: str(value)):
        fold_gap = group_gap(per_well.loc[per_well["fold"] == fold_value])
        if fold_gap is not None and fold_gap > 0.0:
            positive_folds += 1
        fold_gaps.append({"fold": _json_scalar(fold_value), "gap_ft": fold_gap})
    passed = bool(overall_gap is not None and overall_gap >= 0.50 and positive_folds >= 4)
    return {
        "overall_high_minus_low_ft": overall_gap,
        "positive_direction_folds": int(positive_folds),
        "fold_gaps": fold_gaps,
        "passed": passed,
    }


def decide_pf_routes(
    per_well: pd.DataFrame,
    overall_correlations: pd.DataFrame,
    path_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """严格按实验卡阈值判断 PF01 与 PF02 是否得到井级证据支持。"""
    if per_well.empty:
        raise ValueError("per_well 不能为空")
    if "missing_gr_fraction" not in per_well.columns:
        work = add_diagnostic_columns(per_well)
    else:
        work = per_well.copy()
    _require_columns(
        work,
        (
            "well_id",
            "fold",
            "hidden_rows",
            "missing_gr_fraction",
            "pf128_mean_sse",
            "sigma_absolute_relative_difference",
        ),
        "per_well",
    )
    if work["well_id"].isna().any() or work["well_id"].duplicated().any():
        raise ValueError("per_well 每口井必须恰好一行")
    if work["fold"].isna().any():
        raise ValueError("per_well.fold 不能为空")
    _finite_values(
        work,
        ("hidden_rows", "missing_gr_fraction", "pf128_mean_sse", "sigma_absolute_relative_difference"),
        "per_well",
    )
    if np.any(work["hidden_rows"].to_numpy(dtype=np.float64) <= 0.0):
        raise ValueError("per_well.hidden_rows 必须大于 0")
    if np.any(work["pf128_mean_sse"].to_numpy(dtype=np.float64) < 0.0):
        raise ValueError("per_well.pf128_mean_sse 不能小于 0")
    missing_fraction = work["missing_gr_fraction"].to_numpy(dtype=np.float64)
    if np.any((missing_fraction < 0.0) | (missing_fraction > 1.0)):
        raise ValueError("per_well.missing_gr_fraction 必须位于 [0, 1]")

    pf01_candidates, pf01_correlation_passed = _correlation_candidates(
        overall_correlations,
        allowed_x={"missing_gr_fraction", "longest_gr_gap_md_ft"},
        allowed_y={"pf128_mean_rmse"},
        rho_threshold=0.20,
        require_significant=False,
    )
    missing_quartile_condition = _missing_quartile_evidence(work)
    sigma_median = float(np.median(work["sigma_absolute_relative_difference"].to_numpy(dtype=np.float64)))
    sigma_condition = {"median": sigma_median, "passed": bool(sigma_median >= 0.10)}
    pf01_supported = bool(
        pf01_correlation_passed
        or missing_quartile_condition["passed"]
        or sigma_condition["passed"]
    )

    ess_columns = {f"scale_{temperature}_effective_sample_size" for temperature in TEMPERATURES}
    pf02_candidates, pf02_correlation_passed = _correlation_candidates(
        overall_correlations,
        allowed_x=ess_columns,
        allowed_y={"hidden_rows", "observed_gr_fraction"},
        rho_threshold=0.25,
        require_significant=True,
    )
    _require_columns(path_metrics, ("path_name", "pooled_rmse"), "path_metrics")
    if path_metrics["path_name"].isna().any() or path_metrics["path_name"].duplicated().any():
        raise ValueError("path_metrics.path_name 必须非空且唯一")
    fixed_scale_names = [f"scale_{temperature}" for temperature in TEMPERATURES]
    fixed_scale_rows = path_metrics.loc[path_metrics["path_name"].isin(fixed_scale_names)].copy()
    oracle_rows = path_metrics.loc[path_metrics["path_name"] == "oracle_scale"].copy()
    if set(fixed_scale_rows["path_name"]) != set(fixed_scale_names) or len(fixed_scale_rows) != 4:
        raise ValueError("path_metrics 必须各包含一行 scale_3/5/8/12")
    if len(oracle_rows) != 1:
        raise ValueError("path_metrics 必须恰好包含一行 oracle_scale")
    _finite_values(fixed_scale_rows, ("pooled_rmse",), "fixed_scale_rows")
    _finite_values(oracle_rows, ("pooled_rmse",), "oracle_rows")
    if np.any(fixed_scale_rows["pooled_rmse"].to_numpy(dtype=np.float64) < 0.0):
        raise ValueError("固定温度 pooled_rmse 不能小于 0")
    if float(oracle_rows["pooled_rmse"].iloc[0]) < 0.0:
        raise ValueError("oracle_scale pooled_rmse 不能小于 0")
    best_fixed_position = fixed_scale_rows["pooled_rmse"].astype(float).idxmin()
    best_fixed_row = fixed_scale_rows.loc[best_fixed_position]
    best_fixed_rmse = float(best_fixed_row["pooled_rmse"])
    oracle_rmse = float(oracle_rows["pooled_rmse"].iloc[0])
    oracle_gap = best_fixed_rmse - oracle_rmse
    oracle_gap_condition = {
        "best_fixed_scale": str(best_fixed_row["path_name"]),
        "best_fixed_scale_pooled_rmse": best_fixed_rmse,
        "oracle_scale_pooled_rmse": oracle_rmse,
        "gap_ft": float(oracle_gap),
        "passed": bool(oracle_gap >= 0.25),
        "is_oracle_upper_bound": True,
    }
    pf02_supported = bool(pf02_correlation_passed and oracle_gap_condition["passed"])

    return {
        "scope": "whole_well_only",
        "positional_weight_claim_allowed": False,
        "pf01": {
            "correlation_condition": {
                "rho_threshold": 0.20,
                "minimum_same_direction_folds": 4,
                "candidates": pf01_candidates,
                "passed": bool(pf01_correlation_passed),
            },
            "missing_quartile_condition": missing_quartile_condition,
            "sigma_difference_condition": sigma_condition,
            "supported": pf01_supported,
        },
        "pf02": {
            "ess_correlation_condition": {
                "rho_threshold": 0.25,
                "maximum_permutation_p": 0.05,
                "minimum_same_direction_folds": 4,
                "candidates": pf02_candidates,
                "passed": bool(pf02_correlation_passed),
            },
            "oracle_gap_condition": oracle_gap_condition,
            "supported": pf02_supported,
        },
        "pf01_supported": pf01_supported,
        "pf02_supported": pf02_supported,
    }
