"""统一计算 ROGII 行级和井级指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 计算一组真实值和预测值的 RMSE。
def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """输入同 shape 的一维数组，输出 RMSE 标量。"""

    # 转成 float64，避免 float32 平方和累积误差。
    target_values = np.asarray(target, dtype=np.float64)

    # 预测也转成 float64。
    prediction_values = np.asarray(prediction, dtype=np.float64)

    # shape 不一致说明 ID 对齐或评价行选择出错。
    if target_values.shape != prediction_values.shape:
        raise ValueError("target 与 prediction shape 不一致")

    # 任何 NaN/Inf 都应在评分前暴露。
    if not np.isfinite(target_values).all() or not np.isfinite(prediction_values).all():
        raise ValueError("target 或 prediction 含非有限值")

    # 逐行误差，shape 与输入相同。
    error = target_values - prediction_values

    # 返回平方误差均值的平方根。
    return float(np.sqrt(np.mean(error * error)))


# 建立一井一行的评分表。
def build_per_well_metrics(
    prediction_df: pd.DataFrame,
    target_column: str = "target_tvt",
    prediction_column: str = "pred_tvt",
    baseline_column: str | None = "carry_tvt",
) -> pd.DataFrame:
    """输入隐藏评价行预测表，输出逐井 rows、SSE、RMSE 和可选基线指标。"""

    # 固定必需列，fold 用于后续每折汇总。
    required_columns = {"well_id", "fold", target_column, prediction_column}

    # 找出缺少的列。
    missing_columns = required_columns - set(prediction_df.columns)

    # 缺列时停止，避免用错误 schema 评分。
    if missing_columns:
        raise ValueError(f"预测文件缺少列：{sorted(missing_columns)}")

    # 保存逐井结果，每个字典对应一口完整验证井。
    per_well_rows: list[dict] = []

    # 按井遍历，禁止把一口井拆成多个统计单位。
    for well_id, well_df in prediction_df.groupby("well_id", sort=True):
        # 同一口井必须只属于一个固定 fold。
        fold_values = well_df["fold"].astype(int).unique()

        # 多个 fold 说明注册表合并错误。
        if len(fold_values) != 1:
            raise ValueError(f"井 {well_id} 出现在多个 fold")

        # 读取该井真实 TVT。
        target_values = well_df[target_column].to_numpy(dtype=np.float64)

        # 读取该井模型预测 TVT。
        prediction_values = well_df[prediction_column].to_numpy(dtype=np.float64)

        # 逐行误差用于计算 SSE。
        error = target_values - prediction_values

        # 模型 SSE 是该井所有评价行平方误差之和。
        prediction_sse = float(np.sum(error * error))

        # 建立基础逐井记录。
        result_row = {
            "well_id": str(well_id),
            "fold": int(fold_values[0]),
            "rows": int(len(well_df)),
            "prediction_sse": prediction_sse,
            "prediction_rmse": float(np.sqrt(prediction_sse / max(len(well_df), 1))),
        }

        # 只有实际存在基线列时才计算胜井率所需指标。
        if baseline_column is not None and baseline_column in well_df.columns:
            # 读取该井基线 TVT。
            baseline_values = well_df[baseline_column].to_numpy(dtype=np.float64)

            # 基线误差 shape 与评价行相同。
            baseline_error = target_values - baseline_values

            # 计算该井基线 SSE。
            baseline_sse = float(np.sum(baseline_error * baseline_error))

            # 保存基线 SSE。
            result_row["baseline_sse"] = baseline_sse

            # 保存基线逐井 RMSE。
            result_row["baseline_rmse"] = float(np.sqrt(baseline_sse / max(len(well_df), 1)))

            # 模型 RMSE 更小时该井记为获胜。
            result_row["prediction_wins"] = bool(
                result_row["prediction_rmse"] < result_row["baseline_rmse"]
            )

        # 加入逐井结果列表。
        per_well_rows.append(result_row)

    # 转成按井排序的 DataFrame。
    return pd.DataFrame(per_well_rows).sort_values("well_id").reset_index(drop=True)


# 将逐井 SSE 汇总成主指标和尾部指标。
def summarize_per_well_metrics(per_well_df: pd.DataFrame) -> dict:
    """输入逐井评分表，输出 micro、macro、median、P90、worst 和可选胜井率。"""

    # 总评价行数决定 micro RMSE 的分母。
    total_rows = int(per_well_df["rows"].sum())

    # 所有井 SSE 相加得到全评价集 SSE。
    total_prediction_sse = float(per_well_df["prediction_sse"].sum())

    # 逐井 RMSE 数组用于 macro 和分位数。
    well_rmse = per_well_df["prediction_rmse"].to_numpy(dtype=np.float64)

    # 建立统一主指标。
    summary = {
        "rows": total_rows,
        "wells": int(len(per_well_df)),
        "micro_rmse": float(np.sqrt(total_prediction_sse / max(total_rows, 1))),
        "macro_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p90_well_rmse": float(np.quantile(well_rmse, 0.90)),
        "worst_well_rmse": float(np.max(well_rmse)),
    }

    # 如果有基线 SSE，就同时报告基线 micro 和胜井率。
    if "baseline_sse" in per_well_df.columns:
        # 汇总基线全行 SSE。
        total_baseline_sse = float(per_well_df["baseline_sse"].sum())

        # 保存基线 micro RMSE。
        summary["baseline_micro_rmse"] = float(
            np.sqrt(total_baseline_sse / max(total_rows, 1))
        )

        # 正值表示模型比基线更差，负值表示模型更好。
        summary["micro_rmse_delta_vs_baseline"] = (
            summary["micro_rmse"] - summary["baseline_micro_rmse"]
        )

        # 胜井率按井等权。
        summary["well_win_rate"] = float(per_well_df["prediction_wins"].mean())

    # 返回普通字典，便于保存 JSON。
    return summary


# 按 outer fold 计算同一套汇总。
def summarize_by_fold(per_well_df: pd.DataFrame) -> list[dict]:
    """输入逐井表，输出每折统一指标列表。"""

    # 保存每折结果。
    fold_summaries: list[dict] = []

    # 按固定 fold 升序遍历。
    for fold_id, fold_df in per_well_df.groupby("fold", sort=True):
        # 复用整体汇总逻辑。
        fold_summary = summarize_per_well_metrics(fold_df)

        # 显式写入 fold 编号。
        fold_summary["fold"] = int(fold_id)

        # 加入列表。
        fold_summaries.append(fold_summary)

    # 返回五折结果。
    return fold_summaries


# 以井为簇做配对 bootstrap，评价 micro RMSE 差值稳定性。
def paired_well_bootstrap(
    per_well_df: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """输入带模型和基线 SSE 的逐井表，输出配对 micro RMSE 差值置信区间。"""

    # bootstrap 必须同时有模型和基线 SSE。
    required_columns = {"rows", "prediction_sse", "baseline_sse"}

    # 缺列时无法进行配对比较。
    if not required_columns.issubset(per_well_df.columns):
        raise ValueError("paired bootstrap 需要 rows、prediction_sse 和 baseline_sse")

    # 创建固定随机数生成器。
    random_generator = np.random.default_rng(int(seed))

    # 逐井评价行数数组。
    rows = per_well_df["rows"].to_numpy(dtype=np.int64)

    # 逐井模型 SSE 数组。
    prediction_sse = per_well_df["prediction_sse"].to_numpy(dtype=np.float64)

    # 逐井基线 SSE 数组。
    baseline_sse = per_well_df["baseline_sse"].to_numpy(dtype=np.float64)

    # 井数量是每次有放回抽样的样本数。
    well_count = int(len(per_well_df))

    # 保存每次抽样的 candidate minus baseline micro RMSE。
    deltas = np.empty(int(n_resamples), dtype=np.float64)

    # 重复固定次数的井级配对抽样。
    for sample_index in range(int(n_resamples)):
        # 有放回抽取井索引；一口井的全部行始终一起进入。
        sampled_indices = random_generator.integers(0, well_count, size=well_count)

        # 抽样后的总评价行数。
        sampled_rows = int(rows[sampled_indices].sum())

        # 抽样后的模型 micro RMSE。
        prediction_rmse = float(
            np.sqrt(prediction_sse[sampled_indices].sum() / max(sampled_rows, 1))
        )

        # 抽样后的基线 micro RMSE。
        baseline_rmse = float(
            np.sqrt(baseline_sse[sampled_indices].sum() / max(sampled_rows, 1))
        )

        # 保存配对差值；负值代表模型更好。
        deltas[sample_index] = prediction_rmse - baseline_rmse

    # 返回均值和 95% 分位数区间。
    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_better": float(np.mean(deltas < 0.0)),
    }
