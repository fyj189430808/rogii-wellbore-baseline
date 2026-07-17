"""RF01 合法诊断：在可见前缀内部提前截断并回放倾角预测。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rf01a_features import fit_huber_slope


def fit_ols_slope(md_values: np.ndarray, u_values: np.ndarray) -> float:
    """返回 U 对 MD 的普通最小二乘斜率；不可识别时返回 NaN。"""

    md_centered = md_values - float(np.mean(md_values))
    denominator = float(np.sum(md_centered * md_centered))
    if len(md_values) < 2 or denominator <= 1e-12:
        return float("nan")

    u_centered = u_values - float(np.mean(u_values))
    slope = float(np.sum(md_centered * u_centered) / denominator)
    return slope


def build_prefix_holdout_predictions(
    horizontal_df: pd.DataFrame,
    cut_fraction: float,
    windows_ft: list[int],
) -> pd.DataFrame:
    """用可见前缀前半拟合倾角，返回后半真实 TVT 与各合法回放预测。"""

    if not 0.0 < float(cut_fraction) < 1.0:
        raise ValueError("prefix holdout 的 cut_fraction 必须位于 0 与 1 之间")

    # 只在原始 TVT_input 非空区域内切分，天然隐藏段 TVT 不进入拟合或评分。
    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()].reset_index(
        drop=True
    )
    if len(visible_df) < 3:
        raise ValueError("prefix holdout 至少需要三个可见点")

    cut_position = int(np.floor(len(visible_df) * float(cut_fraction)))
    cut_position = max(2, min(cut_position, len(visible_df) - 1))
    fit_df = visible_df.iloc[:cut_position]
    holdout_df = visible_df.iloc[cut_position:]

    anchor_row = fit_df.iloc[-1]
    anchor_tvt = float(anchor_row["TVT_input"])
    anchor_md = float(anchor_row["MD"])
    anchor_z = float(anchor_row["Z"])
    holdout_md_delta = holdout_df["MD"].to_numpy(dtype=np.float64) - anchor_md
    holdout_z_delta = holdout_df["Z"].to_numpy(dtype=np.float64) - anchor_z

    predictions = pd.DataFrame(
        {
            "target_tvt": holdout_df["TVT_input"].to_numpy(dtype=np.float64),
            "carry_tvt": np.full(len(holdout_df), anchor_tvt, dtype=np.float64),
            "zero_u_slope_tvt": anchor_tvt - holdout_z_delta,
            "md_since_cut": holdout_md_delta,
        }
    )

    last_fit_md = float(fit_df["MD"].iloc[-1])
    for window_ft in windows_ft:
        window_start_md = last_fit_md - float(window_ft)
        window_df = fit_df.loc[fit_df["MD"] >= window_start_md]
        window_md = window_df["MD"].to_numpy(dtype=np.float64)
        window_u = (
            window_df["TVT_input"].to_numpy(dtype=np.float64)
            + window_df["Z"].to_numpy(dtype=np.float64)
        )

        huber_slope = fit_huber_slope(window_md, window_u)
        ols_slope = fit_ols_slope(window_md, window_u)
        predictions[f"huber_{window_ft}_tvt"] = (
            anchor_tvt
            + huber_slope * holdout_md_delta
            - holdout_z_delta
        )
        predictions[f"ols_{window_ft}_tvt"] = (
            anchor_tvt
            + ols_slope * holdout_md_delta
            - holdout_z_delta
        )

    return predictions

