"""F01：从当前井可见前缀构造 U=TVT_input+Z 的倾角特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


# 三个窗口各产生斜率、拟合误差和逐行外推 delta，再加入一个短长坡差。
F01_FEATURE_COLUMNS = [
    "u_slope_200",
    "u_fit_rmse_200",
    "u_linear_delta_200",
    "u_slope_500",
    "u_fit_rmse_500",
    "u_linear_delta_500",
    "u_slope_1000",
    "u_fit_rmse_1000",
    "u_linear_delta_1000",
    "u_slope_200_minus_1000",
]

# F01 模型严格使用原 12 列加新增 10 列，不自动扫描其他字段。
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + F01_FEATURE_COLUMNS


def fit_visible_u_trend(
    horizontal_df: pd.DataFrame,
    window_ft: float,
) -> tuple[float, float]:
    """在最后 window_ft 可见 MD 上拟合 U 对 MD 的 OLS 斜率和 RMSE。"""

    # visible_df 只含 TVT_input 可见行，因此隐藏真值不可能进入拟合。
    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    if visible_df.empty:
        raise ValueError("F01 需要至少一个 TVT_input 可见点")

    last_visible_md = float(visible_df["MD"].iloc[-1])
    window_start_md = last_visible_md - float(window_ft)
    window_df = visible_df.loc[visible_df["MD"] >= window_start_md]

    # 少于两个不同 MD 时没有可辨识斜率，使用零斜率并保留 U 离散度。
    md_values = window_df["MD"].to_numpy(dtype=np.float64)
    u_values = (
        window_df["TVT_input"].to_numpy(dtype=np.float64)
        + window_df["Z"].to_numpy(dtype=np.float64)
    )
    md_centered = md_values - float(np.mean(md_values))
    denominator = float(np.sum(md_centered * md_centered))

    if denominator <= 1e-12:
        slope = 0.0
        fitted_u = np.full_like(u_values, float(np.mean(u_values)))
    else:
        u_centered = u_values - float(np.mean(u_values))
        slope = float(np.sum(md_centered * u_centered) / denominator)
        fitted_u = float(np.mean(u_values)) + slope * md_centered

    residual = u_values - fitted_u
    fit_rmse = float(np.sqrt(np.mean(residual * residual)))
    return slope, fit_rmse


def build_f01_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """返回隐藏行的原 12 特征、10 个 F01 特征及固定 target。"""

    # 基础 builder 负责可见/隐藏连续性、MD 单调性、元数据和 target。
    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    visible_positions = np.flatnonzero(horizontal_df["TVT_input"].notna().to_numpy())
    last_visible_position = int(visible_positions[-1])
    hidden_positions = result["row_index"].to_numpy(dtype=np.int64)

    last_visible_md = float(horizontal_df["MD"].iloc[last_visible_position])
    last_visible_z = float(horizontal_df["Z"].iloc[last_visible_position])
    hidden_md = horizontal_df["MD"].iloc[hidden_positions].to_numpy(dtype=np.float64)
    hidden_z = horizontal_df["Z"].iloc[hidden_positions].to_numpy(dtype=np.float64)
    md_delta = hidden_md - last_visible_md
    z_delta = hidden_z - last_visible_z

    slopes: dict[int, float] = {}
    for window_ft in [200, 500, 1000]:
        slope, fit_rmse = fit_visible_u_trend(horizontal_df, float(window_ft))
        slopes[window_ft] = slope

        # 由 U=TVT+Z 得到 TVT_delta = U_delta - Z_delta。
        linear_tvt_delta = slope * md_delta - z_delta
        result[f"u_slope_{window_ft}"] = slope
        result[f"u_fit_rmse_{window_ft}"] = fit_rmse
        result[f"u_linear_delta_{window_ft}"] = linear_tvt_delta

    result["u_slope_200_minus_1000"] = slopes[200] - slopes[1000]
    return result
