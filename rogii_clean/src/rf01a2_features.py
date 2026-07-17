"""RF01a2：可见前缀短窗 OLS 与长窗 Huber 的混合 U 倾角。"""

from __future__ import annotations

import pandas as pd

from src.f01_features import fit_visible_u_trend
from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows
from src.rf01a_features import fit_visible_u_huber_slope


# 方法选择完全来自 RF01-D0 的可见前缀伪 holdout，不读取自然隐藏 TVT。
RF01A2_FEATURE_COLUMNS = [
    "u_ols_slope_50",
    "u_ols_slope_100",
    "u_ols_slope_200",
    "u_huber_slope_500",
    "u_huber_slope_1000",
]

# 单模输入仍然是 B00 十二列加五个倾角，共十七列。
ALL_RF01A2_FEATURE_COLUMNS = FEATURE_COLUMNS + RF01A2_FEATURE_COLUMNS


def build_rf01a2_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """追加短窗 OLS 和长窗 Huber 倾角，不加入其他 RF01 特征组。"""

    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    # 50/100/200 ft 在合法伪 holdout 上由 OLS 获得更低 micro RMSE。
    for window_ft in [50, 100, 200]:
        slope, _ = fit_visible_u_trend(
            horizontal_df,
            window_ft=float(window_ft),
        )
        result[f"u_ols_slope_{window_ft}"] = slope

    # 500/1000 ft 上 Huber 对远端异常趋势更稳，伪 holdout 优于对应 OLS。
    for window_ft in [500, 1000]:
        slope = fit_visible_u_huber_slope(
            horizontal_df,
            window_ft=float(window_ft),
        )
        result[f"u_huber_slope_{window_ft}"] = slope

    return result

