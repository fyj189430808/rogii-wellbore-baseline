"""新版路线图 RF01d：把五窗口 Huber U 倾角变成逐隐藏行投影增量。"""

from __future__ import annotations

import pandas as pd

from src.rf01a_features import ALL_RF01A_FEATURE_COLUMNS, build_rf01a_lgbm_rows


# 每一列表示：若当前窗口倾角保持不变，该隐藏行的 TVT 应相对可见末值变化多少 ft。
RF01D_PROJECTION_FEATURE_COLUMNS = [
    "u_projection_delta_50",
    "u_projection_delta_100",
    "u_projection_delta_200",
    "u_projection_delta_500",
    "u_projection_delta_1000",
]

# 模型输入严格为 RF01a 十七列加五个逐行投影增量，共二十二列。
ALL_RF01D_FEATURE_COLUMNS = (
    ALL_RF01A_FEATURE_COLUMNS + RF01D_PROJECTION_FEATURE_COLUMNS
)


def build_rf01d_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """返回隐藏行特征；只用可见前缀倾角、未来 MD 和 Z，不读取隐藏 TVT。"""

    # 先复用 RF01a，得到 B00 十二列、五个井级 Huber 倾角和每行几何差。
    result = build_rf01a_lgbm_rows(horizontal_df, well_id, fold)

    # md_since_visible_end 的 shape 为 [隐藏行数]，单位 ft。
    md_distance = result["md_since_visible_end"]

    # dz_from_visible_end 的 shape 为 [隐藏行数]，单位 ft。
    z_change = result["dz_from_visible_end"]

    # 分窗口生成未经截断和门控的 TVT 投影变化，便于单独检验外推是否泛化。
    for window_ft in [50, 100, 200, 500, 1000]:
        # 同一口井的 slope 在所有隐藏行相同，单位为 ft/ft。
        slope = result[f"u_huber_slope_{window_ft}"]

        # U 的预测变化为 slope×ΔMD；由 TVT=U-Z，再减去井眼 Z 的实际变化。
        projected_tvt_change = slope * md_distance - z_change

        # 输出 shape 仍为 [隐藏行数]，单位 ft；不使用隐藏 TVT 真值。
        result[f"u_projection_delta_{window_ft}"] = projected_tvt_change

    return result
