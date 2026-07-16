"""F02：用可见前缀的分块 U 斜率衡量局部倾角是否稳定。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.f01_features import fit_visible_u_trend
from src.f01b_features import ALL_F01B_FEATURE_COLUMNS, build_f01b_lgbm_rows


# 每个局部块长度固定为 50 ft；本实验不同时搜索块长度，避免增加调参自由度。
BLOCK_SIZE_FT = 50.0

# 六列只描述已知前缀的斜率变化和方向一致性，不将斜率外推到隐藏远端。
F02_FEATURE_COLUMNS = [
    "u_slope_100_minus_500",
    "u_block_slope_std_500",
    "u_block_slope_std_1000",
    "u_positive_slope_ratio_500",
    "u_positive_slope_ratio_1000",
    "u_max_block_slope_jump_1000",
]

# F02 模型输入严格等于 F01b 的十九列加上述六列，共二十五列。
ALL_F02_FEATURE_COLUMNS = ALL_F01B_FEATURE_COLUMNS + F02_FEATURE_COLUMNS


def fit_u_slope_from_arrays(
    md_values: np.ndarray,
    u_values: np.ndarray,
) -> float | None:
    """对一段 U=TVT_input+Z 与 MD 做 OLS；点数不足时返回 None。"""

    # 两个不同 MD 是识别直线斜率的最低要求。
    if len(md_values) < 2:
        return None

    md_centered = md_values - float(np.mean(md_values))
    denominator = float(np.sum(md_centered * md_centered))
    if denominator <= 1e-12:
        return None

    u_centered = u_values - float(np.mean(u_values))
    slope = float(np.sum(md_centered * u_centered) / denominator)
    return slope


def calculate_visible_block_slopes(
    horizontal_df: pd.DataFrame,
    window_ft: float,
    block_size_ft: float = BLOCK_SIZE_FT,
) -> np.ndarray:
    """把末端可见窗口按 MD 划成固定块，返回从旧到新的有效 U 斜率。"""

    # 只保留推理时已知的 TVT_input；隐藏 TVT 列不会进入任何计算。
    visible_df = horizontal_df.loc[horizontal_df["TVT_input"].notna()]
    if visible_df.empty:
        raise ValueError("F02 需要至少一个 TVT_input 可见点")

    visible_md = visible_df["MD"].to_numpy(dtype=np.float64)
    visible_u = (
        visible_df["TVT_input"].to_numpy(dtype=np.float64)
        + visible_df["Z"].to_numpy(dtype=np.float64)
    )

    last_visible_md = float(visible_md[-1])
    window_start_md = last_visible_md - float(window_ft)
    block_count = int(np.ceil(float(window_ft) / float(block_size_ft)))
    block_slopes: list[float] = []

    # 从窗口最旧端向当前井底推进，使相邻差分保持真实时间顺序。
    for block_index in range(block_count):
        block_start_md = window_start_md + block_index * float(block_size_ft)
        block_end_md = min(
            block_start_md + float(block_size_ft),
            last_visible_md,
        )

        # 使用左开右闭区间，避免相邻块重复使用边界采样点。
        block_mask = (visible_md > block_start_md) & (visible_md <= block_end_md)
        block_slope = fit_u_slope_from_arrays(
            visible_md[block_mask],
            visible_u[block_mask],
        )
        if block_slope is not None:
            block_slopes.append(block_slope)

    return np.asarray(block_slopes, dtype=np.float64)


def build_f02_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在 F01b 十九列上增加六个可见前缀斜率稳定性特征。"""

    # F01b 提供完全相同的基础特征、标签和隐藏行顺序。
    result = build_f01b_lgbm_rows(horizontal_df, well_id, fold)

    slope_100, _ = fit_visible_u_trend(horizontal_df, window_ft=100.0)
    slope_500 = float(result["u_slope_500"].iloc[0])
    slopes_500 = calculate_visible_block_slopes(horizontal_df, window_ft=500.0)
    slopes_1000 = calculate_visible_block_slopes(horizontal_df, window_ft=1000.0)

    # 无有效块只会出现在极短或异常前缀，使用有限的零值以保持输入可检查。
    std_500 = float(np.std(slopes_500)) if len(slopes_500) > 0 else 0.0
    std_1000 = float(np.std(slopes_1000)) if len(slopes_1000) > 0 else 0.0
    positive_ratio_500 = (
        float(np.mean(slopes_500 > 0.0)) if len(slopes_500) > 0 else 0.0
    )
    positive_ratio_1000 = (
        float(np.mean(slopes_1000 > 0.0)) if len(slopes_1000) > 0 else 0.0
    )

    if len(slopes_1000) >= 2:
        max_slope_jump_1000 = float(np.max(np.abs(np.diff(slopes_1000))))
    else:
        max_slope_jump_1000 = 0.0

    # 六个值均为井级前缀统计，对当前井的所有隐藏行重复，但不读取隐藏真值。
    result["u_slope_100_minus_500"] = slope_100 - slope_500
    result["u_block_slope_std_500"] = std_500
    result["u_block_slope_std_1000"] = std_1000
    result["u_positive_slope_ratio_500"] = positive_ratio_500
    result["u_positive_slope_ratio_1000"] = positive_ratio_1000
    result["u_max_block_slope_jump_1000"] = max_slope_jump_1000

    return result

