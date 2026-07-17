"""RF02a-null：用井内循环平移的错误 GR mask 生成同维负对照。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lgbm_features import build_simple_lgbm_rows
from src.rf02a_gr_missing_features import (
    ALL_RF02A_FEATURE_COLUMNS,
    RF02A_MISSING_FEATURE_COLUMNS,
    calculate_gr_missing_geometry,
    calculate_missing_run_lengths,
)


# 真实规模井的 mask 至少平移 201 行，超过最大的 200 ft 局部有效率窗口。
MINIMUM_MASK_SHIFT_ROWS = 201

# 负对照必须与真实 RF02a 使用完全相同的二十一列和列顺序。
ALL_RF02A_CONTROL_FEATURE_COLUMNS = list(ALL_RF02A_FEATURE_COLUMNS)


def choose_mask_shift_rows(number_of_rows: int) -> int:
    """根据井长返回确定性的循环平移行数；真实规模井至少平移 201 行。"""

    # 至少两行才能构造非零且不等于整圈的循环平移。
    if number_of_rows < 2:
        raise ValueError("GR mask 平移负对照需要至少两行")

    # 小型单元测试井无法平移 201 行，改用约三分之一井长的非零位移。
    if number_of_rows <= MINIMUM_MASK_SHIFT_ROWS + 1:
        return max(1, number_of_rows // 3)

    # 真实井优先移动约三分之一井长，同时保证越过全部 200 ft 局部窗口。
    shift_rows = max(MINIMUM_MASK_SHIFT_ROWS, number_of_rows // 3)

    # 防御性取模确保 np.roll 位移小于井长；正常真实井不会得到 0。
    shift_rows = shift_rows % number_of_rows
    if shift_rows == 0:
        shift_rows = 1

    return int(shift_rows)


def circular_shift_observed_mask(observed_mask: np.ndarray) -> np.ndarray:
    """返回井内循环平移后的 observed mask，并严格保持 True 总数。"""

    # observed_array 的 shape 为 [整口井行数]，True 表示真实 GR 非缺失。
    observed_array = np.asarray(observed_mask, dtype=bool)

    # 位移只由井长决定，使每次运行和每个 outer fold 都得到相同负对照。
    shift_rows = choose_mask_shift_rows(len(observed_array))

    # np.roll 在井尾回卷到井首，因此不改变缺失率和缺失 run 的总多重集合。
    shifted_mask = np.roll(observed_array, shift_rows)

    return shifted_mask


def build_rf02a_mask_shift_control_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """保留真实 B00 输入，仅把新增九列替换为错误位置 mask 的统计。"""

    # result 的 gr_raw/gr_missing 仍来自真实原始 GR，这是与 RF02a 公平比较的底座。
    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    # true_observed_mask 的 shape 为 [总行数]，随后只用于构造确定性负对照。
    true_observed_mask = horizontal_df["GR"].notna().to_numpy()

    # shifted_observed_mask 与真实 mask 的 True 数量完全相同，但所在 MD 错误。
    shifted_observed_mask = circular_shift_observed_mask(true_observed_mask)

    # mask_df 只保留计算几何所需的 MD，并用 0/NaN 编码错误位置 mask。
    mask_df = horizontal_df.loc[:, ["MD", "GR"]].copy()
    mask_df["GR"] = np.where(shifted_observed_mask, 0.0, np.nan)

    # control_geometry 的列公式、窗口和维度与真实 RF02a 完全相同。
    control_geometry = calculate_gr_missing_geometry(
        mask_df,
        windows_ft=[50, 100, 200],
    )

    # hidden_positions 将整井负对照几何对齐到自然隐藏评价后缀。
    hidden_positions = result["row_index"].to_numpy(dtype=np.int64)
    for feature_name in RF02A_MISSING_FEATURE_COLUMNS[:7]:
        result[feature_name] = control_geometry[feature_name].to_numpy()[
            hidden_positions
        ]

    # hidden_mask 仍由真实 TVT_input 定义，只决定输出哪些行，不读取隐藏 TVT。
    hidden_mask = horizontal_df["TVT_input"].isna().to_numpy()

    # 在自然隐藏后缀中抽取错误位置的 observed mask，计算同名井级统计。
    hidden_shifted_observed_mask = shifted_observed_mask[hidden_mask]
    result["well_hidden_gr_valid_fraction"] = float(
        np.mean(hidden_shifted_observed_mask)
    )

    # 最长缺口同样只在隐藏后缀内计算，公式与真实 RF02a 保持一致。
    hidden_md = horizontal_df.loc[hidden_mask, "MD"].to_numpy(dtype=np.float64)
    hidden_gap_lengths = calculate_missing_run_lengths(
        hidden_md,
        hidden_shifted_observed_mask,
    )
    result["well_hidden_longest_gr_gap_ft"] = float(
        np.max(hidden_gap_lengths)
    )

    return result
