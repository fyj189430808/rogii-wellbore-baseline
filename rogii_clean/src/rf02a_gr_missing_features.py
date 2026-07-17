"""RF02a：从原始 GR 缺失 mask 构造逐行缺口几何和井级可观测率。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


# 这九列只描述 GR 在哪里被真实观测，不使用 GR 数值或隐藏 TVT。
RF02A_MISSING_FEATURE_COLUMNS = [
    "gr_gap_length_ft",
    "gr_distance_to_left_observed_ft",
    "gr_distance_to_right_observed_ft",
    "gr_relative_position_inside_gap",
    "gr_valid_fraction_50",
    "gr_valid_fraction_100",
    "gr_valid_fraction_200",
    "well_hidden_gr_valid_fraction",
    "well_hidden_longest_gr_gap_ft",
]

# 单模输入严格为 B00 十二列加九个缺失几何特征，共二十一列。
ALL_RF02A_FEATURE_COLUMNS = FEATURE_COLUMNS + RF02A_MISSING_FEATURE_COLUMNS


def calculate_missing_run_lengths(
    md_values: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """返回每行所在缺失 run 的物理长度；实测行返回 0，单位 ft。"""

    # md_array 的 shape 为 [采样行数]，单位 ft。
    md_array = np.asarray(md_values, dtype=np.float64)

    # observed_array 的 shape 为 [采样行数]；True 代表该行原始 GR 非缺失。
    observed_array = np.asarray(observed_mask, dtype=bool)

    # 两个数组必须逐行对应，否则缺口起止位置没有意义。
    if md_array.shape != observed_array.shape:
        raise ValueError("GR 缺失 mask 与 MD shape 不一致")

    # 当前比赛每口井都至少有一行；空数组不应静默进入特征缓存。
    if len(md_array) == 0:
        raise ValueError("GR 缺失几何需要至少一个采样点")

    # 当前数据严格按 MD 递增；搜索邻域和物理缺口长度都依赖这一条件。
    md_step = np.diff(md_array)
    if len(md_step) > 0 and np.any(md_step <= 0.0):
        raise ValueError("GR 缺失几何要求 MD 严格递增")

    # 单行井用 1 ft 作为最小采样宽度；正常井使用实际 MD 间隔中位数。
    sampling_step_ft = 1.0
    if len(md_step) > 0:
        sampling_step_ft = float(np.median(md_step))

    # 实测行保持 0；缺失行随后被其所在连续 run 的长度覆盖。
    run_length_ft = np.zeros(len(md_array), dtype=np.float64)

    # row_position 指向尚未检查的下一行。
    row_position = 0
    while row_position < len(md_array):
        # 实测行不属于缺口，直接移动到下一行。
        if observed_array[row_position]:
            row_position += 1
            continue

        # run_start 是当前连续缺失段第一行的位置。
        run_start = row_position

        # 向右移动，直到井尾或遇到下一行实测 GR。
        while row_position < len(md_array) and not observed_array[row_position]:
            row_position += 1

        # run_end 是半开区间末端；最后一个缺失位置为 run_end-1。
        run_end = row_position

        # 当前严格 1 ft 数据中，该公式等于缺失行数，名称与单位都保持为 ft。
        physical_length_ft = (
            float(md_array[run_end - 1] - md_array[run_start])
            + sampling_step_ft
        )

        # 同一连续缺口中的每一行共享同一个物理长度。
        run_length_ft[run_start:run_end] = physical_length_ft

    return run_length_ft


def calculate_gr_missing_geometry(
    horizontal_df: pd.DataFrame,
    windows_ft: list[int],
) -> pd.DataFrame:
    """为整口井每一行计算缺口长度、邻近实测距离和多窗口有效率。"""

    # 这里只读取 MD 和原始 GR；隐藏 TVT 即使存在也不会被访问。
    md_values = horizontal_df["MD"].to_numpy(dtype=np.float64)
    gr_values = horizontal_df["GR"].to_numpy(dtype=np.float64)

    # observed_mask 的 shape 为 [总行数]；True 代表 GR 是真实输入数值。
    observed_mask = ~np.isnan(gr_values)

    # 缺口长度在实测行等于 0，在缺失行等于其连续缺失 run 的物理长度。
    gap_length_ft = calculate_missing_run_lengths(md_values, observed_mask)

    # 左距离默认 NaN，仅在左边存在真实观测或当前行实测时赋值。
    left_distance_ft = np.full(len(md_values), np.nan, dtype=np.float64)

    # last_observed_md 保存扫描到当前位置前最近的实测 GR 所在 MD。
    last_observed_md = float("nan")
    for row_position in range(len(md_values)):
        if observed_mask[row_position]:
            last_observed_md = float(md_values[row_position])
            left_distance_ft[row_position] = 0.0
        elif np.isfinite(last_observed_md):
            left_distance_ft[row_position] = (
                float(md_values[row_position]) - last_observed_md
            )

    # 右距离同理，从井尾向井首扫描最近的真实观测。
    right_distance_ft = np.full(len(md_values), np.nan, dtype=np.float64)
    next_observed_md = float("nan")
    for row_position in range(len(md_values) - 1, -1, -1):
        if observed_mask[row_position]:
            next_observed_md = float(md_values[row_position])
            right_distance_ft[row_position] = 0.0
        elif np.isfinite(next_observed_md):
            right_distance_ft[row_position] = (
                next_observed_md - float(md_values[row_position])
            )

    # 相对位置只对左右都有实测支撑的内部缺口定义；实测行保持 NaN。
    relative_position = np.full(len(md_values), np.nan, dtype=np.float64)
    internal_gap_mask = (
        (~observed_mask)
        & np.isfinite(left_distance_ft)
        & np.isfinite(right_distance_ft)
    )
    interpolation_span_ft = (
        left_distance_ft[internal_gap_mask]
        + right_distance_ft[internal_gap_mask]
    )
    relative_position[internal_gap_mask] = (
        left_distance_ft[internal_gap_mask] / interpolation_span_ft
    )

    # 先保存四个逐行缺口几何量；DataFrame 行号与原始 CSV 整数位置一致。
    geometry = pd.DataFrame(
        {
            "gr_gap_length_ft": gap_length_ft,
            "gr_distance_to_left_observed_ft": left_distance_ft,
            "gr_distance_to_right_observed_ft": right_distance_ft,
            "gr_relative_position_inside_gap": relative_position,
        }
    )

    # observed_prefix[k] 表示原始前 k 行中有多少行真实观测了 GR。
    observed_prefix = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(observed_mask, dtype=np.int64)]
    )

    # 每个窗口以当前 MD 为中心，窗口首尾超出井范围时按实际存在行数计算分母。
    for window_ft in windows_ft:
        half_window_ft = float(window_ft) / 2.0
        left_boundary_md = md_values - half_window_ft
        right_boundary_md = md_values + half_window_ft
        left_indices = np.searchsorted(md_values, left_boundary_md, side="left")
        right_indices = np.searchsorted(md_values, right_boundary_md, side="right")
        observed_count = observed_prefix[right_indices] - observed_prefix[left_indices]
        available_count = right_indices - left_indices
        valid_fraction = observed_count.astype(np.float64) / available_count
        geometry[f"gr_valid_fraction_{window_ft}"] = valid_fraction

    return geometry


def build_rf02a_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在 B00 隐藏行上追加九个由原始 GR mask 决定的合法特征。"""

    # B00 提供隐藏行索引、十二个原特征和训练时才使用的标签。
    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    # geometry 的 shape 为 [整口井行数, 7]，包含三个固定窗口有效率。
    geometry = calculate_gr_missing_geometry(
        horizontal_df,
        windows_ft=[50, 100, 200],
    )

    # hidden_positions 与 result 每一行一一对应，值是原始 CSV 的整数位置。
    hidden_positions = result["row_index"].to_numpy(dtype=np.int64)

    # 七个逐行特征按隐藏位置抽取，shape 变为 [隐藏行数, 7]。
    for feature_name in RF02A_MISSING_FEATURE_COLUMNS[:7]:
        result[feature_name] = geometry[feature_name].to_numpy()[hidden_positions]

    # hidden_mask 的 shape 为 [总行数]，只用于定义当前井自然隐藏评价后缀。
    hidden_mask = horizontal_df["TVT_input"].isna().to_numpy()

    # hidden_observed_mask 的 shape 为 [隐藏行数]，只读取隐藏段原始 GR mask。
    hidden_observed_mask = horizontal_df.loc[hidden_mask, "GR"].notna().to_numpy()

    # 井级隐藏 GR 有效率是一个无量纲标量，复制到该井所有隐藏行。
    well_hidden_valid_fraction = float(np.mean(hidden_observed_mask))
    result["well_hidden_gr_valid_fraction"] = well_hidden_valid_fraction

    # 单独在隐藏后缀内计算最长缺口，避免把可见前缀缺口混入井级统计。
    hidden_md = horizontal_df.loc[hidden_mask, "MD"].to_numpy(dtype=np.float64)
    hidden_gap_lengths = calculate_missing_run_lengths(
        hidden_md,
        hidden_observed_mask,
    )
    well_hidden_longest_gap_ft = float(np.max(hidden_gap_lengths))
    result["well_hidden_longest_gr_gap_ft"] = well_hidden_longest_gap_ft

    return result
