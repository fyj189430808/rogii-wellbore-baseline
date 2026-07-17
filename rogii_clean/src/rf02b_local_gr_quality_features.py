"""RF02b：从原始非缺失 GR 计算多尺度局部 MAD 与总体方差。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


# 少于十个真实观测时，离散程度估计不稳定，因此保留 NaN 交给 LightGBM。
MINIMUM_OBSERVED_COUNT = 10

# 模型只读取三个窗口的精确 MAD 与总体方差，不把 observed count 混入假设。
RF02B_QUALITY_FEATURE_COLUMNS = [
    "gr_local_mad_50",
    "gr_local_variance_50",
    "gr_local_mad_100",
    "gr_local_variance_100",
    "gr_local_mad_200",
    "gr_local_variance_200",
]

# 单模输入严格为 B00 十二列加六个局部 GR 数值特征，共十八列。
ALL_RF02B_FEATURE_COLUMNS = FEATURE_COLUMNS + RF02B_QUALITY_FEATURE_COLUMNS


def calculate_local_gr_quality(
    horizontal_df: pd.DataFrame,
    windows_ft: list[int],
    minimum_observed_count: int,
) -> pd.DataFrame:
    """返回整井逐行的 observed count、精确 MAD 和 ddof=0 总体方差。"""

    # md_values 的 shape 为 [总行数]，当前比赛数据单位为 ft。
    md_values = horizontal_df["MD"].to_numpy(dtype=np.float64)

    # gr_values 的 shape 为 [总行数]，NaN 表示该点没有真实 GR 观测。
    gr_values = horizontal_df["GR"].to_numpy(dtype=np.float64)

    # 当前全量审计确认采样严格为 1 ft；固定行窗口只在该条件下等于物理窗口。
    md_step = np.diff(md_values)
    if len(md_step) > 0 and not np.allclose(md_step, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("RF02b 精确窗口要求 MD 以 1 ft 严格采样")

    # Inf 不是合法 GR 缺失编码，不能静默当作普通观测或 NaN。
    if np.isinf(gr_values).any():
        raise ValueError("RF02b 的原始 GR 含 Inf")

    # 支持量必须为正整数，否则所有窗口定义都没有可解释意义。
    if minimum_observed_count < 1:
        raise ValueError("minimum_observed_count 必须至少为 1")

    # quality 的行号与原始 CSV 整数位置一致，列在循环中按窗口追加。
    quality = pd.DataFrame(index=np.arange(len(horizontal_df)))

    # 每个窗口单独处理并立即释放临时矩阵，峰值内存只由最大 200 ft 窗口决定。
    for window_ft in windows_ft:
        # 偶数物理宽度才能对称地包含中心点及左右各 W/2 行。
        if window_ft <= 0 or window_ft % 2 != 0:
            raise ValueError("RF02b 窗口必须是正偶数 ft")

        # 50 ft 对应左右各 25 行加中心行，共 51 个采样位置。
        half_window_rows = window_ft // 2
        window_rows = window_ft + 1

        # 井首尾用 NaN 填充，使每个原始行都有一个同宽中心窗口。
        padded_gr = np.pad(
            gr_values,
            (half_window_rows, half_window_rows),
            mode="constant",
            constant_values=np.nan,
        )

        # windows 的虚拟 shape 为 [总行数, window_rows]，不复制底层 padded_gr。
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_gr,
            window_rows,
        )

        # observed_count 的 shape 为 [总行数]，统计窗口中原始非缺失 GR 数量。
        observed_count = np.sum(np.isfinite(windows), axis=1).astype(np.int32)

        # 全 NaN 或低支持窗口会触发 NumPy RuntimeWarning；其值随后明确置 NaN。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # window_median 是每个窗口真实 GR 的中位数，shape 为 [总行数]。
            window_median = np.nanmedian(windows, axis=1)

            # absolute_deviation 的 shape 与 windows 相同，单位与 GR 相同。
            absolute_deviation = np.abs(windows - window_median[:, None])

            # MAD 是绝对偏差的中位数，不乘 1.4826，单位仍与 GR 相同。
            local_mad = np.nanmedian(absolute_deviation, axis=1)

            # 方差使用总体定义 ddof=0，单位为 GR 的平方。
            local_variance = np.nanvar(windows, axis=1, ddof=0)

        # 低于预注册支持量时，不用 0 冒充局部曲线平稳。
        insufficient_support = observed_count < int(minimum_observed_count)
        local_mad[insufficient_support] = np.nan
        local_variance[insufficient_support] = np.nan

        # count 只保存在缓存供诊断，不属于 RF02B_QUALITY_FEATURE_COLUMNS。
        quality[f"gr_local_observed_count_{window_ft}"] = observed_count
        quality[f"gr_local_mad_{window_ft}"] = local_mad
        quality[f"gr_local_variance_{window_ft}"] = local_variance

    return quality


def build_rf02b_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """在 B00 隐藏行后追加三尺度 GR count/MAD/方差；模型只选择后两者。"""

    # result 先包含 B00 十二列、隐藏行位置和训练时才使用的目标。
    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    # quality 的 shape 为 [整口井行数, 9]，只从 MD 和原始 GR 计算。
    quality = calculate_local_gr_quality(
        horizontal_df,
        windows_ft=[50, 100, 200],
        minimum_observed_count=MINIMUM_OBSERVED_COUNT,
    )

    # hidden_positions 把整井质量统计精确对齐到自然隐藏后缀。
    hidden_positions = result["row_index"].to_numpy(dtype=np.int64)

    # 诊断 count 和六个模型特征一起写入缓存，便于检查 NaN 是否来自支持不足。
    for window_ft in [50, 100, 200]:
        feature_names = [
            f"gr_local_observed_count_{window_ft}",
            f"gr_local_mad_{window_ft}",
            f"gr_local_variance_{window_ft}",
        ]
        for feature_name in feature_names:
            result[feature_name] = quality[feature_name].to_numpy()[hidden_positions]

    return result
