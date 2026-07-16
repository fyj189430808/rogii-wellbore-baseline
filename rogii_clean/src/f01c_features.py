"""F01c：只在隐藏段前 1000 ft 提供局部斜率的线性外推特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.f01_features import build_f01_lgbm_rows
from src.f01b_features import ALL_F01B_FEATURE_COLUMNS


# 1000 ft 来自 F01 的距离诊断：局部斜率在近端有用，长距离误差开始累积。
GATED_PROJECTION_LIMIT_FT = 1000.0

# 三列分别使用 200、500、1000 ft 可见窗口斜率，但共享同一个未来距离门槛。
F01C_GATED_FEATURE_COLUMNS = [
    "u_linear_delta_200_within_1000",
    "u_linear_delta_500_within_1000",
    "u_linear_delta_1000_within_1000",
]

# F01c 的模型输入严格为 F01b 的十九列加三列门控外推，共二十二列。
ALL_F01C_FEATURE_COLUMNS = ALL_F01B_FEATURE_COLUMNS + F01C_GATED_FEATURE_COLUMNS


def build_f01c_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """构造 F01 特征，将三列无界外推改名并在 1000 ft 后设为缺失。"""

    # 复用 F01 的斜率和 TVT_delta 公式，避免本实验同时改变数值计算方式。
    result = build_f01_lgbm_rows(horizontal_df, well_id, fold)

    # 该布尔数组形状为 [隐藏行数]；True 表示局部趋势已经超出允许使用距离。
    outside_gate = (
        result["md_since_visible_end"].to_numpy(dtype=np.float64)
        > GATED_PROJECTION_LIMIT_FT
    )

    # 每个窗口只改变特征名称和可用距离，1000 ft 内的原始数值保持逐位一致。
    for window_ft in [200, 500, 1000]:
        raw_feature_name = f"u_linear_delta_{window_ft}"
        gated_feature_name = f"u_linear_delta_{window_ft}_within_1000"

        # pop 同时取出并删除无界版本，保证它不会意外进入缓存或模型。
        gated_values = result.pop(raw_feature_name).to_numpy(dtype=np.float64)

        # 缺失值由 LightGBM 原生处理，避免把局部斜率乘到数千英尺之外。
        gated_values[outside_gate] = np.nan
        result[gated_feature_name] = gated_values

    return result

