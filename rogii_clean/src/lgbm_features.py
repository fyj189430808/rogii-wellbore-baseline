"""为 B00 的单口水平井构造固定且合法的 12 个简单特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 这是模型唯一允许读取的 12 个特征；顺序固定，避免训练和推理时列错位。
FEATURE_COLUMNS = [
    "last_visible_tvt",
    "md_since_visible_end",
    "hidden_fraction",
    "x_current",
    "y_current",
    "z_current",
    "dx_from_visible_end",
    "dy_from_visible_end",
    "dz_from_visible_end",
    "dxy_from_visible_end",
    "gr_raw",
    "gr_missing",
]

# 这些原始列是构造特征、锚点和训练目标所必需的输入。
REQUIRED_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]


# 输入是一口完整训练井、井号和外层 fold；输出只包含隐藏后缀行，不修改输入表。
def build_simple_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """构造隐藏行的 12 个合法特征、可见末值锚点和训练目标。"""

    # 找出缺少的必需列；集合中的每个元素都是一个原始字段名。
    missing_columns = set(REQUIRED_COLUMNS) - set(horizontal_df.columns)

    # 缺列时立即停止，防止后面用错误或不完整的数据静默生成特征。
    if missing_columns:
        raise ValueError(f"水平井缺少列：{sorted(missing_columns)}")

    # visible_mask 形状为 [总行数]；True 表示 TVT_input 在推理时可见。
    visible_mask = horizontal_df["TVT_input"].notna().to_numpy()

    # hidden_mask 形状为 [总行数]；True 表示该行属于自然隐藏评价段。
    hidden_mask = horizontal_df["TVT_input"].isna().to_numpy()

    # 一口合法训练井必须同时提供可见前缀和隐藏后缀。
    if not visible_mask.any() or not hidden_mask.any():
        raise ValueError(f"井 {well_id} 必须同时有可见前缀和隐藏后缀")

    # visible_positions 形状为 [可见行数]，保存可见行在输入表中的整数位置。
    visible_positions = np.flatnonzero(visible_mask)

    # hidden_positions 形状为 [隐藏行数]，保存隐藏行在输入表中的整数位置。
    hidden_positions = np.flatnonzero(hidden_mask)

    # 最后一个可见位置是所有隐藏特征和 target_delta 共用的锚点。
    last_visible_position = int(visible_positions[-1])

    # 合法可见区必须从第 0 行连续延伸到最后一个可见位置。
    expected_visible = np.arange(last_visible_position + 1)

    # 合法隐藏区必须紧跟可见区，并一直连续到输入表末尾。
    expected_hidden = np.arange(last_visible_position + 1, len(horizontal_df))

    # 若可见行中间夹有隐藏行，就不能把最后可见行当作统一锚点。
    if not np.array_equal(visible_positions, expected_visible):
        raise ValueError(f"井 {well_id} 的 TVT_input 可见区不是连续前缀")

    # 若隐藏行不是连续后缀，hidden_fraction 的起止定义就不成立。
    if not np.array_equal(hidden_positions, expected_hidden):
        raise ValueError(f"井 {well_id} 的 TVT_input 隐藏区不是连续后缀")

    # md_all 形状为 [总行数]，单位沿用原始 MD，通常为 ft。
    md_all = horizontal_df["MD"].to_numpy(dtype=np.float64)

    # MD 必须随行号单调非降，否则“距可见末端的 MD”会倒退。
    if np.any(np.diff(md_all) < 0.0):
        raise ValueError(f"井 {well_id} 的 MD 不是单调非降")

    # hidden_df 形状为 [隐藏行数, 原始列数]，只保留需要预测的连续后缀。
    hidden_df = horizontal_df.iloc[hidden_positions]

    # last_visible_row 是一维 Series，提供可见末端的坐标、MD 和 TVT 锚点。
    last_visible_row = horizontal_df.iloc[last_visible_position]

    # hidden_md 形状为 [隐藏行数]，单位沿用原始 MD，通常为 ft。
    hidden_md = hidden_df["MD"].to_numpy(dtype=np.float64)

    # hidden_span 是隐藏段首尾 MD 跨度；至少取 1，避免单行隐藏段除以零。
    hidden_span = max(float(hidden_md[-1] - hidden_md[0]), 1.0)

    # dx 形状为 [隐藏行数]，表示每个隐藏点相对可见末点的 X 位移。
    dx = hidden_df["X"].to_numpy(dtype=np.float64) - float(last_visible_row["X"])

    # dy 形状为 [隐藏行数]，表示每个隐藏点相对可见末点的 Y 位移。
    dy = hidden_df["Y"].to_numpy(dtype=np.float64) - float(last_visible_row["Y"])

    # dz 形状为 [隐藏行数]，表示每个隐藏点相对可见末点的 Z 位移。
    dz = hidden_df["Z"].to_numpy(dtype=np.float64) - float(last_visible_row["Z"])

    # last_visible_tvt 是推理时合法可见的最后一个 TVT_input，作为 carry 锚点。
    last_visible_tvt = float(last_visible_row["TVT_input"])

    # target_tvt 形状为 [隐藏行数]；它只用于训练标签和评分，绝不进入特征列。
    target_tvt = hidden_df["TVT"].to_numpy(dtype=np.float64)

    # gr_raw 形状为 [隐藏行数]，保留原始缺失值，让模型结合缺失指示特征使用。
    gr_raw = hidden_df["GR"].to_numpy(dtype=np.float64)

    # 每一行输出对应一个隐藏采样点；标签列和元数据列不属于 FEATURE_COLUMNS。
    result = pd.DataFrame(
        {
            "well_id": str(well_id),
            "fold": int(fold),
            "row_index": hidden_positions.astype(np.int32),
            "md": hidden_md,
            "target_tvt": target_tvt,
            "carry_tvt": np.full(len(hidden_positions), last_visible_tvt),
            "target_delta": target_tvt - last_visible_tvt,
            "last_visible_tvt": last_visible_tvt,
            "md_since_visible_end": hidden_md - float(last_visible_row["MD"]),
            "hidden_fraction": (hidden_md - hidden_md[0]) / hidden_span,
            "x_current": hidden_df["X"].to_numpy(dtype=np.float64),
            "y_current": hidden_df["Y"].to_numpy(dtype=np.float64),
            "z_current": hidden_df["Z"].to_numpy(dtype=np.float64),
            "dx_from_visible_end": dx,
            "dy_from_visible_end": dy,
            "dz_from_visible_end": dz,
            "dxy_from_visible_end": np.sqrt(dx * dx + dy * dy),
            "gr_raw": gr_raw,
            "gr_missing": np.isnan(gr_raw).astype(np.float64),
        }
    )

    # 返回隐藏行表；调用者可用 FEATURE_COLUMNS 精确选择模型输入。
    return result
