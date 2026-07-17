"""RF01e/f：从可见前缀提取去冗余的 Huber U 倾角窗口子集。"""

from __future__ import annotations

import pandas as pd

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows
from src.rf01a_features import fit_visible_u_huber_slope


# RF01e 覆盖短、中、长尺度，同时删除与相邻尺度高度共线的 100/200 ft。
RF01E_SLOPE_FEATURE_COLUMNS = [
    "u_huber_slope_50",
    "u_huber_slope_500",
    "u_huber_slope_1000",
]

# RF01e 单模输入为 B00 十二列加三个 Huber 倾角，共十五列。
ALL_RF01E_FEATURE_COLUMNS = FEATURE_COLUMNS + RF01E_SLOPE_FEATURE_COLUMNS

# RF01f 是路线停止前的正对照，只保留全五折诊断最稳的 500 ft 倾角。
RF01F_SLOPE_FEATURE_COLUMNS = ["u_huber_slope_500"]

# RF01f 单模输入为 B00 十二列加一个 Huber 倾角，共十三列。
ALL_RF01F_FEATURE_COLUMNS = FEATURE_COLUMNS + RF01F_SLOPE_FEATURE_COLUMNS


def build_selected_huber_slope_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
    windows_ft: list[int],
) -> pd.DataFrame:
    """在 B00 行特征后追加指定窗口的井级 Huber U 倾角。"""

    # 先构造每个隐藏行都能在测试时获得的 B00 十二列。
    result = build_simple_lgbm_rows(horizontal_df, well_id, fold)

    # 每个 slope 只使用当前井可见前缀中的 MD、TVT_input 和 Z。
    for window_ft in windows_ft:
        # slope 是一个井级标量，单位 ft/ft，之后复制到该井所有隐藏行。
        slope = fit_visible_u_huber_slope(
            horizontal_df,
            window_ft=float(window_ft),
        )

        # 新列 shape 为 [隐藏行数]，不读取隐藏段 TVT 真值。
        result[f"u_huber_slope_{window_ft}"] = slope

    return result


def build_rf01e_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """返回 B00 加 50/500/1000 ft 三窗口倾角的十五列隐藏行特征。"""

    return build_selected_huber_slope_rows(
        horizontal_df=horizontal_df,
        well_id=well_id,
        fold=fold,
        windows_ft=[50, 500, 1000],
    )


def build_rf01f_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """返回 B00 加 500 ft 单窗口倾角的十三列隐藏行特征。"""

    return build_selected_huber_slope_rows(
        horizontal_df=horizontal_df,
        well_id=well_id,
        fold=fold,
        windows_ft=[500],
    )
