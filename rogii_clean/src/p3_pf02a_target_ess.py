"""P3-PF02a：按固定目标 ESS 从四条冻结 PF 路径中离散选一条。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


SCALE_TO_PATH_COLUMN = {
    3.0: "pf128_scale_3_delta",
    5.0: "pf128_scale_5_delta",
    8.0: "pf128_scale_8_delta",
    12.0: "pf128_scale_12_delta",
}


def choose_scale_for_target_ess(
    ess_by_scale: Mapping[float, float],
    target_ess: float,
) -> tuple[float, float]:
    """只看无标签 ESS，返回与固定目标最接近的冻结温度及其 ESS。"""
    if not np.isfinite(target_ess) or target_ess <= 0.0:
        raise ValueError("target_ess 必须为有限正数")
    if set(float(scale) for scale in ess_by_scale) != set(SCALE_TO_PATH_COLUMN):
        raise ValueError("ess_by_scale 必须正好包含 3、5、8、12 四个温度")

    candidates: list[tuple[float, float]] = []
    for scale in sorted(SCALE_TO_PATH_COLUMN):
        effective_sample_size = float(ess_by_scale[scale])
        if not np.isfinite(effective_sample_size) or effective_sample_size <= 0.0:
            raise ValueError("每个候选 ESS 都必须为有限正数")
        candidates.append((scale, effective_sample_size))

    # 距离完全相同时选择更大的温度，让权重更分散；该规则不读取任何标签。
    selected_scale, selected_ess = min(
        candidates,
        key=lambda item: (abs(item[1] - float(target_ess)), -item[0]),
    )
    return float(selected_scale), float(selected_ess)


def build_target_ess_path(
    frozen_paths: pd.DataFrame,
    ess_by_scale: Mapping[float, float],
    target_ess: float,
) -> pd.DataFrame:
    """从一口井四条冻结路径中复制 ESS 最接近目标的一条，不重新运行 PF。"""
    required_columns = [
        "well_id",
        "row_index",
        "last_visible_tvt",
        *SCALE_TO_PATH_COLUMN.values(),
    ]
    missing_columns = [
        column for column in required_columns if column not in frozen_paths.columns
    ]
    if missing_columns:
        raise ValueError(f"冻结路径缓存缺少列: {missing_columns}")
    if frozen_paths.empty:
        raise ValueError("冻结路径缓存不能为空")
    if frozen_paths["well_id"].astype(str).nunique() != 1:
        raise ValueError("一次只能为一口井选择路径")

    selected_scale, selected_ess = choose_scale_for_target_ess(
        ess_by_scale=ess_by_scale,
        target_ess=target_ess,
    )
    selected_path_column = SCALE_TO_PATH_COLUMN[selected_scale]
    selected_delta = frozen_paths[selected_path_column].to_numpy(dtype=np.float32)
    if not np.isfinite(selected_delta).all():
        raise ValueError("选中的冻结路径含 NaN 或无穷值")

    result = frozen_paths[["well_id", "row_index", "last_visible_tvt"]].copy()
    result["pf128_target_ess_delta"] = selected_delta
    result["selected_scale"] = np.float32(selected_scale)
    result["selected_ess"] = np.float32(selected_ess)
    return result

