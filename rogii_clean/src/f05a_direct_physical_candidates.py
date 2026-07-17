"""读取 notebook 已冻结的合法 PF、Beam 和多尺度候选输出。"""

from __future__ import annotations

import numpy as np
import pandas as pd


DIRECT_CANDIDATE_COLUMNS = [
    "pf_ancc_delta",
    "pf_ancc_std",
    "pf_z_delta",
    "pf_vs_z",
    "beam_cons_d",
    "beam_loose_d",
    "beam_vcons_d",
    "beam_sm5_d",
    "beam_vloose_d",
    "beam_mid_d",
    "beam_stiff_d",
    "beam_mean_d",
    "beam_std_d",
    "beam_med_d",
    "sc8_d",
    "sc8_sc",
    "sc15_d",
    "sc15_sc",
    "sc25_d",
    "sc25_sc",
    "sc_cons_d",
    "sc_ens_d",
    "sc_trust",
    "hyb_d",
]

DIRECT_SOURCE_USE_COLUMNS = ["well", "id", *DIRECT_CANDIDATE_COLUMNS]


def build_direct_candidate_features(source_df: pd.DataFrame) -> pd.DataFrame:
    """返回键和 24 个直接候选特征，忽略输入中的其余列。"""

    missing_columns = set(DIRECT_SOURCE_USE_COLUMNS) - set(source_df.columns)
    if missing_columns:
        raise ValueError(f"候选数据缺少列：{sorted(missing_columns)}")

    well_ids = source_df["well"].astype(str).to_numpy()
    id_parts = source_df["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if id_parts.shape[1] != 2:
        raise ValueError("候选数据 id 必须以 _row_index 结尾")
    id_wells = id_parts.iloc[:, 0].to_numpy(dtype=str)
    if not np.array_equal(well_ids, id_wells):
        raise ValueError("候选数据的 well 与 id 前缀不一致")
    row_indices = pd.to_numeric(id_parts.iloc[:, 1], errors="raise").to_numpy(
        dtype=np.int32
    )

    feature_values = source_df[DIRECT_CANDIDATE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        raise ValueError("直接物理候选特征含 NaN 或 Inf")

    result = pd.DataFrame({"well_id": well_ids, "row_index": row_indices})
    for feature_index, feature_name in enumerate(DIRECT_CANDIDATE_COLUMNS):
        result[feature_name] = feature_values[:, feature_index]
    return result[["well_id", "row_index", *DIRECT_CANDIDATE_COLUMNS]]

