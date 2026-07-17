"""从四条代表性物理候选路径提取逐行的路径族分歧特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd


REPRESENTATIVE_PATH_COLUMNS = [
    "pf_ancc_delta",
    "pf_z_delta",
    "beam_mean_d",
    "sc_ens_d",
]

FAMILY_DISAGREEMENT_COLUMNS = [
    "pf_ancc_minus_beam",
    "pf_ancc_minus_sc",
    "beam_minus_sc",
    "abs_pf_ancc_minus_beam",
    "abs_pf_ancc_minus_sc",
    "abs_beam_minus_sc",
    "family_path_std",
    "family_path_range",
    "family_path_mad",
]

SOURCE_COLUMNS = ["well_id", "row_index", *REPRESENTATIVE_PATH_COLUMNS]


def build_family_disagreement_features(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """仅使用冻结候选缓存中的四条代表路径，返回键和九个分歧特征。"""

    missing_columns = set(SOURCE_COLUMNS) - set(candidate_rows.columns)
    if missing_columns:
        raise ValueError(f"确定性候选缓存缺少列：{sorted(missing_columns)}")

    representative_paths = candidate_rows[REPRESENTATIVE_PATH_COLUMNS].to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(representative_paths).all():
        raise ValueError("四条代表性候选路径含 NaN 或 Inf")

    pf_ancc = representative_paths[:, 0]
    beam_mean = representative_paths[:, 2]
    scale_ensemble = representative_paths[:, 3]

    pf_ancc_minus_beam = pf_ancc - beam_mean
    pf_ancc_minus_sc = pf_ancc - scale_ensemble
    beam_minus_sc = beam_mean - scale_ensemble

    path_median = np.median(representative_paths, axis=1)
    path_mad = np.median(
        np.abs(representative_paths - path_median[:, None]),
        axis=1,
    )

    result = pd.DataFrame(
        {
            "well_id": candidate_rows["well_id"].astype(str).to_numpy(),
            "row_index": pd.to_numeric(
                candidate_rows["row_index"], errors="raise"
            ).to_numpy(dtype=np.int64),
            "pf_ancc_minus_beam": pf_ancc_minus_beam,
            "pf_ancc_minus_sc": pf_ancc_minus_sc,
            "beam_minus_sc": beam_minus_sc,
            "abs_pf_ancc_minus_beam": np.abs(pf_ancc_minus_beam),
            "abs_pf_ancc_minus_sc": np.abs(pf_ancc_minus_sc),
            "abs_beam_minus_sc": np.abs(beam_minus_sc),
            "family_path_std": np.std(representative_paths, axis=1),
            "family_path_range": np.ptp(representative_paths, axis=1),
            "family_path_mad": path_mad,
        }
    )
    for feature_name in FAMILY_DISAGREEMENT_COLUMNS:
        result[feature_name] = result[feature_name].astype(np.float32)
    return result[["well_id", "row_index", *FAMILY_DISAGREEMENT_COLUMNS]]

