"""从冻结的 PF/Beam/多尺度候选路径中提取汇总几何特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd


CANDIDATE_PATH_COLUMNS = [
    "pf_ancc_delta",
    "pf_z_delta",
    "beam_cons_d",
    "beam_loose_d",
    "beam_vcons_d",
    "beam_sm5_d",
    "beam_vloose_d",
    "beam_mid_d",
    "beam_stiff_d",
    "sc8_d",
    "sc15_d",
    "sc25_d",
    "sc_cons_d",
    "sc_ens_d",
    "hyb_d",
]

BEAM_PATH_COLUMNS = [
    "beam_cons_d",
    "beam_loose_d",
    "beam_vcons_d",
    "beam_sm5_d",
    "beam_vloose_d",
    "beam_mid_d",
    "beam_stiff_d",
]

SCALE_PATH_COLUMNS = ["sc8_d", "sc15_d", "sc25_d", "sc_cons_d", "sc_ens_d"]

SCORE_COLUMNS = [
    "pf_ancc_std",
    "pf_vs_z",
    "sc8_sc",
    "sc15_sc",
    "sc25_sc",
]

SOURCE_USE_COLUMNS = ["well", "id", *CANDIDATE_PATH_COLUMNS, *SCORE_COLUMNS]

ROW_GEOMETRY_COLUMNS = [
    "candidate_median",
    "candidate_iqr",
    "candidate_mad",
    "candidate_span",
    "candidate_largest_gap",
    "candidate_typical_separation",
    "candidate_tail_separation",
    "pf_vs_candidate_median_abs",
    "beam_group_vs_scale_group_abs",
    "pf_seed_std",
    "abs_pf_vs_z",
    "scale_score_std",
    "candidate_span_growth",
]

ENDPOINT_GEOMETRY_COLUMNS = [
    "endpoint_candidate_median",
    "endpoint_candidate_iqr",
    "endpoint_candidate_mad",
    "endpoint_candidate_span",
    "endpoint_candidate_largest_gap",
]

F05A_FEATURE_COLUMNS = [*ROW_GEOMETRY_COLUMNS, *ENDPOINT_GEOMETRY_COLUMNS]


def _quantile_from_sorted(sorted_values: np.ndarray, quantile: float) -> np.ndarray:
    """对已排序且有限的二维数组逐行做线性插值分位数，不重复排序。"""

    position = quantile * (sorted_values.shape[1] - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    upper_weight = np.float32(position - lower_index)
    lower_values = sorted_values[:, lower_index]
    upper_values = sorted_values[:, upper_index]
    return lower_values + upper_weight * (upper_values - lower_values)


def _parse_row_keys(source_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """从 `well` 和 `id` 解析 B00 使用的 well_id、row_index。"""

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
    return well_ids, row_indices


def build_candidate_geometry(source_df: pd.DataFrame) -> pd.DataFrame:
    """为一批完整井生成 18 个候选几何特征，不读取目标列。"""

    missing_columns = set(SOURCE_USE_COLUMNS) - set(source_df.columns)
    if missing_columns:
        raise ValueError(f"候选数据缺少列：{sorted(missing_columns)}")

    well_ids, row_indices = _parse_row_keys(source_df)
    candidates = source_df[CANDIDATE_PATH_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(candidates).all():
        raise ValueError("冻结候选路径含 NaN 或 Inf")
    sorted_candidates = np.sort(candidates, axis=1)
    candidate_median = _quantile_from_sorted(sorted_candidates, 0.50)
    candidate_q25 = _quantile_from_sorted(sorted_candidates, 0.25)
    candidate_q75 = _quantile_from_sorted(sorted_candidates, 0.75)
    candidate_iqr = candidate_q75 - candidate_q25
    candidate_q10 = _quantile_from_sorted(sorted_candidates, 0.10)
    candidate_q90 = _quantile_from_sorted(sorted_candidates, 0.90)
    candidate_tail_separation = candidate_q90 - candidate_q10
    sorted_absolute_deviations = np.sort(
        np.abs(candidates - candidate_median[:, None]), axis=1
    )
    candidate_mad = _quantile_from_sorted(sorted_absolute_deviations, 0.50)
    candidate_min = sorted_candidates[:, 0]
    candidate_max = sorted_candidates[:, -1]
    candidate_span = candidate_max - candidate_min

    adjacent_gaps = np.diff(sorted_candidates, axis=1)
    candidate_largest_gap = np.max(adjacent_gaps, axis=1)

    sorted_beam_values = np.sort(
        source_df[BEAM_PATH_COLUMNS].to_numpy(dtype=np.float32), axis=1
    )
    sorted_scale_values = np.sort(
        source_df[SCALE_PATH_COLUMNS].to_numpy(dtype=np.float32), axis=1
    )
    beam_median = _quantile_from_sorted(sorted_beam_values, 0.50)
    scale_median = _quantile_from_sorted(sorted_scale_values, 0.50)
    scale_scores = source_df[["sc8_sc", "sc15_sc", "sc25_sc"]].to_numpy(
        dtype=np.float32
    )

    result = pd.DataFrame(
        {
            "well_id": well_ids,
            "row_index": row_indices,
            "candidate_median": candidate_median,
            "candidate_iqr": candidate_iqr,
            "candidate_mad": candidate_mad,
            "candidate_span": candidate_span,
            "candidate_largest_gap": candidate_largest_gap,
            "candidate_typical_separation": candidate_iqr,
            "candidate_tail_separation": candidate_tail_separation,
            "pf_vs_candidate_median_abs": np.abs(
                candidates[:, 0] - candidate_median
            ),
            "beam_group_vs_scale_group_abs": np.abs(beam_median - scale_median),
            "pf_seed_std": source_df["pf_ancc_std"].to_numpy(dtype=np.float32),
            "abs_pf_vs_z": np.abs(
                source_df["pf_vs_z"].to_numpy(dtype=np.float32)
            ),
            "scale_score_std": np.std(scale_scores, axis=1),
        }
    )

    result["candidate_span_growth"] = result["candidate_span"] - result.groupby(
        "well_id", sort=False
    )["candidate_span"].transform("first")

    endpoint_source_columns = [
        "candidate_median",
        "candidate_iqr",
        "candidate_mad",
        "candidate_span",
        "candidate_largest_gap",
    ]
    endpoint_names = dict(zip(endpoint_source_columns, ENDPOINT_GEOMETRY_COLUMNS))
    endpoint_rows = result.groupby("well_id", sort=False).tail(1)
    endpoint_lookup = endpoint_rows.set_index("well_id")[endpoint_source_columns]
    endpoint_lookup = endpoint_lookup.rename(columns=endpoint_names)
    result = result.join(endpoint_lookup, on="well_id")

    for feature_name in F05A_FEATURE_COLUMNS:
        result[feature_name] = result[feature_name].astype(np.float32)

    return result[["well_id", "row_index", *F05A_FEATURE_COLUMNS]]
