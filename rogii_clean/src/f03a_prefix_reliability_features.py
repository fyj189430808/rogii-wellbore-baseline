"""把 RF03-D0 的可见前缀诊断结果整理成一井一行的 F03a 特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd


SCOPE_PREFIXES = {
    "all_visible": "all",
    "tail_1000ft": "tail",
}

SCOPE_FEATURE_SUFFIXES = [
    "zero_raw_ncc",
    "zero_affine_mae",
    "valid_pair_count",
    "valid_pair_fraction",
    "ncc_margin",
    "mae_margin",
]

F03A_FEATURE_COLUMNS = [
    *[
        f"f03a_{scope_prefix}_{suffix}"
        for scope_prefix in ("all", "tail")
        for suffix in SCOPE_FEATURE_SUFFIXES
    ],
    "f03a_tail_minus_all_raw_ncc",
    "f03a_tail_minus_all_affine_mae",
    "f03a_tail_minus_all_valid_pair_count",
    "f03a_tail_minus_all_valid_pair_fraction",
    "f03a_tail_minus_all_ncc_margin",
    "f03a_tail_minus_all_mae_margin",
]


def _validate_unique_scope_rows(table: pd.DataFrame, table_name: str) -> None:
    """确认每口井在每个范围内只有一行。"""

    if table.duplicated(["well_id", "scope"]).any():
        raise ValueError(f"{table_name} 存在重复的 well_id/scope")


def build_well_prefix_reliability_features(
    offset_scores: pd.DataFrame,
    margins: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """返回一井一行的 F03a 特征；全部信息都来自可见前缀。"""

    zero_scores = offset_scores.loc[
        np.isclose(offset_scores["offset_ft"].to_numpy(dtype=np.float64), 0.0),
        [
            "well_id",
            "fold",
            "scope",
            "n_points",
            "raw_ncc",
            "affine_median_ae",
        ],
    ].copy()
    selected_margins = margins[
        [
            "well_id",
            "fold",
            "scope",
            "n_points",
            "ncc_margin_vs_best_wrong",
            "mae_margin_vs_best_wrong",
        ]
    ].copy()

    zero_scores["well_id"] = zero_scores["well_id"].astype(str)
    selected_margins["well_id"] = selected_margins["well_id"].astype(str)
    registry_keys = registry[["well_id", "fold", "visible_rows"]].copy()
    registry_keys["well_id"] = registry_keys["well_id"].astype(str)

    _validate_unique_scope_rows(zero_scores, "offset_scores 的零偏移行")
    _validate_unique_scope_rows(selected_margins, "per_well_margins")

    expected_scopes = set(SCOPE_PREFIXES)
    if set(zero_scores["scope"].unique()) != expected_scopes:
        raise ValueError("offset_scores 的 scope 不完整")
    if set(selected_margins["scope"].unique()) != expected_scopes:
        raise ValueError("per_well_margins 的 scope 不完整")

    merged = zero_scores.merge(
        selected_margins,
        on=["well_id", "fold", "scope", "n_points"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(registry_keys) * len(expected_scopes)
    if len(merged) != expected_rows:
        raise ValueError(
            f"D0 特征行数不完整：actual={len(merged)}, expected={expected_rows}"
        )

    merged = merged.merge(
        registry_keys,
        on=["well_id", "fold"],
        how="inner",
        validate="many_to_one",
    )
    if len(merged) != expected_rows:
        raise ValueError("D0 结果与固定 fold 注册表不一致")

    merged["valid_pair_fraction"] = (
        merged["n_points"].to_numpy(dtype=np.float64)
        / merged["visible_rows"].to_numpy(dtype=np.float64)
    )
    merged = merged.rename(
        columns={
            "raw_ncc": "zero_raw_ncc",
            "affine_median_ae": "zero_affine_mae",
            "n_points": "valid_pair_count",
            "ncc_margin_vs_best_wrong": "ncc_margin",
            "mae_margin_vs_best_wrong": "mae_margin",
        }
    )

    scope_tables: dict[str, pd.DataFrame] = {}
    for scope_name, scope_prefix in SCOPE_PREFIXES.items():
        scope_table = merged.loc[
            merged["scope"] == scope_name,
            ["well_id", "fold", *SCOPE_FEATURE_SUFFIXES],
        ].copy()
        scope_table = scope_table.rename(
            columns={
                suffix: f"f03a_{scope_prefix}_{suffix}"
                for suffix in SCOPE_FEATURE_SUFFIXES
            }
        )
        scope_tables[scope_prefix] = scope_table

    features = scope_tables["all"].merge(
        scope_tables["tail"],
        on=["well_id", "fold"],
        how="inner",
        validate="one_to_one",
    )
    difference_sources = {
        "raw_ncc": "zero_raw_ncc",
        "affine_mae": "zero_affine_mae",
        "valid_pair_count": "valid_pair_count",
        "valid_pair_fraction": "valid_pair_fraction",
        "ncc_margin": "ncc_margin",
        "mae_margin": "mae_margin",
    }
    for output_suffix, source_suffix in difference_sources.items():
        features[f"f03a_tail_minus_all_{output_suffix}"] = (
            features[f"f03a_tail_{source_suffix}"]
            - features[f"f03a_all_{source_suffix}"]
        )

    features = features[["well_id", "fold", *F03A_FEATURE_COLUMNS]]
    if len(features) != len(registry_keys):
        raise ValueError("F03a 最终井级特征行数与注册表不一致")
    return features.sort_values("well_id").reset_index(drop=True)
