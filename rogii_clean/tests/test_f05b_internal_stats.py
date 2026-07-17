from __future__ import annotations

import numpy as np
import pandas as pd

from src.f05a_candidate_reproduction import _particle_filter_ancc
from src.f05b_internal_stats import (
    F05B_FEATURE_COLUMNS,
    _particle_filter_ancc_with_diagnostics,
    build_internal_stats_features,
)


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造带有两个隐藏 GR 缺失点的小井。"""

    typewell_tvt = np.arange(11000.0, 11080.5, 0.5, dtype=np.float64)
    typewell_gr = (
        70.0
        + 20.0 * np.sin((typewell_tvt - 11000.0) / 4.0)
        + 8.0 * np.cos((typewell_tvt - 11000.0) / 1.7)
    )
    typewell_df = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})

    visible_count = 80
    hidden_count = 20
    md = np.arange(visible_count + hidden_count, dtype=np.float64)
    z = 1000.0 + 0.02 * md + 0.0002 * md**2
    true_tvt = 11025.0 + 0.025 * md + 0.20 * np.sin(md / 15.0)
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)
    horizontal_gr = horizontal_gr + 1.5 * np.sin(md / 3.0)
    horizontal_gr[[84, 91]] = np.nan
    tvt_input = true_tvt.copy()
    tvt_input[visible_count:] = np.nan
    horizontal_df = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": horizontal_gr,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
            "ANCC": true_tvt + z - 11373.0,
        }
    )
    return horizontal_df, typewell_df


def test_internal_stats_have_fixed_row_level_shape_and_valid_ranges() -> None:
    horizontal_df, typewell_df = _synthetic_well()

    result = build_internal_stats_features(horizontal_df, typewell_df, seed=42)

    assert len(F05B_FEATURE_COLUMNS) == 10
    assert list(result.columns) == ["row_index", *F05B_FEATURE_COLUMNS]
    assert result["row_index"].tolist() == list(range(80, 100))
    assert result.shape == (20, 11)
    values = result[list(F05B_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()

    unit_interval_columns = [
        column
        for column in F05B_FEATURE_COLUMNS
        if column != "f05b_mean_loglik_gap"
    ]
    unit_values = result[unit_interval_columns].to_numpy(dtype=np.float64)
    assert (unit_values >= 0.0).all()
    assert (unit_values <= 1.0).all()
    assert (result["f05b_mean_loglik_gap"] >= 0.0).all()


def test_internal_stats_are_running_trajectories_and_track_missing_gr() -> None:
    horizontal_df, typewell_df = _synthetic_well()

    result = build_internal_stats_features(horizontal_df, typewell_df, seed=42)

    # 隐藏段第 5 行（原索引 84）缺 GR，所以累计有效更新比例应从 1 降为 4/5。
    missing_row = result.loc[result["row_index"] == 84].iloc[0]
    assert np.isclose(missing_row["f05b_valid_update_fraction"], 4.0 / 5.0)
    assert np.isclose(missing_row["f05b_steps_since_valid_update_ratio"], 1.0 / 5.0)

    # 下一行重新出现有效 GR，距上次更新的比例归零。
    next_valid_row = result.loc[result["row_index"] == 85].iloc[0]
    assert np.isclose(next_valid_row["f05b_steps_since_valid_update_ratio"], 0.0)

    # 这些列必须随轨迹推进变化，不能退化成整井常数复制。
    varying_columns = [
        "f05b_mean_valid_ess_ratio",
        "f05b_resample_fraction",
        "f05b_valid_update_fraction",
    ]
    assert any(result[column].nunique() > 1 for column in varying_columns)


def test_same_seed_reproduces_identical_internal_stats() -> None:
    horizontal_df, typewell_df = _synthetic_well()

    first = build_internal_stats_features(horizontal_df, typewell_df, seed=17)
    second = build_internal_stats_features(horizontal_df, typewell_df, seed=17)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_hidden_tvt_and_surface_mutation_or_removal_cannot_change_stats() -> None:
    horizontal_df, typewell_df = _synthetic_well()
    hidden_mask = horizontal_df["TVT_input"].isna()

    mutated = horizontal_df.copy()
    mutated.loc[hidden_mask, "TVT"] = np.linspace(-1e6, 1e6, hidden_mask.sum())
    mutated.loc[hidden_mask, "ANCC"] = np.linspace(1e9, -1e9, hidden_mask.sum())
    removed = horizontal_df.drop(columns=["TVT", "ANCC"])

    original_result = build_internal_stats_features(horizontal_df, typewell_df, seed=29)
    mutated_result = build_internal_stats_features(mutated, typewell_df, seed=29)
    removed_result = build_internal_stats_features(removed, typewell_df, seed=29)

    pd.testing.assert_frame_equal(original_result, mutated_result, check_exact=True)
    pd.testing.assert_frame_equal(original_result, removed_result, check_exact=True)


def test_diagnostics_do_not_change_the_original_ancc_pf_path() -> None:
    hidden_md = np.arange(80.0, 100.0, dtype=np.float64)
    hidden_z = 1000.0 + 0.02 * hidden_md
    grid_minimum = 11000.0
    grid_step = 0.2
    typewell_tvt = np.arange(11000.0, 11080.2, grid_step)
    typewell_gr_grid = 70.0 + 20.0 * np.sin((typewell_tvt - 11000.0) / 4.0)
    hidden_tvt = 11027.0 + 0.02 * (hidden_md - 80.0)
    hidden_gr = np.interp(hidden_tvt, typewell_tvt, typewell_gr_grid)
    hidden_gr[[4, 11]] = np.nan
    arguments = (
        hidden_md,
        hidden_z,
        hidden_gr,
        typewell_gr_grid,
        grid_minimum,
        grid_step,
        20.0,
        hidden_tvt[0] + hidden_z[0],
        0.04,
        600,
        42,
    )

    expected_path, expected_std = _particle_filter_ancc(*arguments)
    actual_path, actual_std, diagnostics = _particle_filter_ancc_with_diagnostics(
        *arguments
    )

    np.testing.assert_array_equal(actual_path, expected_path)
    np.testing.assert_array_equal(actual_std, expected_std)
    assert diagnostics.shape == (20, 10)
