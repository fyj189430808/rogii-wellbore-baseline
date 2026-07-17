from __future__ import annotations

import numpy as np
import pandas as pd

import src.f06_prefix_holdout as f06


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 40 行可见前缀和 5 行自然隐藏段；可见 MD 跨度足够支持 1000 ft。"""

    row_count = 45
    visible_count = 40
    md = np.arange(row_count, dtype=np.float64) * 50.0
    true_tvt = 11020.0 + np.arange(row_count, dtype=np.float64)
    tvt_input = true_tvt.copy()
    tvt_input[visible_count:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": 1000.0 + 0.1 * md,
            "GR": 60.0 + np.sin(md / 80.0),
            "TVT_input": tvt_input,
            "TVT": true_tvt,
            "ANCC": true_tvt + 1000.0,
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.arange(10900.0, 11200.5, 0.5),
            "GR": 60.0 + np.sin(np.arange(10900.0, 11200.5, 0.5) / 20.0),
        }
    )
    return horizontal, typewell


def _perfectly_controlled_generator(true_tvt: np.ndarray):
    """返回误差固定为 1/2/3/4 ft 的四类候选，便于手算 RMSE。"""

    def fake_build_candidate_features(
        horizontal_df: pd.DataFrame,
        typewell_df: pd.DataFrame,
        seed: int = 42,
    ) -> pd.DataFrame:
        del typewell_df, seed
        hidden_mask = horizontal_df["TVT_input"].isna()
        hidden_indices = horizontal_df.index[hidden_mask].to_numpy(dtype=np.int64)
        anchor_tvt = float(horizontal_df.loc[~hidden_mask, "TVT_input"].iloc[-1])
        true_delta = true_tvt[hidden_indices] - anchor_tvt
        return pd.DataFrame(
            {
                "row_index": hidden_indices,
                "pf_z_delta": true_delta + 1.0,
                "pf_ancc_delta": true_delta + 2.0,
                "beam_mean_d": true_delta + 3.0,
                "sc_ens_d": true_delta + 4.0,
            }
        )

    return fake_build_candidate_features


def test_feature_schema_has_six_cuts_five_families_and_fixed_order() -> None:
    assert f06.F06_CUT_NAMES == (
        "ratio50",
        "ratio65",
        "ratio75",
        "horizon250",
        "horizon500",
        "horizon1000",
    )
    assert f06.F06_FAMILY_NAMES == ("carry", "geometry", "pf", "beam", "typewell")
    assert len(f06.F06_FEATURE_COLUMNS) == 174
    assert f06.F06_FEATURE_COLUMNS[:5] == (
        "f06_ratio50_carry_rmse",
        "f06_ratio50_carry_gain_vs_carry",
        "f06_ratio50_carry_gain_vs_pf",
        "f06_ratio50_carry_rank",
        "f06_ratio50_carry_coverage",
    )
    assert f06.F06_FEATURE_COLUMNS[-4:] == (
        "f06_rank_spearman_mean",
        "f06_rank_kendall_mean",
        "f06_winner_stability",
        "f06_modal_top2_set_rate",
    )


def test_each_cut_rebuilds_candidates_and_keeps_post_cut_gr(monkeypatch) -> None:
    horizontal, typewell = _synthetic_well()
    captured_inputs: list[pd.DataFrame] = []
    true_tvt = horizontal["TVT"].to_numpy(dtype=np.float64)
    controlled_generator = _perfectly_controlled_generator(true_tvt)

    def capturing_generator(
        horizontal_df: pd.DataFrame,
        typewell_df: pd.DataFrame,
        seed: int = 42,
    ) -> pd.DataFrame:
        captured_inputs.append(horizontal_df.copy())
        return controlled_generator(horizontal_df, typewell_df, seed)

    monkeypatch.setattr(f06, "build_candidate_features", capturing_generator)

    f06.build_prefix_holdout_features(horizontal, typewell, seed=42)

    assert len(captured_inputs) == 6
    expected_visible_counts = [20, 26, 30, 35, 30, 20]
    for rebuilt, expected_visible_count in zip(captured_inputs, expected_visible_counts):
        assert int(rebuilt["TVT_input"].notna().sum()) == expected_visible_count
        assert rebuilt.loc[expected_visible_count:, "TVT_input"].isna().all()
        np.testing.assert_array_equal(
            rebuilt["GR"].to_numpy(),
            horizontal["GR"].to_numpy(),
        )


def test_cut_metrics_follow_the_hand_calculated_formulas(monkeypatch) -> None:
    horizontal, typewell = _synthetic_well()
    true_tvt = horizontal["TVT"].to_numpy(dtype=np.float64)
    monkeypatch.setattr(
        f06,
        "build_candidate_features",
        _perfectly_controlled_generator(true_tvt),
    )

    result = f06.build_prefix_holdout_features(horizontal, typewell, seed=42).iloc[0]

    # ratio50 在第 20 行前切开，伪 holdout 的真实增量为 1..20。
    expected_carry_rmse = float(np.sqrt(np.mean(np.arange(1.0, 21.0) ** 2)))
    assert np.isclose(result["f06_ratio50_carry_rmse"], expected_carry_rmse)
    assert np.isclose(result["f06_ratio50_geometry_rmse"], 1.0)
    assert np.isclose(result["f06_ratio50_pf_rmse"], 2.0)
    assert np.isclose(result["f06_ratio50_beam_rmse"], 3.0)
    assert np.isclose(result["f06_ratio50_typewell_rmse"], 4.0)
    assert np.isclose(
        result["f06_ratio50_geometry_gain_vs_carry"],
        expected_carry_rmse - 1.0,
    )
    assert np.isclose(result["f06_ratio50_geometry_gain_vs_pf"], 1.0)
    assert result["f06_ratio50_geometry_rank"] == 1.0
    assert result["f06_ratio50_carry_coverage"] == 1.0
    assert 0.0 <= result["f06_winner_stability"] <= 1.0
    assert 0.0 <= result["f06_modal_top2_set_rate"] <= 1.0


def test_hidden_tvt_and_surface_are_never_read(monkeypatch) -> None:
    horizontal, typewell = _synthetic_well()
    true_tvt = horizontal["TVT"].to_numpy(dtype=np.float64)
    monkeypatch.setattr(
        f06,
        "build_candidate_features",
        _perfectly_controlled_generator(true_tvt),
    )
    hidden_mask = horizontal["TVT_input"].isna()
    mutated = horizontal.copy()
    mutated.loc[hidden_mask, "TVT"] = np.linspace(-1e9, 1e9, hidden_mask.sum())
    mutated.loc[hidden_mask, "ANCC"] = np.linspace(1e12, -1e12, hidden_mask.sum())
    removed = horizontal.drop(columns=["TVT", "ANCC"])

    expected = f06.build_prefix_holdout_features(horizontal, typewell, seed=11)
    mutated_result = f06.build_prefix_holdout_features(mutated, typewell, seed=11)
    removed_result = f06.build_prefix_holdout_features(removed, typewell, seed=11)

    pd.testing.assert_frame_equal(expected, mutated_result, check_exact=True)
    pd.testing.assert_frame_equal(expected, removed_result, check_exact=True)


def test_coverage_below_95_percent_invalidates_that_family_cut(monkeypatch) -> None:
    horizontal, typewell = _synthetic_well()
    true_tvt = horizontal["TVT"].to_numpy(dtype=np.float64)
    controlled_generator = _perfectly_controlled_generator(true_tvt)
    call_count = 0

    def generator_with_low_first_cut_coverage(
        horizontal_df: pd.DataFrame,
        typewell_df: pd.DataFrame,
        seed: int = 42,
    ) -> pd.DataFrame:
        nonlocal call_count
        candidates = controlled_generator(horizontal_df, typewell_df, seed)
        if call_count == 0:
            pseudo_indices = np.arange(20, 40, dtype=np.int64)
            candidates.loc[
                candidates["row_index"].isin(pseudo_indices[:2]),
                "pf_z_delta",
            ] = np.nan
        call_count += 1
        return candidates

    monkeypatch.setattr(f06, "build_candidate_features", generator_with_low_first_cut_coverage)

    result = f06.build_prefix_holdout_features(horizontal, typewell).iloc[0]

    assert np.isclose(result["f06_ratio50_geometry_coverage"], 0.9)
    assert np.isnan(result["f06_ratio50_geometry_rmse"])
    assert np.isnan(result["f06_ratio50_geometry_gain_vs_carry"])
    assert np.isnan(result["f06_ratio50_geometry_gain_vs_pf"])
    assert np.isnan(result["f06_ratio50_geometry_rank"])
    assert np.isfinite(result["f06_ratio50_pf_rmse"])


def test_unsupported_physical_horizons_have_only_expected_nans(monkeypatch) -> None:
    horizontal, typewell = _synthetic_well()
    horizontal["MD"] = np.arange(len(horizontal), dtype=np.float64) * 5.0
    true_tvt = horizontal["TVT"].to_numpy(dtype=np.float64)
    monkeypatch.setattr(
        f06,
        "build_candidate_features",
        _perfectly_controlled_generator(true_tvt),
    )

    result = f06.build_prefix_holdout_features(horizontal, typewell).iloc[0]

    for horizon in (250, 500, 1000):
        for family in f06.F06_FAMILY_NAMES:
            assert result[f"f06_horizon{horizon}_{family}_coverage"] == 0.0
            for metric in ("rmse", "gain_vs_carry", "gain_vs_pf", "rank"):
                assert np.isnan(result[f"f06_horizon{horizon}_{family}_{metric}"])
    finite_or_nan = result.to_numpy(dtype=np.float64)
    assert not np.isinf(finite_or_nan).any()

