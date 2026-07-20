"""UP08 严格 OOF PFS 权重的数值契约。"""

from __future__ import annotations

import numpy as np
import pytest

from src.p3_up08_strict_oof_pfs_weight import (
    build_well_features,
    optimal_alpha,
    shrink_alpha,
)


def test_optimal_alpha_matches_closed_form_and_reports_direction_energy() -> None:
    target = np.array([10.5, 11.0, 11.5])
    up01 = np.array([10.0, 10.0, 10.0])
    direction = np.array([1.0, 2.0, 3.0])

    alpha, energy = optimal_alpha(target, up01, direction)

    assert alpha == 0.5
    assert energy == 14.0


def test_optimal_alpha_uses_conservative_center_when_direction_has_zero_energy() -> None:
    alpha, energy = optimal_alpha(
        np.array([12.0, 13.0]),
        np.array([10.0, 10.0]),
        np.zeros(2),
    )

    assert alpha == 0.25
    assert energy == 0.0


def test_optimal_alpha_clips_raw_training_target_to_preregistered_range() -> None:
    alpha, energy = optimal_alpha(
        np.array([20.0, 30.0]),
        np.array([10.0, 10.0]),
        np.array([1.0, 2.0]),
    )

    assert alpha == 1.0
    assert energy == 5.0


def test_build_well_features_reports_path_disagreement_and_runtime_quality() -> None:
    md = np.array([0.0, 100.0, 200.0, 300.0])
    base = np.full(4, 10.0)
    features = build_well_features(
        md,
        base,
        {
            "lag250": np.array([11.0, 13.0, 16.0, 20.0]),
            "lag500": np.array([12.0, 14.0, 15.0, 18.0]),
            "lag1000": np.array([11.0, 12.0, 13.0, 14.0]),
        },
        {
            "gr_observed_fraction": 0.75,
            "hidden_rows": 4,
            "resample_count": 3,
            "lag250_smooth_rows": 9,
            "lag250_fallback_rows": 3,
            "lag500_smooth_rows": 6,
            "lag500_fallback_rows": 6,
            "lag1000_smooth_rows": 3,
            "lag1000_fallback_rows": 9,
        },
    )

    assert features["hidden_rows"] == 4.0
    assert features["hidden_md_span_ft"] == 300.0
    assert features["d_rms_ft"] == np.sqrt(7.5)
    assert features["d_mean_ft"] == 2.5
    assert features["d_abs_mean_ft"] == 2.5
    assert features["d_std_ft"] == np.sqrt(1.25)
    assert features["d_end_ft"] == 4.0
    assert features["d_slope_ft_per_1000ft"] == 10.0
    assert features["lag250_lag500_rms_ft"] == np.sqrt(7.0 / 4.0)
    assert features["lag250_lag1000_rms_ft"] == np.sqrt(46.0 / 4.0)
    assert features["lag500_lag1000_rms_ft"] == 2.5
    assert features["three_lag_spread_rms_ft"] == np.sqrt(50.0 / 4.0)
    assert features["gr_observed_fraction"] == 0.75
    assert features["resample_rate"] == 0.25
    assert features["lag250_fallback_fraction"] == 0.25
    assert features["lag500_fallback_fraction"] == 0.5
    assert features["lag1000_fallback_fraction"] == 0.75


def test_shrink_alpha_moves_toward_center_then_clips_output_range() -> None:
    raw = np.array([-2.0, 0.25, 1.25, 3.0])

    result = shrink_alpha(raw)

    np.testing.assert_allclose(result, np.array([-0.25, 0.25, 0.75, 0.75]))


def test_build_well_features_rejects_non_mapping_or_missing_pfs_lag() -> None:
    md = np.array([0.0, 100.0])
    base = np.array([10.0, 10.0])
    runtime = {
        "gr_observed_fraction": 1.0,
        "hidden_rows": 2,
        "resample_count": 0,
        "lag250_smooth_rows": 2,
        "lag250_fallback_rows": 0,
        "lag500_smooth_rows": 2,
        "lag500_fallback_rows": 0,
        "lag1000_smooth_rows": 2,
        "lag1000_fallback_rows": 0,
    }

    with pytest.raises(ValueError, match="pfs_paths 必须为映射"):
        build_well_features(md, base, np.array([11.0, 11.0]), runtime)
    with pytest.raises(ValueError, match="pfs_paths 缺少必要 lag: lag1000"):
        build_well_features(
            md,
            base,
            {"lag250": base, "lag500": base},
            runtime,
        )


def test_build_well_features_rejects_missing_runtime_key_with_clear_value_error() -> None:
    md = np.array([0.0, 100.0])
    base = np.array([10.0, 10.0])
    with pytest.raises(ValueError, match="runtime 缺少必要键: resample_count"):
        build_well_features(
            md,
            base,
            {"lag250": base, "lag500": base, "lag1000": base},
            {"gr_observed_fraction": 1.0, "hidden_rows": 2},
        )
