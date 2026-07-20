"""UP08 严格 OOF PFS 权重的数值契约。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.p3_up08_strict_oof_pfs_weight import (
    build_well_features,
    optimal_alpha,
    shrink_alpha,
)
import scripts.run_p3_up08_strict_oof_pfs_weight as up08_runner
from scripts.run_p3_up08_strict_oof_pfs_weight import (
    add_paths_and_features,
    validate_nested_runtime_fingerprint,
    outer_train_preprocess,
    require_outer0_nested_prediction_path,
    shuffle_training_targets,
    validate_nested_partition,
)


def test_add_paths_converts_blended_u_back_to_tvt_exactly_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(up08_runner, "PFS_RUNTIME_DIR", tmp_path)
    (tmp_path / "well-a.json").write_text(
        '{"hidden_rows": 3, "resample_count": 0, '
        '"lag250_smooth_rows": 3, "lag250_fallback_rows": 0, '
        '"lag500_smooth_rows": 3, "lag500_fallback_rows": 0, '
        '"lag1000_smooth_rows": 3, "lag1000_fallback_rows": 0}',
        encoding="utf-8",
    )
    keys = {
        "well_id": ["well-a"] * 3,
        "fold": [0] * 3,
        "row_index": [0, 1, 2],
    }
    predictions = pd.DataFrame(
        {**keys, "md": [0.0, 1.0, 2.0], "pred_tvt": [100.0, 101.0, 102.0]}
    )
    pfs = pd.DataFrame(
        {
            **keys,
            "last_visible_tvt": [100.0] * 3,
            "pfs_lag250_delta": [0.0, 1.0, 2.0],
            "pfs_lag500_delta": [0.0, 1.0, 2.0],
            "pfs_lag1000_delta": [0.0, 1.0, 2.0],
        }
    )
    context = pd.DataFrame(
        {
            "well_id": ["well-a"] * 3,
            "row_index": [0, 1, 2],
            "z_current": [-90.0, -91.0, -92.0],
            "gr_missing": [False] * 3,
        }
    )

    result, _ = add_paths_and_features(predictions, pfs, context)

    # U 恒为 10，二次投影不应改变路径；还原 TVT 时只能减一次 Z。
    np.testing.assert_allclose(result["up01_tvt"], predictions["pred_tvt"], atol=1e-12)


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


def test_require_outer0_nested_prediction_path_rejects_non_outer0_cache() -> None:
    with pytest.raises(ValueError, match="outer_0"):
        require_outer0_nested_prediction_path(
            "artifacts/P3_R01a_nested_linear_residual_v1/base_models/outer_1/base/fold_0/predictions.parquet"
        )


def test_validate_nested_partition_rejects_shadow_and_train_validation_overlap() -> None:
    outer = pd.DataFrame({"well_id": ["outer-a", "shadow-a"], "fold": [0, 0]})
    inner = pd.DataFrame({"well_id": ["outer-a", "inner-a"], "fold": [1, 1]})

    with pytest.raises(ValueError, match="shadow"):
        validate_nested_partition(inner, outer, forbidden_well_ids={"shadow-a"})

    with pytest.raises(ValueError, match="交叉"):
        validate_nested_partition(
            inner.loc[inner["well_id"].eq("outer-a")],
            outer.loc[outer["well_id"].eq("outer-a")],
            forbidden_well_ids=set(),
        )


def test_validate_nested_partition_rejects_well_absent_from_frozen_pfs_manifest() -> None:
    inner = pd.DataFrame({"well_id": ["inner-a"], "fold": [1]})
    outer = pd.DataFrame({"well_id": ["outer-a"], "fold": [0]})

    with pytest.raises(ValueError, match="冻结 PFS 开发井清单"):
        validate_nested_partition(
            inner,
            outer,
            forbidden_well_ids=set(),
            allowed_well_ids={"inner-a"},
        )


def test_validate_nested_runtime_fingerprint_rejects_wrong_expected_fingerprint() -> None:
    runtime = {"fingerprint": "actual-nested-fingerprint"}

    with pytest.raises(RuntimeError, match="fingerprint"):
        validate_nested_runtime_fingerprint(runtime, "expected-nested-fingerprint")


def test_outer_train_preprocess_uses_train_median_and_scaler_only() -> None:
    train = pd.DataFrame({"stable": [0.0, 2.0, 4.0], "missing": [1.0, np.nan, 5.0]})
    validation = pd.DataFrame({"stable": [1000.0], "missing": [100.0]})

    prepared_train, prepared_validation, audit = outer_train_preprocess(train, validation)

    assert audit["medians"] == {"stable": 2.0, "missing": 3.0}
    np.testing.assert_allclose(prepared_train.mean(axis=0), np.zeros(2), atol=1e-12)
    scale = np.sqrt(8.0 / 3.0)
    np.testing.assert_allclose(prepared_validation["stable"].to_numpy(), np.array([(1000.0 - 2.0) / scale]))
    np.testing.assert_allclose(prepared_validation["missing"].to_numpy(), np.array([(100.0 - 3.0) / scale]))


def test_shuffle_training_targets_is_fixed_to_seed_42() -> None:
    targets = np.arange(10, dtype=np.float64)

    first = shuffle_training_targets(targets)
    second = shuffle_training_targets(targets)

    np.testing.assert_array_equal(first, np.random.default_rng(42).permutation(targets))
    np.testing.assert_array_equal(first, second)
