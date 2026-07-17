from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.f07r_neighbor_prior import (
    F07R_FEATURE_COLUMNS,
    NeighborProfile,
    allowed_source_ids,
    build_neighbor_prior_features,
    build_source_profile,
    compute_pair_geometry,
)


def _make_well(
    well_id: str,
    pad_id: str,
    y_offset: float,
    *,
    reverse: bool = False,
    stop_x: float = 2500.0,
    base_u: float = 100.0,
    linear_u: float = 0.02,
    quadratic_u: float = 2.0e-6,
) -> tuple[pd.DataFrame, str, str]:
    """构造一条每 25 ft 采样的平行水平井，前 1000 ft 为可见前缀。"""

    x = np.arange(0.0, stop_x + 0.1, 25.0, dtype=np.float64)
    if reverse:
        x = x[::-1].copy()
    md = np.arange(len(x), dtype=np.float64) * 25.0
    z = np.full(len(x), 1000.0, dtype=np.float64)
    u = base_u + linear_u * x + quadratic_u * x**2
    tvt = u - z
    visible_count = 41
    tvt_input = tvt.copy()
    tvt_input[visible_count:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": np.full(len(x), y_offset),
            "Z": z,
            "TVT": tvt,
            "TVT_input": tvt_input,
            "ANCC": np.linspace(1.0e6, 2.0e6, len(x)),
        }
    )
    return horizontal, well_id, pad_id


def _profile(
    well_id: str,
    pad_id: str,
    y_offset: float,
    **kwargs: object,
) -> NeighborProfile:
    horizontal, _, _ = _make_well(
        well_id,
        pad_id,
        y_offset,
        **kwargs,
    )
    return NeighborProfile(
        well_id=well_id,
        pad_id=pad_id,
        rows=build_source_profile(horizontal),
    )


def test_feature_schema_is_the_frozen_ten_column_group() -> None:
    assert F07R_FEATURE_COLUMNS == (
        "neighbor_relative_U_prior",
        "neighbor_relative_slope",
        "neighbor_curvature",
        "neighbor_MAD",
        "nearest_trajectory_distance",
        "azimuth_difference",
        "parallel_overlap_length",
        "support_well_count",
        "support_effective_weight",
        "neighbor_disagreement",
    )


def test_outer_fold_source_selection_excludes_validation_fold_self_and_same_pad() -> None:
    registry = pd.DataFrame(
        {
            "well_id": ["target", "same_pad", "allowed", "validation_other"],
            "pad_id": ["pad_a", "pad_a", "pad_b", "pad_c"],
            "fold": [0, 1, 1, 0],
        }
    )

    selected = allowed_source_ids(
        registry,
        outer_fold=0,
        target_well_id="target",
    )

    assert selected == ["allowed"]


def test_parallel_reversed_trajectory_has_zero_axial_azimuth_difference() -> None:
    target, _, _ = _make_well("target", "pad_a", 0.0)
    reversed_source = _profile(
        "reversed",
        "pad_b",
        100.0,
        reverse=True,
    )

    geometry = compute_pair_geometry(target, reversed_source.rows)

    assert geometry["azimuth_difference"] < 1.0e-8
    assert geometry["parallel_overlap_length"] >= 2400.0
    assert 99.0 <= geometry["trajectory_min_distance"] <= 101.0


def test_one_parallel_neighbor_recovers_calibrated_relative_u_without_extrapolation() -> None:
    target, _, _ = _make_well("target", "pad_a", 0.0)
    source = _profile(
        "source",
        "pad_b",
        100.0,
        base_u=500.0,
        linear_u=0.03,
        quadratic_u=2.0e-6,
    )

    result = build_neighbor_prior_features(
        target,
        [source],
        target_well_id="target",
        target_pad_id="pad_a",
    )
    hidden_indices = target.index[target["TVT_input"].isna()].to_numpy()
    anchor_index = int(target.index[target["TVT_input"].notna()][-1])
    expected_relative_u = (
        target.loc[hidden_indices, "TVT"].to_numpy()
        + target.loc[hidden_indices, "Z"].to_numpy()
        - float(target.loc[anchor_index, "TVT_input"] + target.loc[anchor_index, "Z"])
    )

    assert list(result.columns) == ["row_index", *F07R_FEATURE_COLUMNS]
    np.testing.assert_array_equal(result["row_index"].to_numpy(), hidden_indices)
    np.testing.assert_allclose(
        result["neighbor_relative_U_prior"].to_numpy(),
        expected_relative_u,
        atol=1.0e-8,
    )
    assert result["support_well_count"].eq(1.0).all()
    assert result["support_effective_weight"].eq(1.0).all()
    assert result["neighbor_MAD"].eq(0.0).all()
    assert result["neighbor_disagreement"].eq(0.0).all()

    short_source = _profile(
        "short",
        "pad_c",
        100.0,
        stop_x=1800.0,
        base_u=500.0,
        linear_u=0.03,
        quadratic_u=2.0e-6,
    )
    short_result = build_neighbor_prior_features(
        target,
        [short_source],
        target_well_id="target",
        target_pad_id="pad_a",
    )
    unsupported = short_result["row_index"] > 72
    assert short_result.loc[unsupported, "neighbor_relative_U_prior"].eq(0.0).all()
    assert short_result.loc[unsupported, "support_well_count"].eq(0.0).all()


def test_target_hidden_tvt_surface_and_contact_cannot_change_features() -> None:
    target, _, _ = _make_well("target", "pad_a", 0.0)
    source = _profile("source", "pad_b", 100.0, base_u=500.0)
    hidden = target["TVT_input"].isna()

    mutated = target.copy()
    mutated.loc[hidden, "TVT"] = np.linspace(-1.0e12, 1.0e12, int(hidden.sum()))
    mutated.loc[hidden, "ANCC"] = np.linspace(9.0e15, -9.0e15, int(hidden.sum()))
    mutated["contact"] = np.arange(len(mutated), dtype=np.float64)
    removed = target.drop(columns=["TVT", "ANCC"])

    build_kwargs = {
        "target_well_id": "target",
        "target_pad_id": "pad_a",
    }
    expected = build_neighbor_prior_features(target, [source], **build_kwargs)
    mutated_result = build_neighbor_prior_features(mutated, [source], **build_kwargs)
    removed_result = build_neighbor_prior_features(removed, [source], **build_kwargs)

    pd.testing.assert_frame_equal(expected, mutated_result, check_exact=True)
    pd.testing.assert_frame_equal(expected, removed_result, check_exact=True)


def test_two_neighbors_report_support_and_disagreement() -> None:
    target, _, _ = _make_well("target", "pad_a", 0.0)
    first = _profile(
        "first",
        "pad_b",
        100.0,
        base_u=500.0,
        quadratic_u=2.0e-6,
    )
    second = _profile(
        "second",
        "pad_c",
        200.0,
        base_u=700.0,
        quadratic_u=5.0e-6,
    )

    result = build_neighbor_prior_features(
        target,
        [first, second],
        target_well_id="target",
        target_pad_id="pad_a",
    )
    supported = result["support_well_count"] == 2.0

    assert supported.any()
    assert result.loc[supported, "support_effective_weight"].between(1.0, 2.0).all()
    assert (result.loc[supported, "neighbor_MAD"] > 0.0).any()
    assert (result.loc[supported, "neighbor_disagreement"] > 0.0).any()
    assert np.isfinite(result[list(F07R_FEATURE_COLUMNS)].to_numpy()).all()


def test_builder_rejects_self_same_pad_and_duplicate_sources() -> None:
    target, _, _ = _make_well("target", "pad_a", 0.0)
    self_source = _profile("target", "pad_b", 100.0)
    same_pad_source = _profile("other", "pad_a", 100.0)
    allowed_source = _profile("allowed", "pad_b", 100.0)

    with pytest.raises(ValueError, match="目标井自身"):
        build_neighbor_prior_features(
            target,
            [self_source],
            target_well_id="target",
            target_pad_id="pad_a",
        )

    with pytest.raises(ValueError, match="同一 pad"):
        build_neighbor_prior_features(
            target,
            [same_pad_source],
            target_well_id="target",
            target_pad_id="pad_a",
        )

    with pytest.raises(ValueError, match="重复"):
        build_neighbor_prior_features(
            target,
            [allowed_source, allowed_source],
            target_well_id="target",
            target_pad_id="pad_a",
        )


def test_no_neighbor_uses_finite_fallback_and_zero_support() -> None:
    target, _, _ = _make_well("target", "pad_a", 0.0)

    result = build_neighbor_prior_features(
        target,
        [],
        target_well_id="target",
        target_pad_id="pad_a",
    )

    values = result[list(F07R_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    assert result["neighbor_relative_U_prior"].eq(0.0).all()
    assert result["neighbor_relative_slope"].eq(0.0).all()
    assert result["neighbor_curvature"].eq(0.0).all()
    assert result["neighbor_MAD"].eq(0.0).all()
    assert result["nearest_trajectory_distance"].eq(1_000_000.0).all()
    assert result["azimuth_difference"].eq(90.0).all()
    assert result["parallel_overlap_length"].eq(0.0).all()
    assert result["support_well_count"].eq(0.0).all()
    assert result["support_effective_weight"].eq(0.0).all()
    assert result["neighbor_disagreement"].eq(0.0).all()
