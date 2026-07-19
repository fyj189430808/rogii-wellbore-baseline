from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_mode01_three_seed_path_modes import (  # noqa: E402
    build_three_mode_features,
    compute_half_seed_stability,
    find_mode_split_fraction,
    validate_shared_cache_arrays,
)


def test_three_mode_features_have_exact_formal_schema_and_simple_mean_centers() -> None:
    hidden_md = np.linspace(1000.0, 1100.0, 9, dtype=np.float32)
    progress = np.linspace(0.0, 1.0, 9, dtype=np.float64)

    seed_delta = np.empty((128, 9), dtype=np.float32)
    seed_delta[:50] = np.asarray([10.0 * progress + index * 1e-4 for index in range(50)])
    seed_delta[50:90] = np.asarray([-6.0 * progress + index * 1e-4 for index in range(40)])
    seed_delta[90:] = np.asarray([2.0 * np.sin(np.pi * progress) + index * 1e-4 for index in range(38)])

    final_ll = np.concatenate(
        [
            np.zeros(50, dtype=np.float64),
            np.full(40, -1.0, dtype=np.float64),
            np.full(38, -2.0, dtype=np.float64),
        ]
    )
    seed_ids = np.arange(128, dtype=np.int32)

    features, diagnostics = build_three_mode_features(
        seed_delta=seed_delta,
        hidden_md=hidden_md,
        final_ll=final_ll,
        seed_ids=seed_ids,
    )

    assert list(features) == [
        "pf_mode1_delta",
        "pf_mode2_delta",
        "pf_mode3_delta",
        "pf_mode1_mass",
        "pf_mode2_mass",
        "pf_mode12_margin",
        "pf_mode12_separation",
        "pf_mode_split_fraction",
    ]
    expected_mode1 = seed_delta[:50].mean(axis=0, dtype=np.float64)
    np.testing.assert_allclose(features["pf_mode1_delta"], expected_mode1, atol=1e-6)
    np.testing.assert_allclose(
        features["pf_mode12_separation"],
        features["pf_mode1_delta"] - features["pf_mode2_delta"],
    )
    assert diagnostics["mode1_seed_count"] == 50
    assert np.all(features["pf_mode1_mass"] == features["pf_mode1_mass"][0])


def test_split_fraction_requires_three_consecutive_points_at_fixed_threshold() -> None:
    mode1 = np.zeros(8, dtype=np.float64)
    mode2 = np.asarray([0.0, 1.2, 0.5, 1.1, 1.2, 1.3, 0.0, 0.0])

    split_fraction = find_mode_split_fraction(mode1, mode2)

    assert split_fraction == 3.0 / 7.0


def test_shared_cache_schema_is_exact_and_dtype_strict() -> None:
    arrays = {
        "seed_delta": np.zeros((128, 5), dtype=np.float32),
        "row_index": np.arange(5, dtype=np.int32),
        "hidden_md": np.arange(5, dtype=np.float32),
        "last_tvt": np.asarray([100.0], dtype=np.float64),
        "final_ll": np.zeros(128, dtype=np.float64),
        "seed_ids": np.arange(128, dtype=np.int32),
    }

    validate_shared_cache_arrays(arrays)

    with_extra = dict(arrays)
    with_extra["unexpected"] = np.asarray([1])
    try:
        validate_shared_cache_arrays(with_extra)
    except ValueError as error:
        assert "字段" in str(error)
    else:
        raise AssertionError("额外字段必须被拒绝")

    wrong_dtype = dict(arrays)
    wrong_dtype["seed_delta"] = arrays["seed_delta"].astype(np.float64)
    try:
        validate_shared_cache_arrays(wrong_dtype)
    except ValueError as error:
        assert "dtype" in str(error)
    else:
        raise AssertionError("错误 dtype 必须被拒绝")


def test_half_seed_stability_is_deterministic() -> None:
    hidden_md = np.linspace(0.0, 100.0, 11, dtype=np.float32)
    progress = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    seed_delta = np.empty((128, 11), dtype=np.float32)
    seed_delta[:44] = np.asarray([8.0 * progress + index * 1e-4 for index in range(44)])
    seed_delta[44:87] = np.asarray([-5.0 * progress + index * 1e-4 for index in range(43)])
    seed_delta[87:] = np.asarray([2.0 * np.sin(np.pi * progress) + index * 1e-4 for index in range(41)])
    final_ll = np.linspace(0.0, -2.0, 128, dtype=np.float64)
    seed_ids = np.arange(128, dtype=np.int32)

    first = compute_half_seed_stability(
        seed_delta,
        hidden_md,
        final_ll,
        seed_ids,
        repeats=3,
        random_seed=123,
    )
    second = compute_half_seed_stability(
        seed_delta,
        hidden_md,
        final_ll,
        seed_ids,
        repeats=3,
        random_seed=123,
    )

    assert first == second
    assert first["stability_repeats"] == 3
    assert 0.0 <= first["stability_mean_ari"] <= 1.0
    assert first["stability_center_rmse_mean"] >= 0.0
