"""P3-PF03：正式 128-seed 分段似然路径的关键数值测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_pf03_segmented_likelihood import (
    LOCAL_PATH_COLUMNS,
    aggregate_local_likelihood_path,
    path_observation_row_log_likelihoods,
    read_pf03_shared_cache,
    write_pf03_shared_cache_atomic,
)


def test_path_observation_ll_uses_frozen_gr_residual_formula() -> None:
    """局部 LL 必须直接由 seed TVT 路径上的 Typewell GR 残差计算。"""

    seed_predictions = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    horizontal_gr = np.array([10.0, 14.0], dtype=np.float64)
    typewell_gr_grid = np.array([10.0, 12.0, 14.0], dtype=np.float64)

    actual = path_observation_row_log_likelihoods(
        seed_predictions=seed_predictions,
        horizontal_gr=horizontal_gr,
        typewell_gr_grid=typewell_gr_grid,
        typewell_min_tvt=0.0,
        typewell_step_ft=1.0,
        gr_sigma=2.0,
        squared_gr_residual_cap=600.0,
        likelihood_floor=1e-300,
    )

    expected = np.array([[0.0, -0.5], [-0.5, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=1e-15)


@pytest.mark.filterwarnings("error")
def test_local_target_ess_weights_can_change_along_md() -> None:
    """前后窗口偏好不同 seed 时，聚合路径也必须随 MD 平滑改变。"""

    seed_count = 32
    row_count = 11
    md = np.arange(row_count, dtype=np.float64)
    seed_values = np.arange(seed_count, dtype=np.float64)
    seed_predictions = np.repeat(seed_values[:, None], row_count, axis=1)
    row_log_likelihoods = np.empty((seed_count, row_count), dtype=np.float64)
    for row_index in range(row_count):
        preferred_seed = 2.0 if row_index <= 4 else 29.0
        row_log_likelihoods[:, row_index] = -np.square(
            seed_values - preferred_seed
        )

    delta, quality = aggregate_local_likelihood_path(
        seed_predictions=seed_predictions,
        row_log_likelihoods=row_log_likelihoods,
        md=md,
        observed_gr_mask=np.ones(row_count, dtype=bool),
        last_visible_tvt=0.0,
        window_ft=4.0,
        target_ess=16.0,
        center_step_ft=2.0,
    )

    assert delta.dtype == np.float32
    assert float(delta[1]) < float(delta[-2])
    assert quality["median_achieved_ess"] == pytest.approx(16.0, abs=1e-5)
    assert quality["no_observation_window_count"] == 0


def test_window_without_original_gr_returns_uniform_seed_mean() -> None:
    """原始 GR 完全缺失时不得使用插值 GR 挑 seed，而要逐行回到均值路径。"""

    seed_predictions = np.array(
        [
            [100.0, 101.0, 102.0],
            [110.0, 111.0, 112.0],
            [120.0, 121.0, 122.0],
            [130.0, 131.0, 132.0],
        ],
        dtype=np.float64,
    )
    row_log_likelihoods = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0],
            [-3.0, -3.0, -3.0],
            [-4.0, -4.0, -4.0],
        ],
        dtype=np.float64,
    )

    delta, quality = aggregate_local_likelihood_path(
        seed_predictions=seed_predictions,
        row_log_likelihoods=row_log_likelihoods,
        md=np.array([0.0, 100.0, 200.0]),
        observed_gr_mask=np.zeros(3, dtype=bool),
        last_visible_tvt=100.0,
        window_ft=250.0,
        target_ess=2.0,
        center_step_ft=100.0,
    )

    expected_absolute_path = seed_predictions.mean(axis=0).astype(np.float32)
    expected_delta = expected_absolute_path - np.float32(100.0)
    np.testing.assert_array_equal(delta, expected_delta)
    assert quality["no_observation_window_count"] == quality["window_count"]
    assert quality["uniform_fallback_row_count"] == 3


def test_formal_path_names_are_frozen() -> None:
    assert LOCAL_PATH_COLUMNS == [
        "pf128_local250_delta",
        "pf128_local500_delta",
        "pf128_local1000_delta",
    ]


def test_shared_cache_round_trip_and_fingerprint_guard(tmp_path: Path) -> None:
    """共享缓存必须原样保留 128 路径所需数组，并拒绝错误配置指纹。"""

    cache_path = tmp_path / "well_a.npz"
    arrays = {
        "seed_delta": np.arange(12, dtype=np.float32).reshape(4, 3),
        "row_index": np.array([8, 9, 10], dtype=np.int32),
        "hidden_md": np.array([100.0, 101.0, 102.0], dtype=np.float32),
        "last_tvt": np.array([11000.0], dtype=np.float64),
        "final_ll": np.arange(4, dtype=np.float64),
        "seed_ids": np.arange(4, dtype=np.int32),
    }

    write_pf03_shared_cache_atomic(cache_path, arrays, fingerprint="abc123")
    loaded = read_pf03_shared_cache(cache_path, expected_fingerprint="abc123")

    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)
    with pytest.raises(ValueError, match="指纹"):
        read_pf03_shared_cache(cache_path, expected_fingerprint="different")
