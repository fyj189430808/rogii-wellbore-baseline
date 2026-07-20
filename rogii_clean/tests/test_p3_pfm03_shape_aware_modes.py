"""P3-PFM03 形状聚类核心的冻结行为测试。"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src import p3_pfm03_shape_aware_modes as shape_modes
from src.p3_pfm01_ordered_pf_modes import cluster_seed_descriptors


def _shape_fixture(rows: int = 33) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造均值和端点近似相同、但中段形状明显分成三类的 128 条路径。"""

    progress = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    hidden_md = 10_000.0 + 800.0 * progress
    reference_delta = 4.0 * progress
    seed_ids = np.arange(128, dtype=np.int64)
    group_ids = np.repeat(np.arange(3), [43, 42, 43])
    shape_library = np.stack(
        [
            -7.0 * np.sin(2.0 * np.pi * progress),
            np.zeros_like(progress),
            7.0 * np.sin(2.0 * np.pi * progress),
        ]
    )
    paths = reference_delta[None, :] + shape_library[group_ids]
    # 极小的确定性扰动只用于避免 Ward 面对完全重复行；它不改变三种形状。
    paths = paths + ((seed_ids % 7) - 3)[:, None] * 1e-5 * np.sin(np.pi * progress)
    final_ll = np.linspace(-6.0, 3.0, 128, dtype=np.float64)
    return paths, hidden_md, final_ll, seed_ids, reference_delta


def test_public_builder_has_no_hidden_truth_argument_and_outputs_only_three_paths() -> None:
    parameters = inspect.signature(shape_modes.build_shape_mode_paths).parameters
    forbidden = {"target", "target_tvt", "true_tvt", "residual", "oracle"}

    assert forbidden.isdisjoint(parameters)
    paths, hidden_md, final_ll, seed_ids, reference = _shape_fixture()
    features, diagnostics = shape_modes.build_shape_mode_paths(
        paths,
        hidden_md,
        final_ll,
        seed_ids,
        reference,
    )

    assert features.columns.tolist() == list(shape_modes.FORMAL_FEATURE_COLUMNS)
    assert features.shape == (paths.shape[1], 3)
    assert np.isfinite(features.to_numpy()).all()
    assert diagnostics["hidden_tvt_read"] is False


def test_shape_clusters_use_interior_trajectory_not_old_mean_endpoint_descriptor() -> None:
    paths, hidden_md, final_ll, seed_ids, reference = _shape_fixture()

    shape_result = shape_modes.cluster_shape_residuals(paths, hidden_md, reference)
    old_labels = cluster_seed_descriptors(paths)

    # 真正的三种形状在中点符号不同；PFM03 应完整恢复这三组。
    expected_groups = np.repeat(np.arange(3), [43, 42, 43])
    for expected_group in range(3):
        assert np.unique(shape_result.labels[expected_groups == expected_group]).size == 1
    # 旧两描述符在这个构造上不能恢复同一成员关系，证明新实现没有退化回位置聚类。
    old_same_group_pairs = old_labels[:, None] == old_labels[None, :]
    new_same_group_pairs = shape_result.labels[:, None] == shape_result.labels[None, :]
    assert not np.array_equal(old_same_group_pairs, new_same_group_pairs)


def test_ll_weighted_centers_are_deterministic_and_sorted_low_middle_high() -> None:
    paths, hidden_md, final_ll, seed_ids, reference = _shape_fixture()

    first_features, first_diagnostics = shape_modes.build_shape_mode_paths(
        paths,
        hidden_md,
        final_ll,
        seed_ids,
        reference,
    )
    second_features, second_diagnostics = shape_modes.build_shape_mode_paths(
        paths,
        hidden_md,
        final_ll,
        seed_ids,
        reference,
    )

    np.testing.assert_array_equal(first_features.to_numpy(), second_features.to_numpy())
    assert first_diagnostics == second_diagnostics
    relative_means = [
        float(np.mean(first_features[column].to_numpy() - reference))
        for column in shape_modes.FORMAL_FEATURE_COLUMNS
    ]
    assert relative_means == sorted(relative_means)
    assert first_diagnostics["representative"] == "final_ll_scale8_weighted_center"
    assert first_diagnostics["sample_points"] == 64
    assert first_diagnostics["ward_k"] == 3
    assert first_diagnostics["per_dimension_standardization"] is False

