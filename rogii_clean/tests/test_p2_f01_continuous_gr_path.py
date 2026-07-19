from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# clean 项目根目录，用于直接导入尚未安装成包的新模块。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_f01_continuous_gr_path import (  # noqa: E402
    build_legal_continuous_gr_path,
    build_emission_costs,
    interpolate_block_values,
    solve_second_order_path,
)


def test_all_missing_scores_fall_back_to_zero_offset() -> None:
    """GR 完全缺失时，观测代价只能让所有控制块回到 PF 的零 offset。"""

    offsets = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    scores = np.full((5, 4, 5), np.nan)
    pair_counts = np.zeros_like(scores)
    block_rows = np.full(4, 50.0)

    emission_cost, support = build_emission_costs(
        scale_scores=scores,
        scale_pair_counts=pair_counts,
        block_rows=block_rows,
        offsets_ft=offsets,
        minimum_valid_scales=3,
    )
    solution = solve_second_order_path(
        emission_cost=emission_cost,
        offsets_ft=offsets,
        allowed_changes_ft=np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        maximum_change_acceleration_ft=2.0,
        initial_offset_ft=0.0,
        initial_change_ft=0.0,
    )

    np.testing.assert_allclose(support, 0.0)
    np.testing.assert_allclose(solution.offset_path_ft, 0.0)


def test_second_order_solver_recovers_a_known_smooth_path() -> None:
    """强观测峰位于一条满足速度和曲率约束的路径时，求解器应精确恢复。"""

    offsets = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    known_path = np.array([0.0, 2.0, 4.0, 4.0, 2.0, 0.0])
    emission_cost = np.full((len(known_path), len(offsets)), 10.0)
    for block_position, expected_offset in enumerate(known_path):
        offset_position = int(np.flatnonzero(offsets == expected_offset)[0])
        emission_cost[block_position, offset_position] = 0.0

    solution = solve_second_order_path(
        emission_cost=emission_cost,
        offsets_ft=offsets,
        allowed_changes_ft=np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        maximum_change_acceleration_ft=2.0,
        initial_offset_ft=0.0,
        initial_change_ft=0.0,
    )

    np.testing.assert_allclose(solution.offset_path_ft, known_path)
    np.testing.assert_allclose(solution.change_path_ft, [0.0, 2.0, 2.0, 0.0, -2.0, -2.0])
    assert np.all(solution.forward_backward_margin >= 0.0)


def test_emission_cost_prefers_high_ncc_when_support_is_complete() -> None:
    """五个尺度都有完整支撑时，最高 NCC 的 offset 应有最低观测代价。"""

    offsets = np.array([-2.0, 0.0, 2.0])
    scores = np.zeros((5, 1, 3), dtype=np.float64)
    scores[:, 0, :] = np.array([0.2, 0.9, 0.4])
    pair_counts = np.full_like(scores, 50.0)

    emission_cost, support = build_emission_costs(
        scale_scores=scores,
        scale_pair_counts=pair_counts,
        block_rows=np.array([50.0]),
        offsets_ft=offsets,
        minimum_valid_scales=3,
    )

    assert int(np.argmin(emission_cost[0])) == 1
    np.testing.assert_allclose(support, 1.0)


def test_block_values_are_linearly_interpolated_to_rows() -> None:
    """50 ft 控制块的 offset 应按 MD 连续插回逐行，而不是形成阶梯跳变。"""

    row_md = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    block_md_mid = np.array([0.0, 50.0, 100.0])
    block_values = np.array([0.0, 10.0, 0.0])

    interpolated = interpolate_block_values(
        row_md=row_md,
        block_md_mid=block_md_mid,
        block_values=block_values,
    )

    np.testing.assert_allclose(interpolated, [0.0, 5.0, 10.0, 5.0, 0.0])


def test_full_synthetic_typewell_path_is_recovered_within_one_grid_step() -> None:
    """用非重复合成 Typewell 串联检查 scorer、动态规划和逐行插值的方向。"""

    md = np.arange(1000, dtype=np.float64)
    block_md_mid = np.arange(24.5, 1000.0, 50.0)
    known_block_offset = np.array(
        [0, 2, 4, 6, 8, 10, 12, 12, 10, 8, 6, 4, 2, 0, 0, 2, 4, 4, 2, 0],
        dtype=np.float64,
    )
    known_row_offset = np.interp(md, block_md_mid, known_block_offset)
    center_tvt = 100.0 + 0.2 * md
    true_tvt = center_tvt + known_row_offset
    typewell_tvt = np.arange(0.0, 400.01, 0.2)
    typewell_gr = (
        80.0
        + 20.0 * np.sin(0.002 * np.square(typewell_tvt))
        + 10.0 * np.sin(typewell_tvt / 3.7)
        + 0.05 * typewell_tvt
    )
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)

    result = build_legal_continuous_gr_path(
        row_index=np.arange(len(md), dtype=np.int64),
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=np.arange(-40.0, 42.0, 2.0),
        block_width_ft=50.0,
        smoothing_widths_ft=[5.0, 11.0, 21.0, 51.0, 101.0],
        minimum_valid_pairs=30,
        minimum_valid_scales=3,
        allowed_changes_ft=np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        maximum_change_acceleration_ft=2.0,
        initial_offset_ft=0.0,
        initial_change_ft=0.0,
    )

    recovered_offset = result.row_path["f01_offset_path"].to_numpy(dtype=np.float64)
    offset_rmse = float(np.sqrt(np.mean(np.square(recovered_offset - known_row_offset))))
    assert offset_rmse <= 2.0
    assert result.scale_scores.shape == (5, 20, 41)
    assert result.scale_pair_counts.shape == (5, 20, 41)
    assert "TVT" not in result.row_path.columns
