"""P2-F01b：同一 MD 轴平滑得分面和连续路径的核心测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# 将 clean 项目根目录放入模块搜索路径，测试可直接导入 src。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_f01b_path_domain_smoothing import (  # noqa: E402
    build_legal_path_domain_gr_path,
    build_path_domain_score_landscape,
)


def _make_nonperiodic_typewell() -> tuple[np.ndarray, np.ndarray]:
    """生成带多个不对称标志层的 Typewell，避免周期正弦造成假峰。"""

    # Typewell 每 0.25 ft 一个点，覆盖合成路径的全部候选 TVT。
    typewell_tvt = np.arange(0.0, 260.25, 0.25, dtype=np.float64)

    # 低频背景只提供平滑趋势，不能单独决定 offset。
    background = 65.0 + 0.018 * typewell_tvt

    # 三个位置、宽度和振幅均不同的标志层让正确对齐具有唯一性。
    marker_one = 30.0 * np.exp(-0.5 * np.square((typewell_tvt - 86.0) / 2.3))
    marker_two = -21.0 * np.exp(-0.5 * np.square((typewell_tvt - 118.0) / 4.7))
    marker_three = 17.0 * np.exp(-0.5 * np.square((typewell_tvt - 151.0) / 1.6))

    # 两个不同频率的小波纹打破宽峰内部的完全对称。
    texture = 5.0 * np.sin(typewell_tvt / 5.3) + 2.0 * np.cos(typewell_tvt / 2.7)
    typewell_gr = background + marker_one + marker_two + marker_three + texture
    return typewell_tvt, typewell_gr


def test_score_surface_is_translation_equivariant() -> None:
    """同一绝对候选 TVT 改写成不同 center/offset 后，原始得分必须相同。"""

    typewell_tvt, typewell_gr = _make_nonperiodic_typewell()

    # MD 每 1 ft 一个点，但 TVT 每 1 ft MD 只变化 0.08 ft，故两个坐标并非 1:1。
    md = np.arange(0.0, 401.0, 1.0, dtype=np.float64)
    center_tvt = 82.0 + 0.08 * md

    # 水平井 GR 来自中心上方 6 ft 的已知轨迹。
    horizontal_gr = np.interp(center_tvt + 6.0, typewell_tvt, typewell_gr)
    offsets = np.arange(-20.0, 22.0, 2.0, dtype=np.float64)

    original = build_path_domain_score_landscape(
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=offsets,
        block_width_ft=50.0,
        smoothing_widths_md_ft=[5.0, 11.0, 21.0, 51.0, 101.0],
        minimum_valid_pairs=30,
    )

    # 中心整体加 12 ft 后，offset 减 12 ft 仍表示完全相同的绝对 TVT。
    shifted = build_path_domain_score_landscape(
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt + 12.0,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=offsets,
        block_width_ft=50.0,
        smoothing_widths_md_ft=[5.0, 11.0, 21.0, 51.0, 101.0],
        minimum_valid_pairs=30,
    )

    # shifted 的 [-20, 8] 与 original 的 [-8, 20] 表示同一批绝对候选 TVT。
    original_positions = np.flatnonzero((offsets >= -8.0) & (offsets <= 20.0))
    shifted_positions = np.flatnonzero((offsets >= -20.0) & (offsets <= 8.0))
    original_scores = original.scores[:, :, original_positions]
    shifted_scores = shifted.scores[:, :, shifted_positions]
    original_counts = original.pair_counts[:, :, original_positions]
    shifted_counts = shifted.pair_counts[:, :, shifted_positions]

    # 缺失位置和公共有效行数也必须完全相同，不能只比较有限得分。
    assert np.array_equal(np.isfinite(original_scores), np.isfinite(shifted_scores))
    assert np.array_equal(original_counts, shifted_counts)
    np.testing.assert_allclose(original_scores, shifted_scores, atol=1e-10, rtol=0.0)


def test_known_smooth_offset_path_is_recovered() -> None:
    """在 MD/TVT 比例不为 1 的合成井上，完整路径误差应小于一个网格步长。"""

    typewell_tvt, typewell_gr = _make_nonperiodic_typewell()
    md = np.arange(0.0, 1001.0, 1.0, dtype=np.float64)
    center_tvt = 75.0 + 0.075 * md

    # 21 个 50 ft 控制点上的 offset 满足冻结的一阶和二阶硬约束。
    block_md_mid = np.arange(25.0, 1026.0, 50.0, dtype=np.float64)
    known_block_offset = np.array(
        [
            0.0, -2.0, -4.0, -6.0, -8.0, -8.0, -6.0,
            -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0,
            8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0,
        ],
        dtype=np.float64,
    )
    known_row_offset = np.interp(md, block_md_mid, known_block_offset)
    true_tvt = center_tvt + known_row_offset
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)

    result = build_legal_path_domain_gr_path(
        row_index=np.arange(len(md), dtype=np.int64),
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=np.arange(-20.0, 22.0, 2.0, dtype=np.float64),
        block_width_ft=50.0,
        smoothing_widths_md_ft=[5.0, 11.0, 21.0, 51.0, 101.0],
        minimum_valid_pairs=30,
        minimum_valid_scales=3,
        allowed_changes_ft=np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        maximum_change_acceleration_ft=2.0,
        initial_offset_ft=0.0,
        initial_change_ft=0.0,
    )

    recovered_offset = result.row_path["f01b_offset_path"].to_numpy(dtype=np.float64)
    offset_rmse = float(np.sqrt(np.mean(np.square(recovered_offset - known_row_offset))))
    assert offset_rmse <= 2.0


def test_all_missing_horizontal_gr_returns_zero_offset() -> None:
    """没有任何水平井 GR 证据时，支持度回退必须严格保留 PF 中心。"""

    typewell_tvt, typewell_gr = _make_nonperiodic_typewell()
    md = np.arange(0.0, 301.0, 1.0, dtype=np.float64)
    center_tvt = 90.0 + 0.05 * md

    result = build_legal_path_domain_gr_path(
        row_index=np.arange(len(md), dtype=np.int64),
        md=md,
        horizontal_gr=np.full(len(md), np.nan, dtype=np.float64),
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=np.arange(-20.0, 22.0, 2.0, dtype=np.float64),
        block_width_ft=50.0,
        smoothing_widths_md_ft=[5.0, 11.0, 21.0],
        minimum_valid_pairs=20,
        minimum_valid_scales=2,
        allowed_changes_ft=np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
        maximum_change_acceleration_ft=2.0,
        initial_offset_ft=0.0,
        initial_change_ft=0.0,
    )

    recovered_offset = result.row_path["f01b_offset_path"].to_numpy(dtype=np.float64)
    assert float(np.max(np.abs(recovered_offset))) == 0.0


def test_score_tensor_shapes_match_rows_blocks_offsets_and_scales() -> None:
    """合法缓存所需的每个维度必须与自然行、控制块和冻结网格严格对齐。"""

    typewell_tvt, typewell_gr = _make_nonperiodic_typewell()
    md = np.arange(0.0, 126.0, 1.0, dtype=np.float64)
    center_tvt = 100.0 + 0.04 * md
    horizontal_gr = np.interp(center_tvt, typewell_tvt, typewell_gr)
    offsets = np.arange(-10.0, 12.0, 2.0, dtype=np.float64)
    scales = [5.0, 11.0, 21.0]

    landscape = build_path_domain_score_landscape(
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=offsets,
        block_width_ft=50.0,
        smoothing_widths_md_ft=scales,
        minimum_valid_pairs=20,
    )

    assert landscape.scores.shape == (len(scales), 3, len(offsets))
    assert landscape.pair_counts.shape == landscape.scores.shape
    assert landscape.block_index.shape == (len(md),)
    assert landscape.block_table["block_rows"].sum() == len(md)

