"""P2-D01 PF 中心分块多尺度 GR 得分面测试。"""

from __future__ import annotations

import inspect
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

RUNNER_PATH = CLEAN_ROOT / "scripts" / "diagnose_p2_d01_pf_centered_gr.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_d01_pf_centered_multiscale_gr_v1.json"

from src.p2_d01_pf_centered_gr import (  # noqa: E402
    attach_oracle_diagnostics,
    build_block_score_landscape,
)


def _synthetic_shift_case() -> tuple[np.ndarray, ...]:
    """构造一条具有多尺度形状、真实 TVT 比中心高 8 ft 的人工井。"""

    typewell_tvt = np.arange(0.0, 1000.0, 1.0, dtype=np.float64)
    typewell_gr = (
        75.0
        + 13.0 * np.sin(typewell_tvt / 17.0)
        + 7.0 * np.sin(typewell_tvt / 43.0)
        + 18.0 * np.exp(-((typewell_tvt - 410.0) / 24.0) ** 2)
        - 12.0 * np.exp(-((typewell_tvt - 615.0) / 31.0) ** 2)
    )
    md = np.arange(0.0, 600.0, 1.0, dtype=np.float64)
    center_tvt = 180.0 + 0.9 * md
    true_tvt = center_tvt + 8.0
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)
    return md, horizontal_gr, center_tvt, true_tvt, typewell_tvt, typewell_gr


def test_pf_centered_multiscale_landscape_recovers_known_shift() -> None:
    """真实 +8 ft 偏移应被多尺度平均分稳定找回。"""

    md, horizontal_gr, center_tvt, true_tvt, typewell_tvt, typewell_gr = (
        _synthetic_shift_case()
    )
    landscape = build_block_score_landscape(
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=np.arange(-40.0, 42.0, 2.0),
        block_width_ft=100.0,
        smoothing_widths_ft=[5.0, 11.0, 21.0, 51.0, 101.0],
        minimum_valid_pairs=30,
        second_peak_minimum_distance_ft=6.0,
    )
    oracle = attach_oracle_diagnostics(
        landscape,
        true_tvt=true_tvt,
        center_tvt=center_tvt,
    )
    ensemble = oracle.loc[oracle["scale_label"] == "ensemble"]

    assert len(ensemble) == 6
    assert np.nanmedian(np.abs(ensemble["best_offset_ft"] - 8.0)) <= 2.0
    assert ensemble["true_offset_top5"].mean() >= 0.8
    assert ensemble["true_offset_in_grid"].all()


def test_circular_shift_control_has_weaker_ensemble_match() -> None:
    """循环平移保留 GR 自相关，但应破坏真实层位对应并降低最佳 NCC。"""

    md, horizontal_gr, center_tvt, _, typewell_tvt, typewell_gr = (
        _synthetic_shift_case()
    )
    common_kwargs = {
        "md": md,
        "center_tvt": center_tvt,
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "offsets_ft": np.arange(-40.0, 42.0, 2.0),
        "block_width_ft": 100.0,
        "smoothing_widths_ft": [5.0, 11.0, 21.0, 51.0, 101.0],
        "minimum_valid_pairs": 30,
        "second_peak_minimum_distance_ft": 6.0,
    }
    real = build_block_score_landscape(
        horizontal_gr=horizontal_gr,
        **common_kwargs,
    )
    shifted = build_block_score_landscape(
        horizontal_gr=np.roll(horizontal_gr, len(horizontal_gr) // 2),
        **common_kwargs,
    )
    real_scores = real.summary.loc[
        real.summary["scale_label"] == "ensemble", "best_ncc"
    ]
    shifted_scores = shifted.summary.loc[
        shifted.summary["scale_label"] == "ensemble", "best_ncc"
    ]

    assert float(real_scores.median()) > float(shifted_scores.median()) + 0.15


def test_oracle_truth_is_attached_after_legal_scores_are_frozen() -> None:
    """改变隐藏真值只能改变 oracle 列，不能反向改变 legal score 表。"""

    md, horizontal_gr, center_tvt, true_tvt, typewell_tvt, typewell_gr = (
        _synthetic_shift_case()
    )
    landscape = build_block_score_landscape(
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=np.arange(-40.0, 42.0, 2.0),
        block_width_ft=100.0,
        smoothing_widths_ft=[5.0, 11.0],
        minimum_valid_pairs=30,
        second_peak_minimum_distance_ft=6.0,
    )
    legal_before = landscape.summary.copy(deep=True)
    first = attach_oracle_diagnostics(
        landscape,
        true_tvt=true_tvt,
        center_tvt=center_tvt,
    )
    second = attach_oracle_diagnostics(
        landscape,
        true_tvt=true_tvt - 20.0,
        center_tvt=center_tvt,
    )

    pd.testing.assert_frame_equal(landscape.summary, legal_before)
    assert not first["true_offset_median_ft"].equals(
        second["true_offset_median_ft"]
    )
    assert "true_tvt" not in inspect.signature(
        build_block_score_landscape
    ).parameters


def test_insufficient_pairs_produce_missing_scores_instead_of_fake_values() -> None:
    """公共有效点少于固定门槛时不得制造 NCC。"""

    md, horizontal_gr, center_tvt, _, typewell_tvt, typewell_gr = (
        _synthetic_shift_case()
    )
    horizontal_gr[:] = np.nan
    horizontal_gr[:10] = 70.0
    landscape = build_block_score_landscape(
        md=md,
        horizontal_gr=horizontal_gr,
        center_tvt=center_tvt,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        offsets_ft=np.arange(-40.0, 42.0, 2.0),
        block_width_ft=100.0,
        smoothing_widths_ft=[5.0],
        minimum_valid_pairs=30,
        second_peak_minimum_distance_ft=6.0,
    )

    assert landscape.summary["best_ncc"].isna().all()


def test_mismatched_legal_array_lengths_are_rejected() -> None:
    """MD、GR 和中心路径必须逐行对齐。"""

    md, horizontal_gr, center_tvt, _, typewell_tvt, typewell_gr = (
        _synthetic_shift_case()
    )

    with pytest.raises(ValueError, match="长度"):
        build_block_score_landscape(
            md=md[:-1],
            horizontal_gr=horizontal_gr,
            center_tvt=center_tvt,
            typewell_tvt=typewell_tvt,
            typewell_gr=typewell_gr,
            offsets_ft=np.arange(-40.0, 42.0, 2.0),
            block_width_ft=100.0,
            smoothing_widths_ft=[5.0],
            minimum_valid_pairs=30,
            second_peak_minimum_distance_ft=6.0,
        )


def _load_runner():
    """按文件路径加载 P2-D01 runner。"""

    assert RUNNER_PATH.is_file(), "P2-D01 runner 尚未实现"
    specification = importlib.util.spec_from_file_location(
        "diagnose_p2_d01_pf_centered_gr",
        RUNNER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_restore_pf_center_uses_exact_hidden_row_keys() -> None:
    """PF 中心必须由末个可见 TVT 加冻结 delta，并按原始 row_index 对齐。"""

    runner = _load_runner()
    tvt_input = np.asarray([100.0, 101.0, np.nan, np.nan], dtype=np.float64)
    candidate_rows = pd.DataFrame(
        {
            "row_index": [2, 3],
            "pf_ancc_delta": [4.0, 7.0],
        }
    )

    hidden_positions, center_tvt = runner.restore_pf_center(
        tvt_input,
        candidate_rows,
    )

    assert hidden_positions.tolist() == [2, 3]
    assert center_tvt.tolist() == [105.0, 108.0]


def test_restore_pf_center_rejects_reordered_or_missing_keys() -> None:
    """候选缓存键只要少一行或乱序就必须停止。"""

    runner = _load_runner()
    tvt_input = np.asarray([100.0, 101.0, np.nan, np.nan], dtype=np.float64)
    candidate_rows = pd.DataFrame(
        {
            "row_index": [3, 2],
            "pf_ancc_delta": [7.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="row_index"):
        runner.restore_pf_center(tvt_input, candidate_rows)


def test_p2_d01_config_freezes_search_and_control_design() -> None:
    """D01 配置必须与预注册设计一致，不能在看结果后改网格或尺度。"""

    assert CONFIG_PATH.is_file(), "P2-D01 配置尚未实现"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["experiment_id"] == "P2_D01_pf_centered_multiscale_gr_v1"
    assert config["center_path"] == "last_visible_tvt_plus_pf_ancc_delta"
    assert config["offset_grid_ft"] == list(np.arange(-40.0, 42.0, 2.0))
    assert config["block_widths_ft"] == [50.0, 100.0]
    assert config["smoothing_widths_ft"] == [5.0, 11.0, 21.0, 51.0, 101.0]
    assert config["minimum_valid_pairs"] == 30
    assert config["negative_control"] == "hidden_gr_circular_shift_half_segment"
    assert config["hidden_tvt_access"] == "oracle_attachment_only"
