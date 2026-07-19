"""P2-M01：模板指纹、marker 转移、q 坐标和候选排名的核心测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 测试直接导入 clean 项目的 src 模块。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_m01_marker_normalized_coordinate import (  # noqa: E402
    aggregate_outer_train_markers,
    deterministic_contiguous_gate,
    marker_coordinate_to_tvt,
    normalized_gated_rank,
    typewell_template_fingerprint,
    tvt_to_marker_coordinate,
)


MARKER_NAMES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def test_cropped_typewells_share_the_same_deep_tail_fingerprint() -> None:
    """只删除浅部行不应改变最后 500 行 TVT-GR 的模板身份。"""

    tvt = 10000.0 + 0.5 * np.arange(1200, dtype=np.float64)
    gr = 70.0 + 8.0 * np.sin(np.arange(1200) / 17.0)
    full = pd.DataFrame({"TVT": tvt, "GR": gr})
    shallow_crop = full.iloc[250:].reset_index(drop=True)

    full_hash = typewell_template_fingerprint(full, tail_rows=500)
    cropped_hash = typewell_template_fingerprint(shallow_crop, tail_rows=500)
    assert full_hash == cropped_hash

    # 深部尾段改变一个真实 GR 点后必须得到不同模板，不能只看行数或末端 TVT。
    changed = shallow_crop.copy()
    changed.loc[len(changed) - 10, "GR"] += 0.01
    assert typewell_template_fingerprint(changed, tail_rows=500) != full_hash


def test_outer_train_marker_transfer_excludes_query_and_detects_ambiguity() -> None:
    """验证井自身不能成为 donor，outer-train marker 不一致时必须回退。"""

    donor_rows: list[dict[str, object]] = []
    marker_values = [11000.0, 11150.0, 11180.0, 11250.0, 11290.0, 11420.0]
    for well_id, fold_id, adjustment in (
        ("query", 0, 100.0),
        ("donor_a", 1, 0.0),
        ("donor_b", 2, 0.4),
    ):
        for marker_name, marker_tvt in zip(MARKER_NAMES, marker_values):
            donor_rows.append(
                {
                    "well_id": well_id,
                    "fold": fold_id,
                    "template_fingerprint": "same_template",
                    "marker_name": marker_name,
                    "marker_tvt": marker_tvt + adjustment,
                    "marker_boundary_observed": True,
                }
            )
    donor_table = pd.DataFrame(donor_rows)

    transferred = aggregate_outer_train_markers(
        query_well_id="query",
        query_fold=0,
        query_template_fingerprint="same_template",
        donor_markers=donor_table,
        marker_names=MARKER_NAMES,
        maximum_marker_range_ft=1.0,
    )

    # query 的 +100 ft 自身 marker 被严格排除，中位数只来自 donor_a 和 donor_b。
    np.testing.assert_allclose(
        transferred["marker_tvt"].to_numpy(dtype=np.float64),
        np.asarray(marker_values) + 0.2,
    )
    assert transferred["donor_well_count"].eq(2).all()
    assert transferred["marker_usable"].all()

    # 将一个 donor 的 BUDA 改动 3 ft，极差超过 1 ft 后该 marker 必须记为歧义。
    ambiguous_table = donor_table.copy()
    ambiguous_mask = ambiguous_table["well_id"].eq("donor_b") & ambiguous_table[
        "marker_name"
    ].eq("BUDA")
    ambiguous_table.loc[ambiguous_mask, "marker_tvt"] += 3.0
    ambiguous = aggregate_outer_train_markers(
        query_well_id="query",
        query_fold=0,
        query_template_fingerprint="same_template",
        donor_markers=ambiguous_table,
        marker_names=MARKER_NAMES,
        maximum_marker_range_ft=1.0,
    )
    buda = ambiguous.loc[ambiguous["marker_name"].eq("BUDA")].iloc[0]
    assert bool(buda["marker_ambiguous"]) is True
    assert bool(buda["marker_usable"]) is False


def test_marker_coordinate_roundtrip_and_zone_boundaries() -> None:
    """六 marker 内的 TVT 必须可无损映射到 q，并按左闭右开区间编号。"""

    markers = np.array([100.0, 160.0, 190.0, 260.0, 300.0, 430.0])
    tvt = np.array([100.0, 130.0, 159.999, 160.0, 225.0, 429.999])

    q, zone, valid = tvt_to_marker_coordinate(tvt, markers)
    assert valid.all()
    assert zone.tolist() == [0, 0, 0, 1, 2, 4]
    recovered_tvt, recovered_valid = marker_coordinate_to_tvt(q, markers)
    assert recovered_valid.all()
    np.testing.assert_allclose(recovered_tvt, tvt, atol=1e-10, rtol=0.0)

    outside_q, outside_zone, outside_valid = tvt_to_marker_coordinate(
        np.array([99.9, 430.0]),
        markers,
    )
    assert not outside_valid.any()
    assert np.isnan(outside_q).all()
    assert outside_zone.tolist() == [-1, -1]


def test_gated_rank_penalizes_excluding_the_true_candidate() -> None:
    """候选门控若排除真实 offset，normalized rank 必须记最差而非静默丢行。"""

    costs = np.array([0.40, 0.10, 0.30, 0.20], dtype=np.float64)

    included = normalized_gated_rank(
        costs=costs,
        true_candidate_position=2,
        eligible_mask=np.array([False, True, True, False]),
    )
    assert included["true_included"] is True
    assert included["candidate_count"] == 2
    assert included["rank"] == 2
    assert included["normalized_rank"] == 1.0

    excluded = normalized_gated_rank(
        costs=costs,
        true_candidate_position=2,
        eligible_mask=np.array([True, True, False, False]),
    )
    assert excluded["true_included"] is False
    assert excluded["candidate_count"] == 2
    assert excluded["normalized_rank"] == 1.0


def test_deterministic_random_gate_has_fixed_contiguous_candidate_count() -> None:
    """等候选数负对照必须可复现、连续且与合法 gate 候选数完全相同。"""

    gate_a = deterministic_contiguous_gate(
        number_of_candidates=41,
        gate_size=9,
        key="well_a:block_7",
        seed=29,
    )
    gate_b = deterministic_contiguous_gate(
        number_of_candidates=41,
        gate_size=9,
        key="well_a:block_7",
        seed=29,
    )
    assert np.array_equal(gate_a, gate_b)
    selected_positions = np.flatnonzero(gate_a)
    assert len(selected_positions) == 9
    assert np.all(np.diff(selected_positions) == 1)

