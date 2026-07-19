"""P3-PF03a 五条冻结路径局部混合代理的最小行为测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_p3_pf03a_five_path_local500_proxy as runner
from src.p3_pf03a_five_path_local500_proxy import (
    CANDIDATE_PATH_COLUMNS,
    build_local500_proxy_path,
    interpolate_segment_weights,
    score_one_segment,
)


def make_frozen_paths(rows: int) -> pd.DataFrame:
    data: dict[str, object] = {
        "well_id": ["well_a"] * rows,
        "row_index": np.arange(2, rows + 2),
        "last_visible_tvt": np.full(rows, 100.0),
    }
    for candidate_index, column in enumerate(CANDIDATE_PATH_COLUMNS):
        data[column] = np.full(rows, float(candidate_index))
    return pd.DataFrame(data)


def make_horizontal(rows: int, hidden_gr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(rows + 2, dtype=np.float64),
            "Z": np.zeros(rows + 2, dtype=np.float64),
            "GR": np.concatenate(([10.0, 10.0], hidden_gr)),
            "TVT_input": np.concatenate(([100.0, 100.0], np.full(rows, np.nan))),
        }
    )


def make_typewell() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TVT": np.arange(95.0, 111.0),
            "GR": np.arange(95.0, 111.0) * 10.0,
        }
    )


def test_scored_segment_reaches_fixed_ess_four() -> None:
    candidate_tvt = np.column_stack(
        [np.full(100, 100.0 + index) for index in range(5)]
    )
    observed_gr = np.full(100, 1000.0)

    weights, report = score_one_segment(
        observed_gr=observed_gr,
        candidate_tvt=candidate_tvt,
        typewell_tvt=make_typewell()["TVT"].to_numpy(),
        typewell_gr=make_typewell()["GR"].to_numpy(),
        gr_sigma=10.0,
        target_ess=4.0,
        minimum_observed_rows=50,
        squared_residual_cap=600.0,
    )

    assert report["fallback_used"] is False
    assert abs(report["achieved_ess"] - 4.0) < 1e-6
    assert weights[0] > weights[-1]
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_segment_with_fewer_than_50_gr_rows_falls_back_to_mean_path() -> None:
    frozen = make_frozen_paths(49)
    horizontal = make_horizontal(49, np.full(49, 1000.0))

    output, report = build_local500_proxy_path(
        frozen_paths=frozen,
        horizontal_well=horizontal,
        typewell=make_typewell(),
        gr_sigma=10.0,
        segment_length_ft=500.0,
        minimum_observed_rows=50,
        target_ess=4.0,
        squared_residual_cap=600.0,
    )

    np.testing.assert_array_equal(output["row_index"], frozen["row_index"])
    np.testing.assert_allclose(
        output["pf128_local500_delta"], frozen["pf128_mean_delta"]
    )
    assert report["fallback_segments"] == 1
    assert report["scored_segments"] == 0


def test_segment_center_weights_are_linearly_interpolated_and_normalized() -> None:
    row_md = np.array([0.0, 250.0, 500.0, 750.0, 1000.0])
    centers = np.array([250.0, 750.0])
    segment_weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    row_weights = interpolate_segment_weights(row_md, centers, segment_weights)

    np.testing.assert_allclose(row_weights[1], segment_weights[0])
    np.testing.assert_allclose(row_weights[2], [0.5, 0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(row_weights[3], segment_weights[1])
    np.testing.assert_allclose(row_weights.sum(axis=1), 1.0)


def test_hidden_tvt_column_cannot_change_proxy_path() -> None:
    frozen = make_frozen_paths(60)
    horizontal = make_horizontal(60, np.full(60, 1000.0))
    horizontal["TVT"] = np.arange(len(horizontal), dtype=np.float64)
    before, _ = build_local500_proxy_path(
        frozen, horizontal, make_typewell(), 10.0, 500.0, 50, 4.0, 600.0
    )
    horizontal.loc[horizontal["TVT_input"].isna(), "TVT"] = -1e12
    after, _ = build_local500_proxy_path(
        frozen, horizontal, make_typewell(), 10.0, 500.0, 50, 4.0, 600.0
    )
    np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())


def test_window_choice_uses_fold0_only_even_when_another_window_wins_fold1() -> None:
    comparison = pd.DataFrame(
        [
            {"segment_length_ft": 250.0, "fold": 0, "local500_rmse": 9.0, "improvement_ft": 0.3},
            {"segment_length_ft": 250.0, "fold": 1, "local500_rmse": 20.0, "improvement_ft": -5.0},
            {"segment_length_ft": 500.0, "fold": 0, "local500_rmse": 9.2, "improvement_ft": 0.2},
            {"segment_length_ft": 500.0, "fold": 1, "local500_rmse": 8.0, "improvement_ft": 1.0},
            {"segment_length_ft": 1000.0, "fold": 0, "local500_rmse": 9.4, "improvement_ft": 0.1},
            {"segment_length_ft": 1000.0, "fold": 1, "local500_rmse": 1.0, "improvement_ft": 8.0},
        ]
    )

    selection = runner.choose_window_using_fold0(comparison, minimum_improvement_ft=0.2)

    assert selection["selected_segment_length_ft"] == 250.0
    assert selection["selection_used_fold1"] is False
