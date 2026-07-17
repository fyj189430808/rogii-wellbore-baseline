from __future__ import annotations

import numpy as np
import pandas as pd

from src.f04_branch_aware_self_template import (
    F04_FEATURE_COLUMNS,
    build_f04_self_template_rows,
    merge_short_direction_runs,
)


def test_short_direction_noise_is_merged_into_the_long_visit() -> None:
    directions = np.array([1] * 25 + [-1] * 3 + [1] * 25, dtype=np.int8)

    merged = merge_short_direction_runs(directions, minimum_run_rows=20)

    assert np.array_equal(merged, np.ones(len(directions), dtype=np.int8))


def test_branch_aware_template_uses_nearest_same_direction_visit() -> None:
    first_up_tvt = 100.0 + 0.5 * np.arange(40)
    down_tvt = first_up_tvt[::-1]
    recent_up_tvt = 80.0 + 0.5 * np.arange(40)
    visible_tvt = np.concatenate([first_up_tvt, down_tvt, recent_up_tvt])

    first_up_gr = 20.0 + 2.0 * (first_up_tvt - 100.0)
    down_gr = 120.0 + (down_tvt - 100.0)
    recent_up_gr = 220.0 + (recent_up_tvt - 80.0)
    visible_gr = np.concatenate([first_up_gr, down_gr, recent_up_gr])

    hidden_count = 20
    hidden_candidate_tvt = 100.0 + 0.5 * np.arange(hidden_count)
    hidden_gr = 20.0 + 2.0 * (hidden_candidate_tvt - 100.0)
    all_gr = np.concatenate([visible_gr, hidden_gr])

    md = np.arange(len(all_gr), dtype=np.float64)
    z = np.full(len(all_gr), 1000.0, dtype=np.float64)
    visible_end = len(visible_tvt) - 1
    z[visible_end + 1 :] = 1000.0 - (
        hidden_candidate_tvt - float(visible_tvt[-1])
    )
    tvt_input = np.concatenate(
        [visible_tvt, np.full(hidden_count, np.nan, dtype=np.float64)]
    )
    horizontal_df = pd.DataFrame(
        {"MD": md, "Z": z, "GR": all_gr, "TVT_input": tvt_input}
    )

    rows = build_f04_self_template_rows(horizontal_df, "synthetic", 0)

    assert rows.columns.tolist() == ["well_id", "fold", "row_index", *F04_FEATURE_COLUMNS]
    assert rows["row_index"].tolist() == list(range(visible_end + 1, len(all_gr)))
    assert np.allclose(rows["f04_self_predicted_gr"], hidden_gr)
    assert np.allclose(rows["f04_self_abs_error"], 0.0)
    assert np.allclose(rows["f04_coverage_flag"], 1.0)
    assert float(rows.iloc[0]["f04_same_direction_support_count"]) == 1.0
    assert float(rows.iloc[0]["f04_opposite_direction_support_count"]) == 1.0
    assert float(rows.iloc[0]["f04_number_of_visits"]) == 2.0
    assert abs(float(rows["f04_best_offset"].median())) < 1e-12
    assert not np.isinf(rows[F04_FEATURE_COLUMNS].to_numpy(dtype=np.float64)).any()


def test_missing_same_direction_falls_back_without_claiming_coverage() -> None:
    visible_tvt = 100.0 + 0.5 * np.arange(40)
    hidden_count = 10
    hidden_candidate_tvt = 119.0 - 0.5 * np.arange(hidden_count)
    tvt_input = np.concatenate(
        [visible_tvt, np.full(hidden_count, np.nan, dtype=np.float64)]
    )
    gr = np.concatenate(
        [visible_tvt, hidden_candidate_tvt]
    )
    md = np.arange(len(tvt_input), dtype=np.float64)
    z = np.full(len(tvt_input), 1000.0, dtype=np.float64)
    visible_end = len(visible_tvt) - 1
    z[visible_end + 1 :] = 1000.0 - (
        hidden_candidate_tvt - float(visible_tvt[-1])
    )
    horizontal_df = pd.DataFrame(
        {"MD": md, "Z": z, "GR": gr, "TVT_input": tvt_input}
    )

    rows = build_f04_self_template_rows(horizontal_df, "fallback", 2)

    assert np.allclose(rows["f04_coverage_flag"], 0.0)
    assert rows["f04_self_predicted_gr"].notna().all()
    assert not np.isinf(rows[F04_FEATURE_COLUMNS].to_numpy(dtype=np.float64)).any()
