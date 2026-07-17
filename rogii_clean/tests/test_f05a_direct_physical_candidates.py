from __future__ import annotations

import pandas as pd

from src.f05a_direct_physical_candidates import (
    DIRECT_CANDIDATE_COLUMNS,
    build_direct_candidate_features,
)


def _source_rows() -> pd.DataFrame:
    rows = []
    for row_number, row_index in enumerate([1442, 1443]):
        row: dict[str, object] = {
            "well": "000d7d20",
            "id": f"000d7d20_{row_index}",
            "target": 99999.0,
            "sig_std": -999.0,
            "sig_mean_d": -999.0,
        }
        for feature_number, feature_name in enumerate(DIRECT_CANDIDATE_COLUMNS):
            row[feature_name] = row_number * 100.0 + feature_number
        rows.append(row)
    return pd.DataFrame(rows)


def test_direct_candidates_keep_only_keys_and_fixed_24_features() -> None:
    result = build_direct_candidate_features(_source_rows())

    assert len(DIRECT_CANDIDATE_COLUMNS) == 24
    assert list(result.columns) == ["well_id", "row_index", *DIRECT_CANDIDATE_COLUMNS]
    assert result["well_id"].tolist() == ["000d7d20", "000d7d20"]
    assert result["row_index"].tolist() == [1442, 1443]
    assert "target" not in result.columns
    assert "sig_std" not in result.columns
    assert "sig_mean_d" not in result.columns


def test_target_and_forbidden_signal_changes_cannot_change_features() -> None:
    original = _source_rows()
    changed = original.copy()
    changed["target"] = [-1.0, 2.0]
    changed["sig_std"] = [1000.0, 2000.0]
    changed["sig_mean_d"] = [3000.0, 4000.0]

    pd.testing.assert_frame_equal(
        build_direct_candidate_features(original),
        build_direct_candidate_features(changed),
    )

