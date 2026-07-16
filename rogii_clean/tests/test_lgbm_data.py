from pathlib import Path

import pandas as pd
import pytest

from src.lgbm_data import build_feature_table, load_and_validate_registry


def test_registry_rejects_a_pad_split_across_folds(tmp_path: Path) -> None:
    registry_path = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "pad_id": ["same", "same"],
            "fold": [0, 1],
            "hidden_rows": [1, 1],
        }
    ).to_csv(registry_path, index=False)
    with pytest.raises(ValueError, match="pad"):
        load_and_validate_registry(registry_path, expected_wells=None, expected_rows=None)


def test_build_feature_table_matches_registry_hidden_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = pd.DataFrame(
        {"well_id": ["a"], "pad_id": ["a"], "fold": [0], "hidden_rows": [2]}
    )
    well_path = tmp_path / "a__horizontal_well.csv"
    pd.DataFrame(
        {
            "MD": [1.0, 2.0, 3.0],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [10.0, 11.0, 12.0],
            "GR": [50.0, 51.0, 52.0],
            "TVT": [100.0, 101.0, 102.0],
            "TVT_input": [100.0, None, None],
        }
    ).to_csv(well_path, index=False)
    result = build_feature_table(registry, tmp_path, progress_interval=0)
    assert len(result) == 2
    assert result["well_id"].tolist() == ["a", "a"]
