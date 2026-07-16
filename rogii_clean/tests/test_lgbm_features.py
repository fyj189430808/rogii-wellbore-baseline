import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


def make_small_well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [100.0, 101.0, 102.0, 103.0, 104.0],
            "X": [10.0, 11.0, 12.0, 14.0, 17.0],
            "Y": [20.0, 20.0, 20.0, 23.0, 27.0],
            "Z": [1000.0, 1001.0, 1002.0, 1004.0, 1007.0],
            "GR": [50.0, 51.0, np.nan, 70.0, 80.0],
            "TVT_input": [200.0, 201.0, np.nan, np.nan, np.nan],
            "TVT": [200.0, 201.0, 202.0, 204.0, 207.0],
            "ANCC": [9999.0] * 5,
        }
    )


def test_build_simple_lgbm_rows_uses_exact_formulas() -> None:
    result = build_simple_lgbm_rows(make_small_well(), "well_a", 2)
    assert list(result[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert result["row_index"].tolist() == [2, 3, 4]
    np.testing.assert_allclose(result["last_visible_tvt"], [201.0] * 3)
    np.testing.assert_allclose(result["md_since_visible_end"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result["hidden_fraction"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result["dx_from_visible_end"], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(result["dy_from_visible_end"], [0.0, 3.0, 7.0])
    np.testing.assert_allclose(result["dz_from_visible_end"], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(
        result["dxy_from_visible_end"], [1.0, np.sqrt(18.0), np.sqrt(85.0)]
    )
    assert result["gr_missing"].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(result.loc[0, "gr_raw"])
    np.testing.assert_allclose(result["target_delta"], [1.0, 3.0, 6.0])


def test_hidden_tvt_changes_only_targets_not_features() -> None:
    original = make_small_well()
    changed = make_small_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0
    original_rows = build_simple_lgbm_rows(original, "well_a", 2)
    changed_rows = build_simple_lgbm_rows(changed, "well_a", 2)
    assert_frame_equal(
        original_rows[FEATURE_COLUMNS], changed_rows[FEATURE_COLUMNS], check_exact=True
    )
    assert not np.array_equal(original_rows["target_delta"], changed_rows["target_delta"])


def test_extra_surface_column_never_enters_features() -> None:
    result = build_simple_lgbm_rows(make_small_well(), "well_a", 2)
    assert "ANCC" not in FEATURE_COLUMNS
    assert "ANCC" not in result.columns
    assert len(FEATURE_COLUMNS) == 12
