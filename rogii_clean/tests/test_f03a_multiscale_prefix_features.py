import numpy as np
import pandas as pd

from src.f03a_multiscale_prefix_features import (
    F03A_MULTISCALE_FEATURE_COLUMNS,
    build_multiscale_prefix_features,
)


def test_builds_nine_prefix_features_without_using_hidden_rows() -> None:
    visible_md = np.arange(0.0, 41.0, 1.0)
    hidden_md = np.arange(41.0, 46.0, 1.0)
    all_md = np.concatenate([visible_md, hidden_md])
    visible_tvt = 1000.0 + visible_md
    reference_visible_gr = np.sin(visible_md / 4.0) + 0.02 * visible_md
    horizontal_visible_gr = 2.0 * reference_visible_gr + 5.0

    horizontal_df = pd.DataFrame(
        {
            "MD": all_md,
            "GR": np.concatenate([horizontal_visible_gr, np.full(5, 9999.0)]),
            "TVT_input": np.concatenate([visible_tvt, np.full(5, np.nan)]),
        }
    )
    typewell_tvt = np.arange(990.0, 1061.0, 1.0)
    typewell_md_equivalent = typewell_tvt - 1000.0
    typewell_df = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": np.sin(typewell_md_equivalent / 4.0)
            + 0.02 * typewell_md_equivalent,
        }
    )

    actual = build_multiscale_prefix_features(horizontal_df, typewell_df)

    assert list(actual) == F03A_MULTISCALE_FEATURE_COLUMNS
    assert actual["f03a_prefix_valid_pair_count"] == 41.0
    assert actual["f03a_prefix_valid_pair_fraction"] == 1.0
    for feature_name in [
        "f03a_prefix_raw_ncc",
        "f03a_prefix_2ft_ncc",
        "f03a_prefix_5ft_ncc",
        "f03a_prefix_10ft_ncc",
        "f03a_prefix_20ft_ncc",
        "f03a_prefix_derivative_ncc",
    ]:
        assert np.isclose(actual[feature_name], 1.0, atol=1e-10)
    assert np.isclose(actual["f03a_prefix_affine_median_ae"], 0.0, atol=1e-10)

    mutated_hidden = horizontal_df.copy()
    mutated_hidden.loc[mutated_hidden["TVT_input"].isna(), "GR"] = -123456.0
    mutated = build_multiscale_prefix_features(mutated_hidden, typewell_df)
    assert mutated == actual
