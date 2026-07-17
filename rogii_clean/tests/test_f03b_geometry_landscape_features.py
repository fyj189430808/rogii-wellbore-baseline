from __future__ import annotations

import numpy as np
import pandas as pd

from src.f03b_geometry_landscape_features import (
    F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    build_geometry_landscape_features,
)


def test_geometry_landscape_recovers_shift_and_changes_by_hidden_row() -> None:
    md = np.arange(161, dtype=np.float64)
    z = 1000.0 + md
    visible_end = 59
    anchor_tvt = 11500.0
    candidate_tvt = anchor_tvt - (z - z[visible_end])

    typewell_tvt = np.arange(11250.0, 11620.0, dtype=np.float64)
    typewell_gr = (
        70.0
        + 13.0 * np.sin(typewell_tvt / 17.0)
        + 7.0 * np.cos(typewell_tvt / 31.0)
        + 0.015 * (typewell_tvt - 11400.0)
    )
    true_offset = 8.0
    horizontal_gr = np.interp(
        candidate_tvt + true_offset,
        typewell_tvt,
        typewell_gr,
    )
    tvt_input = candidate_tvt.copy()
    tvt_input[visible_end + 1 :] = np.nan

    horizontal_df = pd.DataFrame(
        {"MD": md, "Z": z, "GR": horizontal_gr, "TVT_input": tvt_input}
    )
    typewell_df = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})

    features = build_geometry_landscape_features(horizontal_df, typewell_df)

    assert features["row_index"].tolist() == list(range(visible_end + 1, len(md)))
    assert features.columns.tolist() == [
        "row_index",
        *F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS,
    ]
    assert abs(float(features["f03b_best_offset"].median()) - true_offset) <= 2.0
    assert features["f03b_best_ncc"].nunique(dropna=True) > 1
    assert not np.isinf(
        features[F03B_GEOMETRY_LANDSCAPE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).any()
