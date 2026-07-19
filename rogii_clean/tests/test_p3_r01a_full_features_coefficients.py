from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pf02_target_ess_lgbm_cv import (  # noqa: E402
    load_development_registry,
)
from scripts.run_p3_r01a_full_features_coefficients import (  # noqa: E402
    _load_typewell_features,
)


def test_typewell_features_use_source_full_registry_then_filter_development() -> None:
    development, shadow_ids = load_development_registry(
        CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv",
        CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    )

    features = _load_typewell_features(development)

    scores = pd.read_csv(
        CLEAN_ROOT / "artifacts/RF03_D0_prefix_alignment_v1/offset_scores.csv",
        dtype={"well_id": str},
    )
    old_fold = scores[["well_id", "fold"]].drop_duplicates()
    fold_comparison = old_fold.merge(
        development[["well_id", "fold"]],
        on="well_id",
        suffixes=("_old", "_current"),
    )
    assert int((fold_comparison["fold_old"] != fold_comparison["fold_current"]).sum()) == 538
    assert len(features) == 657
    assert features["well_id"].nunique() == 657
    assert not set(features["well_id"]).intersection(shadow_ids)
    assert features.shape[1] == 5
    first_well = str(development["well_id"].iloc[0])
    expected_ncc = scores.loc[
        scores["well_id"].eq(first_well)
        & scores["scope"].eq("tail_1000ft")
        & np.isclose(scores["offset_ft"], 0.0),
        "raw_ncc",
    ].iloc[0]
    actual_ncc = features.loc[
        features["well_id"].eq(first_well), "f03a_tail_zero_raw_ncc"
    ].iloc[0]
    assert np.isclose(actual_ncc, expected_ncc, equal_nan=True)
