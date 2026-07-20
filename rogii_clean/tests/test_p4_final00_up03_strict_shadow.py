"""P4 FINAL00 严格影子评分的训练分区与无目标候选合同。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_p4_final00_up03_strict_shadow as strict


def test_strict_contract_is_frozen_to_same_model_and_postprocess() -> None:
    assert strict.EXPERIMENT_ID == "P4_FINAL00_UP03_strict_shadow_v1"
    assert strict.CANDIDATE_ID == "P4_FINAL00_UP03_v1"
    assert len(strict.P3B00_FEATURES) == 41
    assert strict.FROZEN_MODEL_PARAMS["n_estimators"] == 1734
    assert strict.FROZEN_MODEL_PARAMS["random_state"] == 29
    assert strict.UP01_DEGREE == 2
    assert strict.UP01_BLEND == 0.5
    assert strict.PFS_CORRECTION == 0.25


def test_strict_split_excludes_all_shadow_and_current_fold_from_training() -> None:
    development = pd.DataFrame(
        {
            "well_id": [f"dev-{fold}" for fold in range(5)],
            "fold": list(range(5)),
        }
    )
    shadow = pd.DataFrame(
        {
            "well_id": [f"shadow-{fold}" for fold in range(5)],
            "fold": list(range(5)),
        }
    )

    train_ids, validation_ids = strict.strict_well_split(development, shadow, fold=2)

    assert train_ids == {"dev-0", "dev-1", "dev-3", "dev-4"}
    assert validation_ids == {"shadow-2"}
    assert not train_ids.intersection(set(shadow["well_id"]))


def test_restore_strict_absolute_prediction_uses_only_visible_anchor() -> None:
    predicted_delta = np.array([1.5, -2.0, 0.0])
    anchor = np.array([100.0, 100.0, 100.0])

    result = strict.restore_strict_tvt(anchor, predicted_delta)

    np.testing.assert_allclose(result, np.array([101.5, 98.0, 100.0]))


def test_frozen_lightgbm_features_allow_nan_but_reject_infinity() -> None:
    strict.validate_feature_matrix(np.array([[1.0, np.nan], [2.0, 3.0]]))

    with pytest.raises(ValueError, match="Inf"):
        strict.validate_feature_matrix(np.array([[1.0, np.inf]]))


def test_remap_frozen_fold_overwrites_stale_cache_fold() -> None:
    cached = pd.DataFrame(
        {"well_id": ["a", "a", "b"], "fold": [4, 4, 0], "row_index": [1, 2, 1]}
    )
    registry = pd.DataFrame({"well_id": ["a", "b"], "fold": [2, 3]})

    result = strict.remap_frozen_folds(cached, registry)

    assert result["fold"].tolist() == [2, 2, 3]
