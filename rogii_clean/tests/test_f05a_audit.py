import numpy as np
import pandas as pd

from scripts.finalize_f05a_audit import (
    _read_fold_micro_rmse,
    _strongest_finite_correlation,
)


def test_constant_feature_has_no_strongest_existing_correlation():
    correlations = pd.Series({"base_a": np.nan, "base_b": np.nan})

    feature_name, correlation = _strongest_finite_correlation(correlations)

    assert feature_name is None
    assert np.isnan(correlation)


def test_read_fold_micro_rmse_accepts_metrics_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"folds": [{"fold": 0, "micro_rmse": 12.5}]}',
        encoding="utf-8",
    )

    assert _read_fold_micro_rmse(tmp_path, fold_id=0) == 12.5
