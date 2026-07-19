from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p3_d00_residual_structure import (  # noqa: E402
    load_development_predictions,
    run_audit,
)


def _write_shadow(path: Path, shadow_wells: list[str]) -> None:
    rows = [
        {"well_id": well_id, "fold": 0, "is_shadow": True}
        for well_id in shadow_wells
    ]
    pd.DataFrame(rows, columns=["well_id", "fold", "is_shadow"]).to_csv(
        path,
        index=False,
    )


def test_loader_excludes_shadow_before_returning_target_rows(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "well_id": ["dev_a", "shadow_b", "dev_c"],
            "fold": [0, 1, 2],
            "row_index": [10, 20, 30],
            "md": [100.0, 200.0, 300.0],
            "target_tvt": [1.0, 999999.0, 4.0],
            "pred_tvt": [1.5, -999999.0, 3.5],
        }
    )
    prediction_path = tmp_path / "predictions.parquet"
    shadow_path = tmp_path / "shadow.csv"
    predictions.to_parquet(prediction_path, index=False)
    _write_shadow(shadow_path, ["shadow_b"])

    development = load_development_predictions(prediction_path, shadow_path)

    assert set(development["well_id"]) == {"dev_a", "dev_c"}
    assert "shadow_b" not in development["well_id"].tolist()
    assert 999999.0 not in development["target_tvt"].tolist()


def test_loader_rejects_duplicate_natural_hidden_keys(tmp_path: Path) -> None:
    duplicate_rows = pd.DataFrame(
        {
            "well_id": ["dev_a", "dev_a"],
            "fold": [0, 0],
            "row_index": [10, 10],
            "md": [100.0, 100.0],
            "target_tvt": [1.0, 1.0],
            "pred_tvt": [1.5, 1.5],
        }
    )
    prediction_path = tmp_path / "predictions.parquet"
    shadow_path = tmp_path / "shadow.csv"
    duplicate_rows.to_parquet(prediction_path, index=False)
    _write_shadow(shadow_path, [])

    with pytest.raises(ValueError, match="重复"):
        load_development_predictions(prediction_path, shadow_path)


def test_run_audit_uses_pooled_sse_not_mean_well_rmse() -> None:
    predictions = pd.DataFrame(
        {
            "well_id": ["short", "long", "long", "long"],
            "fold": [0, 1, 1, 1],
            "row_index": [0, 0, 1, 2],
            "md": [0.0, 0.0, 1.0, 2.0],
            "target_tvt": [10.0, 12.0, 12.0, 12.0],
            "pred_tvt": [10.0, 10.0, 10.0, 10.0],
        }
    )
    source_metrics = {"overall": {"micro_rmse": 10.305704992073148}}

    per_well, per_fold, summary = run_audit(predictions, source_metrics)

    expected_pooled = float(np.sqrt((0.0 + 4.0 + 4.0 + 4.0) / 4.0))
    observed = summary["development_overall"]["baseline"]["pooled_rmse"]
    assert observed == pytest.approx(expected_pooled)
    assert observed != pytest.approx(per_well["baseline_rmse"].mean())
    assert set(per_fold["fold"].astype(int)) == {0, 1}
    assert summary["lineage_full_773_micro_rmse"] == 10.305704992073148


def test_run_audit_summary_is_json_serializable() -> None:
    predictions = pd.DataFrame(
        {
            "well_id": ["one", "one", "one", "one"],
            "fold": [0, 0, 0, 0],
            "row_index": [0, 1, 2, 3],
            "md": [0.0, 1.0, 2.0, 3.0],
            "target_tvt": [10.0, 11.0, 12.0, 13.0],
            "pred_tvt": [10.0, 10.0, 10.0, 10.0],
        }
    )

    _per_well, _per_fold, summary = run_audit(
        predictions,
        {"overall": {"micro_rmse": 10.305704992073148}},
    )

    json.dumps(summary, allow_nan=False)
