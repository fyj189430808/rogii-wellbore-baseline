from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_shadow_holdout import (  # noqa: E402
    collect_legal_well_metadata,
    select_balanced_shadow,
)


def _write_well(train_dir: Path, well_id: str, gr_shift: float = 0.0) -> None:
    horizontal = pd.DataFrame(
        {
            "MD": [100.0, 150.0, 200.0, 260.0],
            "X": [0.0, 10.0, 20.0, 30.0],
            "Y": [0.0, 5.0, 10.0, 15.0],
            "GR": [40.0, 45.0, np.nan, 55.0],
            "TVT_input": [1000.0, 1001.0, np.nan, np.nan],
            # This forbidden source column must never become metadata.
            "TVT": [1000.0, 1001.0, 1003.0, 1005.0],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": [0.0, 10.0, 20.0],
            "GR": [30.0 + gr_shift, 40.0 + gr_shift, 50.0 + gr_shift],
            "surface_1": [1.0, 2.0, 3.0],
        }
    )
    horizontal.to_csv(train_dir / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(train_dir / f"{well_id}__typewell.csv", index=False)


def _legal_metadata() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for number in range(4):
            rows.append(
                {
                    "well_id": f"well_{fold}_{number}",
                    "fold": fold,
                    "hidden_row_count": 50 + 10 * number,
                    "md_span": 1000.0 + 100.0 * number,
                    "hidden_gr_observed_rate": 0.25 * number,
                    "x_median": 100.0 * fold,
                    "y_median": 10.0 * number,
                    "azimuth_deg": 45.0 * number,
                    "typewell_fingerprint": f"fp_{number % 2}",
                }
            )
    return pd.DataFrame(rows)


def test_collects_only_legal_well_metadata_and_stable_typewell_fingerprint(
    tmp_path: Path,
) -> None:
    _write_well(tmp_path, "well_a")
    _write_well(tmp_path, "well_b")
    registry = pd.DataFrame({"well_id": ["well_a", "well_b"], "fold": [0, 1]})

    first = collect_legal_well_metadata(tmp_path, registry)
    second = collect_legal_well_metadata(tmp_path, registry.sample(frac=1.0, random_state=7))

    assert first.equals(second)
    assert first.loc[first["well_id"] == "well_a", "typewell_fingerprint"].item() == first.loc[
        first["well_id"] == "well_b", "typewell_fingerprint"
    ].item()
    assert set(first.columns) == {
        "well_id",
        "fold",
        "hidden_row_count",
        "md_span",
        "hidden_gr_observed_rate",
        "x_median",
        "y_median",
        "azimuth_deg",
        "typewell_fingerprint",
    }
    well_a = first.loc[first["well_id"] == "well_a"].iloc[0]
    assert int(well_a["hidden_row_count"]) == 2
    assert float(well_a["md_span"]) == pytest.approx(60.0)
    assert float(well_a["hidden_gr_observed_rate"]) == pytest.approx(0.5)


def test_shadow_selection_is_deterministic_exact_and_covers_every_fold() -> None:
    metadata = _legal_metadata()

    first = select_balanced_shadow(metadata, number_of_shadow_wells=10, salt="p3-shadow-v1")
    second = select_balanced_shadow(
        metadata.sample(frac=1.0, random_state=17),
        number_of_shadow_wells=10,
        salt="p3-shadow-v1",
    )

    assert first.equals(second)
    assert len(first) == 10
    assert set(first["fold"].astype(int)) == {0, 1, 2, 3, 4}
    assert first["well_id"].is_unique
    assert first["is_shadow"].eq(True).all()


@pytest.mark.parametrize(
    "forbidden_column",
    ["TVT", "target_delta", "surface_3", "residual", "prediction_error", "rmse", "oracle_score"],
)
def test_shadow_selection_rejects_forbidden_metadata_columns(forbidden_column: str) -> None:
    metadata = _legal_metadata().assign(**{forbidden_column: 1.0})

    with pytest.raises(ValueError, match="forbidden"):
        select_balanced_shadow(metadata, number_of_shadow_wells=10, salt="p3-shadow-v1")
