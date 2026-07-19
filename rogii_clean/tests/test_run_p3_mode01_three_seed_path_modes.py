from __future__ import annotations

import sys
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_mode01_three_seed_path_modes import (  # noqa: E402
    build_mode_cache_frame,
    read_selected_targets,
    separate_oracle_metrics,
    select_available_registry,
)


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mode01_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _shared_arrays() -> dict[str, np.ndarray]:
    hidden_md = np.linspace(0.0, 100.0, 9, dtype=np.float32)
    progress = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    paths = np.empty((128, 9), dtype=np.float32)
    paths[:50] = np.asarray([10.0 * progress + index * 1e-4 for index in range(50)])
    paths[50:90] = np.asarray([-6.0 * progress + index * 1e-4 for index in range(40)])
    paths[90:] = np.asarray([2.0 * np.sin(np.pi * progress) + index * 1e-4 for index in range(38)])
    return {
        "seed_delta": paths,
        "row_index": np.arange(100, 109, dtype=np.int32),
        "hidden_md": hidden_md,
        "last_tvt": np.asarray([1000.0], dtype=np.float64),
        "final_ll": np.concatenate(
            [np.zeros(50), np.full(40, -1.0), np.full(38, -2.0)]
        ).astype(np.float64),
        "seed_ids": np.arange(128, dtype=np.int32),
    }


def test_smoke_prefers_frozen_well_and_only_uses_existing_shared_cache(
    workspace_tmp_path: Path,
) -> None:
    development = pd.DataFrame(
        {
            "well_id": ["000d7d20", "other_a", "other_b"],
            "fold": [0, 0, 1],
            "hidden_rows": [9, 10, 11],
        }
    )
    (workspace_tmp_path / "000d7d20.npz").touch()
    (workspace_tmp_path / "other_b.npz").touch()

    selected = select_available_registry(
        development,
        shared_cache_dir=workspace_tmp_path,
        mode="smoke",
        smoke_well_ids=["000d7d20", "other_a", "other_b"],
    )

    assert selected["well_id"].tolist() == ["000d7d20", "other_b"]


def test_mode_cache_contains_only_eight_formal_features_and_legal_keys() -> None:
    frame, diagnostics = build_mode_cache_frame(
        well_id="well_a",
        fold=2,
        arrays=_shared_arrays(),
        experiment_fingerprint="abc123",
    )

    assert frame.columns.tolist() == [
        "well_id",
        "fold",
        "row_index",
        "last_visible_tvt",
        "pf_mode1_delta",
        "pf_mode2_delta",
        "pf_mode3_delta",
        "pf_mode1_mass",
        "pf_mode2_mass",
        "pf_mode12_margin",
        "pf_mode12_separation",
        "pf_mode_split_fraction",
        "_cache_fingerprint",
    ]
    assert len(frame) == 9
    assert "target_tvt" not in frame
    assert diagnostics["number_of_modes"] == 3


def test_target_loader_excludes_shadow_during_arrow_scan(
    workspace_tmp_path: Path,
) -> None:
    prediction_path = workspace_tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "well_id": ["dev_a", "dev_a", "shadow_x"],
            "fold": [0, 0, 1],
            "row_index": [10, 11, 20],
            "target_tvt": [100.0, 101.0, 999.0],
            "pred_tvt": [99.5, 100.5, 998.5],
        }
    ).to_parquet(prediction_path, index=False)
    selected = pd.DataFrame(
        {"well_id": ["dev_a"], "fold": [0], "hidden_rows": [2]}
    )

    targets = read_selected_targets(
        prediction_path,
        selected=selected,
        shadow_ids={"shadow_x"},
    )

    assert targets["well_id"].tolist() == ["dev_a", "dev_a"]
    assert targets["target_tvt"].tolist() == [100.0, 101.0]


def test_oracle_summary_is_removed_from_root_metrics() -> None:
    combined = {
        "experiment_id": "mode01",
        "wells": 3,
        "hidden_rows": 100,
        "path_rmse": {"mode1": 5.0},
        "per_well_best_old_oracle_rmse": 4.0,
        "per_well_best_mode_oracle_rmse": 3.0,
    }

    legal, oracle = separate_oracle_metrics(combined)

    assert not any("oracle" in key for key in legal)
    assert oracle["per_well_best_old_oracle_rmse"] == 4.0
    assert oracle["per_well_best_mode_oracle_rmse"] == 3.0
