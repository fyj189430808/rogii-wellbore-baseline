from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_p3_final_candidate_shadow.py"
CLEAN_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_final_candidate_shadow as runner  # noqa: E402


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="final_shadow_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def test_dedicated_shadow_runner_exists() -> None:
    assert SCRIPT.is_file(), "一次性影子验证必须使用独立新 runner"


def test_contract_is_fully_frozen() -> None:
    assert runner.EXPERIMENT_ID == "P4_FINAL00_UP03_shadow_v1"
    assert runner.CANDIDATE_ID == "P4_FINAL00_UP03_v1"
    assert runner.EXPECTED_SHADOW_WELLS == 116
    assert runner.EXPECTED_SHADOW_FOLD_COUNTS == {0: 24, 1: 23, 2: 23, 3: 23, 4: 23}
    assert runner.UP01_DEGREE == 2
    assert runner.UP01_BLEND == 0.50
    assert runner.PFS_LAG_FT == 1000.0
    assert runner.PFS_CORRECTION == 0.25
    assert runner.GATE == {
        "minimum_pooled_improvement_ft": 0.15,
        "minimum_improved_folds": 3,
        "minimum_well_win_rate": 0.52,
        "maximum_bootstrap_ci_upper_ft": 0.0,
        "maximum_p90_degradation_ft": 0.20,
    }


def _formal_shadow_tables(root: Path) -> tuple[Path, Path, pd.DataFrame]:
    fold_counts = [24, 23, 23, 23, 23]
    rows: list[dict[str, object]] = []
    for fold, count in enumerate(fold_counts):
        for number in range(count):
            rows.append(
                {
                    "well_id": f"s{fold}_{number:02d}",
                    "fold": fold,
                    "hidden_rows": number + 2,
                }
            )
    folds = pd.DataFrame(rows)
    shadow = folds.rename(columns={"hidden_rows": "hidden_row_count"}).copy()
    shadow["is_shadow"] = True
    shadow["selection_rank"] = np.arange(len(shadow))
    fold_path = root / "folds.csv"
    shadow_path = root / "shadow.csv"
    folds.to_csv(fold_path, index=False)
    shadow.to_csv(shadow_path, index=False)
    return fold_path, shadow_path, folds


def test_shadow_registry_selects_exact_frozen_116_wells(workspace_tmp_path: Path) -> None:
    fold_path, shadow_path, expected = _formal_shadow_tables(workspace_tmp_path)

    actual = runner.load_shadow_registry(fold_path, shadow_path)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected[["well_id", "fold", "hidden_rows"]].sort_values("well_id").reset_index(drop=True),
    )


def test_shadow_registry_rejects_fold_or_count_drift(workspace_tmp_path: Path) -> None:
    fold_path, shadow_path, _ = _formal_shadow_tables(workspace_tmp_path)
    shadow = pd.read_csv(shadow_path)
    shadow.loc[0, "fold"] = 4
    shadow.to_csv(shadow_path, index=False)

    with pytest.raises(ValueError, match="fold|116|冻结"):
        runner.load_shadow_registry(fold_path, shadow_path)


def test_legal_p2_loader_physically_excludes_target_column(workspace_tmp_path: Path) -> None:
    prediction_path = workspace_tmp_path / "predictions.parquet"
    source = pd.DataFrame(
        {
            "well_id": ["shadow_a", "shadow_a", "dev_b"],
            "fold": [0, 0, 1],
            "row_index": [2, 3, 4],
            "md": [12.0, 13.0, 14.0],
            "target_tvt": [-999.0, -998.0, -997.0],
            "carry_tvt": [90.0, 90.0, 91.0],
            "pred_tvt": [100.0, 101.0, 102.0],
        }
    )
    source.to_parquet(prediction_path, index=False)
    registry = pd.DataFrame({"well_id": ["shadow_a"], "fold": [0], "hidden_rows": [2]})

    legal = runner.load_legal_p2_rows(prediction_path, registry)

    assert tuple(legal.columns) == ("well_id", "fold", "row_index", "md", "pred_tvt")
    assert "target_tvt" not in legal.columns
    assert legal["well_id"].unique().tolist() == ["shadow_a"]


def test_build_candidate_applies_only_frozen_up01_and_up03_formulas() -> None:
    legal = pd.DataFrame(
        {
            "well_id": ["a"] * 5,
            "fold": [0] * 5,
            "row_index": np.arange(5),
            "md": np.arange(5, dtype=float),
            "pred_tvt": np.array([100.0, 103.0, 101.0, 105.0, 104.0]),
            "z_current": np.array([10.0, 9.0, 8.0, 7.0, 6.0]),
        }
    )
    pfs = pd.DataFrame(
        {
            "well_id": ["a"] * 5,
            "fold": [0] * 5,
            "row_index": np.arange(5),
            "last_visible_tvt": [99.0] * 5,
            "pfs_lag1000_delta": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    candidate = runner.build_legal_candidate(legal, pfs)

    from src.p3_up01_robust_u_projection import robust_polynomial_projection

    base_u = legal["pred_tvt"].to_numpy() + legal["z_current"].to_numpy()
    projected_u = robust_polynomial_projection(legal["md"].to_numpy(), base_u, degree=2)
    expected_up01 = base_u + 0.50 * (projected_u - base_u) - legal["z_current"].to_numpy()
    expected_pfs = pfs["last_visible_tvt"].to_numpy() + pfs["pfs_lag1000_delta"].to_numpy()
    expected_up03 = expected_up01 + 0.25 * (expected_pfs - legal["pred_tvt"].to_numpy())
    np.testing.assert_allclose(candidate["up01_degree2_blend50_tvt"], expected_up01)
    np.testing.assert_allclose(candidate["pfs_lag1000_abs_tvt"], expected_pfs)
    np.testing.assert_allclose(candidate["up03_pred_tvt"], expected_up03)
    assert "target_tvt" not in candidate.columns
    assert not any("up02" in column.lower() or "up08" in column.lower() for column in candidate.columns)


def test_target_loader_requires_matching_landed_candidate_hash(
    workspace_tmp_path: Path,
) -> None:
    candidate_path = workspace_tmp_path / "legal_candidates.parquet"
    manifest_path = workspace_tmp_path / "legal_generation_manifest.json"
    prediction_path = workspace_tmp_path / "predictions.parquet"
    legal = pd.DataFrame(
        {
            "well_id": ["shadow-a"],
            "fold": [0],
            "row_index": [7],
            "md": [100.0],
            "p3b00_pred_tvt": [110.0],
            "up01_degree2_blend50_tvt": [111.0],
            "pfs_lag1000_abs_tvt": [112.0],
            "up03_pred_tvt": [111.5],
        }
    )
    legal.to_parquet(candidate_path, index=False)
    pd.DataFrame(
        {
            "well_id": ["shadow-a"],
            "fold": [0],
            "row_index": [7],
            "target_tvt": [113.0],
        }
    ).to_parquet(prediction_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "candidate_generation_complete": True,
                "hidden_target_read": False,
                "wells": 1,
                "rows": 1,
                "legal_candidates_sha256": "wrong-hash",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SHA-256|哈希"):
        runner.load_targets_after_candidate_landed(
            candidate_path, manifest_path, prediction_path
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["legal_candidates_sha256"] = runner.file_sha256(candidate_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = runner.load_targets_after_candidate_landed(
        candidate_path, manifest_path, prediction_path
    )
    assert tuple(target.columns) == ("well_id", "fold", "row_index", "target_tvt")
    assert target["target_tvt"].tolist() == [113.0]


def test_shadow_gate_is_fixed_and_requires_every_component() -> None:
    passing = {
        "pooled_improvement_ft": 0.16,
        "improved_folds": 3,
        "well_win_rate": 0.53,
        "bootstrap_ci95_upper_ft": -0.001,
        "p90_degradation_ft": 0.19,
    }
    assert runner.evaluate_shadow_gate(passing)["passed"] is True

    for key, failing_value in {
        "pooled_improvement_ft": 0.149,
        "improved_folds": 2,
        "well_win_rate": 0.519,
        "bootstrap_ci95_upper_ft": 0.001,
        "p90_degradation_ft": 0.201,
    }.items():
        failing = {**passing, key: failing_value}
        assert runner.evaluate_shadow_gate(failing)["passed"] is False


def test_score_shadow_reports_pooled_fold_well_bootstrap_and_p90() -> None:
    legal_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for fold in range(5):
        for row_index in range(2):
            keys = {"well_id": f"well-{fold}", "fold": fold, "row_index": row_index}
            legal_rows.append(
                {
                    **keys,
                    "md": float(row_index),
                    "p3b00_pred_tvt": 2.0,
                    "up01_degree2_blend50_tvt": 1.0,
                    "pfs_lag1000_abs_tvt": 1.0,
                    "up03_pred_tvt": 1.0,
                }
            )
            target_rows.append({**keys, "target_tvt": 0.0})

    metrics, per_fold, per_well, scored = runner.score_shadow(
        pd.DataFrame(legal_rows), pd.DataFrame(target_rows), bootstrap_resamples=100
    )

    assert metrics["p3b00_pooled_rmse"] == 2.0
    assert metrics["up03_pooled_rmse"] == 1.0
    assert metrics["pooled_improvement_ft"] == 1.0
    assert metrics["improved_folds"] == 5
    assert metrics["well_win_rate"] == 1.0
    assert metrics["bootstrap_ci95_low_ft"] == -1.0
    assert metrics["bootstrap_ci95_upper_ft"] == -1.0
    assert metrics["p90_degradation_ft"] == -1.0
    assert metrics["gate"]["passed"] is True
    assert len(per_fold) == 5 and len(per_well) == 5 and len(scored) == 10
