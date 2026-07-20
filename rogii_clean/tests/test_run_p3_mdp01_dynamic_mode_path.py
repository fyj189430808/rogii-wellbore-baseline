"""P3-MDP01 合法路径运行器的输入边界、自然键和缓存合同测试。"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_mdp01_dynamic_mode_path as runner


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mdp01_runner_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def test_runner_exposes_only_legal_horizontal_and_typewell_columns() -> None:
    assert runner.LEGAL_HORIZONTAL_COLUMNS == ["MD", "Z", "GR", "TVT_input"]
    assert runner.LEGAL_TYPEWELL_COLUMNS == ["TVT", "GR"]
    assert "target_tvt" not in runner.P2_LEGAL_COLUMNS
    assert "TVT" not in runner.LEGAL_HORIZONTAL_COLUMNS

    source = Path(runner.__file__).read_text(encoding="utf-8")
    legal_reader_source = inspect.getsource(runner.read_legal_inputs)
    assert "usecols=LEGAL_HORIZONTAL_COLUMNS" in legal_reader_source
    assert "usecols=LEGAL_TYPEWELL_COLUMNS" in legal_reader_source
    assert 'columns=P2_LEGAL_COLUMNS' in inspect.getsource(runner.load_p2_legal_rows)
    assert '"target_tvt"' not in source


def test_development_selection_removes_shadow_and_formal_screen_is_folds_1_2(
    workspace_tmp_path: Path,
) -> None:
    tmp_path = workspace_tmp_path
    folds_path = tmp_path / "folds.csv"
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame(
        {
            "well_id": ["00001234", "well-b", "well-c", "shadow-well"],
            "fold": [1, 2, 0, 1],
            "hidden_rows": [2, 3, 4, 5],
        }
    ).to_csv(folds_path, index=False)
    pd.DataFrame({"well_id": ["shadow-well"]}).to_csv(shadow_path, index=False)

    development, shadow_ids = runner.load_development_registry(folds_path, shadow_path)
    selected = runner.select_registry(development, "folds12")

    assert development["well_id"].tolist() == ["00001234", "well-b", "well-c"]
    assert shadow_ids == {"shadow-well"}
    assert selected["well_id"].tolist() == ["00001234", "well-b"]
    assert selected["fold"].tolist() == [1, 2]


def test_candidate_assembly_requires_exact_natural_key_alignment() -> None:
    p2 = pd.DataFrame(
        {
            "well_id": ["00001234"] * 3,
            "fold": [1] * 3,
            "row_index": [7, 8, 9],
            "pred_tvt": [100.0, 101.0, 102.0],
        }
    )
    mode = pd.DataFrame(
        {
            "well_id": ["00001234"] * 3,
            "fold": [1] * 3,
            "row_index": [7, 8, 9],
            "last_visible_tvt": [90.0] * 3,
            "pf_mode_low_delta": [1.0, 2.0, 3.0],
            "pf_mode_middle_delta": [4.0, 5.0, 6.0],
            "pf_mode_high_delta": [7.0, 8.0, 9.0],
            "_cache_fingerprint": ["mode-fingerprint"] * 3,
        }
    )

    row_index, last_tvt, candidate_tvt = runner.assemble_candidate_tvt(
        p2,
        mode,
        well_id="00001234",
        fold=1,
        hidden_rows=3,
        expected_mode_fingerprint="mode-fingerprint",
    )

    np.testing.assert_array_equal(row_index, [7, 8, 9])
    assert last_tvt == 90.0
    np.testing.assert_allclose(
        candidate_tvt,
        np.asarray(
            [
                [100.0, 91.0, 94.0, 97.0],
                [101.0, 92.0, 95.0, 98.0],
                [102.0, 93.0, 96.0, 99.0],
            ]
        ),
    )

    misaligned = mode.copy()
    misaligned.loc[1, "row_index"] = 88
    with pytest.raises(ValueError, match="row_index"):
        runner.assemble_candidate_tvt(
            p2,
            misaligned,
            well_id="00001234",
            fold=1,
            hidden_rows=3,
            expected_mode_fingerprint="mode-fingerprint",
        )


def test_cache_validation_rejects_wrong_fingerprint(workspace_tmp_path: Path) -> None:
    tmp_path = workspace_tmp_path
    cache_path = tmp_path / "well.parquet"
    runtime_path = tmp_path / "well.json"
    frame = pd.DataFrame(
        {
            "well_id": ["00001234"],
            "fold": [1],
            "row_index": [7],
            "md": [10_000.0],
            "last_visible_tvt": [90.0],
            "mdp_dynamic_delta": [2.0],
            "mdp_selected_state": [0],
            "mdp_margin": [np.nan],
            "mdp_safe_10_delta": [0.2],
            "mdp_safe_25_delta": [0.5],
            "mdp_gr_shift_dynamic_delta": [3.0],
            "mdp_gr_shift_selected_state": [1],
            "mdp_gr_shift_safe_10_delta": [0.3],
            "mdp_gr_shift_safe_25_delta": [0.75],
            "mdp_cost_permutation_dynamic_delta": [4.0],
            "mdp_cost_permutation_selected_state": [2],
            "mdp_cost_permutation_safe_10_delta": [0.4],
            "mdp_cost_permutation_safe_25_delta": [1.0],
            "_cache_fingerprint": ["wrong"],
        }
    )
    frame.to_parquet(cache_path, index=False)
    runtime_path.write_text("{}", encoding="utf-8")

    assert (
        runner.validate_cache_hit(
            cache_path,
            runtime_path,
            well_id="00001234",
            fold=1,
            hidden_rows=1,
            fingerprint="expected",
            row_index=np.asarray([7]),
            source_signatures={},
        )
        is None
    )
