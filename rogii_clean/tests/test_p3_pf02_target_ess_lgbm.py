"""P3-PF02 正式模型入口的最小合同测试。"""

from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path
from collections.abc import Iterator

import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pf02_target_ess_lgbm_cv import (
    NEW_ESS_FEATURES,
    P3B00_FEATURES,
    build_formal_feature_names,
    load_development_registry,
    merge_pf02_legal_cache,
    parse_fold_spec,
    read_development_parquet,
)


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """避开当前 Windows 用户临时目录的权限问题。"""

    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pf02_model_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def test_feature_contract_keeps_all_41_baseline_columns_and_appends_four_ess_paths() -> None:
    features = build_formal_feature_names()

    assert features[:41] == P3B00_FEATURES
    assert features[41:] == NEW_ESS_FEATURES
    assert len(features) == 45
    assert len(set(features)) == 45


def test_fold_spec_only_accepts_registered_screen_and_full_modes() -> None:
    assert parse_fold_spec("0,1") == [0, 1]
    assert parse_fold_spec("all") == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="0,1"):
        parse_fold_spec("0")


def test_development_registry_removes_shadow_before_any_target_table_is_loaded(
    workspace_tmp_path: Path,
) -> None:
    fold_path = workspace_tmp_path / "folds.csv"
    shadow_path = workspace_tmp_path / "shadow.csv"
    pd.DataFrame(
        {
            "well_id": ["dev_a", "shadow_x", "dev_b"],
            "pad_id": ["dev_a", "shadow_x", "dev_b"],
            "fold": [0, 1, 2],
            "hidden_rows": [2, 3, 4],
        }
    ).to_csv(fold_path, index=False)
    pd.DataFrame({"well_id": ["shadow_x"], "is_shadow": [True]}).to_csv(
        shadow_path,
        index=False,
    )

    development, shadow_ids = load_development_registry(fold_path, shadow_path)

    assert development["well_id"].tolist() == ["dev_a", "dev_b"]
    assert shadow_ids == {"shadow_x"}


def test_arrow_reader_returns_only_registered_development_wells(
    workspace_tmp_path: Path,
) -> None:
    parquet_path = workspace_tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "well_id": ["dev_a", "shadow_x", "dev_a"],
            "row_index": [1, 1, 2],
            "target_tvt": [10.0, 999999.0, 11.0],
        }
    ).to_parquet(parquet_path, index=False)

    loaded = read_development_parquet(
        parquet_path,
        well_ids=["dev_a"],
        columns=["well_id", "row_index", "target_tvt"],
    )

    assert loaded["well_id"].tolist() == ["dev_a", "dev_a"]
    assert 999999.0 not in loaded["target_tvt"].tolist()


def test_pf02_cache_merge_adds_only_four_legal_paths_and_checks_anchor(
    workspace_tmp_path: Path,
) -> None:
    cache_dir = workspace_tmp_path / "legal_cache"
    runtime_dir = workspace_tmp_path / "legal_runtime"
    cache_dir.mkdir()
    runtime_dir.mkdir()
    base = pd.DataFrame(
        {
            "well_id": ["dev_a", "dev_a"],
            "row_index": [10, 11],
            "last_visible_tvt": [100.0, 100.0],
            "old_feature": [1.0, 2.0],
        }
    )
    registry = pd.DataFrame(
        {"well_id": ["dev_a"], "fold": [0], "hidden_rows": [2]}
    )
    legal = pd.DataFrame(
        {
            "well_id": ["dev_a", "dev_a"],
            "fold": [0, 0],
            "row_index": [10, 11],
            "last_visible_tvt": [100.0, 100.0],
            "pf128_ess2_delta": [1.0, 2.0],
            "pf128_ess8_delta": [1.1, 2.1],
            "pf128_ess32_delta": [1.2, 2.2],
            "pf128_ess96_delta": [1.3, 2.3],
            "_cache_fingerprint": ["fp", "fp"],
        }
    )
    legal.to_parquet(cache_dir / "dev_a.parquet", index=False)

    merged, fingerprint = merge_pf02_legal_cache(
        base,
        registry,
        cache_dir,
        runtime_dir,
        shadow_ids={"shadow_x"},
    )

    assert fingerprint == "fp"
    assert "old_feature" in merged.columns
    assert merged[NEW_ESS_FEATURES].shape == (2, 4)


def test_pf02_cache_rejects_any_target_column(workspace_tmp_path: Path) -> None:
    cache_dir = workspace_tmp_path / "legal_cache"
    runtime_dir = workspace_tmp_path / "legal_runtime"
    cache_dir.mkdir()
    runtime_dir.mkdir()
    registry = pd.DataFrame(
        {"well_id": ["dev_a"], "fold": [0], "hidden_rows": [1]}
    )
    bad = pd.DataFrame(
        {
            "well_id": ["dev_a"],
            "fold": [0],
            "row_index": [10],
            "last_visible_tvt": [100.0],
            "pf128_ess2_delta": [1.0],
            "pf128_ess8_delta": [1.0],
            "pf128_ess32_delta": [1.0],
            "pf128_ess96_delta": [1.0],
            "target_tvt": [101.0],
            "_cache_fingerprint": ["fp"],
        }
    )
    bad.to_parquet(cache_dir / "dev_a.parquet", index=False)

    with pytest.raises(ValueError, match="schema"):
        merge_pf02_legal_cache(
            pd.DataFrame(
                {
                    "well_id": ["dev_a"],
                    "row_index": [10],
                    "last_visible_tvt": [100.0],
                }
            ),
            registry,
            cache_dir,
            runtime_dir,
            shadow_ids=set(),
        )
