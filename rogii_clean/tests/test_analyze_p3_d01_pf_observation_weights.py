"""P3-D01 独立评分脚本的目标隔离、路径对齐和门控测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import analyze_p3_d01_pf_observation_weights as analysis_runner  # noqa: E402


def _write_path_cache(path: Path, well_id: str, row_indices: list[int]) -> None:
    """写入可手算的旧 P2-P01 合法路径。"""

    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "well_id": [well_id] * len(row_indices),
            "row_index": row_indices,
            "last_visible_tvt": [10.0] * len(row_indices),
            "pf128_mean_tvt": [11.0, 13.0][: len(row_indices)],
            "pf128_scale_3_delta": [1.0, 3.0][: len(row_indices)],
            "pf128_scale_5_delta": [1.5, 2.5][: len(row_indices)],
            "pf128_scale_8_delta": [2.0, 2.0][: len(row_indices)],
            "pf128_scale_12_delta": [2.5, 1.5][: len(row_indices)],
        }
    )
    frame.to_parquet(path / f"{well_id}.parquet", index=False)


def test_load_development_targets_filters_shadow_during_arrow_scan(tmp_path: Path, monkeypatch) -> None:
    """极端影子真值必须在 Arrow 扫描阶段被排除。"""

    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "well_id": ["dev_a", "dev_a", "shadow_x"],
            "fold": [0, 0, 1],
            "row_index": [2, 1, 9],
            "target_tvt": [12.0, 11.0, 1.0e300],
            "pred_tvt": [12.5, 10.5, -1.0e300],
        }
    ).to_parquet(predictions_path, index=False)
    selected = pd.DataFrame({"well_id": ["dev_a"], "fold": [0], "hidden_rows": [2]})

    original_dataset = analysis_runner.ds.dataset
    observed_filters: list[object] = []

    class DatasetProbe:
        def __init__(self, dataset):
            self._dataset = dataset

        def to_table(self, *, columns, filter):
            observed_filters.append(filter)
            return self._dataset.to_table(columns=columns, filter=filter)

    def probed_dataset(*args, **kwargs):
        return DatasetProbe(original_dataset(*args, **kwargs))

    monkeypatch.setattr(analysis_runner.ds, "dataset", probed_dataset)
    loaded = analysis_runner.load_development_targets(
        predictions_path,
        selected,
        shadow_ids={"shadow_x"},
    )

    assert observed_filters and observed_filters[0] is not None
    assert loaded["well_id"].tolist() == ["dev_a", "dev_a"]
    assert loaded["row_index"].tolist() == [1, 2]
    assert np.max(np.abs(loaded["target_tvt"].to_numpy())) < 100.0


def test_score_development_paths_aligns_keys_and_returns_one_row_per_well(tmp_path: Path) -> None:
    """目标乱序时仍按 row_index 对齐，且结果保持一井一行。"""

    cache_dir = tmp_path / "cache"
    _write_path_cache(cache_dir, "a", [1, 2])
    _write_path_cache(cache_dir, "b", [4, 3])
    targets = pd.DataFrame(
        {
            "well_id": ["b", "a", "b", "a"],
            "fold": [1, 0, 1, 0],
            "row_index": [3, 2, 4, 1],
            "target_tvt": [11.0, 14.0, 13.0, 11.0],
            "pred_tvt": [10.0, 15.0, 13.0, 12.0],
        }
    )
    selected = pd.DataFrame(
        {"well_id": ["a", "b"], "fold": [0, 1], "hidden_rows": [2, 2]}
    )

    scored = analysis_runner.score_development_paths(targets, selected, cache_dir)

    assert scored["well_id"].tolist() == ["a", "b"]
    assert scored["well_id"].is_unique
    a_row = scored.loc[scored["well_id"] == "a"].iloc[0]
    assert a_row["p2p02_sse"] == pytest.approx(2.0)
    assert a_row["pf128_mean_sse"] == pytest.approx(1.0)
    assert bool(a_row["best_scale_is_oracle"]) is True


def test_legal_readiness_fails_before_target_loader(tmp_path: Path, monkeypatch) -> None:
    """legal runtime 未完成时，不能接触目标 Parquet。"""

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "runtime_smoke.json").write_text(
        json.dumps({"completed": False, "errors": [{"well_id": "a"}]}),
        encoding="utf-8",
    )
    selected = pd.DataFrame({"well_id": ["a"], "fold": [0], "hidden_rows": [2]})
    target_called = False

    def forbidden_target_loader(*args, **kwargs):
        nonlocal target_called
        target_called = True
        raise AssertionError("不应读取目标")

    monkeypatch.setattr(analysis_runner, "load_development_targets", forbidden_target_loader)
    with pytest.raises(RuntimeError, match="legal"):
        analysis_runner.analyze_mode(
            mode="smoke",
            selected=selected,
            shadow_ids={"shadow_x"},
            predictions_path=tmp_path / "unused.parquet",
            source_path_cache_dir=tmp_path / "unused_cache",
            artifact_dir=artifact_dir,
        )
    assert target_called is False


def test_validate_legal_table_rejects_target_derived_columns() -> None:
    """legal 表禁止混入任何目标、误差、RMSE 或 oracle 列。"""

    selected = pd.DataFrame({"well_id": ["a"], "fold": [0], "hidden_rows": [2]})
    legal = selected.assign(target_rmse=[1.0])
    with pytest.raises(ValueError, match="禁止"):
        analysis_runner.validate_legal_table(legal, selected)


def test_mode_output_paths_are_isolated_and_all_has_contract_names(tmp_path: Path) -> None:
    """三种模式互不覆盖，只有 all 产生无后缀最终合同名。"""

    smoke = analysis_runner.output_paths(tmp_path, "smoke")
    fold01 = analysis_runner.output_paths(tmp_path, "fold01")
    all_paths = analysis_runner.output_paths(tmp_path, "all")

    assert set(smoke.values()).isdisjoint(set(fold01.values()))
    assert smoke["summary"].name == "summary_smoke.json"
    assert fold01["summary"].name == "summary_fold01.json"
    assert all_paths["summary"].name == "summary_all.json"
    assert all_paths["final_summary"].name == "summary.json"
    assert "final_summary" not in smoke
    assert "final_summary" not in fold01
