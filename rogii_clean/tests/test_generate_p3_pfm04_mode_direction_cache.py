"""PFM04 三个井级摘要的合法生成与来源指纹测试。"""

from __future__ import annotations

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

from scripts import generate_p3_pfm04_mode_direction_cache as generator
from src.p3_pfm01_ordered_pf_modes import build_ordered_mode_features


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """把临时文件放在项目盘，避免 Windows 系统临时目录权限不稳定。"""

    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pfm04_cache_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_seed_cache(path: Path, row_index: np.ndarray) -> None:
    """构造三个位置清楚的 PF 模式，便于核对摘要公式。"""

    rows = len(row_index)
    progress = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    groups = np.repeat(np.arange(3), [43, 42, 43])
    mode_offsets = np.asarray([-4.0, 0.0, 5.0], dtype=np.float64)
    seed_delta = mode_offsets[groups, None] + progress[None, :]
    np.savez(
        path,
        seed_delta=seed_delta.astype(np.float32),
        row_index=row_index.astype(np.int32),
        hidden_md=np.linspace(10_000.0, 10_800.0, rows).astype(np.float32),
        last_tvt=np.asarray([100.0], dtype=np.float64),
        final_ll=np.linspace(-4.0, 2.0, 128, dtype=np.float64),
        seed_ids=np.arange(128, dtype=np.int32),
        _cache_fingerprint=np.asarray([generator.SHARED_FINGERPRINT]),
        _format_version=np.asarray([1], dtype=np.int64),
    )


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """建立两口开发井、一口影子井及仅含合法列的 P2 路径。"""

    wells = ["00001234", "well-b", "shadow-well"]
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame(
        {"well_id": wells, "fold": [1, 2, 3], "hidden_rows": [9, 9, 9]}
    ).to_csv(folds_path, index=False)
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame({"well_id": ["shadow-well"]}).to_csv(shadow_path, index=False)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    row_index = np.arange(1400, 1409, dtype=np.int64)
    p2_rows: list[pd.DataFrame] = []
    for well_id in wells:
        _write_seed_cache(shared_dir / f"{well_id}.npz", row_index)
        p2_rows.append(
            pd.DataFrame(
                {
                    "well_id": [well_id] * 9,
                    "fold": [folds_path and {"00001234": 1, "well-b": 2, "shadow-well": 3}[well_id]] * 9,
                    "row_index": row_index,
                    "pred_tvt": 100.0 + np.linspace(-1.0, 2.0, 9),
                }
            )
        )
    p2_path = tmp_path / "p2_predictions.parquet"
    pd.concat(p2_rows, ignore_index=True).to_parquet(p2_path, index=False)
    return folds_path, shadow_path, shared_dir, p2_path


def test_generator_writes_only_three_finite_features_and_preserves_natural_well_id(
    workspace_tmp_path: Path,
) -> None:
    """合法缓存应一井一行，且准确包含冻结的三个数值特征。"""

    folds_path, shadow_path, shared_dir, p2_path = _make_inputs(workspace_tmp_path)
    output_dir = workspace_tmp_path / "output"
    summary = generator.generate_mode_direction_cache(
        folds_path,
        shadow_path,
        shared_dir,
        p2_path,
        output_dir,
        max_wells=2,
        workers=1,
    )

    legal = pd.read_parquet(output_dir / "smoke_2" / "legal" / "per_well.parquet")
    assert legal.columns.tolist() == generator.LEGAL_COLUMNS
    assert legal["well_id"].tolist() == ["00001234", "well-b"]
    assert legal["well_id"].nunique() == 2
    assert np.isfinite(legal[generator.NEW_FEATURES].to_numpy(dtype=np.float64)).all()
    assert summary["wells"] == 2
    assert summary["hidden_tvt_read"] is False
    assert set(generator.P2_LEGAL_COLUMNS) == {"well_id", "fold", "row_index", "pred_tvt"}
    forbidden = {"target_tvt", "tvt", "oracle", "rmse", "error"}
    assert forbidden.isdisjoint({name.lower() for name in generator.NEW_FEATURES})


def test_changed_p2_source_invalidates_existing_checkpoint(workspace_tmp_path: Path) -> None:
    """输入路径发生变化时，不得把旧摘要误报为缓存命中。"""

    folds_path, shadow_path, shared_dir, p2_path = _make_inputs(workspace_tmp_path)
    output_dir = workspace_tmp_path / "output"
    first = generator.generate_mode_direction_cache(
        folds_path, shadow_path, shared_dir, p2_path, output_dir, max_wells=2, workers=1
    )
    second = generator.generate_mode_direction_cache(
        folds_path, shadow_path, shared_dir, p2_path, output_dir, max_wells=2, workers=1
    )
    changed = pd.read_parquet(p2_path)
    changed.loc[changed["well_id"].eq("00001234"), "pred_tvt"] += 0.25
    changed.to_parquet(p2_path, index=False)
    third = generator.generate_mode_direction_cache(
        folds_path, shadow_path, shared_dir, p2_path, output_dir, max_wells=2, workers=1
    )

    assert first["cache_hits"] == 0
    assert second["cache_hits"] == 2
    assert third["cache_hits"] == 0
    assert third["experiment_fingerprint"] != first["experiment_fingerprint"]


def _extreme_likelihood_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """制造三个清楚模式，其中两个模式相对最佳似然低到全局 softmax 下溢。"""

    rows = 11
    progress = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    groups = np.repeat(np.arange(3), [43, 42, 43])
    seed_delta = np.asarray([-8.0, 0.0, 9.0])[groups, None] + progress[None, :]
    final_ll = np.asarray([0.0, -7_000.0, -14_000.0])[groups]
    seed_ids = np.arange(128, dtype=np.int64)
    p2_delta = np.linspace(-1.0, 2.0, rows, dtype=np.float64)
    return seed_delta, final_ll, seed_ids, p2_delta


def test_old_global_softmax_reproduces_extreme_likelihood_underflow() -> None:
    """RED 的根因对照：旧入口确实会把远模式质量算成零并抛错。"""

    seed_delta, final_ll, seed_ids, p2_delta = _extreme_likelihood_inputs()
    with pytest.raises(RuntimeError, match="模式质量必须是有限正数"):
        build_ordered_mode_features(
            seed_delta,
            final_ll,
            seed_ids,
            p2_pred_tvt=100.0 + p2_delta,
            last_visible_tvt=100.0,
        )


def test_pfm04_stable_summary_survives_extreme_likelihood_range() -> None:
    """PFM04 自身应允许数学上极小的 mode mass 记为零，摘要仍保持有限。"""

    seed_delta, final_ll, seed_ids, p2_delta = _extreme_likelihood_inputs()
    summary = generator.stable_mode_direction_summaries(
        seed_delta,
        final_ll,
        seed_ids,
        p2_delta,
    )

    assert list(summary) == generator.NEW_FEATURES
    assert np.isfinite(np.asarray(list(summary.values()), dtype=np.float64)).all()
    assert summary["direction_score"] == pytest.approx(-1.0, abs=1e-15)
    assert summary["high_minus_low_separation"] == pytest.approx(17.0)


def test_stable_summary_matches_old_entry_in_normal_likelihood_range() -> None:
    """数值正常时，新实现不得改变已成功井上的三个摘要定义。"""

    rows = 9
    progress = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    groups = np.repeat(np.arange(3), [43, 42, 43])
    seed_delta = np.asarray([-4.0, 0.0, 5.0])[groups, None] + progress[None, :]
    final_ll = np.linspace(-4.0, 2.0, 128, dtype=np.float64)
    seed_ids = np.arange(128, dtype=np.int64)
    p2_delta = np.linspace(-1.0, 2.0, rows, dtype=np.float64)
    _, old_summary, _ = build_ordered_mode_features(
        seed_delta,
        final_ll,
        seed_ids,
        p2_pred_tvt=100.0 + p2_delta,
        last_visible_tvt=100.0,
    )

    stable = generator.stable_mode_direction_summaries(
        seed_delta,
        final_ll,
        seed_ids,
        p2_delta,
    )

    for feature_name in generator.NEW_FEATURES:
        assert stable[feature_name] == pytest.approx(old_summary[feature_name], abs=1e-12)
