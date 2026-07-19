"""P3-PF03：正式 128-seed 分段似然路径的关键数值测试。"""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_pf03_segmented_likelihood import (
    LOCAL_PATH_COLUMNS,
    aggregate_local_likelihood_path,
    path_observation_row_log_likelihoods,
    read_pf03_shared_cache,
    write_pf03_shared_cache_atomic,
)
import scripts.run_p3_pf03_segmented_likelihood as pf03_runner  # noqa: E402


def test_path_observation_ll_uses_frozen_gr_residual_formula() -> None:
    """局部 LL 必须直接由 seed TVT 路径上的 Typewell GR 残差计算。"""

    seed_predictions = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    horizontal_gr = np.array([10.0, 14.0], dtype=np.float64)
    typewell_gr_grid = np.array([10.0, 12.0, 14.0], dtype=np.float64)

    actual = path_observation_row_log_likelihoods(
        seed_predictions=seed_predictions,
        horizontal_gr=horizontal_gr,
        typewell_gr_grid=typewell_gr_grid,
        typewell_min_tvt=0.0,
        typewell_step_ft=1.0,
        gr_sigma=2.0,
        squared_gr_residual_cap=600.0,
        likelihood_floor=1e-300,
    )

    expected = np.array([[0.0, -0.5], [-0.5, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=1e-15)


@pytest.mark.filterwarnings("error")
def test_local_target_ess_weights_can_change_along_md() -> None:
    """前后窗口偏好不同 seed 时，聚合路径也必须随 MD 平滑改变。"""

    seed_count = 32
    row_count = 11
    md = np.arange(row_count, dtype=np.float64)
    seed_values = np.arange(seed_count, dtype=np.float64)
    seed_predictions = np.repeat(seed_values[:, None], row_count, axis=1)
    row_log_likelihoods = np.empty((seed_count, row_count), dtype=np.float64)
    for row_index in range(row_count):
        preferred_seed = 2.0 if row_index <= 4 else 29.0
        row_log_likelihoods[:, row_index] = -np.square(
            seed_values - preferred_seed
        )

    delta, quality = aggregate_local_likelihood_path(
        seed_predictions=seed_predictions,
        row_log_likelihoods=row_log_likelihoods,
        md=md,
        observed_gr_mask=np.ones(row_count, dtype=bool),
        last_visible_tvt=0.0,
        window_ft=4.0,
        target_ess=16.0,
        center_step_ft=2.0,
    )

    assert delta.dtype == np.float32
    assert float(delta[1]) < float(delta[-2])
    assert quality["median_achieved_ess"] == pytest.approx(16.0, abs=1e-5)
    assert quality["no_observation_window_count"] == 0


def test_window_without_original_gr_returns_uniform_seed_mean() -> None:
    """原始 GR 完全缺失时不得使用插值 GR 挑 seed，而要逐行回到均值路径。"""

    seed_predictions = np.array(
        [
            [100.0, 101.0, 102.0],
            [110.0, 111.0, 112.0],
            [120.0, 121.0, 122.0],
            [130.0, 131.0, 132.0],
        ],
        dtype=np.float64,
    )
    row_log_likelihoods = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0],
            [-3.0, -3.0, -3.0],
            [-4.0, -4.0, -4.0],
        ],
        dtype=np.float64,
    )

    delta, quality = aggregate_local_likelihood_path(
        seed_predictions=seed_predictions,
        row_log_likelihoods=row_log_likelihoods,
        md=np.array([0.0, 100.0, 200.0]),
        observed_gr_mask=np.zeros(3, dtype=bool),
        last_visible_tvt=100.0,
        window_ft=250.0,
        target_ess=2.0,
        center_step_ft=100.0,
    )

    expected_absolute_path = seed_predictions.mean(axis=0).astype(np.float32)
    expected_delta = expected_absolute_path - np.float32(100.0)
    np.testing.assert_array_equal(delta, expected_delta)
    assert quality["no_observation_window_count"] == quality["window_count"]
    assert quality["uniform_fallback_row_count"] == 3


def test_formal_path_names_are_frozen() -> None:
    assert LOCAL_PATH_COLUMNS == [
        "pf128_local250_delta",
        "pf128_local500_delta",
        "pf128_local1000_delta",
    ]


def test_shared_cache_round_trip_and_fingerprint_guard() -> None:
    """共享缓存必须原样保留 128 路径所需数组，并拒绝错误配置指纹。"""

    with tempfile.TemporaryDirectory(dir=CLEAN_ROOT / "tests") as temporary_directory:
        cache_path = Path(temporary_directory) / "well_a.npz"
        arrays = {
            "seed_delta": np.arange(12, dtype=np.float32).reshape(4, 3),
            "row_index": np.array([8, 9, 10], dtype=np.int32),
            "hidden_md": np.array([100.0, 101.0, 102.0], dtype=np.float32),
            "last_tvt": np.array([11000.0], dtype=np.float64),
            "final_ll": np.arange(4, dtype=np.float64),
            "seed_ids": np.arange(4, dtype=np.int32),
        }

        write_pf03_shared_cache_atomic(cache_path, arrays, fingerprint="abc123")
        loaded = read_pf03_shared_cache(cache_path, expected_fingerprint="abc123")

        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)
        with pytest.raises(ValueError, match="指纹"):
            read_pf03_shared_cache(cache_path, expected_fingerprint="different")


def test_generation_only_argument_is_mutually_exclusive_and_default_stays_smoke() -> None:
    default_args = pf03_runner.parse_args([])
    assert default_args.mode == "smoke"
    assert default_args.generation_only_fold is None

    regular_args = pf03_runner.parse_args(["--mode", "fold01"])
    assert regular_args.mode == "fold01"
    assert regular_args.generation_only_fold is None

    generation_args = pf03_runner.parse_args(["--generation-only-fold", "0"])
    assert generation_args.mode is None
    assert generation_args.generation_only_fold == 0

    with pytest.raises(SystemExit):
        pf03_runner.parse_args(
            ["--mode", "fold01", "--generation-only-fold", "0"]
        )


def test_generation_only_fold0_writes_legal_runtime_and_never_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir=CLEAN_ROOT / "tests") as temporary_directory:
        root = Path(temporary_directory)
        artifact_dir = root / "artifact"
        source_pf_config = root / "source_pf_config.json"
        source_pf_config.write_text(
            json.dumps({"particle_filter": {"number_of_seeds": 128}}),
            encoding="utf-8",
        )
        config_path = root / "config.json"
        config = {
            "experiment_id": pf03_runner.EXPERIMENT_ID,
            "window_sizes_ft": [250.0, 500.0, 1000.0],
            "target_ess": 16.0,
            "center_step_ft": 100.0,
            "model_training": False,
            "shadow_target_access": False,
            "source_pf_core": str(root / "pf_core.py"),
            "source_pf_core_sha256": "frozen-core-sha",
            "source_pf_config": str(source_pf_config),
            "fold_registry": str(root / "folds.csv"),
            "shadow_registry": str(root / "shadow.csv"),
            "shared_seed_cache_dir": str(root / "shared"),
            "source_d01_fingerprint": "d01-fingerprint",
            "raw_train_dir": str(root / "raw"),
            "source_d01_likelihood_cache_dir": str(root / "d01"),
            "source_pf_legal_cache_dir": str(root / "old"),
            "source_model_predictions": str(root / "predictions.parquet"),
            "workers": 8,
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        hidden_rows = [5001, *([4976] * 130)]
        development = pd.DataFrame(
            {
                "well_id": [f"fold0_{number:03d}" for number in range(131)]
                + ["fold1_extra"],
                "fold": [0] * 131 + [1],
                "hidden_rows": hidden_rows + [99],
            }
        )
        assert sum(hidden_rows) == 651_881
        written_csv: dict[Path, pd.DataFrame] = {}
        written_json: dict[Path, dict] = {}

        monkeypatch.setattr(pf03_runner, "file_sha256", lambda path: "frozen-core-sha")
        monkeypatch.setattr(
            pf03_runner,
            "load_development_registry",
            lambda *args, **kwargs: development.copy(),
        )

        def fake_generation(tasks, workers):
            assert workers == 8
            assert len(tasks) == 131
            assert {task["fold"] for task in tasks} == {0}
            assert sum(task["hidden_rows"] for task in tasks) == 651_881
            return [
                {
                    "well_id": task["well_id"],
                    "fold": task["fold"],
                    "hidden_rows": task["hidden_rows"],
                    "cache_hit": number < 46,
                    "elapsed_seconds": 0.0,
                    "quality": {},
                }
                for number, task in enumerate(tasks)
            ]

        monkeypatch.setattr(pf03_runner, "run_generation", fake_generation)
        monkeypatch.setattr(
            pf03_runner,
            "write_csv_atomic",
            lambda path, frame: written_csv.__setitem__(Path(path), frame.copy()),
        )
        monkeypatch.setattr(
            pf03_runner,
            "write_json_atomic",
            lambda path, value: written_json.__setitem__(Path(path), dict(value)),
        )

        def forbidden_score(*args, **kwargs):
            raise AssertionError("generation-only 不得调用 score_paths")

        def forbidden_target_read(*args, **kwargs):
            raise AssertionError("generation-only 不得读取 target")

        monkeypatch.setattr(pf03_runner, "score_paths", forbidden_score)
        monkeypatch.setattr(pf03_runner, "read_selected_targets", forbidden_target_read)

        pf03_runner.main(
            [
                "--generation-only-fold",
                "0",
                "--config",
                str(config_path),
                "--artifact-dir",
                str(artifact_dir),
            ]
        )

        legal_path = artifact_dir / "legal" / "per_well_generation_only_fold0.csv"
        runtime_path = artifact_dir / "runtime_generation_only_fold0.json"
        assert legal_path in written_csv
        assert len(written_csv[legal_path]) == 131
        runtime = written_json[runtime_path]
        assert runtime["wells"] == 131
        assert runtime["rows"] == 651_881
        assert runtime["cache_hits"] == 46
        assert runtime["new"] == 85
        assert runtime["generated"] == 131
        assert runtime["completed"] is True
        assert set(runtime["fingerprints"]) == {"experiment", "shared_cache"}
