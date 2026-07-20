"""P3-PFS01 固定滞后粒子祖先平滑的数值合同测试。"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_p01_multiseed_pf import particle_filter_all_seeds_numba  # noqa: E402
from src.p3_pfs01_fixed_lag_smoothing import (  # noqa: E402
    DIAGNOSTIC_FIXED_COLUMN_COUNT,
    DIAGNOSTIC_HISTORY_ROWS_REORDERED,
    DIAGNOSTIC_MAX_ACTIVE_ROWS,
    DIAGNOSTIC_REPEATED_SOURCE_RESAMPLES,
    DIAGNOSTIC_RESAMPLE_COUNT,
    particle_filter_fixed_lag_all_seeds_numba,
    particle_filter_fixed_lag_all_seeds_with_diagnostics_numba,
)
from scripts.run_p3_pfs01_fixed_lag_particle_smoothing import (  # noqa: E402
    CACHE_COLUMNS,
    FROZEN_GENERATION_RUNNER_SHA256,
    FORMAL_LAG_DISTANCES_FT,
    HORIZONTAL_USECOLS,
    SMOKE_WELLS,
    TYPEWELL_USECOLS,
    build_cache_fingerprint,
    build_legal_cache_frame,
    load_development_registry,
    read_legal_inputs,
    validate_cache_hit,
    write_json_atomic,
    write_parquet_atomic,
)


def _small_kernel_arguments(
    md: np.ndarray | None = None,
    number_of_particles: int = 24,
    number_of_seeds: int = 2,
) -> dict[str, object]:
    """构造可快速编译、又能触发真实状态转移的小型冻结 PF 输入。"""

    hidden_md = (
        np.asarray(md, dtype=np.float64)
        if md is not None
        else np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    )
    number_of_rows = len(hidden_md)
    return {
        "md": hidden_md,
        "z": np.linspace(1000.0, 1000.3, number_of_rows, dtype=np.float64),
        "horizontal_gr": np.linspace(68.0, 78.0, number_of_rows, dtype=np.float64),
        "typewell_gr_grid": np.linspace(55.0, 90.0, 141, dtype=np.float64),
        "typewell_min_tvt": 10970.0,
        "typewell_step_ft": 0.25,
        "gr_sigma": 12.0,
        "initial_u": 12000.0,
        "initial_rate": 0.01,
        "number_of_particles": number_of_particles,
        "number_of_seeds": number_of_seeds,
        "seed_base": 3,
        "rate_momentum": 0.998,
        "rate_noise": 0.002,
        "position_noise_ft": 0.005,
        "resample_position_noise_ft": 0.1,
        "resample_rate_noise": 0.001,
        "resample_effective_fraction": 0.5,
        "initial_position_spread_ft": 4.5,
        "position_limit_beyond_typewell_ft": 100.0,
        "initial_rate_std": 0.01,
        "minimum_md_step_ft": 1.0,
        "squared_gr_residual_cap": 600.0,
        "likelihood_floor": 1e-300,
    }


def test_public_kernel_has_no_hidden_truth_argument() -> None:
    """正式内核只接收测试时合法数组，不得出现隐藏 TVT/target 参数。"""

    expected_parameter_names = {
        "md",
        "z",
        "horizontal_gr",
        "typewell_gr_grid",
        "typewell_min_tvt",
        "typewell_step_ft",
        "gr_sigma",
        "initial_u",
        "initial_rate",
        "number_of_particles",
        "number_of_seeds",
        "seed_base",
        "rate_momentum",
        "rate_noise",
        "position_noise_ft",
        "resample_position_noise_ft",
        "resample_rate_noise",
        "resample_effective_fraction",
        "initial_position_spread_ft",
        "position_limit_beyond_typewell_ft",
        "initial_rate_std",
        "minimum_md_step_ft",
        "squared_gr_residual_cap",
        "likelihood_floor",
        "lag_distances_ft",
    }
    actual_parameter_names = set(
        inspect.signature(
            particle_filter_fixed_lag_all_seeds_numba.py_func
        ).parameters
    )
    assert actual_parameter_names == expected_parameter_names


@pytest.mark.parametrize(
    "illegal_lags",
    [
        np.array([-1.0], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        np.array([], dtype=np.float64),
    ],
)
def test_illegal_lag_is_rejected(illegal_lags: np.ndarray) -> None:
    """内核层允许测试用 lag=0，但负数、非有限值和空列表必须拒绝。"""

    arguments = _small_kernel_arguments(number_of_particles=8, number_of_seeds=1)
    with pytest.raises(ValueError):
        particle_filter_fixed_lag_all_seeds_numba(
            **arguments,
            lag_distances_ft=illegal_lags,
        )


def test_lag_zero_and_final_likelihood_match_old_kernel_bitwise() -> None:
    """lag=0 必须逐位复刻旧 PF，证明没有增加或重排随机数调用。"""

    arguments = _small_kernel_arguments(number_of_particles=12, number_of_seeds=2)
    old_filtered, old_final_ll = particle_filter_all_seeds_numba(**arguments)
    lag_distances = np.array([0.0, 1.5, 20.0], dtype=np.float64)
    filtered, smoothed, final_ll = particle_filter_fixed_lag_all_seeds_numba(
        **arguments,
        lag_distances_ft=lag_distances,
    )

    assert filtered.shape == old_filtered.shape
    assert smoothed.shape == (3, *old_filtered.shape)
    np.testing.assert_array_equal(filtered, old_filtered)
    np.testing.assert_array_equal(smoothed[0], old_filtered)
    np.testing.assert_array_equal(final_ll, old_final_ll)


def test_future_evidence_changes_old_state_under_repeated_resampling() -> None:
    """未来强证据应通过重复祖先重采样，实际改写较早位置的后验均值。"""

    arguments = {
        "md": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        "z": np.zeros(3, dtype=np.float64),
        "horizontal_gr": np.array([0.0, 7.0, -7.0], dtype=np.float64),
        "typewell_gr_grid": np.arange(-10.0, 10.5, 0.5, dtype=np.float64),
        "typewell_min_tvt": -10.0,
        "typewell_step_ft": 0.5,
        "gr_sigma": 0.25,
        "initial_u": 0.0,
        "initial_rate": 0.0,
        "number_of_particles": 64,
        "number_of_seeds": 1,
        "seed_base": 11,
        "rate_momentum": 1.0,
        "rate_noise": 0.0,
        "position_noise_ft": 0.0,
        "resample_position_noise_ft": 3.0,
        "resample_rate_noise": 0.0,
        "resample_effective_fraction": 1.01,
        "initial_position_spread_ft": 3.0,
        "position_limit_beyond_typewell_ft": 0.0,
        "initial_rate_std": 0.0,
        "minimum_md_step_ft": 1.0,
        "squared_gr_residual_cap": 600.0,
        "likelihood_floor": 1e-300,
    }
    filtered, smoothed, final_ll, diagnostics = (
        particle_filter_fixed_lag_all_seeds_with_diagnostics_numba(
            **arguments,
            lag_distances_ft=np.array([1.0], dtype=np.float64),
        )
    )

    assert diagnostics[0, DIAGNOSTIC_RESAMPLE_COUNT] == len(arguments["md"])
    assert diagnostics[0, DIAGNOSTIC_REPEATED_SOURCE_RESAMPLES] > 0
    assert diagnostics[0, DIAGNOSTIC_HISTORY_ROWS_REORDERED] == 2
    assert diagnostics[0, DIAGNOSTIC_MAX_ACTIVE_ROWS] == 2
    assert smoothed[0, 0, 0] != filtered[0, 0]
    assert np.isfinite(final_ll).all()


def test_irregular_md_finalizes_multiple_rows_and_falls_back_per_seed() -> None:
    """跨越很大的 MD 步长时可一次定稿多行，末尾则逐 seed 回退过滤路径。"""

    md = np.array([0.0, 10.0, 20.0, 300.0, 305.0], dtype=np.float64)
    arguments = _small_kernel_arguments(
        md=md,
        number_of_particles=12,
        number_of_seeds=3,
    )
    lag_distances = np.array([15.0, 250.0, 1000.0], dtype=np.float64)
    filtered, smoothed, _, diagnostics = (
        particle_filter_fixed_lag_all_seeds_with_diagnostics_numba(
            **arguments,
            lag_distances_ft=lag_distances,
        )
    )

    assert filtered.shape == (3, 5)
    assert smoothed.shape == (3, 3, 5)
    assert np.all(diagnostics[:, DIAGNOSTIC_MAX_ACTIVE_ROWS] == 5)

    smoothed_count_start = DIAGNOSTIC_FIXED_COLUMN_COUNT
    fallback_count_start = DIAGNOSTIC_FIXED_COLUMN_COUNT + len(lag_distances)
    expected_smoothed_counts = np.array([3, 3, 0], dtype=np.int64)
    expected_fallback_counts = np.array([2, 2, 5], dtype=np.int64)
    for seed_index in range(3):
        np.testing.assert_array_equal(
            diagnostics[
                seed_index,
                smoothed_count_start:fallback_count_start,
            ],
            expected_smoothed_counts,
        )
        np.testing.assert_array_equal(
            diagnostics[seed_index, fallback_count_start:],
            expected_fallback_counts,
        )

        np.testing.assert_array_equal(smoothed[0, seed_index, 3:], filtered[seed_index, 3:])
        np.testing.assert_array_equal(smoothed[1, seed_index, 3:], filtered[seed_index, 3:])
        np.testing.assert_array_equal(smoothed[2, seed_index], filtered[seed_index])


def test_runner_contract_uses_fixed_smoke_wells_lags_and_legal_columns() -> None:
    """正式 runner 的三井、lag 和 CSV 读取列必须与预注册合同完全相同。"""

    assert SMOKE_WELLS == (
        ("fba7683c", 1, 407),
        ("cdc31d65", 4, 4840),
        ("ea3a0e38", 0, 10052),
    )
    assert FORMAL_LAG_DISTANCES_FT == (250.0, 500.0, 1000.0)
    assert HORIZONTAL_USECOLS == ("MD", "Z", "GR", "TVT_input")
    assert TYPEWELL_USECOLS == ("TVT", "GR")
    assert CACHE_COLUMNS == (
        "well_id",
        "fold",
        "row_index",
        "last_visible_tvt",
        "pfs_lag250_delta",
        "pfs_lag500_delta",
        "pfs_lag1000_delta",
        "_cache_fingerprint",
    )


def test_fingerprint_ignores_workers_and_output_directory() -> None:
    """线程数和落盘目录只影响运行方式，不得让同一算法产生不同缓存身份。"""

    common = {
        "experiment_id": "P3_PFS01_fixed_lag_particle_smoothing_v1",
        "fold_version": "balanced_well_5fold_v1",
        "lag_distances_ft": [250.0, 500.0, 1000.0],
        "likelihood_scale": 8.0,
        "particle_filter": {"number_of_particles": 500, "number_of_seeds": 128},
    }
    first = dict(common, workers=1, output_dir="one")
    second = dict(common, workers=8, output_dir="two")
    source_hashes = {"runner": "a", "core": "b", "fold": "c", "shadow": "d"}
    assert build_cache_fingerprint(first, source_hashes) == build_cache_fingerprint(
        second,
        source_hashes,
    )


def test_validation_only_changes_keep_frozen_generation_cache_identity() -> None:
    """收紧续跑校验不改变已经生成的三井路径缓存身份。"""

    assert FROZEN_GENERATION_RUNNER_SHA256 == (
        "75d94d26cb3e2ca1cd31f524829fe4db"
        "5241a4e55cf260486b424cf9305a8183"
    )


def test_registry_excludes_shadow_and_selects_fixed_smoke_tuple(tmp_path: Path) -> None:
    """开发井必须先排除 shadow；smoke 必须按井号取固定三井而不是 head(3)。"""

    folds = pd.DataFrame(
        {
            "well_id": ["shadow", "ea3a0e38", "other", "fba7683c", "cdc31d65"],
            "fold": [0, 0, 2, 1, 4],
            "hidden_rows": [10, 10052, 11, 407, 4840],
        }
    )
    shadow = pd.DataFrame({"well_id": ["shadow"]})
    folds_path = tmp_path / "folds.csv"
    shadow_path = tmp_path / "shadow.csv"
    folds.to_csv(folds_path, index=False)
    shadow.to_csv(shadow_path, index=False)

    development, smoke = load_development_registry(
        folds_path,
        shadow_path,
        enforce_formal_counts=False,
    )

    assert "shadow" not in set(development["well_id"])
    assert list(smoke.itertuples(index=False, name=None)) == list(SMOKE_WELLS)


def test_legal_csv_read_is_unchanged_when_hidden_tvt_is_mutated(tmp_path: Path) -> None:
    """隐藏 TVT 即使被改写，usecols 物理隔离后的合法输入也必须逐位不变。"""

    well_id = "demo"
    horizontal_path = tmp_path / f"{well_id}__horizontal_well.csv"
    typewell_path = tmp_path / f"{well_id}__typewell.csv"
    horizontal = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0],
            "Z": [100.0, 101.0, 102.0],
            "GR": [70.0, np.nan, 72.0],
            "TVT_input": [11000.0, np.nan, np.nan],
            "TVT": [11000.0, 11001.0, 11002.0],
        }
    )
    typewell = pd.DataFrame({"TVT": [10990.0, 11010.0], "GR": [65.0, 80.0]})
    horizontal.to_csv(horizontal_path, index=False)
    typewell.to_csv(typewell_path, index=False)
    before_horizontal, before_typewell = read_legal_inputs(tmp_path, well_id)

    horizontal.loc[horizontal["TVT_input"].isna(), "TVT"] += 9999.0
    horizontal.to_csv(horizontal_path, index=False)
    after_horizontal, after_typewell = read_legal_inputs(tmp_path, well_id)

    pd.testing.assert_frame_equal(before_horizontal, after_horizontal)
    pd.testing.assert_frame_equal(before_typewell, after_typewell)


def test_atomic_cache_has_exact_schema_and_strict_resume_validation(tmp_path: Path) -> None:
    """合法缓存必须原子写入，行键/指纹/文件哈希全匹配才算 cache hit。"""

    fingerprint = "f" * 64
    frame = build_legal_cache_frame(
        well_id="demo",
        fold=2,
        row_index=np.array([3, 4], dtype=np.int64),
        last_visible_tvt=11000.0,
        smoothed_tvt_paths=np.array(
            [[11001.0, 11002.0], [11003.0, 11004.0], [11005.0, 11006.0]],
            dtype=np.float64,
        ),
        fingerprint=fingerprint,
    )
    cache_path = tmp_path / "demo.parquet"
    runtime_path = tmp_path / "demo.json"
    write_parquet_atomic(frame, cache_path)
    runtime = {
        "experiment_id": "P3_PFS01_fixed_lag_particle_smoothing_v1",
        "well_id": "demo",
        "fold": 2,
        "hidden_rows": 2,
        "legal_horizontal_columns": list(HORIZONTAL_USECOLS),
        "typewell_columns": list(TYPEWELL_USECOLS),
        "experiment_fingerprint": fingerprint,
        "cache_sha256": __import__("hashlib").sha256(cache_path.read_bytes()).hexdigest(),
        "hidden_tvt_read": False,
        "generation_seconds": 1.25,
        "resume_check_seconds": 0.01,
        "resample_count": 3,
        "history_rows_reordered_total": 4,
        "history_particle_copies": 2000,
        "max_active_rows": 2,
        "history_bytes_peak": 16000,
        "lag250_smooth_rows": 0,
        "lag250_fallback_rows": 256,
        "lag500_smooth_rows": 0,
        "lag500_fallback_rows": 256,
        "lag1000_smooth_rows": 0,
        "lag1000_fallback_rows": 256,
    }
    write_json_atomic(runtime, runtime_path)

    loaded = validate_cache_hit(
        cache_path=cache_path,
        runtime_path=runtime_path,
        well_id="demo",
        fold=2,
        expected_row_index=np.array([3, 4], dtype=np.int64),
        fingerprint=fingerprint,
    )

    assert tuple(pd.read_parquet(cache_path).columns) == CACHE_COLUMNS
    assert loaded is not None
    assert not list(tmp_path.glob("*.tmp"))
    bad_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    bad_runtime["experiment_fingerprint"] = "bad"
    write_json_atomic(bad_runtime, runtime_path)
    assert validate_cache_hit(
        cache_path=cache_path,
        runtime_path=runtime_path,
        well_id="demo",
        fold=2,
        expected_row_index=np.array([3, 4], dtype=np.int64),
        fingerprint=fingerprint,
    ) is None


def _make_valid_runtime_contract(cache_path: Path, fingerprint: str) -> dict[str, object]:
    """为破坏性续跑测试构造一份字段和值都完整的逐井 runtime。"""

    return {
        "experiment_id": "P3_PFS01_fixed_lag_particle_smoothing_v1",
        "well_id": "demo",
        "fold": 2,
        "hidden_rows": 2,
        "legal_horizontal_columns": list(HORIZONTAL_USECOLS),
        "typewell_columns": list(TYPEWELL_USECOLS),
        "experiment_fingerprint": fingerprint,
        "cache_sha256": __import__("hashlib").sha256(cache_path.read_bytes()).hexdigest(),
        "hidden_tvt_read": False,
        "generation_seconds": 1.25,
        "resume_check_seconds": 0.01,
        "resample_count": 3,
        "history_rows_reordered_total": 4,
        "history_particle_copies": 2000,
        "max_active_rows": 2,
        "history_bytes_peak": 16000,
        "lag250_smooth_rows": 0,
        "lag250_fallback_rows": 256,
        "lag500_smooth_rows": 0,
        "lag500_fallback_rows": 256,
        "lag1000_smooth_rows": 0,
        "lag1000_fallback_rows": 256,
    }


def _write_valid_cache_for_runtime_damage_test(
    tmp_path: Path,
) -> tuple[Path, Path, str, dict[str, object]]:
    """写入一份可命中的合法 parquet，供每个 runtime 破坏测试独立使用。"""

    fingerprint = "a" * 64
    frame = build_legal_cache_frame(
        well_id="demo",
        fold=2,
        row_index=np.array([3, 4], dtype=np.int64),
        last_visible_tvt=11000.0,
        smoothed_tvt_paths=np.array(
            [[11001.0, 11002.0], [11003.0, 11004.0], [11005.0, 11006.0]],
            dtype=np.float64,
        ),
        fingerprint=fingerprint,
    )
    cache_path = tmp_path / "demo.parquet"
    runtime_path = tmp_path / "demo.json"
    write_parquet_atomic(frame, cache_path)
    runtime = _make_valid_runtime_contract(cache_path, fingerprint)
    write_json_atomic(runtime, runtime_path)
    assert validate_cache_hit(
        cache_path,
        runtime_path,
        "demo",
        2,
        np.array([3, 4], dtype=np.int64),
        fingerprint,
    ) is not None
    return cache_path, runtime_path, fingerprint, runtime


def test_cache_misses_when_runtime_hidden_rows_is_damaged(tmp_path: Path) -> None:
    """runtime 自报隐藏行数与实际缓存不符时，绝不能继续复用。"""

    cache_path, runtime_path, fingerprint, runtime = (
        _write_valid_cache_for_runtime_damage_test(tmp_path)
    )
    runtime["hidden_rows"] = 999
    write_json_atomic(runtime, runtime_path)
    assert validate_cache_hit(
        cache_path,
        runtime_path,
        "demo",
        2,
        np.array([3, 4], dtype=np.int64),
        fingerprint,
    ) is None


def test_cache_misses_when_runtime_cache_sha_is_damaged(tmp_path: Path) -> None:
    """runtime 记录的 parquet SHA 不匹配时，绝不能继续复用。"""

    cache_path, runtime_path, fingerprint, runtime = (
        _write_valid_cache_for_runtime_damage_test(tmp_path)
    )
    runtime["cache_sha256"] = "0" * 64
    write_json_atomic(runtime, runtime_path)
    assert validate_cache_hit(
        cache_path,
        runtime_path,
        "demo",
        2,
        np.array([3, 4], dtype=np.int64),
        fingerprint,
    ) is None


def test_cache_misses_when_required_runtime_field_is_deleted(tmp_path: Path) -> None:
    """任一必需运行诊断缺失时，都应重算而不是静默命中。"""

    cache_path, runtime_path, fingerprint, runtime = (
        _write_valid_cache_for_runtime_damage_test(tmp_path)
    )
    del runtime["generation_seconds"]
    write_json_atomic(runtime, runtime_path)
    assert validate_cache_hit(
        cache_path,
        runtime_path,
        "demo",
        2,
        np.array([3, 4], dtype=np.int64),
        fingerprint,
    ) is None
