"""P2-P01 多随机种子 PF 的确定性、合法输入和输出含义测试。"""

from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_p01_multiseed_pf import (  # noqa: E402
    build_multiseed_pf_features,
    particle_filter_all_seeds_numba,
    particle_filter_all_seeds_reference,
    prepare_particle_filter_inputs,
)


def _load_parameters() -> dict:
    config_path = CLEAN_ROOT / "configs" / "p2_p01_multiseed_pf_mean_v1.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["particle_filter"]


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 8 个可见点和 7 个隐藏点的小井，便于快速跑完整算法。"""

    md = np.arange(15, dtype=np.float64)
    z = 1000.0 + 0.15 * md
    true_tvt = 11000.0 + 0.03 * md
    typewell_tvt = np.arange(10980.0, 11030.5, 0.5)
    typewell_gr = 70.0 + 12.0 * np.sin((typewell_tvt - 10980.0) / 3.7)
    horizontal_gr = np.interp(true_tvt, typewell_tvt, typewell_gr)
    tvt_input = true_tvt.copy()
    tvt_input[8:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": horizontal_gr,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell


def test_numba_kernel_matches_direct_reference_on_tiny_arrays() -> None:
    """同一随机种子和调用顺序下，快速内核必须复现直接循环公式。"""

    md = np.array([8.0, 9.0, 11.0], dtype=np.float64)
    z = np.array([1000.0, 1000.2, 1000.4], dtype=np.float64)
    gr = np.array([68.0, 72.0, 75.0], dtype=np.float64)
    reference_gr = np.linspace(60.0, 80.0, 101, dtype=np.float64)
    arguments = dict(
        md=md,
        z=z,
        horizontal_gr=gr,
        typewell_gr_grid=reference_gr,
        typewell_min_tvt=10980.0,
        typewell_step_ft=0.2,
        gr_sigma=15.0,
        initial_u=12000.0,
        initial_rate=0.01,
        number_of_particles=12,
        number_of_seeds=2,
        seed_base=0,
        rate_momentum=0.998,
        rate_noise=0.002,
        position_noise_ft=0.005,
        resample_position_noise_ft=0.1,
        resample_rate_noise=0.001,
        resample_effective_fraction=0.5,
        initial_position_spread_ft=4.5,
        position_limit_beyond_typewell_ft=100.0,
        initial_rate_std=0.01,
        minimum_md_step_ft=1.0,
        squared_gr_residual_cap=600.0,
        likelihood_floor=1e-300,
    )
    fast_predictions, fast_likelihoods = particle_filter_all_seeds_numba(**arguments)
    reference_predictions, reference_likelihoods = particle_filter_all_seeds_reference(
        **arguments
    )
    np.testing.assert_allclose(fast_predictions, reference_predictions, atol=1e-12)
    np.testing.assert_allclose(fast_likelihoods, reference_likelihoods, atol=1e-12)


def test_hidden_tvt_is_not_an_input_and_row_keys_match_hidden_rows() -> None:
    """改写隐藏真值不能改变特征，输出键必须正好是自然隐藏行。"""

    horizontal, typewell = _synthetic_well()
    parameters = _load_parameters()
    parameters["number_of_particles"] = 24
    parameters["number_of_seeds"] = 3

    original, _ = build_multiseed_pf_features(horizontal, typewell, parameters)
    changed = horizontal.copy()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 9999.0
    mutated, _ = build_multiseed_pf_features(changed, typewell, parameters)
    without_truth, _ = build_multiseed_pf_features(
        horizontal.drop(columns="TVT"), typewell, parameters
    )

    expected_rows = np.flatnonzero(horizontal["TVT_input"].isna())
    assert original["row_index"].tolist() == expected_rows.tolist()
    pd.testing.assert_frame_equal(original, mutated, check_exact=True)
    pd.testing.assert_frame_equal(original, without_truth, check_exact=True)


def test_repeat_and_threaded_calls_are_identical() -> None:
    """固定 seed 后，重复调用以及两个线程并发调用都必须逐位一致。"""

    horizontal, typewell = _synthetic_well()
    parameters = _load_parameters()
    parameters["number_of_particles"] = 20
    parameters["number_of_seeds"] = 4

    first, first_quality = build_multiseed_pf_features(
        horizontal, typewell, parameters
    )
    second, second_quality = build_multiseed_pf_features(
        horizontal, typewell, parameters
    )
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first_quality == second_quality

    def run_once() -> pd.DataFrame:
        result, _ = build_multiseed_pf_features(horizontal, typewell, parameters)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent_results = list(executor.map(lambda _: run_once(), range(2)))
    pd.testing.assert_frame_equal(first, concurrent_results[0], check_exact=True)
    pd.testing.assert_frame_equal(first, concurrent_results[1], check_exact=True)


def test_output_contains_only_finite_legal_path_features() -> None:
    """正式均值、四个 scale、seed 标准差和审计常数必须完整且有限。"""

    horizontal, typewell = _synthetic_well()
    parameters = copy.deepcopy(_load_parameters())
    parameters["number_of_particles"] = 20
    parameters["number_of_seeds"] = 3
    prepared = prepare_particle_filter_inputs(horizontal, typewell, parameters)
    assert "TVT" not in prepared

    features, quality = build_multiseed_pf_features(horizontal, typewell, parameters)
    expected_columns = {
        "row_index",
        "last_visible_tvt",
        "pf128_mean_tvt",
        "pf128_mean_delta",
        "pf128_seed0_delta",
        "pf128_scale_3_delta",
        "pf128_scale_5_delta",
        "pf128_scale_8_delta",
        "pf128_scale_12_delta",
        "pf128_seed_std",
    }
    assert set(features.columns) == expected_columns
    assert np.isfinite(features.drop(columns="row_index").to_numpy()).all()
    float_columns = [column for column in features.columns if column != "row_index"]
    assert all(features[column].dtype == np.float32 for column in float_columns)
    assert set(quality) == {"pf_best_ll_per_row", "pf_ll_spread", "pf_gr_sigma"}
    assert np.isfinite(list(quality.values())).all()


def test_missing_gr_preprocessing_matches_frozen_notebook_rules() -> None:
    """可见缺失填 0、隐藏整井插值和 Typewell 均值填充必须被固定。"""

    horizontal, typewell = _synthetic_well()
    horizontal.loc[2, "GR"] = np.nan
    horizontal.loc[10:11, "GR"] = np.nan
    typewell.loc[4, "GR"] = np.nan
    parameters = _load_parameters()
    parameters["number_of_particles"] = 8
    parameters["number_of_seeds"] = 1
    prepared = prepare_particle_filter_inputs(horizontal, typewell, parameters)

    typewell_mean = float(typewell["GR"].mean(skipna=True))
    filled_typewell = typewell["GR"].fillna(typewell_mean).to_numpy(dtype=float)
    expected_visible = horizontal.loc[:7, "GR"].fillna(0.0).to_numpy(dtype=float)
    expected_reference = np.interp(
        horizontal.loc[:7, "TVT_input"], typewell["TVT"], filled_typewell
    )
    expected_sigma = float(np.clip(np.std(expected_visible - expected_reference), 10.0, 60.0))
    assert prepared["gr_sigma"] == expected_sigma

    expected_hidden_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy()[8:]
    )
    np.testing.assert_allclose(prepared["horizontal_gr"], expected_hidden_gr)
