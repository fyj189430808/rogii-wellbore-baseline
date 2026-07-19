"""P3-D01 Task 1：PF 权重与 GR 证据审计纯函数测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p2_p01_multiseed_pf import (  # noqa: E402
    _kernel_arguments,
    particle_filter_all_seeds_numba,
    prepare_particle_filter_inputs,
)
from src.p3_d01_pf_observation_audit import (  # noqa: E402
    hidden_gr_evidence_statistics,
    read_likelihood_cache,
    run_pf_likelihood_audit,
    seed_weights,
    summarize_weight_scale,
    visible_prefix_gr_diagnostics,
    write_likelihood_cache,
)


def _parameters() -> dict:
    path = CLEAN_ROOT / "configs" / "p2_p01_multiseed_pf_mean_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))["particle_filter"]


def _typewell() -> pd.DataFrame:
    return pd.DataFrame({"TVT": np.arange(100.0, 121.0), "GR": np.arange(20.0, 41.0)})


def _well_with_six_hidden_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(8, dtype=float) * 10.0,
            "Z": 1000.0 + np.arange(8, dtype=float),
            "GR": [20.0, 21.0, 22.0, np.nan, np.nan, 25.0, 26.0, 27.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "TVT": np.arange(100.0, 108.0),
        }
    )


def _small_pf_parameters() -> dict:
    parameters = _parameters().copy()
    parameters.update({"number_of_particles": 8, "number_of_seeds": 3})
    return parameters


def _well_with_five_hidden_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(7, dtype=float) * 10.0,
            "Z": 1000.0 + np.arange(7, dtype=float),
            "GR": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "TVT": np.arange(100.0, 107.0),
        }
    )


def test_identical_seed_likelihoods_are_uniform() -> None:
    summary = summarize_weight_scale(np.full(128, -42.0), temperature=3.0)
    weights = seed_weights(np.full(128, -42.0), temperature=3.0)

    np.testing.assert_allclose(weights, np.full(128, 1.0 / 128.0))
    assert summary["effective_sample_size"] == pytest.approx(128.0)
    assert summary["max_weight"] == pytest.approx(1.0 / 128.0)
    assert summary["normalized_entropy"] == pytest.approx(1.0)
    assert summary["normalized_effective_sample_size"] == pytest.approx(1.0)
    assert summary["entropy"] == pytest.approx(np.log(128.0))
    assert summary["perplexity"] == pytest.approx(128.0)
    assert summary["top4_mass"] == pytest.approx(4.0 / 128.0)


def test_seed_weights_are_stable_shift_invariant_and_temperature_ordered() -> None:
    likelihoods = np.array([-1.0e6, -1.0e6 - 2.0, -1.0e6 - 5.0])
    weights = seed_weights(likelihoods, temperature=3.0)
    shifted = seed_weights(likelihoods + 1.0e3, temperature=3.0)
    cool = summarize_weight_scale(likelihoods, temperature=3.0)
    warm = summarize_weight_scale(likelihoods, temperature=12.0)

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]
    np.testing.assert_allclose(weights, shifted)
    assert warm["effective_sample_size"] >= cool["effective_sample_size"]
    assert warm["normalized_entropy"] >= cool["normalized_entropy"]
    assert warm["max_weight"] <= cool["max_weight"]


@pytest.mark.parametrize(
    ("log_likelihoods", "temperature"),
    [
        (np.ones((2, 2)), 3.0),
        (np.array([]), 3.0),
        (np.array([0.0, np.nan]), 3.0),
        (np.array([0.0, np.inf]), 3.0),
        (np.array([0.0]), 0.0),
        (np.array([0.0]), -1.0),
    ],
)
def test_seed_weights_reject_invalid_inputs(log_likelihoods: np.ndarray, temperature: float) -> None:
    with pytest.raises(ValueError):
        seed_weights(log_likelihoods, temperature)


def test_hidden_gr_statistics_classify_and_count_observations() -> None:
    statistics = hidden_gr_evidence_statistics(_well_with_six_hidden_rows())

    assert statistics["hidden_rows"] == 6
    assert statistics["observed_gr_rows"] == 4
    assert statistics["interpolated_gr_rows"] == 2
    assert statistics["internal_interpolated_gr_rows"] == 2
    assert statistics["edge_filled_gr_rows"] == 0
    assert statistics["typewell_mean_fallback_rows"] == 0
    assert statistics["observed_gr_fraction"] == pytest.approx(4.0 / 6.0)
    assert statistics["longest_gr_gap_rows"] == 2
    assert statistics["longest_gr_gap_md_ft"] == pytest.approx(20.0)
    assert statistics["effective_observation_count"] == 4
    assert statistics["effective_observation_count_q025"] == pytest.approx(4.5)
    assert statistics["current_likelihood_update_count"] == 6
    assert statistics["current_evidence_inflation"] == pytest.approx(1.5)
    assert (
        statistics["observed_gr_rows"]
        + statistics["internal_interpolated_gr_rows"]
        + statistics["edge_filled_gr_rows"]
        + statistics["typewell_mean_fallback_rows"]
        == statistics["hidden_rows"]
    )


def test_hidden_gr_statistics_distinguish_edge_and_typewell_fallback() -> None:
    edge_well = _well_with_six_hidden_rows()
    edge_well.loc[:, "GR"] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 26.0, 27.0]
    edge = hidden_gr_evidence_statistics(edge_well)
    no_gr_well = _well_with_six_hidden_rows()
    no_gr_well.loc[:, "GR"] = np.nan
    fallback = hidden_gr_evidence_statistics(no_gr_well)

    assert edge["edge_filled_gr_rows"] == 4
    assert edge["internal_interpolated_gr_rows"] == 0
    assert fallback["typewell_mean_fallback_rows"] == 6
    assert fallback["observed_gr_rows"] == 0


def test_hidden_gr_statistics_treat_infinite_gr_as_missing_not_interpolation_support() -> None:
    well = _well_with_six_hidden_rows()
    well.loc[0, "GR"] = np.inf
    well.loc[5, "GR"] = -np.inf

    statistics = hidden_gr_evidence_statistics(well)

    assert statistics["observed_gr_rows"] == 3
    assert statistics["internal_interpolated_gr_rows"] == 3
    assert statistics["edge_filled_gr_rows"] == 0


def test_hidden_gr_statistics_report_independent_maximum_gap_rows_and_length() -> None:
    well = pd.DataFrame(
        {
            "MD": [0.0, 10.0, 110.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0],
            "GR": [20.0, np.nan, np.nan, 23.0, np.nan, np.nan, np.nan, 27.0, 28.0, 29.0],
            "TVT_input": [100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    statistics = hidden_gr_evidence_statistics(well)

    assert statistics["longest_gr_gap_rows"] == 3
    assert statistics["longest_gr_gap_md_ft"] == pytest.approx(110.0)


def test_visible_prefix_diagnostics_fit_affine_and_match_legacy_gr_sigma() -> None:
    typewell = _typewell()
    well = _well_with_six_hidden_rows()
    visible = well["TVT_input"].notna()
    well.loc[visible, "GR"] = 2.0 * well.loc[visible, "TVT_input"] - 155.0
    # Typewell GR(TVT)=TVT-80, therefore horizontal GR=2*reference+5.

    diagnostics = visible_prefix_gr_diagnostics(well, typewell, _parameters())
    legacy = prepare_particle_filter_inputs(well, typewell, _parameters())

    assert diagnostics["affine_gr_slope"] == pytest.approx(2.0)
    assert diagnostics["affine_gr_intercept"] == pytest.approx(5.0)
    assert diagnostics["affine_fallback_used"] is False
    assert diagnostics["gr_sigma"] == pytest.approx(legacy["gr_sigma"])
    assert diagnostics["observed_only_gr_sigma"] == pytest.approx(10.0)


@pytest.mark.parametrize("infinite_gr", [np.inf, -np.inf])
def test_visible_prefix_diagnostics_rejects_infinite_visible_gr(infinite_gr: float) -> None:
    well = _well_with_six_hidden_rows()
    well.loc[0, "GR"] = infinite_gr

    with pytest.raises(ValueError, match="可见 GR"):
        visible_prefix_gr_diagnostics(well, _typewell(), _parameters())


@pytest.mark.parametrize(
    "well, typewell",
    [
        (_well_with_six_hidden_rows().assign(GR=[20.0, np.nan, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]), _typewell()),
        (_well_with_six_hidden_rows(), pd.DataFrame({"TVT": [100.0, 101.0], "GR": [30.0, 30.0]})),
    ],
)
def test_visible_prefix_affine_fallback_is_finite_for_insufficient_support_or_constant_reference(
    well: pd.DataFrame, typewell: pd.DataFrame
) -> None:
    diagnostics = visible_prefix_gr_diagnostics(well, typewell, _parameters())

    assert diagnostics["affine_fallback_used"] is True
    assert diagnostics["affine_gr_slope"] == 1.0
    assert diagnostics["affine_gr_intercept"] == 0.0
    assert np.isfinite(diagnostics["gr_sigma"])
    assert np.isfinite(diagnostics["observed_only_gr_sigma"])


def test_diagnostics_ignore_hidden_true_tvt_and_surface() -> None:
    well = _well_with_six_hidden_rows()
    typewell = _typewell()
    baseline_hidden = hidden_gr_evidence_statistics(well)
    baseline_visible = visible_prefix_gr_diagnostics(well, typewell, _parameters())
    changed = well.copy()
    hidden = changed["TVT_input"].isna()
    changed.loc[hidden, "TVT"] = np.linspace(-1e9, 1e9, hidden.sum())
    changed["surface"] = np.where(hidden, -99999.0, 99999.0)

    assert hidden_gr_evidence_statistics(changed) == baseline_hidden
    assert visible_prefix_gr_diagnostics(changed, typewell, _parameters()) == baseline_visible


def test_pf_likelihood_audit_returns_exact_paths_and_well_level_report() -> None:
    likelihoods, report, paths = run_pf_likelihood_audit(
        _well_with_five_hidden_rows(), _typewell(), _small_pf_parameters()
    )

    assert likelihoods.shape == (3,)
    assert np.isfinite(likelihoods).all()
    assert report["hidden_rows"] == 5
    assert set(paths) == {
        "row_index",
        "last_visible_tvt",
        "pf128_mean_tvt",
        "pf128_mean_delta",
        "pf128_scale_3_delta",
        "pf128_scale_5_delta",
        "pf128_scale_8_delta",
        "pf128_scale_12_delta",
    }
    assert all(values.shape == (5,) for values in paths.values())


def test_pf_likelihood_audit_is_bitwise_repeatable_and_ignores_hidden_labels() -> None:
    well = _well_with_five_hidden_rows()
    baseline = run_pf_likelihood_audit(well, _typewell(), _small_pf_parameters())
    repeated = run_pf_likelihood_audit(well, _typewell(), _small_pf_parameters())
    changed = well.copy()
    hidden = changed["TVT_input"].isna()
    changed.loc[hidden, "TVT"] = np.linspace(-1e9, 1e9, hidden.sum())
    changed["surface"] = np.where(hidden, -99999.0, 99999.0)
    hidden_changed = run_pf_likelihood_audit(changed, _typewell(), _small_pf_parameters())

    np.testing.assert_array_equal(baseline[0], repeated[0])
    assert baseline[1] == repeated[1]
    for key in baseline[2]:
        np.testing.assert_array_equal(baseline[2][key], repeated[2][key])
    np.testing.assert_array_equal(baseline[0], hidden_changed[0])
    assert baseline[1] == hidden_changed[1]
    for key in baseline[2]:
        np.testing.assert_array_equal(baseline[2][key], hidden_changed[2][key])


def test_pf_likelihood_audit_uses_hidden_gr_and_matches_legacy_kernel() -> None:
    well = _well_with_five_hidden_rows()
    parameters = _small_pf_parameters()
    likelihoods, report, _ = run_pf_likelihood_audit(well, _typewell(), parameters)
    changed = well.copy()
    changed.loc[changed["TVT_input"].isna(), "GR"] += 100.0
    changed_likelihoods, _, _ = run_pf_likelihood_audit(changed, _typewell(), parameters)
    prepared = prepare_particle_filter_inputs(well, _typewell(), parameters)
    _, direct_likelihoods = particle_filter_all_seeds_numba(**_kernel_arguments(prepared))

    assert not np.array_equal(likelihoods, changed_likelihoods)
    np.testing.assert_array_equal(likelihoods, direct_likelihoods)
    assert report["pf_best_ll_per_row"] == pytest.approx(np.max(direct_likelihoods) / 5.0)
    assert report["pf_ll_spread"] == pytest.approx(np.std(direct_likelihoods))
    assert report["gr_sigma"] == pytest.approx(prepared["gr_sigma"])


def test_likelihood_cache_round_trip_has_only_the_declared_keys(tmp_path: Path) -> None:
    path = tmp_path / "likelihoods.npz"
    values = np.array([-3.0, -2.0, -1.0])
    write_likelihood_cache(path, values, "fingerprint-1")

    loaded = read_likelihood_cache(path, "fingerprint-1", expected_number_of_seeds=3)
    np.testing.assert_array_equal(loaded, values)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"seed_log_likelihoods", "cache_fingerprint"}


@pytest.mark.parametrize(
    ("expected_fingerprint", "expected_number_of_seeds"),
    [("wrong", 3), ("fingerprint-1", 2)],
)
def test_likelihood_cache_rejects_mismatched_metadata(
    tmp_path: Path, expected_fingerprint: str, expected_number_of_seeds: int
) -> None:
    path = tmp_path / "likelihoods.npz"
    write_likelihood_cache(path, np.array([-3.0, -2.0, -1.0]), "fingerprint-1")

    with pytest.raises(ValueError):
        read_likelihood_cache(path, expected_fingerprint, expected_number_of_seeds)


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_likelihood_cache_rejects_invalid_values(tmp_path: Path, values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        write_likelihood_cache(tmp_path / "likelihoods.npz", values, "fingerprint-1")


def test_likelihood_cache_rejects_corrupt_archive(tmp_path: Path) -> None:
    path = tmp_path / "likelihoods.npz"
    path.write_bytes(b"not an npz archive")

    with pytest.raises(ValueError):
        read_likelihood_cache(path, "fingerprint-1", expected_number_of_seeds=3)


def test_likelihood_cache_atomic_write_failure_keeps_no_valid_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "likelihoods.npz"

    def fail_savez(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", fail_savez)
    with pytest.raises(OSError, match="simulated write failure"):
        write_likelihood_cache(path, np.array([-3.0, -2.0, -1.0]), "fingerprint-1")
    assert not path.exists()
