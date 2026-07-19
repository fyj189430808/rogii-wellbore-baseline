"""P3-D01 Task 4A：开发井路径评分与统计分析测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_d01_diagnostic_analysis import (  # noqa: E402
    add_diagnostic_columns,
    aggregate_path_metrics,
    build_binned_metrics,
    correlation_tables,
    decide_pf_routes,
    score_well_paths,
    spearman_report,
)


PATH_NAMES = (
    "p2p02",
    "pf128_mean",
    "scale_3",
    "scale_5",
    "scale_8",
    "scale_12",
    "oracle_scale",
)


def _target_rows() -> pd.DataFrame:
    """构造故意乱序的两行真值，检验评分按 row_index 对齐。"""
    return pd.DataFrame(
        {
            "well_id": ["well_a", "well_a"],
            "fold": [2, 2],
            "row_index": [20, 10],
            "target_tvt": [14.0, 10.0],
            "pred_tvt": [16.0, 9.0],
        }
    )


def _legal_paths() -> pd.DataFrame:
    """构造与真值顺序相反、且四种温度误差可手算的合法路径。"""
    return pd.DataFrame(
        {
            "row_index": [10, 20],
            "last_visible_tvt": [8.0, 8.0],
            "pf128_mean_tvt": [10.0, 13.0],
            "pf128_scale_3_delta": [2.0, 6.0],
            "pf128_scale_5_delta": [1.0, 5.0],
            "pf128_scale_8_delta": [3.0, 7.0],
            "pf128_scale_12_delta": [0.0, 4.0],
        }
    )


def test_score_well_paths_uses_row_key_and_matches_hand_calculation() -> None:
    result = score_well_paths(_target_rows(), _legal_paths())

    assert result["well_id"] == "well_a"
    assert result["fold"] == 2
    assert result["hidden_rows"] == 2
    assert result["p2p02_sse"] == pytest.approx(5.0)
    assert result["p2p02_rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["pf128_mean_sse"] == pytest.approx(1.0)
    assert result["scale_3_sse"] == pytest.approx(0.0)
    assert result["scale_5_sse"] == pytest.approx(2.0)
    assert result["scale_8_sse"] == pytest.approx(2.0)
    assert result["scale_12_sse"] == pytest.approx(8.0)
    assert result["oracle_best_scale"] == 3
    assert result["oracle_scale_sse"] == pytest.approx(0.0)
    assert result["best_scale_is_oracle"] is True
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad_side", ["target_duplicate", "legal_duplicate", "missing_key", "extra_key"])
def test_score_well_paths_rejects_non_bijective_row_keys(bad_side: str) -> None:
    target = _target_rows()
    legal = _legal_paths()
    if bad_side == "target_duplicate":
        target.loc[1, "row_index"] = 20
    elif bad_side == "legal_duplicate":
        legal.loc[1, "row_index"] = 10
    elif bad_side == "missing_key":
        legal = legal.iloc[:1].copy()
    else:
        extra = legal.iloc[[0]].copy()
        extra.loc[:, "row_index"] = 30
        legal = pd.concat([legal, extra], ignore_index=True)

    with pytest.raises(ValueError):
        score_well_paths(target, legal)


@pytest.mark.parametrize(
    ("frame_name", "column", "bad_value"),
    [
        ("target", "target_tvt", np.nan),
        ("target", "pred_tvt", np.inf),
        ("legal", "pf128_mean_tvt", -np.inf),
        ("legal", "pf128_scale_8_delta", np.nan),
    ],
)
def test_score_well_paths_rejects_non_finite_values(
    frame_name: str,
    column: str,
    bad_value: float,
) -> None:
    target = _target_rows()
    legal = _legal_paths()
    selected = target if frame_name == "target" else legal
    selected.loc[selected.index[0], column] = bad_value

    with pytest.raises(ValueError):
        score_well_paths(target, legal)


def test_score_well_paths_breaks_oracle_temperature_ties_in_fixed_order() -> None:
    target = _target_rows()
    legal = _legal_paths()
    legal["pf128_scale_5_delta"] = legal["pf128_scale_3_delta"]

    result = score_well_paths(target, legal)

    assert result["scale_3_sse"] == result["scale_5_sse"]
    assert result["oracle_best_scale"] == 3


def test_aggregate_path_metrics_pools_sse_instead_of_averaging_well_rmse() -> None:
    rows = []
    for well_id, hidden_rows, sse in (("short", 1, 0.0), ("long", 9, 9.0)):
        row: dict[str, object] = {"well_id": well_id, "fold": 0, "hidden_rows": hidden_rows}
        for path_name in PATH_NAMES:
            row[f"{path_name}_sse"] = sse
            row[f"{path_name}_rmse"] = np.sqrt(sse / hidden_rows)
        rows.append(row)

    metrics = aggregate_path_metrics(pd.DataFrame(rows)).set_index("path_name")

    assert metrics.loc["p2p02", "pooled_rmse"] == pytest.approx(np.sqrt(9.0 / 10.0))
    assert metrics.loc["p2p02", "macro_well_rmse"] == pytest.approx(0.5)
    assert bool(metrics.loc["oracle_scale", "is_oracle_upper_bound"])
    assert not bool(metrics.loc["scale_3", "is_oracle_upper_bound"])


def _manual_permutation_p(x: np.ndarray, y: np.ndarray, permutations: int, seed: int) -> float:
    """用独立的简短实现手算置换比例，锁定随机数语义。"""
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    observed = float(np.corrcoef(x_rank, y_rank)[0, 1])
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted_rho = float(np.corrcoef(x_rank, generator.permutation(y_rank))[0, 1])
        exceedances += int(abs(permuted_rho) >= abs(observed))
    return (1 + exceedances) / (permutations + 1)


def test_spearman_report_is_exact_and_reproducible() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    expected_p = _manual_permutation_p(
        frame["x"].to_numpy(),
        frame["y"].to_numpy(),
        permutations=19,
        seed=71,
    )

    first = spearman_report(frame, "x", "y", permutations=19, seed=71)
    second = spearman_report(frame, "x", "y", permutations=19, seed=71)

    assert first == second
    assert first["valid"] is True
    assert first["rho"] == pytest.approx(1.0)
    assert first["permutation_p"] == pytest.approx(expected_p)
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}),
        pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 5.0]}),
    ],
)
def test_spearman_report_marks_unidentifiable_cases_invalid(frame: pd.DataFrame) -> None:
    report = spearman_report(frame, "x", "y", permutations=9, seed=4)

    assert report["valid"] is False
    assert report["rho"] is None
    assert report["permutation_p"] is None
    json.dumps(report, allow_nan=False)


def test_correlation_tables_compute_each_fold_and_count_matching_directions() -> None:
    rows = []
    for fold in range(5):
        for x_value in (1.0, 2.0, 3.0):
            y_value = x_value if fold < 4 else 4.0 - x_value
            rows.append({"fold": fold, "x": x_value, "y": y_value})
    frame = pd.DataFrame(rows)

    overall, per_fold = correlation_tables(frame, [("x", "y")], permutations=9, seed=22)

    assert len(overall) == 1
    assert overall.loc[0, "valid_folds"] == 5
    assert overall.loc[0, "same_direction_folds"] == 4
    assert per_fold.set_index("fold").loc[4, "rho"] == pytest.approx(-1.0)
    assert (per_fold.set_index("fold").loc[[0, 1, 2, 3], "rho"] == 1.0).all()
    repeated = correlation_tables(frame, [("x", "y")], permutations=9, seed=22)
    pdt.assert_frame_equal(overall, repeated[0])
    pdt.assert_frame_equal(per_fold, repeated[1])


def test_correlation_tables_keep_invalid_numeric_results_as_nan() -> None:
    frame = pd.DataFrame({"fold": [0, 0, 0], "x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]})

    overall, per_fold = correlation_tables(frame, [("x", "y")], permutations=9, seed=1)

    assert not bool(overall.loc[0, "valid"])
    assert np.isnan(overall.loc[0, "rho"])
    assert np.isnan(overall.loc[0, "permutation_p"])
    assert np.isnan(per_fold.loc[0, "rho"])


def _diagnostic_frame(number_of_wells: int = 16) -> pd.DataFrame:
    """生成包含评分、ESS、缺失率和 fold 的完整小型诊断表。"""
    rows = []
    for index in range(number_of_wells):
        hidden_rows = 10 + index
        pf_rmse = 1.0 + 0.1 * index
        p2_rmse = 1.2 + 0.1 * index
        observed_fraction = 0.95 - 0.02 * index
        row: dict[str, object] = {
            "well_id": f"well_{index:02d}",
            "fold": index % 5,
            "hidden_rows": hidden_rows,
            "observed_gr_fraction": observed_fraction,
            "longest_gr_gap_md_ft": float(index),
            "gr_sigma": 11.0,
            "observed_only_gr_sigma": 10.0,
            "pf128_mean_sse": pf_rmse**2 * hidden_rows,
            "pf128_mean_rmse": pf_rmse,
            "p2p02_sse": p2_rmse**2 * hidden_rows,
            "p2p02_rmse": p2_rmse,
        }
        for temperature in (3, 5, 8, 12):
            path_rmse = pf_rmse + 0.01 * temperature
            row[f"scale_{temperature}_sse"] = path_rmse**2 * hidden_rows
            row[f"scale_{temperature}_rmse"] = path_rmse
            row[f"scale_{temperature}_effective_sample_size"] = float(1 + index + temperature / 10)
        rows.append(row)
    return pd.DataFrame(rows)


def test_add_diagnostic_columns_uses_frozen_formulas() -> None:
    frame = _diagnostic_frame(4)
    frame.loc[0, "observed_gr_fraction"] = 0.25
    frame.loc[0, "gr_sigma"] = 12.0
    frame.loc[0, "observed_only_gr_sigma"] = 10.0
    frame.loc[0, "scale_3_effective_sample_size"] = 2.0
    frame.loc[0, "scale_3_rmse"] = 2.0
    frame.loc[0, "scale_12_rmse"] = 1.4
    frame.loc[0, "pf128_mean_rmse"] = 1.8

    result = add_diagnostic_columns(frame)

    assert result.loc[0, "missing_gr_fraction"] == pytest.approx(0.75)
    assert result.loc[0, "sigma_absolute_relative_difference"] == pytest.approx(0.2)
    assert bool(result.loc[0, "scale_3_collapsed_severe"])
    assert bool(result.loc[0, "scale_3_collapsed_medium"])
    assert bool(result.loc[0, "scale_3_easy_oversharp"])
    assert not bool(result.loc[3, "scale_3_collapsed_severe"])
    assert "missing_gr_fraction" not in frame.columns


@pytest.mark.parametrize("bad_fraction", [-0.01, 1.01, np.nan, np.inf])
def test_add_diagnostic_columns_rejects_invalid_observed_fraction(bad_fraction: float) -> None:
    frame = _diagnostic_frame(4)
    frame.loc[0, "observed_gr_fraction"] = bad_fraction

    with pytest.raises(ValueError):
        add_diagnostic_columns(frame)


@pytest.mark.parametrize("bad_ess", [0.99, 128.01, np.nan, np.inf])
def test_add_diagnostic_columns_rejects_ess_outside_particle_count(bad_ess: float) -> None:
    frame = _diagnostic_frame(4)
    frame.loc[0, "scale_5_effective_sample_size"] = bad_ess

    with pytest.raises(ValueError):
        add_diagnostic_columns(frame)


def test_build_binned_metrics_is_stable_under_input_shuffle() -> None:
    frame = add_diagnostic_columns(_diagnostic_frame(16))

    original = build_binned_metrics(frame).sort_values(["bin_column", "bin_number"]).reset_index(drop=True)
    shuffled = build_binned_metrics(frame.sample(frac=1.0, random_state=99)).sort_values(
        ["bin_column", "bin_number"]
    ).reset_index(drop=True)

    pdt.assert_frame_equal(original, shuffled)
    assert set(original["bin_column"]) == {
        "hidden_rows",
        "missing_gr_fraction",
        "longest_gr_gap_md_ft",
        "p2p02_rmse",
    }
    assert set(original["bin_number"]) == {1, 2, 3, 4}
    first_bin = original[
        (original["bin_column"] == "hidden_rows") & (original["bin_number"] == 1)
    ].iloc[0]
    selected = frame.sort_values(["hidden_rows", "well_id"]).head(4)
    expected_pf_rmse = np.sqrt(selected["pf128_mean_sse"].sum() / selected["hidden_rows"].sum())
    assert first_bin["pf128_mean_pooled_rmse"] == pytest.approx(expected_pf_rmse)
    assert first_bin["well_count"] == 4
    assert "scale_3_severe_collapse_rate" in original.columns
    assert "scale_12_medium_collapse_rate" in original.columns


def test_build_binned_metrics_handles_duplicate_index_and_repeated_values_stably() -> None:
    frame = add_diagnostic_columns(_diagnostic_frame(16))
    frame["hidden_rows"] = np.repeat([10, 20, 30, 40], 4)
    # 同一个 index 标签故意跨越四个分组，旧实现会让后写入的分组覆盖前面的行。
    frame.index = np.tile(np.arange(4), 4)

    original = build_binned_metrics(frame).sort_values(["bin_column", "bin_number"]).reset_index(drop=True)
    shuffled = build_binned_metrics(frame.sample(frac=1.0, random_state=13)).sort_values(
        ["bin_column", "bin_number"]
    ).reset_index(drop=True)

    pdt.assert_frame_equal(original, shuffled)
    hidden_bins = original.loc[original["bin_column"] == "hidden_rows"]
    assert hidden_bins["bin_number"].tolist() == [1, 2, 3, 4]
    assert hidden_bins["well_count"].tolist() == [4, 4, 4, 4]


def _supporting_correlations() -> pd.DataFrame:
    rows = [
        {
            "x_column": "missing_gr_fraction",
            "y_column": "pf128_mean_rmse",
            "valid": True,
            "rho": 0.30,
            "permutation_p": 0.01,
            "same_direction_folds": 4,
            "valid_folds": 5,
        }
    ]
    for temperature in (3, 5, 8, 12):
        rows.append(
            {
                "x_column": f"scale_{temperature}_effective_sample_size",
                "y_column": "hidden_rows",
                "valid": True,
                "rho": -0.30 if temperature == 3 else -0.10,
                "permutation_p": 0.01 if temperature == 3 else 0.50,
                "same_direction_folds": 4,
                "valid_folds": 5,
            }
        )
    return pd.DataFrame(rows)


def _path_metrics_with_oracle_gap() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path_name": ["scale_3", "scale_5", "scale_8", "scale_12", "oracle_scale"],
            "pooled_rmse": [10.0, 9.8, 9.9, 10.1, 9.4],
        }
    )


def test_decide_pf_routes_applies_preregistered_thresholds_and_is_json_safe() -> None:
    per_well = add_diagnostic_columns(_diagnostic_frame(40))

    decision = decide_pf_routes(per_well, _supporting_correlations(), _path_metrics_with_oracle_gap())

    assert decision["pf01_supported"] is True
    assert decision["pf01"]["correlation_condition"]["passed"] is True
    assert decision["pf01"]["sigma_difference_condition"]["median"] == pytest.approx(0.1)
    assert decision["pf02_supported"] is True
    assert decision["pf02"]["ess_correlation_condition"]["passed"] is True
    assert decision["pf02"]["oracle_gap_condition"]["gap_ft"] == pytest.approx(0.4)
    assert decision["scope"] == "whole_well_only"
    json.dumps(decision, allow_nan=False)


def test_decide_pf_routes_does_not_pass_invalid_correlations_or_small_oracle_gap() -> None:
    per_well = add_diagnostic_columns(_diagnostic_frame(20))
    per_well["gr_sigma"] = per_well["observed_only_gr_sigma"]
    per_well = add_diagnostic_columns(per_well.drop(columns=[
        "missing_gr_fraction",
        "sigma_absolute_relative_difference",
        *[f"scale_{temperature}_collapsed_severe" for temperature in (3, 5, 8, 12)],
        *[f"scale_{temperature}_collapsed_medium" for temperature in (3, 5, 8, 12)],
        "scale_3_easy_oversharp",
    ]))
    correlations = _supporting_correlations()
    correlations.loc[:, "valid"] = False
    correlations.loc[:, "rho"] = np.nan
    correlations.loc[:, "permutation_p"] = np.nan
    correlations.loc[:, "same_direction_folds"] = 0
    metrics = _path_metrics_with_oracle_gap()
    metrics.loc[metrics["path_name"] == "oracle_scale", "pooled_rmse"] = 9.7

    decision = decide_pf_routes(per_well, correlations, metrics)

    assert decision["pf01"]["correlation_condition"]["passed"] is False
    assert decision["pf01"]["sigma_difference_condition"]["passed"] is False
    assert decision["pf02"]["ess_correlation_condition"]["passed"] is False
    assert decision["pf02"]["oracle_gap_condition"]["passed"] is False
    assert decision["pf02_supported"] is False
    json.dumps(decision, allow_nan=False)


def test_decide_pf_routes_rejects_duplicate_path_metric_names() -> None:
    per_well = add_diagnostic_columns(_diagnostic_frame(20))
    metrics = _path_metrics_with_oracle_gap()
    duplicate_irrelevant_path = pd.DataFrame(
        {"path_name": ["p2p02", "p2p02"], "pooled_rmse": [10.0, 10.0]}
    )
    metrics = pd.concat([metrics, duplicate_irrelevant_path], ignore_index=True)

    with pytest.raises(ValueError):
        decide_pf_routes(per_well, _supporting_correlations(), metrics)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("rho", np.nan),
        ("rho", np.inf),
        ("rho", 1.01),
        ("permutation_p", np.nan),
        ("permutation_p", -0.01),
        ("permutation_p", 1.01),
        ("same_direction_folds", -1),
        ("same_direction_folds", 6),
        ("same_direction_folds", 1.5),
    ],
)
def test_decide_pf_routes_rejects_malformed_valid_correlation_rows(
    column: str,
    bad_value: float,
) -> None:
    per_well = add_diagnostic_columns(_diagnostic_frame(20))
    correlations = _supporting_correlations()
    if column == "same_direction_folds":
        correlations[column] = correlations[column].astype(float)
    correlations.loc[0, column] = bad_value

    with pytest.raises(ValueError):
        decide_pf_routes(per_well, correlations, _path_metrics_with_oracle_gap())


def test_decide_pf_routes_safely_nulls_nonfinite_values_on_invalid_correlations() -> None:
    per_well = add_diagnostic_columns(_diagnostic_frame(20))
    correlations = _supporting_correlations()
    correlations.loc[0, "valid"] = False
    correlations.loc[0, "rho"] = np.inf
    correlations.loc[0, "permutation_p"] = np.nan
    correlations.loc[0, "same_direction_folds"] = 0

    decision = decide_pf_routes(per_well, correlations, _path_metrics_with_oracle_gap())

    candidate = decision["pf01"]["correlation_condition"]["candidates"][0]
    assert candidate["rho"] is None
    assert candidate["permutation_p"] is None
    json.dumps(decision, allow_nan=False)


def test_decide_pf_routes_rejects_empty_per_well_table_explicitly() -> None:
    empty = add_diagnostic_columns(_diagnostic_frame(4)).iloc[:0]

    with pytest.raises(ValueError, match="不能为空"):
        decide_pf_routes(empty, _supporting_correlations(), _path_metrics_with_oracle_gap())
