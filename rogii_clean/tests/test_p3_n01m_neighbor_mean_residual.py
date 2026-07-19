from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_n01m_neighbor_mean_residual import (  # noqa: E402
    aggregate_neighbor_mean,
    build_within_fold_derangement,
    compute_cached_trajectory_geometry,
    select_top_neighbors,
    summarize_source_signal,
    trajectory_direction,
    weighted_median,
)
from src.p3_n01a_neighbor_residual_audit import (  # noqa: E402
    compute_trajectory_geometry,
    geometry_is_eligible,
    neighbor_weight,
)
from scripts.run_p3_n01m_neighbor_mean_residual import (  # noqa: E402
    SourceRecord,
    _all_eligible_neighbors,
    _build_source_pair_signal,
    _candidate_neighbors,
    _read_selected_outer_scoring_predictions,
    _resolve_output_dir,
    _run_scope_config,
)


class CountingSpatialIndex:
    def __init__(self, xy: np.ndarray) -> None:
        self.tree = cKDTree(xy)
        self.calls = 0

    def query(self, *args, **kwargs):
        self.calls += 1
        return self.tree.query(*args, **kwargs)


def test_weighted_median_uses_first_value_reaching_half_weight() -> None:
    assert weighted_median(np.array([10.0, -2.0, 3.0]), np.array([1.0, 3.0, 2.0])) == -2.0


def test_neighbor_mean_uses_eta_equal_total_weight_over_weight_plus_one() -> None:
    result = aggregate_neighbor_mean(
        residuals=np.array([-4.0, 2.0, 8.0]),
        weights=np.array([1.0, 2.0, 1.0]),
    )

    assert result.weighted_median_residual == 2.0
    assert result.total_weight == 4.0
    assert result.eta == 4.0 / 5.0
    assert result.predicted_offset_ft == 1.6


def test_no_neighbors_returns_exact_zero() -> None:
    result = aggregate_neighbor_mean(np.array([]), np.array([]))

    assert result.predicted_offset_ft == 0.0
    assert result.total_weight == 0.0
    assert result.eta == 0.0


def test_derangement_is_deterministic_has_no_fixed_points_and_stays_within_fold() -> None:
    sources = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d", "e", "f"],
            "fold": [1, 1, 1, 2, 2, 2],
        }
    )

    first = build_within_fold_derangement(sources, seed=20260719)
    second = build_within_fold_derangement(sources.sample(frac=1.0), seed=20260719)

    pd.testing.assert_frame_equal(first, second)
    assert (first["source_well_id"] != first["donor_well_id"]).all()
    fold_by_well = sources.set_index("well_id")["fold"].to_dict()
    assert all(
        fold_by_well[source] == fold_by_well[donor]
        for source, donor in first[["source_well_id", "donor_well_id"]].itertuples(index=False)
    )


def test_top8_sort_is_weight_then_distance_then_well_id_and_excludes_self() -> None:
    candidates = pd.DataFrame(
        {
            "source_well_id": ["target", "z", "b", "a", "c", "d", "e", "f", "g", "h", "i"],
            "weight": [99.0, 0.5, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "distance_ft": [1.0, 1.0, 100.0, 100.0, 50.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
        }
    )

    selected = select_top_neighbors(candidates, target_well_id="target", maximum_neighbors=8)

    assert len(selected) == 8
    assert "target" not in set(selected["source_well_id"])
    assert selected["source_well_id"].tolist()[:3] == ["c", "a", "b"]


def test_source_signal_keeps_all_eligible_pairs_before_formal_top8() -> None:
    target_xy = np.array([[0.0, 0.0], [1000.0, 0.0]])
    sources = {
        f"s{number:02d}": SourceRecord(
            well_id=f"s{number:02d}",
            fold=1 + number % 4,
            hidden_rows=2,
            hidden_xy=np.array([[0.0, float(number + 1)], [1000.0, float(number + 1)]]),
            bbox=(0.0, float(number + 1), 1000.0, float(number + 1)),
            mean_residual=float(number),
            shuffled_residual=float(-number),
            typewell_tail_fingerprint="same",
            spatial_index=CountingSpatialIndex(
                np.array([[0.0, float(number + 1)], [1000.0, float(number + 1)]])
            ),
            direction=np.array([1.0, 0.0]),
        )
        for number in range(10)
    }

    all_eligible = _all_eligible_neighbors("target", target_xy, "same", sources)
    formal = _candidate_neighbors("target", target_xy, "same", sources)
    for source in sources.values():
        source.spatial_index.calls = 0
    source_signal = _build_source_pair_signal(sources)

    assert len(all_eligible) == 10
    assert len(formal) == 8
    assert set(formal["source_well_id"]).issubset(set(all_eligible["source_well_id"]))
    assert len(source_signal) == 90
    assert {
        "distance_ft",
        "azimuth_difference_deg",
        "parallel_overlap_ft",
        "same_typewell_tail",
        "weight",
        "target_m",
        "neighbor_m",
        "shuffled_neighbor_m",
        "real_absdiff",
        "shuffled_absdiff",
    }.issubset(source_signal.columns)
    assert sum(source.spatial_index.calls for source in sources.values()) == 45


def test_cached_geometry_matches_old_geometry_in_both_directions() -> None:
    x_a = np.linspace(0.0, 1000.0, 21)
    a = np.column_stack([x_a, np.zeros_like(x_a)])
    x_b = np.linspace(1100.0, 100.0, 21)
    b = np.column_stack([x_b, 50.0 + 0.1 * x_b])
    direction_a = trajectory_direction(a)
    direction_b = trajectory_direction(b)
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)

    old_ab = compute_trajectory_geometry(a, b)
    old_ba = compute_trajectory_geometry(b, a)
    new_ab = compute_cached_trajectory_geometry(
        a, direction_a, b, direction_b, tree_b
    )
    symmetric_distance = float(tree_b.query(a, k=1)[0].min())
    new_ba = compute_cached_trajectory_geometry(
        b,
        direction_b,
        a,
        direction_a,
        tree_a,
        minimum_distance_ft=symmetric_distance,
    )

    for old, new in [(old_ab, new_ab), (old_ba, new_ba)]:
        assert old.minimum_distance_ft == new.minimum_distance_ft
        assert old.azimuth_difference_deg == new.azimuth_difference_deg
        assert old.parallel_overlap_ft == new.parallel_overlap_ft
        assert old.reverse_source_controls == new.reverse_source_controls
        assert geometry_is_eligible(old) == geometry_is_eligible(new)
        assert neighbor_weight(
            old.minimum_distance_ft,
            old.azimuth_difference_deg,
            old.parallel_overlap_ft,
            True,
        ) == neighbor_weight(
            new.minimum_distance_ft,
            new.azimuth_difference_deg,
            new.parallel_overlap_ft,
            True,
        )


def test_unordered_source_pairs_match_old_directed_full_pair_rows() -> None:
    trajectories = {
        "a": np.array([[0.0, 0.0], [1000.0, 0.0]]),
        "b": np.array([[1100.0, 100.0], [100.0, 0.0]]),
        "c": np.array([[100.0, 250.0], [1100.0, 250.0]]),
    }
    sources = {
        well_id: SourceRecord(
            well_id=well_id,
            fold=1,
            hidden_rows=2,
            hidden_xy=xy,
            bbox=(
                float(xy[:, 0].min()),
                float(xy[:, 1].min()),
                float(xy[:, 0].max()),
                float(xy[:, 1].max()),
            ),
            mean_residual=float(number + 1),
            shuffled_residual=float(-(number + 1)),
            typewell_tail_fingerprint="same" if number < 2 else "different",
            spatial_index=CountingSpatialIndex(xy),
            direction=trajectory_direction(xy),
        )
        for number, (well_id, xy) in enumerate(trajectories.items())
    }
    expected_rows = []
    for target_id in sorted(sources):
        target = sources[target_id]
        for neighbor_id in sorted(sources):
            if target_id == neighbor_id:
                continue
            neighbor = sources[neighbor_id]
            geometry = compute_trajectory_geometry(target.hidden_xy, neighbor.hidden_xy)
            if not geometry_is_eligible(geometry):
                continue
            same_typewell = (
                target.typewell_tail_fingerprint == neighbor.typewell_tail_fingerprint
            )
            expected_rows.append(
                {
                    "target_source_well_id": target_id,
                    "neighbor_source_well_id": neighbor_id,
                    "distance_ft": geometry.minimum_distance_ft,
                    "azimuth_difference_deg": geometry.azimuth_difference_deg,
                    "parallel_overlap_ft": geometry.parallel_overlap_ft,
                    "same_typewell_tail": same_typewell,
                    "weight": neighbor_weight(
                        geometry.minimum_distance_ft,
                        geometry.azimuth_difference_deg,
                        geometry.parallel_overlap_ft,
                        same_typewell,
                    ),
                    "target_m": target.mean_residual,
                    "neighbor_m": neighbor.mean_residual,
                    "shuffled_neighbor_m": neighbor.shuffled_residual,
                    "real_absdiff": abs(target.mean_residual - neighbor.mean_residual),
                    "shuffled_absdiff": abs(
                        target.mean_residual - neighbor.shuffled_residual
                    ),
                }
            )
    expected = pd.DataFrame(expected_rows).sort_values(
        ["target_source_well_id", "neighbor_source_well_id"]
    ).reset_index(drop=True)
    actual = _build_source_pair_signal(sources).sort_values(
        ["target_source_well_id", "neighbor_source_well_id"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert sum(source.spatial_index.calls for source in sources.values()) == 3


def test_source_signal_metrics_use_frozen_non_overlapping_slices() -> None:
    pairs = pd.DataFrame(
        {
            "distance_ft": [400.0, 700.0, 1500.0, 2000.0],
            "azimuth_difference_deg": [10.0, 20.0, 30.0, 5.0],
            "same_typewell_tail": [True, False, True, False],
            "real_absdiff": [1.0, 3.0, 2.0, 4.0],
            "shuffled_absdiff": [4.0, 5.0, 6.0, 8.0],
        }
    )

    metrics = summarize_source_signal(pairs)

    assert metrics["distance_slices"]["le_500ft"]["pairs"] == 1
    assert metrics["distance_slices"]["500_to_1000ft"]["pairs"] == 1
    assert metrics["distance_slices"]["1000_to_2500ft"]["pairs"] == 2
    assert metrics["azimuth_slices"]["le_15deg"]["pairs"] == 2
    assert metrics["azimuth_slices"]["15_to_45deg"]["pairs"] == 2
    assert metrics["typewell_slices"]["same_template"]["real_mean_absdiff"] == 1.5
    assert metrics["typewell_slices"]["different_template"]["shuffled_median_absdiff"] == 6.5


def test_smoke_forces_an_independent_output_directory_and_separates_contract() -> None:
    base = Path("artifacts/P3_N01m_neighbor_mean_residual_v1")

    assert _resolve_output_dir(base, None) == base
    assert _resolve_output_dir(base, 3) == base / "_smoke_max_wells_3"
    scope = _run_scope_config(max_wells=3, actual_wells=3, actual_rows=12)
    assert scope["frozen"] == {"outer_wells": 131, "outer_rows": 651881}
    assert scope["actual"] == {
        "mode": "smoke",
        "max_wells": 3,
        "outer_wells": 3,
        "outer_rows": 12,
    }


def test_selected_outer_target_reader_uses_only_requested_wells(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir=CLEAN_ROOT / "tests") as temporary_directory:
        path = Path(temporary_directory) / "outer.parquet"
        full = pd.DataFrame(
            {
                "well_id": ["a", "a", "b", "b"],
                "fold": [0, 0, 0, 0],
                "row_index": [1, 2, 1, 2],
                "md": [10.0, 11.0, 20.0, 21.0],
                "pred_tvt": [100.0, 101.0, 200.0, 201.0],
                "target_tvt": [100.5, 101.5, 200.5, 201.5],
            }
        )
        full.to_parquet(path, index=False)
        selected_feature = full.loc[full["well_id"].eq("b")].drop(columns="target_tvt")

        def fail_full_pandas_read(*args, **kwargs):
            raise AssertionError("不得用 pandas 先读取完整 outer0 再筛井")

        monkeypatch.setattr(pd, "read_parquet", fail_full_pandas_read)
        selected = _read_selected_outer_scoring_predictions(
            path, selected_feature, selected_wells=["b"]
        )

        assert set(selected["well_id"]) == {"b"}
        assert len(selected) == 2
