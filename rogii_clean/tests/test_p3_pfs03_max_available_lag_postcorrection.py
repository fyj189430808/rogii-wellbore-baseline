import numpy as np

from src.p3_pfs03_max_available_lag_postcorrection import (
    build_fixed_candidate,
    choose_dynamic_pfs_absolute,
)


def test_choose_longest_lag_with_complete_future_evidence_at_boundaries():
    remaining_md = np.array([1001.0, 1000.0, 999.9, 500.0, 499.9, 250.0, 249.9, 0.0])
    pf_scale8 = np.full(8, 8.0)
    lag250 = np.full(8, 250.0)
    lag500 = np.full(8, 500.0)
    lag1000 = np.full(8, 1000.0)

    selected, selected_lag = choose_dynamic_pfs_absolute(
        remaining_md, pf_scale8, lag250, lag500, lag1000
    )

    np.testing.assert_array_equal(
        selected,
        np.array([1000.0, 1000.0, 500.0, 500.0, 250.0, 250.0, 8.0, 8.0]),
    )
    np.testing.assert_array_equal(
        selected_lag,
        np.array([1000, 1000, 500, 500, 250, 250, 0, 0]),
    )


def test_fixed_candidate_uses_locked_quarter_correction_from_p3b00():
    up01 = np.array([10.0, 20.0])
    p3b00 = np.array([8.0, 18.0])
    dynamic_pfs = np.array([12.0, 14.0])

    actual = build_fixed_candidate(up01, p3b00, dynamic_pfs, correction_fraction=0.25)

    np.testing.assert_allclose(actual, np.array([11.0, 19.0]), rtol=0.0, atol=0.0)

