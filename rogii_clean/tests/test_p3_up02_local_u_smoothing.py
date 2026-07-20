import numpy as np

from src.p3_up02_local_u_smoothing import (
    build_local_smoothing_candidates,
    smooth_u_on_md_grid,
)


def test_irregular_md_is_resampled_in_feet_and_preserves_linear_u() -> None:
    """反射高斯对远离边界的线性 U 不应制造局部弯曲。"""

    md = np.array([0.0, 0.7, 2.2, 3.1, 5.8, 8.0, 11.4, 15.0, 20.0])
    u = 100.0 + 0.03 * md

    smoothed = smooth_u_on_md_grid(md, u, smoothing_window_ft=4.0)

    assert smoothed.shape == u.shape
    assert np.isfinite(smoothed).all()
    # 反射边界只会影响两端；内部线性趋势应基本保持。
    np.testing.assert_allclose(smoothed[2:-2], u[2:-2], atol=0.015, rtol=0.0)


def test_each_well_must_be_smoothed_separately() -> None:
    """分别调用两口井时，第二口井的巨大 U 不得污染第一口井。"""

    md = np.arange(0.0, 21.0, 1.0)
    first_u = np.zeros_like(md)
    second_u = np.full_like(md, 1000.0)

    first_candidates = build_local_smoothing_candidates(
        md,
        first_u,
        smoothing_windows_ft=(250.0,),
        blend_fractions=(0.5,),
    )
    second_candidates = build_local_smoothing_candidates(
        md,
        second_u,
        smoothing_windows_ft=(250.0,),
        blend_fractions=(0.5,),
    )

    np.testing.assert_allclose(first_candidates["smooth250_blend50"], 0.0)
    np.testing.assert_allclose(second_candidates["smooth250_blend50"], 1000.0)


def test_candidate_order_and_blend_formula_are_fixed() -> None:
    md = np.arange(0.0, 1001.0, 1.0)
    u = 0.002 * md + 3.0 * np.sin(md / 15.0)

    candidates = build_local_smoothing_candidates(
        md,
        u,
        smoothing_windows_ft=(250.0, 500.0, 1000.0),
        blend_fractions=(0.25, 0.5, 0.75),
    )

    assert list(candidates) == [
        "smooth250_blend25",
        "smooth250_blend50",
        "smooth250_blend75",
        "smooth500_blend25",
        "smooth500_blend50",
        "smooth500_blend75",
        "smooth1000_blend25",
        "smooth1000_blend50",
        "smooth1000_blend75",
    ]
    smooth250 = smooth_u_on_md_grid(md, u, smoothing_window_ft=250.0)
    np.testing.assert_allclose(
        candidates["smooth250_blend25"],
        0.75 * u + 0.25 * smooth250,
    )


def test_non_monotonic_md_is_rejected() -> None:
    md = np.array([0.0, 2.0, 1.0])
    u = np.array([1.0, 2.0, 3.0])

    try:
        smooth_u_on_md_grid(md, u, smoothing_window_ft=250.0)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("非单调 MD 必须被拒绝")
