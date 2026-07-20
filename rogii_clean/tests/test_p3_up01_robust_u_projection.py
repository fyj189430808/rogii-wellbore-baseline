import numpy as np

from src.p3_up01_robust_u_projection import (
    build_projection_candidates,
    robust_polynomial_projection,
)


def test_quadratic_projection_recovers_quadratic_u_with_one_outlier() -> None:
    md = np.linspace(1000.0, 2000.0, 101)
    x = 2.0 * (md - md[0]) / (md[-1] - md[0]) - 1.0
    true_u = 12000.0 + 30.0 * x - 5.0 * x * x
    noisy_u = true_u.copy()
    noisy_u[50] += 1000.0

    projected = robust_polynomial_projection(md, noisy_u, degree=2)

    normal_rows = np.arange(len(md)) != 50
    assert np.max(np.abs(projected[normal_rows] - true_u[normal_rows])) < 0.1


def test_candidate_fusion_uses_fixed_fractions_and_preserves_input() -> None:
    md = np.linspace(0.0, 10.0, 11)
    base_u = 3.0 + md + 0.2 * md**2 + np.sin(md)
    original = base_u.copy()

    candidates = build_projection_candidates(
        md,
        base_u,
        degrees=(2, 3),
        blend_fractions=(0.25, 0.50, 0.75),
    )

    assert np.array_equal(base_u, original)
    assert list(candidates) == [
        "degree2_blend25",
        "degree2_blend50",
        "degree2_blend75",
        "degree3_blend25",
        "degree3_blend50",
        "degree3_blend75",
    ]
    projected2 = robust_polynomial_projection(md, base_u, degree=2)
    assert np.allclose(candidates["degree2_blend25"], 0.75 * base_u + 0.25 * projected2)


def test_degenerate_md_falls_back_to_original_path() -> None:
    md = np.ones(10)
    base_u = np.linspace(1.0, 2.0, 10)

    projected = robust_polynomial_projection(md, base_u, degree=3)

    assert np.array_equal(projected, base_u)
