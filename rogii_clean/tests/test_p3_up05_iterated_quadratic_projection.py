import numpy as np

from src.p3_up01_robust_u_projection import robust_polynomial_projection
from src.p3_up05_iterated_quadratic_projection import iterated_quadratic_tvt


def test_iterated_projection_is_fixed_half_blend_in_u() -> None:
    md = np.linspace(1000.0, 2000.0, 101)
    z = -7000.0 + 0.2 * (md - md[0])
    base_u = 300.0 + 0.04 * md + 2.0 * np.sin(md / 35.0)
    up03_tvt = base_u - z
    projected_u = robust_polynomial_projection(md, base_u, degree=2)

    candidate_tvt = iterated_quadratic_tvt(md, z, up03_tvt)

    expected_tvt = 0.5 * up03_tvt + 0.5 * (projected_u - z)
    np.testing.assert_allclose(candidate_tvt, expected_tvt, rtol=0.0, atol=1.0e-12)


def test_exact_quadratic_u_is_unchanged() -> None:
    md = np.linspace(0.0, 100.0, 51)
    z = -1000.0 + md
    u = 10.0 + 0.2 * md + 0.003 * md**2
    up03_tvt = u - z

    candidate_tvt = iterated_quadratic_tvt(md, z, up03_tvt)

    np.testing.assert_allclose(candidate_tvt, up03_tvt, rtol=0.0, atol=1.0e-9)
