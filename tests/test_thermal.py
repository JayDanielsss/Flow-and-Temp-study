"""1-D thermal model along the Rotor axis, checked against analytical fins."""

import numpy as np
import pytest

from rotor_design.thermal import fin_profile, solve_tridiagonal

L = 0.02
K_A = 1e-6     # W m / K
H_P = 1e-2     # W / (m K)  ->  m = sqrt(hP / kA) = 100 1/m, mL = 2


def test_uniform_fin_with_adiabatic_tip_matches_analytical_solution():
    # theta(z) / theta_L = cosh(m z) / cosh(m L), tip at z = 0, base at z = L.
    #   T(0)   = 1.2 + 8.8 / cosh 2          = 3.53906 K
    #   T(L/2) = 1.2 + 8.8 cosh 1 / cosh 2   = 4.80936 K
    #   Q_base = kA m theta_L tanh(m L)      = 8.48344e-4 W
    fin = fin_profile(L, lambda z, t: K_A, lambda z, t: H_P, 1.2, 10.0)
    assert fin.temperature_k[0] == pytest.approx(3.53906, rel=1e-3)
    assert np.interp(L / 2, fin.z_m, fin.temperature_k) == pytest.approx(4.80936, rel=1e-3)
    assert fin.base_heat_w == pytest.approx(8.48344e-4, rel=1e-3)


def test_zero_heat_input_gives_uniform_bath_temperature():
    fin = fin_profile(L, lambda z, t: K_A, lambda z, t: H_P, 1.2, 1.2)
    assert np.allclose(fin.temperature_k, 1.2, atol=1e-12)
    assert fin.base_heat_w == pytest.approx(0.0, abs=1e-15)


def test_fin_cooled_at_tip_matches_analytical_convective_tip():
    # Tip conductance G to the sink at z = 0, base at z = L. With G = kA m the
    # convective-tip solution reduces to theta(z) / theta_L = exp(m (z - L)):
    #   T(0)   = 1.2 + 8.8 exp(-2)  = 2.39095 K
    #   Q_base = kA m theta_L       = 8.8e-4 W, of which G theta(0) leaves through the tip
    g_tip = K_A * 100.0
    fin = fin_profile(L, lambda z, t: K_A, lambda z, t: H_P, 1.2, 10.0, tip_conductance=lambda t: g_tip)
    assert fin.temperature_k[0] == pytest.approx(2.39095, rel=1e-3)
    assert fin.base_heat_w == pytest.approx(8.8e-4, rel=1e-3)
    assert fin.tip_heat_out_w == pytest.approx(g_tip * 8.8 * np.exp(-2.0), rel=1e-3)


def test_heat_injected_at_insulated_tip_conducts_out_through_base():
    # No side cooling: T(0) = T_base + Q L / kA = 1.2 + 1e-6 * 0.02 / 1e-6 = 1.22 K
    fin = fin_profile(L, lambda z, t: K_A, lambda z, t: 0.0, 1.2, 1.2, tip_heat_w=1e-6)
    assert fin.temperature_k[0] == pytest.approx(1.22, rel=1e-6)
    assert fin.base_heat_w == pytest.approx(-1e-6, rel=1e-6)


def test_tridiagonal_solve_matches_a_dense_solve():
    rng = np.random.default_rng(0)
    n = 50
    lower, upper = -rng.uniform(0.1, 1.0, n - 1), -rng.uniform(0.1, 1.0, n - 1)
    diag = rng.uniform(2.5, 3.0, n)   # diagonally dominant, as the fin's matrix is
    b = rng.normal(size=n)
    dense = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    assert solve_tridiagonal(lower, diag, upper, b) == pytest.approx(np.linalg.solve(dense, b), rel=1e-12)
