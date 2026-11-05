"""1-D thermal model along the Rotor axis, checked against analytical fins."""

import numpy as np
import pytest

from rotor_design.thermal import fin_profile

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
