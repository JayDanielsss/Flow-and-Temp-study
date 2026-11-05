"""Gas public functions, checked against tabulated and limiting values."""

import pytest

from rotor_design.gas import (gap_heat_transfer_coefficient, knudsen_number, mean_free_path,
                              vapour_thermal_conductivity, vapour_viscosity)


def test_vapour_viscosity_at_tabulated_point():
    # Hurly & Moldover (2000), J. Res. NIST 105, 667, Table A1: 0.3395 uPa s at 1.2 K.
    assert vapour_viscosity(1.2) == pytest.approx(0.3395e-6, rel=1e-4)


def test_mean_free_path_from_hard_sphere_kinetic_theory():
    # lambda = (mu / p) * sqrt(pi * R * T / 2), R = 8.314462618 / 0.004002602 J/(kg K)
    # 1.2 K, 26.6645 Pa (0.2 Torr), mu = 0.3395e-6 Pa s:
    #   sqrt(pi * 2077.26 * 1.2 / 2) = 62.562 m/s
    #   lambda = 0.3395e-6 / 26.6645 * 62.562 = 0.7966e-6 m
    assert mean_free_path(1.2, 26.6645) == pytest.approx(0.7966e-6, rel=1e-3)


def test_knudsen_number_is_mean_free_path_over_length():
    assert knudsen_number(1.2, 26.6645, 0.1e-3) == pytest.approx(0.7966e-2, rel=1e-3)


def test_vapour_thermal_conductivity_at_tabulated_point():
    # Hurly & Moldover (2000), Table A1: 2.713 mW/(m K) at 1.2 K.
    assert vapour_thermal_conductivity(1.2) == pytest.approx(2.713e-3, rel=1e-4)


def test_gap_conduction_continuum_limit_is_k_over_gap():
    # 10 K, 1 bar, 0.1 mm gap: h = 16.44 mW/(m K) / 0.1 mm = 164.4 W/(m^2 K)
    assert gap_heat_transfer_coefficient(10.0, 1e5, 0.1e-3) == pytest.approx(164.4, rel=1e-3)


def test_gap_conduction_free_molecular_limit_matches_kinetic_theory():
    # h = (gamma + 1) / (gamma - 1) * p * sqrt(R / (8 pi T)), full accommodation
    #   1.2 K, 1e-4 Pa: 4 * 1e-4 * sqrt(2077.26 / (8 pi 1.2)) = 3.3197e-3 W/(m^2 K)
    assert gap_heat_transfer_coefficient(1.2, 1e-4, 0.1e-3) == pytest.approx(3.3197e-3, rel=1e-3)
