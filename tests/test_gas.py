"""Gas public functions, checked against tabulated and limiting values."""

import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from rotor_design.gas import (AIR, GASES, HELIUM, NITROGEN, gap_heat_transfer_coefficient, knudsen_number,
                              mean_free_path, stagnation_state, vapour_enthalpy, vapour_thermal_conductivity,
                              vapour_viscosity)


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


def test_vapour_enthalpy_difference_in_the_dilute_limit_is_five_halves_r_delta_t():
    # Dilute monatomic gas: cp = 5/2 R. 10 K to 1.2 K at 0.2 Torr (26.6645 Pa):
    #   2.5 * 2077.26 J/(kg K) * 8.8 K = 45,700 J/kg
    drop = vapour_enthalpy(10.0, 26.6645) - vapour_enthalpy(1.2, 26.6645)
    assert drop == pytest.approx(45_700, rel=1e-3)


def test_gases_carry_specific_gas_constant_and_dilute_cp():
    # R = 8.314462618 / M: helium 2077.26, nitrogen 296.80, air 287.05 J/(kg K)
    assert set(GASES) == {"helium", "nitrogen", "air"}
    assert HELIUM.specific_gas_constant == pytest.approx(2077.26, rel=1e-5)
    assert NITROGEN.specific_gas_constant == pytest.approx(296.80, rel=1e-4)
    assert AIR.specific_gas_constant == pytest.approx(287.05, rel=1e-4)
    # cp = gamma / (gamma - 1) R: 5/2 R for helium, 7/2 R for the diatomic gases
    assert HELIUM.cp_dilute == pytest.approx(2.5 * 2077.26, rel=1e-5)
    assert NITROGEN.cp_dilute == pytest.approx(3.5 * 296.80, rel=1e-4)


@pytest.mark.parametrize("gas, mu, k", [
    # Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table A.4, 1 atm, 300 K.
    (NITROGEN, 178.2e-7, 25.9e-3),
    (AIR, 184.6e-7, 26.3e-3),
])
def test_diatomic_transport_at_300k_matches_tabulated_values(gas, mu, k):
    assert vapour_viscosity(300.0, gas) == pytest.approx(mu, rel=1e-2)
    assert vapour_thermal_conductivity(300.0, gas) == pytest.approx(k, rel=1e-2)


def test_diatomic_enthalpy_difference_near_ideal_is_seven_halves_r_delta_t():
    # Nitrogen at 1 bar, 300 K to 200 K: 3.5 * 296.80 * 100 = 103,880 J/kg (ideal to about 0.5 %)
    drop = vapour_enthalpy(300.0, 1e5, NITROGEN) - vapour_enthalpy(200.0, 1e5, NITROGEN)
    assert drop == pytest.approx(103_880, rel=5e-3)


@pytest.mark.parametrize("call", [
    lambda: vapour_viscosity(10.0, NITROGEN),
    lambda: vapour_thermal_conductivity(50.0, AIR),
    lambda: vapour_enthalpy(1.2, 26.6645, NITROGEN),
    lambda: stagnation_state(10.0, 1e5, AIR),
    lambda: vapour_viscosity(0.9),
])
def test_properties_below_a_gas_data_range_raise(call):
    with pytest.raises(ValueError, match="No .* property data below"):
        call()


def test_helium_transport_properties_have_no_step_where_the_table_hands_over_to_coolprop():
    # Hurly & Moldover's table ends at 10 K; CoolProp, which differs there by a few percent, is
    # scaled to meet it and joins unscaled by 20 K.
    for prop in (vapour_viscosity, vapour_thermal_conductivity):
        assert prop(10.0 + 1e-9) == pytest.approx(prop(10.0), rel=1e-6)
    assert vapour_viscosity(25.0) == pytest.approx(PropsSI("V", "T", 25.0, "P", 100.0, "Helium"), rel=1e-12)
    temperatures = np.array([5.0, 10.0, 15.0, 25.0])
    assert vapour_viscosity(temperatures) == pytest.approx([vapour_viscosity(t) for t in temperatures], rel=1e-12)
