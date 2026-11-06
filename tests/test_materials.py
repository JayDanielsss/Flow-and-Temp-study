"""Zirconia data reproduce the cited source values."""

import pytest

from rotor_design.materials import ZIRCONIA, helium_vapour_pressure


def test_conductivity_follows_t_squared_law_at_lowest_temperatures():
    # Lawless & Gupta (1983): k ~ 1e-5 T^2 W/(cm K) = 1e-3 T^2 W/(m K).
    assert ZIRCONIA.conductivity(1.2) == pytest.approx(1.44e-3, rel=1e-6)


def test_conductivity_matches_measured_point_at_25k():
    # Cahill group data file ysz15.fin: 1.344 W/(m K) at 25.17 K.
    assert ZIRCONIA.conductivity(25.17) == pytest.approx(1.344, rel=1e-6)


def test_conductivity_multiplier_scales_every_temperature():
    assert ZIRCONIA.conductivity(5.0, multiplier=3.0) == pytest.approx(3.0 * ZIRCONIA.conductivity(5.0))


@pytest.mark.parametrize("temperature_k, pressure_pa", [
    # Donnelly & Barenghi (1998), Table 17.3, recommended saturated vapour pressure of
    # helium-4 on T90. Below 1.25 K the low-temperature relation, above it the ITS-90 equation.
    (0.65, 1.101e-1), (0.80, 1.475), (1.00, 1.558e1), (1.10, 3.802e1), (1.20, 8.152e1),
    (1.25, 1.147e2), (1.30, 1.579e2), (1.50, 4.715e2), (2.00, 3.130e3), (2.15, 4.716e3),
    (2.20, 5.335e3), (3.00, 2.405e4), (4.20, 9.923e4), (5.00, 1.960e5),
])
def test_helium_vapour_pressure_matches_tabulated_points(temperature_k, pressure_pa):
    # The table gives four figures; the equations reproduce it to better than 0.1 %.
    assert helium_vapour_pressure(temperature_k) == pytest.approx(pressure_pa, rel=1e-3)


def test_helium_vapour_pressure_at_1k_is_about_0_12_torr():
    assert helium_vapour_pressure(1.0) / 133.322 == pytest.approx(0.117, abs=1e-3)


def test_helium_vapour_pressure_is_continuous_across_its_ranges():
    for t in (1.25, 2.1768):
        assert helium_vapour_pressure(t - 1e-9) == pytest.approx(helium_vapour_pressure(t + 1e-9), rel=1e-3)


@pytest.mark.parametrize("temperature_k", [0.5, 5.3])
def test_helium_vapour_pressure_outside_the_curve_is_rejected(temperature_k):
    with pytest.raises(ValueError, match="vapour-pressure curve"):
        helium_vapour_pressure(temperature_k)
