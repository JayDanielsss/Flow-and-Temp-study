"""Rotordynamics checked against analytical limiting cases."""

import math

import pytest

from rotor_design.rotordynamics import (first_bending_mode_hz, half_frequency_whirl_threshold_hz,
                                        rigid_body_critical_speeds_hz, rotor_mass_properties)


def test_rigid_body_frequencies_are_sqrt_k_over_m_and_conical():
    # Symmetric stations, arms l = 10 mm, K = 1e6 N/m both stations, m = 2 g, I_t - I_p = 5e-8 kg m^2:
    #   bounce  sqrt(K / m) / 2 pi             = 3558.81 Hz
    #   conical sqrt(K l^2 / (I_t - I_p)) / 2 pi = 7116.14 Hz
    speeds = rigid_body_critical_speeds_hz(mass_kg=2e-3, transverse_inertia_kg_m2=7e-8, polar_inertia_kg_m2=2e-8,
                                           stiffness_n_m=1e6, lower_arm_m=0.01, upper_arm_m=0.01)
    assert speeds == pytest.approx((math.sqrt(1e6 / 2e-3) / (2 * math.pi),
                                    math.sqrt(1e6 * 0.01**2 / 5e-8) / (2 * math.pi)))


def test_disc_like_rotor_has_no_conical_critical_speed():
    # I_p >= I_t: gyroscopic stiffening keeps the forward conical whirl above the spin; only bounce remains.
    speeds = rigid_body_critical_speeds_hz(2e-3, 2e-8, 3e-8, 1e6, 0.01, 0.01)
    assert speeds == pytest.approx((math.sqrt(1e6 / 2e-3) / (2 * math.pi),))


def test_free_free_tube_first_bending_mode():
    # f = (4.73004)^2 / (2 pi L^2) sqrt(E I / (rho A)), tube 4 / 3.5 mm radius, L = 20 mm
    ro, ri, length, e, rho = 4e-3, 3.5e-3, 20e-3, 210e9, 6020.0
    i = math.pi * (ro**4 - ri**4) / 4
    area = math.pi * (ro**2 - ri**2)
    expected = 4.73004**2 / (2 * math.pi * length**2) * math.sqrt(e * i / (rho * area))
    assert first_bending_mode_hz(length, ro, ri, e, rho) == pytest.approx(expected, rel=1e-5)


def test_half_frequency_whirl_starts_at_twice_the_lowest_rigid_body_critical_speed():
    assert half_frequency_whirl_threshold_hz((900.0, 1500.0)) == 1800.0


def test_mass_properties_of_a_closed_tube():
    # Tube 4 / 3.5 mm radius, 1 mm Bottom Cap, 3 mm Turbine Cap, 20 mm long, rho = 6000.
    p = rotor_mass_properties(outer_radius_m=4e-3, inner_radius_m=3.5e-3, length_m=20e-3, bottom_cap_m=1e-3,
                              turbine_cap_m=3e-3, density_kg_m3=6000.0)
    tube = 6000.0 * math.pi * (4e-3**2 - 3.5e-3**2) * 16e-3
    bottom, top = (6000.0 * math.pi * 4e-3**2 * h for h in (1e-3, 3e-3))
    assert p.mass_kg == pytest.approx(tube + bottom + top)
    assert p.centre_of_mass_m == pytest.approx((bottom * 0.5e-3 + tube * 9e-3 + top * 18.5e-3) / p.mass_kg)
    assert p.polar_inertia_kg_m2 == pytest.approx(tube * (4e-3**2 + 3.5e-3**2) / 2 + (bottom + top) * 4e-3**2 / 2)
