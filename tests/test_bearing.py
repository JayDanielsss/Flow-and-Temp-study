"""Bearing types, checked against Petroff and Hertz hand calculations."""

import math

import pytest

from rotor_design.bearing import (MECHANICAL_WARNING, GasBearing, MechanicalBearing, TwoCenterPivots,
                                  cone_seat_radial_limit_n, unbalance_force_n)

# Rotor radius 4 mm at 1000 Hz in 0.2 Torr vapour at 1.2 K.
ROTOR = dict(rotor_radius_m=0.004, rotor_weight_n=0.0, spin_rate_hz=1000.0, bath_temperature_k=1.2,
             ambient_pressure_pa=26.6645, accommodation=1.0)


def _gas_bearing(**overrides):
    values = dict(journal_length_mm=3.0, clearance_mm=0.01, gas_temperature_k=10.0, supply_pressure_bar=2.0,
                  mass_flow_kg_s=1e-7, stiffness_coefficient=0.5, radial_space_mm=1.0)
    return GasBearing(**{**values, **overrides})


def test_gas_bearing_friction_matches_petroff():
    # Petroff, per journal: T = 2 pi mu R^3 L omega / c
    #   mu(10 K) = 2.102 uPa s, R = 4 mm, L = 3 mm, c = 10 um, omega = 2 pi 1000
    #   -> 1.59329e-6 N m, heat T omega = 10.0109 mW; two journals in the set.
    p = _gas_bearing().performance(**ROTOR)
    assert p.friction_torque_n_m == pytest.approx(2 * 1.59329e-6, rel=1e-4)
    assert p.bottom_friction_heat_w == pytest.approx(10.0109e-3, rel=1e-4)
    assert p.radial_space_mm == 1.0
    # K = 2 journals * 0.5 * (2e5 - 26.66 Pa) * 3 mm * 8 mm / 10 um = 4.79936e5 N/m
    assert p.stiffness_n_m == pytest.approx(4.79936e5, rel=1e-4)
    assert p.warning == ""


def test_bearing_gas_only_ever_adds_heat():
    warm = _gas_bearing(gas_temperature_k=10.0).performance(**ROTOR)
    cold = _gas_bearing(gas_temperature_k=1.0).performance(**ROTOR)
    assert warm.bottom_gas_heat_w > 0.0
    assert cold.bottom_gas_heat_w == 0.0
    # Bounded by the enthalpy the lower journal's gas carries above the Rotor:
    #   (1e-7 / 2) kg/s * 5193.2 J/(kg K) * 8.8 K = 2.285 mW
    assert warm.bottom_gas_heat_w <= 1e-7 / 2 * 5193.2 * 8.8 * (1 + 1e-4)


def test_unbalance_force_is_m_e_omega_squared():
    # 3 g Rotor, 1 um off its axis, 1000 Hz: 0.003 * 1e-6 * (2 pi 1000)^2 = 0.118435 N
    assert unbalance_force_n(0.003, 1e-6, 1000.0) == pytest.approx(0.118435, rel=1e-5)
    assert unbalance_force_n(0.003, 1e-6, 2000.0) == pytest.approx(4 * 0.118435, rel=1e-5)


def test_cone_seat_reacts_radial_load_up_to_half_the_axial_load_over_tan_alpha():
    # Ring normal force per radian: P / (2 pi sin a) from the axial load, F cos(phi) / (pi cos a)
    # from a radial load; the ring unloads at F = P / (2 tan a). 45 deg: F = P / 2; 60 deg: P / (2 sqrt 3).
    assert cone_seat_radial_limit_n(1.0, 45.0) == pytest.approx(0.5)
    assert cone_seat_radial_limit_n(1.0, 60.0) == pytest.approx(1.0 / (2.0 * math.sqrt(3.0)))
    # Ring force balance done numerically: at the limit the least normal force is zero.
    alpha, axial = math.radians(30.0), 0.7
    radial = cone_seat_radial_limit_n(axial, 30.0)
    phi = [2 * math.pi * i / 3600 for i in range(3600)]
    per_radian = [axial / (2 * math.pi * math.sin(alpha)) - radial * math.cos(f) / (math.pi * math.cos(alpha))
                  for f in phi]
    assert min(per_radian) == pytest.approx(0.0, abs=1e-12)
    assert sum(n * math.sin(alpha) for n in per_radian) * (2 * math.pi / 3600) == pytest.approx(axial)
    assert sum(n * math.cos(alpha) * -math.cos(f) for n, f in zip(per_radian, phi)) * (2 * math.pi / 3600) \
        == pytest.approx(radial)


PIVOTS = TwoCenterPivots(friction_coefficient=0.1, ball_radius_mm=0.5, cone_half_angle_deg=45.0, preload_margin=2.0)
# 3 g Rotor with its centre of mass midway between the seats, Balance Grade 1 um, no weight.
UNBALANCED = dict(ROTOR, rotor_mass_kg=0.003, lower_arm_m=0.01, upper_arm_m=0.01, tilt_angle_deg=0.0,
                  balance_grade_m=1e-6)


def test_pivot_preload_is_sized_from_the_unbalance_load_and_friction_follows_from_it():
    #   unbalance 0.118435 N, half at each seat              = 0.0592176 N
    #   least Preload 2 tan(45) * 0.0592176                   = 0.118435 N; x2 margin = 0.236871 N
    #   friction per seat mu (P / sin a) R cos a = mu P R cot a = 0.1 * 0.236871 * 0.5e-3 = 1.18435e-5 N m
    #   ring radius R cos 45                                  = 0.353553 mm
    p = PIVOTS.performance(**UNBALANCED)
    assert p.unbalance_force_n == pytest.approx(0.118435, rel=1e-5)
    assert p.preload_n == pytest.approx(0.236871, rel=1e-5)
    assert p.friction_torque_n_m == pytest.approx(2 * 1.18435e-5, rel=1e-5)
    assert p.bottom_friction_heat_w == pytest.approx(1.18435e-5 * 2 * math.pi * 1000, rel=1e-5)
    assert p.contact_radius_m == pytest.approx(0.353553e-3, rel=1e-5)
    assert p.radial_space_mm == 0.0 and p.bottom_gas_heat_w == 0.0
    # Each seat is left holding its radial share with the margin.
    assert cone_seat_radial_limit_n(p.preload_n, 45.0) == pytest.approx(2.0 * 0.0592176, rel=1e-5)


def test_tilt_angle_weight_adds_a_radial_share_and_bears_axially_on_the_lower_seat():
    # Weight 0.03 N at the Tilt Angle: 0.03 sin(54.7356) = 0.0244949 N across the axis on top of the
    # unbalance; the 0.03 cos = 0.0173205 N along it bears on the lower seat, adding to its friction.
    tilted = PIVOTS.performance(**{**UNBALANCED, "rotor_weight_n": 0.03, "tilt_angle_deg": 54.7356})
    radial = 0.118435 + 0.0244949
    assert tilted.preload_n == pytest.approx(2.0 * radial, rel=1e-5)
    lower_axial = tilted.preload_n + 0.0173205
    assert tilted.bottom_friction_heat_w == pytest.approx(0.1 * lower_axial * 0.5e-3 * 2 * math.pi * 1000, rel=1e-5)


def test_cone_seat_gives_radial_stiffness_that_grows_with_preload():
    soft = PIVOTS.performance(**UNBALANCED)
    stiff = TwoCenterPivots(0.1, 0.5, 45.0, 4.0).performance(**UNBALANCED)
    assert 1e6 < soft.stiffness_n_m < stiff.stiffness_n_m
    # No unbalance and no weight: nothing to react, no Preload, no friction.
    idle = PIVOTS.performance(**{**UNBALANCED, "balance_grade_m": 0.0})
    assert idle.preload_n == 0.0 and idle.friction_torque_n_m == 0.0


def test_mechanical_bearing_friction_and_warning():
    # T = mu P d_m / 2 per bearing, d_m = 8 mm + 2 mm radial space: 0.01 * 1 N * 10 mm / 2 = 5e-5 N m
    p = MechanicalBearing(friction_coefficient=0.01, preload_n=1.0, stiffness_n_m=1e7,
                          radial_space_mm=2.0).performance(**ROTOR)
    assert p.friction_torque_n_m == pytest.approx(2 * 5e-5)
    assert p.bottom_friction_heat_w == pytest.approx(5e-5 * 2 * math.pi * 1000)
    assert p.stiffness_n_m == 2e7
    assert p.radial_space_mm == 2.0
    assert p.warning == MECHANICAL_WARNING
    for words in ("unlubricated", "1.2 K", "60,000 rpm"):
        assert words in p.warning
