"""Bearing types, checked against Petroff and Hertz hand calculations."""

import math

import pytest

from rotor_design.bearing import (MECHANICAL_WARNING, GasBearing, MechanicalBearing, TwoCenterPivots,
                                  pivot_contact_conductance)
from rotor_design.materials import ZIRCONIA

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


def test_pivot_friction_matches_hertz_hand_calculation():
    # Zirconia sphere tip R = 0.5 mm on a zirconia flat, W = 0.1 N, mu = 0.1:
    #   E* = 210 GPa / (2 (1 - 0.3^2))              = 115.385 GPa
    #   a  = (3 W R / (4 E*))^(1/3)                 = 6.87534 um
    #   T  = (3 pi / 16) mu W a, per pivot          = 4.04991e-8 N m
    #   heat at the lower pivot, T omega            = 2.54463e-4 W
    #   stiffness 2 E* a per pivot                  = 1.58662e6 N/m
    p = TwoCenterPivots(friction_coefficient=0.1, preload_n=0.1, tip_radius_mm=0.5).performance(**ROTOR)
    assert p.contact_radius_m == pytest.approx(6.87534e-6, rel=1e-4)
    assert p.friction_torque_n_m == pytest.approx(2 * 4.04991e-8, rel=1e-4)
    assert p.bottom_friction_heat_w == pytest.approx(4.04991e-8 * 2 * math.pi * 1000, rel=1e-4)
    assert p.stiffness_n_m == pytest.approx(2 * 1.58662e6, rel=1e-4)
    assert p.radial_space_mm == 0.0
    assert p.bottom_gas_heat_w == 0.0


def test_rotor_weight_is_shared_between_the_two_stations():
    preload_only = TwoCenterPivots(0.1, 0.1, 0.5).performance(**ROTOR)
    weighted = TwoCenterPivots(0.1, 0.0, 0.5).performance(**{**ROTOR, "rotor_weight_n": 0.2})
    assert weighted.friction_torque_n_m == pytest.approx(preload_only.friction_torque_n_m)


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


def test_pivot_contact_conductance_is_constriction_of_two_half_spaces():
    # G = 2 k a, k at the mean temperature: k(1.2 K) = 1e-3 * 1.2^2 = 1.44e-3 W/(m K), a = 10 um
    assert pivot_contact_conductance(10e-6, 1.2, 1.2) == pytest.approx(2.88e-8)
    assert pivot_contact_conductance(10e-6, 1.4, 1.0) == pytest.approx(2 * ZIRCONIA.conductivity(1.2) * 10e-6)
