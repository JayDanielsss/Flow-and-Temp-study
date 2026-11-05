"""Turbine torque, checked against hand-worked velocity triangles."""

import math

import pytest

from rotor_design.turbine import FlutedCap, RadialInflow


def test_fluted_cap_euler_torque_for_simple_velocity_triangle():
    # Jet 200 m/s at 60 deg from the tangent: c_theta1 = 200 cos 60 = 100 m/s.
    # Rim radius 4 mm at 1000 Hz: U = 2 pi * 0.004 * 1000 = 25.133 m/s.
    # Straight flutes: gas leaves with c_theta2 = U.
    # T = m r (c_theta1 - U) = 1e-4 * 0.004 * (100 - 25.133) = 2.9947e-5 N m
    torque = FlutedCap().torque(1e-4, 200.0, 60.0, 0.004, 1000.0)
    assert torque == pytest.approx(2.9947e-5, rel=1e-4)


def test_radial_inflow_euler_torque_for_simple_velocity_triangle():
    # Same jet and rim: inlet r1 c_theta1 = 0.004 * 100 = 0.4 m^2/s.
    # Hub exit 3 mm diameter, r_e = 1.5 mm; gas leaves axially with no swirl relative
    # to the Rotor, so its mean angular momentum over the exit disc is omega r_e^2 / 2
    # = 2 pi * 1000 * 1.5e-3^2 / 2 = 7.0686e-3 m^2/s.
    # T = m (r1 c_theta1 - omega r_e^2 / 2) = 1e-4 * (0.4 - 7.0686e-3) = 3.9293e-5 N m
    torque = RadialInflow(hub_exit_diameter_mm=3.0).torque(1e-4, 200.0, 60.0, 0.004, 1000.0)
    assert torque == pytest.approx(3.9293e-5, rel=1e-4)


def test_radial_inflow_with_vanishing_hub_exit_discharges_no_swirl():
    # r_e -> 0: T = m r1 c_theta1 = 1e-4 * 0.004 * 100 = 4e-5 N m.
    torque = RadialInflow(hub_exit_diameter_mm=0.0).torque(1e-4, 200.0, 60.0, 0.004, 1000.0)
    assert torque == pytest.approx(4e-5, rel=1e-9)


def test_efficiency_scales_the_ideal_euler_torque():
    ideal = RadialInflow(hub_exit_diameter_mm=3.0).torque(1e-4, 200.0, 60.0, 0.004, 1000.0)
    assert RadialInflow(hub_exit_diameter_mm=3.0).torque(1e-4, 200.0, 60.0, 0.004, 1000.0,
                                                         efficiency=0.6) == pytest.approx(0.6 * ideal)


def test_hub_exit_takes_its_area_out_of_the_turbine_cap():
    # Fluted Cap uses the whole cap face: pi * 4^2 = 50.265 mm^2.
    # Radial-Inflow blades fill the ring outside the hub exit: pi (4^2 - 1.5^2) = 43.197 mm^2.
    assert FlutedCap().blade_area_m2(0.004) == pytest.approx(math.pi * 0.004**2)
    assert RadialInflow(hub_exit_diameter_mm=3.0).blade_area_m2(0.004) == pytest.approx(43.197e-6, rel=1e-4)
    # A hub exit as wide as the cap leaves no blade area.
    assert RadialInflow(hub_exit_diameter_mm=10.0).blade_area_m2(0.004) == 0.0
