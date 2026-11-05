"""Turbine torque, checked against hand-worked velocity triangles."""

import pytest

from rotor_design.turbine import fluted_cap_torque


def test_fluted_cap_euler_torque_for_simple_velocity_triangle():
    # Jet 200 m/s at 60 deg from the tangent: c_theta1 = 200 cos 60 = 100 m/s.
    # Rim radius 4 mm at 1000 Hz: U = 2 pi * 0.004 * 1000 = 25.133 m/s.
    # Straight flutes: gas leaves with c_theta2 = U.
    # T = m r (c_theta1 - U) = 1e-4 * 0.004 * (100 - 25.133) = 2.9947e-5 N m
    torque = fluted_cap_torque(1e-4, 200.0, 60.0, 0.004, 1000.0)
    assert torque == pytest.approx(2.9947e-5, rel=1e-4)
