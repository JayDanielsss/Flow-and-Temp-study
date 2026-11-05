"""Geometry public functions, checked against hand calculations only."""

import pytest

from rotor_design.geometry import StatorAllowance, max_rotor_diameter, projected_width

TILT_ANGLE_DEG = 54.7356  # arccos(1/sqrt(3))

PLACEHOLDER_ALLOWANCE = StatorAllowance(
    wall=1.5,
    nozzle_plenum=2.0,
    housing_above_turbine_cap=3.0,
    cold_support_plate=2.0,
    probe_wall_clearance=0.5,
)


def test_projected_width_of_10mm_square_stator_at_tilt_angle():
    # Hand calculation: 10 * sin(54.7356 deg) + 10 * cos(54.7356 deg)
    #                 = 8.16497 + 5.77350 = 13.93847 mm
    assert projected_width(10.0, 10.0, TILT_ANGLE_DEG) == pytest.approx(13.9385, abs=1e-3)


def test_largest_rotor_diameter_at_20mm_rotor_length():
    # Hand calculation with the placeholder Stator Allowance:
    #   Stator length   = 20 + 3.0 + 2.0              = 25 mm
    #   usable bore     = 30 - 2 * 0.5                = 29 mm
    #   Stator diameter = (29 - 25 * 0.816497) / 0.577350 = 14.8742 mm
    #   Rotor diameter  = 14.8742 - 2 * (1.5 + 2.0)   = 7.8742 mm
    d = max_rotor_diameter(20.0, PLACEHOLDER_ALLOWANCE, 30.0, TILT_ANGLE_DEG)
    assert d == pytest.approx(7.874, abs=1e-3)
