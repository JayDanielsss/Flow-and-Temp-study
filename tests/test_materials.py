"""Zirconia data reproduce the cited source values."""

import pytest

from rotor_design.materials import ZIRCONIA


def test_conductivity_follows_t_squared_law_at_lowest_temperatures():
    # Lawless & Gupta (1983): k ~ 1e-5 T^2 W/(cm K) = 1e-3 T^2 W/(m K).
    assert ZIRCONIA.conductivity(1.2) == pytest.approx(1.44e-3, rel=1e-6)


def test_conductivity_matches_measured_point_at_25k():
    # Cahill group data file ysz15.fin: 1.344 W/(m K) at 25.17 K.
    assert ZIRCONIA.conductivity(25.17) == pytest.approx(1.344, rel=1e-6)


def test_conductivity_multiplier_scales_every_temperature():
    assert ZIRCONIA.conductivity(5.0, multiplier=3.0) == pytest.approx(3.0 * ZIRCONIA.conductivity(5.0))
