"""Stress in the hollow rotating Rotor, checked against analytical limits."""

import math

import pytest

from rotor_design.stress import rotating_tube_stress

RHO, NU, SPIN = 6020.0, 0.3, 1000.0
OMEGA = 2 * math.pi * SPIN


def test_thin_wall_hoop_stress_tends_to_rho_v_squared():
    # Thin ring: sigma_theta = rho v^2, v = omega R = 2 pi 1000 * 4 mm = 25.13 m/s -> 3.802 MPa
    s = rotating_tube_stress(outer_radius_m=4e-3, inner_radius_m=3.9996e-3, spin_rate_hz=SPIN,
                             density_kg_m3=RHO, poissons_ratio=NU)
    assert s.peak_hoop_pa == pytest.approx(RHO * (OMEGA * 4e-3) ** 2, rel=1e-4)
    assert s.peak_radial_pa == pytest.approx(0.0, abs=1e-6 * s.peak_hoop_pa)


def test_thick_tube_peaks_match_lame_rotating_disc():
    # Plane-stress rotating annulus, a = 2 mm, b = 4 mm:
    #   hoop at the bore      rho w^2 / 4 ((3 + nu) b^2 + (1 - nu) a^2)
    #   radial at r = sqrt(a b)  (3 + nu) / 8 rho w^2 (b - a)^2
    a, b = 2e-3, 4e-3
    s = rotating_tube_stress(b, a, SPIN, RHO, NU)
    assert s.peak_hoop_pa == pytest.approx(RHO * OMEGA**2 / 4 * ((3 + NU) * b**2 + (1 - NU) * a**2))
    assert s.peak_radial_pa == pytest.approx((3 + NU) / 8 * RHO * OMEGA**2 * (b - a) ** 2)


def test_solid_disc_peaks_at_the_centre():
    # Solid disc: hoop = radial = (3 + nu) / 8 rho w^2 b^2 at r = 0
    s = rotating_tube_stress(4e-3, 0.0, SPIN, RHO, NU)
    expected = (3 + NU) / 8 * RHO * OMEGA**2 * 4e-3**2
    assert s.peak_hoop_pa == pytest.approx(expected)
    assert s.peak_radial_pa == pytest.approx(expected)
