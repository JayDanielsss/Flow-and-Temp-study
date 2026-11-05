"""Windage drag, checked against continuum and free-molecular limits."""

import pytest

from rotor_design.drag import cold_support_shear_torque, windage_torque

# Rotor radius 4 mm, gap 0.1 mm, length 20 mm, 1000 Hz.
R, GAP, L, HZ = 0.004, 0.1e-3, 0.02, 1000.0


def test_continuum_limit_matches_laminar_taylor_couette():
    # Helium at 10 K and 1 bar, mu = 2.102 uPa s (Hurly & Moldover 2000, Table A1).
    # T = 4 pi mu L omega r1^2 r2^2 / (r2^2 - r1^2), r2 = r1 + gap  ->  1.10219e-6 N m
    w = windage_torque(R, L, GAP, HZ, 10.0, 1e5)
    assert w.regime == "continuum"
    assert w.torque_n_m == pytest.approx(1.10219e-6, rel=1e-3)


def test_free_molecular_limit_matches_kinetic_theory():
    # Helium at 1.2 K and 1e-4 Pa (Kn about 2000), full accommodation:
    # tau = p U / sqrt(2 pi R T), U = omega r1;  T = tau * 2 pi r1 L * r1  ->  4.03779e-11 N m
    w = windage_torque(R, L, GAP, HZ, 1.2, 1e-4)
    assert w.regime == "free-molecular"
    assert w.torque_n_m == pytest.approx(4.03779e-11, rel=1e-3)


def test_vapour_at_0_2_torr_is_in_transition_regime():
    # Kn = 0.80 um / 0.1 mm = 0.008 at 1.2 K, 0.2 Torr; a 50 um gap gives Kn = 0.016.
    w = windage_torque(R, L, 50e-6, HZ, 1.2, 26.6645)
    assert w.regime == "transition"
    assert w.knudsen_number == pytest.approx(0.0159, rel=1e-2)


def test_cold_support_shear_continuum_limit_matches_narrow_gap_couette():
    # Disc of radius R turning a narrow gap g from a still plate, helium at 10 K and 1 bar:
    # tau = mu omega r / g, T = integral tau r 2 pi r dr = pi mu omega R^4 / (2 g)
    #   mu = 2.102 uPa s, R = 4 mm, g = 10 um, 1000 Hz  ->  5.31095e-7 N m
    s = cold_support_shear_torque(R, 10e-6, HZ, 10.0, 1e5)
    assert s.regime == "continuum"
    assert s.torque_n_m == pytest.approx(5.31095e-7, rel=1e-3)


def test_cold_support_shear_free_molecular_limit_matches_kinetic_theory():
    # tau = p omega r / sqrt(2 pi R T), T = pi p omega R^4 / (2 sqrt(2 pi R T))
    #   1.2 K, 1e-4 Pa, sqrt(2 pi R T) = 125.149 m/s  ->  2.01889e-12 N m
    s = cold_support_shear_torque(R, GAP, HZ, 1.2, 1e-4)
    assert s.regime == "free-molecular"
    assert s.torque_n_m == pytest.approx(2.01889e-12, rel=1e-3)
