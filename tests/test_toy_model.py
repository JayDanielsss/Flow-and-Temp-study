"""Toy model: one simple, fully specified Design run end to end through evaluate(),
checked against hand calculations written out below.

Room-temperature nitrogen everywhere (Drive Gas, the gas the Rotor turns in, the Stator
and the Turbine Cap all at 300 K, 1 bar), so the gas is ideal to about 0.1 % and the
windage gap is in the continuum. Frictionless two-center pivots and no windage heat
into the Rotor leave the drive chain as the only thing doing work.

    Rotor:   20 mm long, 6 mm diameter (r1 = 3 mm), 0.5 mm wall (bore a = 2.5 mm), zirconia
    Stator:  0.1 mm radial gap (r2 = 3.1 mm)
    Drive:   4 Nozzles, 0.05 mm throat, 20 deg from the rim tangent, Cd 0.8, Fluted Cap,
             ideal Euler torque (efficiency 1), Ducted to 1 bar; 1000 Hz, omega = 6283.19 rad/s
    Gas:     nitrogen, R = 8.314462618 / 0.02801348 = 296.80 J/(kg K), gamma = 1.4,
             mu(300 K) = 17.877 uPa s (CoolProp; Incropera Table A.4 gives 17.82, see test_gas)
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.bearing import TWO_CENTER_PIVOT, TwoCenterPivots
from rotor_design.evaluate import DUCTED, PA_PER_BAR, PA_PER_TORR, VAPOUR_ONLY, Design, evaluate
from rotor_design.turbine import FLUTED_CAP

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"
GAS_REL = 5e-3  # ideal-gas hand calculation against real-gas properties

TOY_DESIGN = Design(
    rotor_length_mm=20.0, rotor_diameter_mm=6.0, wall_thickness_mm=0.5, nozzle_count=4,
    nozzle_throat_mm=0.05, nozzle_angle_deg=20.0, drive_gas="nitrogen", drive_gas_temperature_k=300.0,
    bearing=TWO_CENTER_PIVOT, pivot_to_bath=False, cooling_path=VAPOUR_ONLY, turbine=FLUTED_CAP,
    exhaust_path=DUCTED, back_pressure_bar=1.0,
)


def _toy_assumptions(**overrides):
    a = load_assumptions(ASSUMPTIONS_FILE)
    toy = dict(
        environment_gas="nitrogen", environment_temperature_k=300.0, stator_temperature_k=300.0,
        environment_pressure_torr=PA_PER_BAR / PA_PER_TORR, target_spin_rate_hz=1000.0,
        discharge_coefficient=0.8, turbine_efficiency=1.0, rotor_stator_gap_mm=0.1,
        windage_heat_to_rotor_fraction=0.0, rotor_emissivity=0.0,
        bearings={**a.bearings, TWO_CENTER_PIVOT: TwoCenterPivots(friction_coefficient=0.0, preload_n=0.05,
                                                                  tip_radius_mm=0.5)},
    )
    return replace(a, **{**toy, **overrides})


@pytest.fixture(scope="module")
def toy():
    return evaluate(TOY_DESIGN, _toy_assumptions())


def test_windage_is_laminar_taylor_couette(toy):
    # T_c = 4 pi mu L omega r1^2 r2^2 / (r2^2 - r1^2)
    #     = 4 pi * 17.877e-6 * 0.02 * 6283.19 * (9e-6 * 9.61e-6 / 0.61e-6) = 4.0027e-6 N m
    # Free-molecular limit p omega r1 / sqrt(2 pi R T) * 2 pi r1 L * r1 = 2.850e-3 N m, joined
    # harmonically: 1 / (1 / 4.0027e-6 + 1 / 2.850e-3) = 3.9971e-6 N m. Frictionless pivots add nothing.
    assert toy.windage_regime == "continuum"
    assert toy.bearing_friction_torque_n_m == 0.0
    assert toy.drag_torque_n_m == pytest.approx(3.9971e-6, rel=GAS_REL)


def test_choked_jet_and_euler_torque_per_unit_flow(toy):
    # Choked: v = sqrt(2 gamma R T0 / (gamma + 1)) = sqrt(2 * 1.4 * 296.80 * 300 / 2.4) = 322.31 m/s
    # Rim speed U = omega r1 = 18.850 m/s
    # Fluted Cap: T / m = r1 (v cos 20 - U) = 3e-3 * (302.87 - 18.85) = 0.85206 N m s/kg
    assert toy.nozzle_choked
    assert toy.jet_velocity_m_s == pytest.approx(322.31, rel=GAS_REL)
    assert toy.turbine_torque_per_flow_n_m_s_kg == pytest.approx(0.85206, rel=GAS_REL)


def test_torque_balance_sets_mass_flow_and_supply_pressure(toy):
    # Turbine torque = drag:  m = 3.9971e-6 / 0.85206 = 4.6911e-6 kg/s over all four Nozzles
    # Choked flux per unit area and p0: psi = sqrt(gamma / (R T0)) (2 / (gamma + 1))^3 = 2.2947e-3 s/m
    # A = pi (0.05e-3)^2 / 4 = 1.9635e-9 m^2
    # p0 = m / (N Cd A psi) = 4.6911e-6 / (4 * 0.8 * 1.9635e-9 * 2.2947e-3) = 3.2536 bar,
    # above 1 bar / 0.5283 = 1.893 bar, so the Nozzles are choked as assumed.
    assert toy.turbine_torque_n_m == pytest.approx(toy.drag_torque_n_m, rel=1e-5)
    assert toy.required_mass_flow_kg_s == pytest.approx(4.6911e-6, rel=GAS_REL)
    assert toy.required_supply_pressure_bar == pytest.approx(3.2536, rel=GAS_REL)


def test_stress_is_the_rotating_annulus(toy):
    # rho omega^2 = 6020 * 6283.19^2 = 2.3766e11 kg/(m^3 s^2)
    # hoop at the bore: rho omega^2 / 4 ((3 + nu) b^2 + (1 - nu) a^2)
    #   = 2.3766e11 / 4 * (3.3 * 9e-6 + 0.7 * 6.25e-6) = 2.0246 MPa; safety factor 550 / 2.0246 = 271.7
    # radial at sqrt(a b): (3 + nu) / 8 rho omega^2 (b - a)^2 = 0.024509 MPa
    assert toy.peak_hoop_stress_mpa == pytest.approx(2.0246, rel=1e-4)
    assert toy.peak_radial_stress_mpa == pytest.approx(0.024509, rel=1e-4)
    assert toy.stress_safety_factor == pytest.approx(271.66, rel=1e-4)


def test_bending_is_the_free_free_tube(toy):
    # f = (beta L)^2 / (2 pi L^2) sqrt(E I / (rho A)), beta L = 4.7300
    # I = pi (b^4 - a^4) / 4 = 3.2938e-11 m^4, A = pi (b^2 - a^2) = 8.6394e-6 m^2
    # f = 22.373 / (2 pi * 4e-4) * sqrt(210e9 * 3.2938e-11 / (6020 * 8.6394e-6)) = 102,661 Hz
    assert toy.bending_critical_hz == pytest.approx(102_661, rel=1e-4)


def test_with_no_heat_sources_the_whole_rotor_stays_at_the_common_temperature(toy):
    # Stator, Turbine Cap and gas all at 300 K; nothing does work on the Rotor.
    # The whole body below the Turbine Cap (20 - 3 = 17 mm) is the Sample Zone.
    assert np.allclose(toy.temperature_k, 300.0, atol=1e-6)
    assert toy.sample_zone_mm == pytest.approx(17.0, rel=1e-6)
    assert toy.sample_heat_load_w == pytest.approx(0.0, abs=1e-9)


def test_radiation_between_rotor_and_stator_at_one_temperature_carries_no_net_heat():
    hot = evaluate(TOY_DESIGN, _toy_assumptions(rotor_emissivity=0.9))
    assert np.allclose(hot.temperature_k, 300.0, atol=1e-6)
    assert hot.sample_heat_load_breakdown_w["radiation"] == pytest.approx(0.0, abs=1e-9)


def test_a_warmer_stator_radiates_into_the_rotor():
    # Stator 10 K above everything else: radiation now flows in, and only in.
    base = evaluate(TOY_DESIGN, _toy_assumptions(stator_temperature_k=310.0))
    hot = evaluate(TOY_DESIGN, _toy_assumptions(stator_temperature_k=310.0, rotor_emissivity=0.9))
    assert base.sample_heat_load_breakdown_w["radiation"] == 0.0
    assert hot.sample_heat_load_breakdown_w["radiation"] > 0.0
    assert max(hot.temperature_k) > max(base.temperature_k)
