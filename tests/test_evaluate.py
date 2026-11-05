"""Evaluate one Design: Design plus assumptions in, Results out."""

import math
from pathlib import Path

import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import COLD_SUPPORT, VAPOUR_ONLY, Design, evaluate

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"


@pytest.fixture
def assumptions():
    return load_assumptions(ASSUMPTIONS_FILE)


def test_assumptions_file_holds_spec_placeholders(assumptions):
    a = assumptions
    assert a.sample_zone_delta_t_k == 0.1
    assert a.min_sample_zone_mm == 5.0
    assert a.bath_cooling_power_w == 0.100
    assert a.stator_allowance.wall == 1.5
    assert a.stator_allowance.nozzle_plenum == 2.0
    assert a.stator_allowance.housing_above_turbine_cap == 3.0
    assert a.stator_allowance.cold_support_plate == 2.0
    assert a.stator_allowance.probe_wall_clearance == 0.5
    assert a.environment_temperature_k == 1.2
    assert a.environment_pressure_torr == 0.2
    assert a.drive_gas_temperature_k == 10.0
    assert a.max_supply_pressure_bar == 3.77
    assert a.sample_internal_heat_load_w == 0.0
    assert a.target_spin_rate_hz == 1000.0
    assert a.bore_diameter_mm == 30.0
    assert a.max_rotor_length_mm == 20.0
    assert a.tilt_angle_deg == pytest.approx(54.74, abs=0.005)


def test_20mm_rotor_at_largest_diameter_fits_bore_and_is_feasible(assumptions):
    # Hand calculation in test_geometry: largest Rotor diameter 7.874 mm at 20 mm.
    # The Stator then exactly fills the usable bore: 30 - 2 * 0.5 = 29 mm.
    results = evaluate(Design(rotor_length_mm=20.0), assumptions)
    assert results.max_rotor_diameter_mm == pytest.approx(7.874, abs=1e-3)
    assert results.rotor_diameter_mm == pytest.approx(7.874, abs=1e-3)
    assert results.projected_width_mm == pytest.approx(29.0, abs=1e-3)
    assert results.fits_bore
    assert results.feasible
    assert results.binding_constraint == "Probe Bore"


def test_oversized_stator_is_ruled_out_by_probe_bore(assumptions):
    # A 10 mm Rotor at 20 mm length needs a 17 mm Stator diameter:
    # 25 * 0.816497 + 17 * 0.577350 = 30.227 mm > 29 mm usable bore.
    results = evaluate(Design(rotor_length_mm=20.0, rotor_diameter_mm=10.0), assumptions)
    assert results.projected_width_mm == pytest.approx(30.227, abs=1e-3)
    assert not results.fits_bore
    assert not results.feasible
    assert results.binding_constraint == "Probe Bore"


def test_rotor_length_above_20mm_is_rejected(assumptions):
    with pytest.raises(ValueError, match="Rotor Length"):
        evaluate(Design(rotor_length_mm=20.5), assumptions)


def test_turbine_torque_balances_drag_at_target_spin_rate(assumptions):
    results = evaluate(Design(rotor_length_mm=20.0), assumptions)
    assert results.spin_rate_hz == 1000.0
    assert results.drag_torque_n_m > 0.0
    assert results.turbine_torque_n_m == pytest.approx(results.drag_torque_n_m, rel=1e-3)
    assert results.required_mass_flow_kg_s > 0.0
    back_pressure_bar = 0.2 * 133.322 / 1e5
    assert back_pressure_bar < results.required_supply_pressure_bar <= 3.77
    assert results.feasible


def test_nozzle_too_small_to_drive_rotor_is_ruled_out_by_supply_pressure(assumptions):
    # One 2 um Nozzle passes a few ng/s even when choked at 3.77 bar: far too little torque.
    design = Design(rotor_length_mm=20.0, nozzle_count=1, nozzle_throat_mm=0.002)
    results = evaluate(design, assumptions)
    assert results.fits_bore
    assert results.required_supply_pressure_bar > 3.77
    assert not results.feasible
    assert results.binding_constraint == "Supply Pressure"


def test_drive_gas_temperature_defaults_to_10k_and_warmer_gas_needs_less_flow(assumptions):
    # A warmer supply gives a faster jet, so less mass flow balances the same drag.
    cold = evaluate(Design(rotor_length_mm=20.0), assumptions)
    warm = evaluate(Design(rotor_length_mm=20.0, drive_gas_temperature_k=20.0), assumptions)
    assert cold.drive_gas_temperature_k == 10.0
    assert warm.drive_gas_temperature_k == 20.0
    assert warm.required_mass_flow_kg_s < cold.required_mass_flow_kg_s


HEAT_SOURCES = {"turbine_cap_conduction", "windage", "bearing_friction", "convection", "radiation",
                "internal_load", "cold_support_shear"}


def test_baseline_reports_temperature_profile_sample_zone_and_heat_load(assumptions):
    results = evaluate(Design(rotor_length_mm=20.0), assumptions)
    t = results.temperature_k
    assert len(results.z_mm) == len(t)
    assert min(t) >= 1.2 - 1e-9
    # The fin base is the Turbine Cap, held at the Drive Gas temperature.
    assert t[-1] == pytest.approx(10.0)
    assert 5.0 <= results.sample_zone_mm <= 20.0
    assert results.sample_hottest_temperature_k <= 1.3
    assert set(results.sample_heat_load_breakdown_w) == HEAT_SOURCES
    assert results.sample_heat_load_w == pytest.approx(sum(results.sample_heat_load_breakdown_w.values()))
    assert results.sample_heat_load_breakdown_w["turbine_cap_conduction"] > 0.0
    assert results.feasible


def test_rotor_too_short_for_5mm_sample_zone_is_ruled_out(assumptions):
    results = evaluate(Design(rotor_length_mm=8.0), assumptions)
    assert results.fits_bore
    assert results.sample_zone_mm < 5.0
    assert not results.feasible
    assert results.binding_constraint == "Sample Zone"


def test_internal_load_in_poor_conductor_overheats_sample(assumptions):
    # Centre-to-wall rise of a uniformly heated cylinder: Q / (4 pi k L)
    #   = 1e-5 W / (4 pi * 1e-5 W/(m K) * 5e-3 m) = 15.9 K, far above 1.3 K,
    # while 10 uW barely warms the wall, so the Sample Zone itself survives.
    design = Design(rotor_length_mm=20.0, sample_internal_heat_load_w=1e-5,
                    sample_conductivity_w_m_k=1e-5)
    results = evaluate(design, assumptions)
    assert results.sample_zone_mm >= 5.0
    assert results.sample_hottest_temperature_k > 1.2 + 15.0
    assert not results.feasible
    assert results.binding_constraint == "Sample Temperature"


def test_lower_sample_conductivity_gives_hotter_sample(assumptions):
    good = evaluate(Design(rotor_length_mm=20.0, sample_internal_heat_load_w=1e-5,
                           sample_conductivity_w_m_k=1.0), assumptions)
    poor = evaluate(Design(rotor_length_mm=20.0, sample_internal_heat_load_w=1e-5,
                           sample_conductivity_w_m_k=0.01), assumptions)
    assert poor.sample_hottest_temperature_k > good.sample_hottest_temperature_k
    assert poor.sample_heat_load_breakdown_w["internal_load"] == pytest.approx(1e-5, rel=1e-6)


def test_cooling_path_defaults_to_vapour_only_with_no_cold_support_terms(assumptions):
    results = evaluate(Design(rotor_length_mm=20.0), assumptions)
    assert results.cooling_path == VAPOUR_ONLY
    assert results.cold_support_drag_torque_n_m == 0.0
    assert results.cold_support_heat_to_bath_w == 0.0
    assert results.sample_heat_load_breakdown_w["cold_support_shear"] == 0.0


def test_unknown_cooling_path_is_rejected(assumptions):
    with pytest.raises(ValueError, match="cooling path"):
        evaluate(Design(rotor_length_mm=20.0, cooling_path="liquid"), assumptions)


def test_cold_support_gap_shear_adds_to_drag_torque_and_sample_heat_load(assumptions):
    vapour = evaluate(Design(rotor_length_mm=20.0), assumptions)
    cold = evaluate(Design(rotor_length_mm=20.0, cooling_path=COLD_SUPPORT, cold_support_gap_mm=0.01),
                    assumptions)
    assert cold.cooling_path == COLD_SUPPORT
    assert cold.cold_support_gap_mm == 0.01
    assert cold.cold_support_drag_torque_n_m > 0.0
    assert cold.drag_torque_n_m == pytest.approx(vapour.drag_torque_n_m + cold.cold_support_drag_torque_n_m)
    assert cold.turbine_torque_n_m == pytest.approx(cold.drag_torque_n_m, rel=1e-3)
    assert cold.required_mass_flow_kg_s > vapour.required_mass_flow_kg_s
    # All gap shear heat enters the Rotor at the placeholder windage_heat_to_rotor_fraction of 1.
    shear_heat = cold.cold_support_drag_torque_n_m * 2.0 * math.pi * 1000.0
    assert cold.sample_heat_load_breakdown_w["cold_support_shear"] == pytest.approx(shear_heat)
    assert cold.cold_support_heat_to_bath_w > shear_heat


def test_cold_support_gap_defaults_from_assumptions_file(assumptions):
    results = evaluate(Design(rotor_length_mm=20.0, cooling_path=COLD_SUPPORT), assumptions)
    assert results.cold_support_gap_mm == assumptions.cold_support_gap_mm


def test_smaller_cold_support_gap_cools_sample_until_shear_heating_dominates(assumptions):
    gaps_mm = assumptions.cold_support_gap_sweep_mm
    hottest = [evaluate(Design(rotor_length_mm=20.0, cooling_path=COLD_SUPPORT, cold_support_gap_mm=g),
                        assumptions).sample_hottest_temperature_k for g in gaps_mm]
    vapour = evaluate(Design(rotor_length_mm=20.0), assumptions).sample_hottest_temperature_k
    coolest = min(range(len(gaps_mm)), key=hottest.__getitem__)
    assert 0 < coolest < len(gaps_mm) - 1
    # Wide gaps: each step smaller cools the Sample; the widest barely beats Vapour Cooling alone.
    assert hottest[coolest:] == sorted(hottest[coolest:])
    assert hottest[-1] < vapour
    # Narrow gaps: shear heating grows faster than gap conduction and warms the Sample again.
    assert hottest[:coolest + 1] == sorted(hottest[:coolest + 1], reverse=True)
