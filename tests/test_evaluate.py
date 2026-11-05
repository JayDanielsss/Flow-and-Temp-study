"""Evaluate one Design: Design plus assumptions in, Results out."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.bearing import GAS, MECHANICAL, MECHANICAL_WARNING, TWO_CENTER_PIVOT
from rotor_design.evaluate import (COLD_SUPPORT, DUCTED, VAPOUR_ONLY, VENTED, Design, configurations,
                                  evaluate)
from rotor_design.gas import R_HELIUM, stagnation_state, vapour_enthalpy
from rotor_design.turbine import FLUTED_CAP, RADIAL_INFLOW

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
    assert results.back_pressure_bar < results.required_supply_pressure_bar <= 3.77
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
    # The lower pivot warms the Bottom Cap end, so the Sample's lowest slice can sit just below
    # the Sample Zone; only the part inside the zone counts.
    assert 0.9e-5 < poor.sample_heat_load_breakdown_w["internal_load"] <= 1e-5 * (1 + 1e-6)


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


def test_bearing_type_is_part_of_the_configuration(assumptions):
    default = evaluate(Design(rotor_length_mm=20.0), assumptions)
    assert default.bearing == assumptions.bearing == TWO_CENTER_PIVOT
    assert not default.pivot_to_bath
    for bearing in (GAS, MECHANICAL, TWO_CENTER_PIVOT):
        assert evaluate(Design(rotor_length_mm=20.0, bearing=bearing), assumptions).bearing == bearing


def test_unknown_bearing_type_is_rejected(assumptions):
    with pytest.raises(ValueError, match="Bearing"):
        evaluate(Design(rotor_length_mm=20.0, bearing="magnetic"), assumptions)


def test_each_bearing_reports_friction_stiffness_and_radial_space(assumptions):
    for bearing in (GAS, MECHANICAL, TWO_CENTER_PIVOT):
        r = evaluate(Design(rotor_length_mm=20.0, bearing=bearing), assumptions)
        assert r.bearing_friction_torque_n_m > 0.0
        assert r.bearing_stiffness_n_m > 0.0
        assert r.bearing_radial_space_mm >= 0.0
        assert r.sample_heat_load_breakdown_w["bearing_friction"] > 0.0
        # Bearing friction is part of the drag the turbine must balance.
        assert r.drag_torque_n_m > r.bearing_friction_torque_n_m
        assert r.turbine_torque_n_m == pytest.approx(r.drag_torque_n_m, rel=1e-3)


def test_bearing_radial_space_shrinks_the_largest_rotor_that_fits(assumptions):
    # Each mm of radial space each side takes 2 mm off the Stator diameter budget.
    pivots = evaluate(Design(rotor_length_mm=20.0, bearing=TWO_CENTER_PIVOT), assumptions)
    gas = evaluate(Design(rotor_length_mm=20.0, bearing=GAS), assumptions)
    assert pivots.bearing_radial_space_mm == 0.0
    assert pivots.max_rotor_diameter_mm == pytest.approx(7.874, abs=1e-3)
    assert gas.max_rotor_diameter_mm == pytest.approx(7.874 - 2 * gas.bearing_radial_space_mm, abs=1e-3)
    assert gas.fits_bore


def test_bearing_gas_colder_than_the_rotor_never_cools_it(assumptions):
    def with_gas_at(t_k):
        bearings = {**assumptions.bearings, GAS: replace(assumptions.bearings[GAS], gas_temperature_k=t_k)}
        return evaluate(Design(rotor_length_mm=20.0, bearing=GAS), replace(assumptions, bearings=bearings))

    warm, cold = with_gas_at(10.0), with_gas_at(1.0)
    assert warm.sample_heat_load_breakdown_w["convection"] > 0.0
    assert cold.sample_heat_load_breakdown_w["convection"] == 0.0
    assert min(cold.temperature_k) >= 1.2 - 1e-9


def test_only_the_mechanical_bearing_carries_a_warning(assumptions):
    for bearing in (GAS, MECHANICAL, TWO_CENTER_PIVOT):
        r = evaluate(Design(rotor_length_mm=20.0, bearing=bearing), assumptions)
        assert r.warning == (MECHANICAL_WARNING if bearing == MECHANICAL else "")


def test_pivot_to_bath_conducts_through_the_lower_pivot(assumptions):
    free = evaluate(Design(rotor_length_mm=20.0, bearing=TWO_CENTER_PIVOT, pivot_to_bath=False), assumptions)
    tied = evaluate(Design(rotor_length_mm=20.0, bearing=TWO_CENTER_PIVOT, pivot_to_bath=True), assumptions)
    assert free.pivot_heat_to_bath_w == 0.0
    assert tied.pivot_heat_to_bath_w > 0.0
    assert tied.temperature_k[0] < free.temperature_k[0]


@pytest.mark.parametrize("bearing", [GAS, MECHANICAL])
def test_pivot_to_bath_with_any_other_bearing_is_excluded(assumptions, bearing):
    with pytest.raises(ValueError, match="Excluded Configuration"):
        evaluate(Design(rotor_length_mm=20.0, bearing=bearing, pivot_to_bath=True), assumptions)


def test_configurations_cover_every_allowed_combination_and_no_excluded_one():
    combos = {(c["bearing"], c["pivot_to_bath"], c["cooling_path"], c["turbine"], c["exhaust_path"])
              for c in configurations()}
    # (3 Bearing types x 2 cooling paths, plus the pivot-to-Bath option on two-center pivots)
    # x 2 Turbine types x 2 Exhaust Paths.
    assert len(combos) == 32
    assert all(bearing == TWO_CENTER_PIVOT for bearing, tied, *_ in combos if tied)
    assert {c[3] for c in combos} == {FLUTED_CAP, RADIAL_INFLOW}
    assert {c[4] for c in combos} == {VENTED, DUCTED}


def test_bottom_cap_warmed_by_lower_pivot_leaves_a_cold_sample_zone_above_it(assumptions):
    # The solid zirconia Bottom Cap barely conducts at 1.2 K, so pivot friction warms its face
    # above the limit while the Rotor above stays cold: the Sample Zone need not start at z = 0.
    r = evaluate(Design(rotor_length_mm=20.0, bearing=TWO_CENTER_PIVOT), assumptions)
    assert r.temperature_k[0] > 1.3
    assert r.sample_zone_mm >= 5.0
    assert r.sample_hottest_temperature_k <= 1.3
    assert r.feasible
    assert r.sample_heat_load_breakdown_w["bearing_friction"] > 0.0
    assert r.sample_heat_load_w == pytest.approx(sum(r.sample_heat_load_breakdown_w.values()))


def test_turbine_type_is_part_of_the_configuration(assumptions):
    default = evaluate(Design(rotor_length_mm=20.0), assumptions)
    assert default.turbine == assumptions.turbine == FLUTED_CAP
    assert math.isnan(default.hub_exit_diameter_mm)
    radial = evaluate(Design(rotor_length_mm=20.0, turbine=RADIAL_INFLOW), assumptions)
    assert radial.turbine == RADIAL_INFLOW
    assert radial.hub_exit_diameter_mm == assumptions.turbines[RADIAL_INFLOW].hub_exit_diameter_mm
    assert radial.turbine_torque_n_m == pytest.approx(radial.drag_torque_n_m, rel=1e-3)


def test_unknown_turbine_type_is_rejected(assumptions):
    with pytest.raises(ValueError, match="Turbine"):
        evaluate(Design(rotor_length_mm=20.0, turbine="pelton"), assumptions)


def test_hub_exit_takes_blade_area_from_the_turbine_cap(assumptions):
    fluted = evaluate(Design(rotor_length_mm=20.0, turbine=FLUTED_CAP), assumptions)
    radial = evaluate(Design(rotor_length_mm=20.0, turbine=RADIAL_INFLOW, hub_exit_diameter_mm=3.0), assumptions)
    cap_area = math.pi * (fluted.rotor_diameter_mm / 2) ** 2
    assert fluted.turbine_blade_area_mm2 == pytest.approx(cap_area)
    assert radial.turbine_blade_area_mm2 == pytest.approx(cap_area - math.pi * 1.5**2)


def test_radial_inflow_needs_less_drive_gas_than_fluted_cap(assumptions):
    # Gas leaving at the hub carries away less angular momentum than gas leaving at the rim.
    fluted = evaluate(Design(rotor_length_mm=20.0, turbine=FLUTED_CAP), assumptions)
    radial = evaluate(Design(rotor_length_mm=20.0, turbine=RADIAL_INFLOW), assumptions)
    assert radial.turbine_torque_per_flow_n_m_s_kg > fluted.turbine_torque_per_flow_n_m_s_kg
    assert radial.required_mass_flow_kg_s < fluted.required_mass_flow_kg_s


def test_wider_hub_exit_lowers_torque_per_unit_flow(assumptions):
    narrow, wide = (evaluate(Design(rotor_length_mm=20.0, turbine=RADIAL_INFLOW, hub_exit_diameter_mm=d),
                             assumptions) for d in (1.0, 6.0))
    assert wide.turbine_torque_per_flow_n_m_s_kg < narrow.turbine_torque_per_flow_n_m_s_kg


PA_PER_TORR = 133.322


def test_exhaust_path_is_part_of_the_configuration(assumptions):
    default = evaluate(Design(rotor_length_mm=20.0), assumptions)
    assert default.exhaust_path == assumptions.exhaust_path == DUCTED
    assert default.back_pressure_bar == assumptions.ducted_back_pressure_bar
    vented = evaluate(Design(rotor_length_mm=20.0, exhaust_path=VENTED), assumptions)
    assert vented.exhaust_path == VENTED
    # Vented: the turbine exits into the probe space at the Bath's 0.2 Torr.
    assert vented.back_pressure_bar == pytest.approx(0.2 * PA_PER_TORR / 1e5)


def test_unknown_exhaust_path_is_rejected(assumptions):
    with pytest.raises(ValueError, match="Exhaust Path"):
        evaluate(Design(rotor_length_mm=20.0, exhaust_path="recirculated"), assumptions)


def test_ducted_back_pressure_is_an_input_and_a_higher_one_needs_more_supply_pressure(assumptions):
    low, high = (evaluate(Design(rotor_length_mm=20.0, exhaust_path=DUCTED, back_pressure_bar=p), assumptions)
                 for p in (0.01, 0.5))
    assert low.back_pressure_bar == 0.01
    assert high.back_pressure_bar == 0.5
    assert high.required_supply_pressure_bar > low.required_supply_pressure_bar


def test_vented_reports_pumping_speed_to_hold_the_probe_space_pressure(assumptions):
    vented = evaluate(Design(rotor_length_mm=20.0, exhaust_path=VENTED), assumptions)
    ducted = evaluate(Design(rotor_length_mm=20.0, exhaust_path=DUCTED), assumptions)
    # Volume flow of the vented Drive Gas as vapour at 1.2 K and 0.2 Torr: m R T / p.
    expected = vented.required_mass_flow_kg_s * R_HELIUM * 1.2 / (0.2 * PA_PER_TORR)
    assert vented.pumping_speed_m3_s == pytest.approx(expected)
    assert math.isnan(ducted.pumping_speed_m3_s)
    assert ducted.exhaust_heat_to_bath_w == 0.0


def test_vented_exhaust_heat_is_mass_flow_times_enthalpy_drop_to_the_bath(assumptions):
    # The Drive Gas enters with the supply stagnation enthalpy, gives up the shaft work to the
    # turbine and is cooled to Bath vapour: Q = m (h0 - h_bath) - torque * omega.
    r = evaluate(Design(rotor_length_mm=20.0, exhaust_path=VENTED), assumptions)
    h0, _ = stagnation_state(10.0, r.required_supply_pressure_bar * 1e5)
    h_bath = vapour_enthalpy(1.2, 0.2 * PA_PER_TORR)
    shaft_power = r.turbine_torque_n_m * 2.0 * math.pi * 1000.0
    assert r.exhaust_heat_to_bath_w == pytest.approx(r.required_mass_flow_kg_s * (h0 - h_bath) - shaft_power,
                                                     rel=1e-6)
    # The shaft work returns to the Bath as drag heat; two-center pivots add no bearing gas.
    assert r.bath_heat_load_w == pytest.approx(r.exhaust_heat_to_bath_w + shaft_power, rel=1e-5)


def test_vented_gives_higher_bath_heat_load_than_ducted(assumptions):
    vented = evaluate(Design(rotor_length_mm=20.0, exhaust_path=VENTED), assumptions)
    ducted = evaluate(Design(rotor_length_mm=20.0, exhaust_path=DUCTED), assumptions)
    assert ducted.bath_heat_load_w >= ducted.sample_heat_load_w > 0.0
    assert vented.bath_heat_load_w > ducted.bath_heat_load_w


def test_vented_drive_gas_convects_onto_the_rotor_and_warms_it(assumptions):
    # Drive Gas at 10 K washing the whole body is an upper bound: it leaves no Sample Zone.
    vented = evaluate(Design(rotor_length_mm=20.0, exhaust_path=VENTED), assumptions)
    ducted = evaluate(Design(rotor_length_mm=20.0, exhaust_path=DUCTED), assumptions)
    assert min(vented.temperature_k) > min(ducted.temperature_k)
    assert vented.sample_hottest_temperature_k > ducted.sample_hottest_temperature_k
    assert vented.sample_zone_mm < ducted.sample_zone_mm
    assert not vented.feasible


def test_design_over_the_bath_cooling_power_is_ruled_out_by_bath_heat_load(assumptions):
    design = Design(rotor_length_mm=20.0, exhaust_path=DUCTED)
    assert evaluate(design, assumptions).feasible
    load = evaluate(design, assumptions).bath_heat_load_w
    starved = evaluate(design, replace(assumptions, bath_cooling_power_w=load / 2.0))
    assert starved.bath_heat_load_w == pytest.approx(load)
    assert not starved.feasible
    assert starved.binding_constraint == "Bath Heat Load"


def test_results_report_peak_stress_safety_factor_and_every_critical_speed(assumptions):
    r = evaluate(Design(rotor_length_mm=20.0), assumptions)
    # Surface speed ~25 m/s: hoop stress a few MPa against 550 MPa tensile strength.
    assert 0.0 < r.peak_radial_stress_mpa < r.peak_hoop_stress_mpa < 10.0
    assert r.stress_safety_factor == pytest.approx(550.0 / r.peak_hoop_stress_mpa)
    assert r.rotor_mass_kg > 0.0
    assert 0.0 < r.first_rigid_body_critical_hz <= r.second_rigid_body_critical_hz < r.bending_critical_hz
    assert math.isnan(r.whirl_threshold_hz)
    assert r.critical_speed_separation > 0.2
    assert r.feasible


def test_required_safety_factor_above_the_achieved_one_is_ruled_out_by_stress(assumptions):
    design = Design(rotor_length_mm=20.0)
    achieved = evaluate(design, assumptions).stress_safety_factor
    r = evaluate(design, replace(assumptions, required_safety_factor=2.0 * achieved))
    assert not r.feasible
    assert r.binding_constraint == "Stress"


def _with_mechanical_stiffness_scaled(assumptions, scale):
    mechanical = assumptions.bearings[MECHANICAL]
    bearings = {**assumptions.bearings, MECHANICAL: replace(mechanical, stiffness_n_m=mechanical.stiffness_n_m * scale)}
    return replace(assumptions, bearings=bearings)


@pytest.mark.parametrize("ratio, inside", [(0.85, True), (1.15, True), (0.45, False), (1.25, False)])
def test_critical_speed_within_20_percent_of_target_is_ruled_out(assumptions, ratio, inside):
    # Rigid-body critical speeds scale with sqrt(stiffness): place the lowest at ratio x 1000 Hz.
    # The conical one sits ~1.6x higher, so 0.45 keeps both below the band (450 and ~720 Hz).
    design = Design(rotor_length_mm=20.0, bearing=MECHANICAL)
    lowest = evaluate(design, assumptions).first_rigid_body_critical_hz
    r = evaluate(design, _with_mechanical_stiffness_scaled(assumptions, (ratio * 1000.0 / lowest) ** 2))
    assert r.first_rigid_body_critical_hz == pytest.approx(ratio * 1000.0)
    nearest = min(abs(f - 1000.0) / 1000.0 for f in (r.first_rigid_body_critical_hz,
                                                     r.second_rigid_body_critical_hz, r.bending_critical_hz))
    assert r.critical_speed_separation == pytest.approx(nearest)
    assert (r.critical_speed_separation < 0.2) == inside
    if inside:
        assert not r.feasible
        assert r.binding_constraint == "Critical Speed"
    else:
        assert r.binding_constraint != "Critical Speed"


def test_gas_bearing_reports_whirl_threshold_and_is_ruled_out_below_it(assumptions):
    design = Design(rotor_length_mm=20.0, bearing=GAS)
    baseline = evaluate(design, assumptions)
    assert baseline.whirl_threshold_hz == pytest.approx(2.0 * baseline.first_rigid_body_critical_hz)
    # Rigid-body critical speeds at 450 and ~710 Hz: clear of the band, but whirl sets in at 900 Hz.
    gas = assumptions.bearings[GAS]
    scale = (450.0 / baseline.first_rigid_body_critical_hz) ** 2
    bearings = {**assumptions.bearings, GAS: replace(gas, stiffness_coefficient=gas.stiffness_coefficient * scale)}
    r = evaluate(design, replace(assumptions, bearings=bearings))
    assert r.whirl_threshold_hz == pytest.approx(900.0)
    assert r.critical_speed_separation > 0.2
    assert not r.feasible
    assert r.binding_constraint == "Whirl"


def test_sample_volume_is_the_bore_cylinder_the_sample_fills(assumptions):
    # 6 mm Rotor, 1 mm wall: a 4 mm bore; 5 mm of Sample gives pi * 2^2 * 5 mm^3.
    r = evaluate(Design(rotor_length_mm=20.0, rotor_diameter_mm=6.0, wall_thickness_mm=1.0,
                        sample_length_mm=5.0), assumptions)
    assert r.sample_volume_mm3 == pytest.approx(62.832, abs=1e-3)
