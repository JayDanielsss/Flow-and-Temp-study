"""Evaluate one Design: Design plus assumptions in, Results out."""

from pathlib import Path

import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import Design, evaluate

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
