"""Benchmark Cases: cited case files, evaluated through evaluate(), and the Model-Form Error."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.benchmark import (INPUTS, QUANTITIES, compare, load_case, load_cases, model_form_error)
from rotor_design.evaluate import DUCTED, PA_PER_BAR, Design, evaluate

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS_FILE = ROOT / "assumptions.toml"
BENCHMARKS = ROOT / "benchmarks"
CASES = load_cases(BENCHMARKS)


@pytest.fixture(scope="module")
def assumptions():
    return load_assumptions(ASSUMPTIONS_FILE)


# A made-up, fully specified system for testing the comparison machinery. NOT a Benchmark Case.
FIXTURE_INPUTS = {
    "drive_gas": "helium", "rotor_material": "zirconia", "rotor_length_mm": 20.0, "rotor_diameter_mm": 4.0,
    "wall_thickness_mm": 0.5, "bearing": "gas", "drive_gas_temperature_k": 45.0,
    "environment_temperature_k": 45.0, "environment_pressure_bar": 1.0, "turbine": "pelton",
    "nozzle_count": 4, "nozzle_throat_mm": 0.3, "nozzle_angle_deg": 20.0, "back_pressure_bar": 1.0,
}


def _entry(value):
    literal = f'"{value}"' if isinstance(value, str) else repr(value)
    return f'{{ value = {literal}, source = "fixture", location = "Table 1", flag = "M" }}'


def _write_case(tmp_path, inputs=FIXTURE_INPUTS, points=None, extra_inputs=""):
    lines = ['id = "T1"', 'name = "test fixture"', "", "[sources]", 'fixture = "made up for tests"', "",
             "[inputs]"]
    lines += [f"{k} = {_entry(v) if v is not None else '{}'}" for k, v in inputs.items()]
    lines.append(extra_inputs)
    for point in points if points is not None else [
            'inputs.spin_rate_hz = { value = 8000.0, source = "fixture", location = "Fig. 1", flag = "F" }\n'
            'measured.mass_flow_kg_s = { value = 5e-5, source = "fixture", location = "Fig. 2", flag = "F" }\n'
            'measured.sample_temperature_k = { value = 50.0, source = "fixture", location = "Fig. 3", flag = "M" }\n'
            'measured.bath_heat_load_w = { note = "not reported" }']:
        lines += ["", "[[points]]", point]
    path = tmp_path / "t1.toml"
    path.write_text("\n".join(lines))
    return path


def test_every_case_file_loads_with_every_input_listed_and_every_value_cited():
    assert {c.id for c in CASES} >= {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "K1-1p3", "K1-0p7", "K2", "K3"}
    for case in CASES:
        assert set(INPUTS) <= set(case.inputs)
        values = list(case.inputs.values()) + [v for p in case.points for v in (*p.inputs.values(),
                                                                                  *p.measured.values())]
        for v in values:
            if v.value is not None:
                assert v.source in case.sources and case.sources[v.source]
                assert v.location and v.flag


@pytest.mark.parametrize("bad, message", [
    ('{ value = 1.0, location = "Fig. 1", flag = "M" }', "unknown source"),
    ('{ value = 1.0, source = "nowhere", location = "Fig. 1", flag = "M" }', "unknown source"),
    ('{ value = 1.0, source = "fixture", flag = "M" }', "no location"),
    ('{ value = 1.0, source = "fixture", location = "Fig. 1", flag = "S" }', "not validation truth"),
    ('{ value = 1.0, source = "fixture", location = "Fig. 1", flag = "2nd" }', "not validation truth"),
    ('{ value = 1.0, source = "fixture", location = "Fig. 1" }', "flag"),
])
def test_loader_rejects_a_value_without_a_usable_citation(tmp_path, bad, message):
    inputs = dict(FIXTURE_INPUTS)
    del inputs["wall_thickness_mm"]
    path = _write_case(tmp_path, inputs, extra_inputs=f"wall_thickness_mm = {bad}")
    with pytest.raises(ValueError, match=message):
        load_case(path)


def test_loader_requires_every_input_listed_even_when_empty(tmp_path):
    inputs = dict(FIXTURE_INPUTS)
    del inputs["nozzle_throat_mm"]
    with pytest.raises(ValueError, match="nozzle_throat_mm"):
        load_case(_write_case(tmp_path, inputs))


def test_complete_case_is_predicted_through_evaluate(tmp_path, assumptions):
    comparisons = compare(load_case(_write_case(tmp_path)), assumptions)
    # The empty bath_heat_load_w entry is not compared at all.
    assert [c.quantity for c in comparisons] == ["mass_flow_kg_s", "sample_temperature_k"]
    assert all(not c.skipped for c in comparisons)

    direct = evaluate(
        Design(rotor_length_mm=20.0, rotor_diameter_mm=4.0, wall_thickness_mm=0.5, nozzle_count=4,
               nozzle_throat_mm=0.3, nozzle_angle_deg=20.0, drive_gas="helium", drive_gas_temperature_k=45.0,
               bearing="gas",
               turbine="fluted_cap", exhaust_path=DUCTED, back_pressure_bar=1.0),
        replace(assumptions, target_spin_rate_hz=8000.0, bath_temperature_k=45.0,
                sample_limit_temperature_k=45.2, cold_space_pressure_pa=PA_PER_BAR))
    flow, temperature = comparisons
    assert flow.predicted == pytest.approx(direct.required_mass_flow_kg_s)
    assert flow.model_form_error == pytest.approx((direct.required_mass_flow_kg_s - 5e-5) / 5e-5)
    assert temperature.predicted == pytest.approx(direct.sample_hottest_temperature_k)
    assert temperature.flag == "M"


def test_missing_input_skips_only_the_quantities_that_need_it(tmp_path, assumptions):
    # No Nozzle throat: flow cannot be predicted, but the Rotor temperature does not depend on it.
    path = _write_case(tmp_path, {**FIXTURE_INPUTS, "nozzle_throat_mm": None})
    flow, temperature = compare(load_case(path), assumptions)
    assert flow.skipped == "not reported: nozzle_throat_mm"
    assert math.isnan(flow.predicted) and math.isnan(flow.model_form_error)
    assert not temperature.skipped


def test_point_without_spin_rate_is_skipped(tmp_path, assumptions):
    point = ('inputs.spin_rate_hz = { note = "not given" }\n'
             'measured.mass_flow_kg_s = { value = 5e-5, source = "fixture", location = "Fig. 2", flag = "F" }')
    (flow,) = compare(load_case(_write_case(tmp_path, points=[point])), assumptions)
    assert flow.skipped == "not reported: spin_rate_hz"


@pytest.mark.parametrize("field, value, reason", [
    ("drive_gas", "argon", "helium, nitrogen, air only"),
    ("rotor_material", "sapphire", "zirconia only"),
    ("turbine", "surface grooves", "no model Turbine type"),
    ("bearing", "single drive jet", "no model Bearing type"),
])
def test_system_outside_the_model_is_skipped_with_the_reason(tmp_path, assumptions, field, value, reason):
    path = _write_case(tmp_path, {**FIXTURE_INPUTS, field: value})
    comparisons = compare(load_case(path), assumptions)
    assert comparisons and all(reason in c.skipped for c in comparisons)


def test_model_form_error_is_reported_per_quantity(tmp_path, assumptions):
    comparisons = compare(load_case(_write_case(tmp_path)), assumptions)
    comparisons += compare(load_case(_write_case(tmp_path, {**FIXTURE_INPUTS, "nozzle_throat_mm": None})),
                           assumptions)
    summary = {q.quantity: q for q in model_form_error(comparisons)}
    assert set(summary) == {"mass_flow_kg_s", "sample_temperature_k"}
    flow = summary["mass_flow_kg_s"]
    assert (flow.compared, flow.skipped) == (1, 1)
    assert flow.mean_error == flow.worst_error == comparisons[0].model_form_error
    assert flow.rms_error == pytest.approx(abs(comparisons[0].model_form_error))
    assert summary["sample_temperature_k"].compared == 2


def test_every_measured_quantity_has_a_tolerance(assumptions):
    assert set(assumptions.benchmark_tolerance) == set(QUANTITIES)


BENCHMARK_POINTS = [(case, i, quantity) for case in CASES for i, point in enumerate(case.points)
                    for quantity, measured in point.measured.items() if measured.value is not None]


@pytest.mark.parametrize("case, point, quantity", BENCHMARK_POINTS,
                         ids=[f"{c.id}-{i}-{q}" for c, i, q in BENCHMARK_POINTS])
def test_benchmark_case_prediction_within_tolerance(assumptions, case, point, quantity):
    (comparison,) = [c for c in compare(case, assumptions) if (c.point, c.quantity) == (point, quantity)]
    if comparison.skipped:
        pytest.skip(comparison.skipped)
    tolerance = assumptions.benchmark_tolerance[quantity]
    assert abs(comparison.model_form_error) <= tolerance, (
        f"predicted {comparison.predicted:.4g}, measured {comparison.measured:.4g} ({comparison.flag})")
