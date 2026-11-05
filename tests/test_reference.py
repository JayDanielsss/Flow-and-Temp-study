"""Reference Cases: one cited classic experiment per mechanism, and the Model-Form Error of each."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.benchmark import model_form_error
from rotor_design.reference import MECHANISMS, QUANTITY_MECHANISM, compare, load_case, load_cases

ROOT = Path(__file__).resolve().parents[1]
CASES = {case.id: case for case in load_cases(ROOT / "references")}


@pytest.fixture(scope="module")
def assumptions():
    return load_assumptions(ROOT / "assumptions.toml")


def _only(case_id, assumptions, point=0):
    return [c for c in compare(CASES[case_id], assumptions) if c.point == point][0]


def test_every_mechanism_has_a_cited_reference_case():
    assert {case.mechanism for case in CASES.values()} == set(MECHANISMS)
    for case in CASES.values():
        assert set(MECHANISMS[case.mechanism].inputs) <= set(case.inputs)
        values = list(case.inputs.values()) + [v for p in case.points for v in (*p.inputs.values(),
                                                                                  *p.measured.values())]
        for v in values:
            if v.value is not None:
                assert v.source in case.sources and case.sources[v.source]
                assert v.location and v.flag


def _write(tmp_path, mechanism, body):
    path = tmp_path / "r.toml"
    path.write_text(f'id = "T"\nname = "test fixture"\nmechanism = "{mechanism}"\n'
                    '[sources]\nfixture = "made up for tests"\n' + body)
    return path


@pytest.mark.parametrize("mechanism, body, message", [
    ("magnetism", "", "mechanism"),
    ("conduction", "[inputs]\n", "wall_temperature_k"),
    ("conduction", "[inputs]\nwall_temperature_k = {}\nwall_emissivity = {}\n", "unknown inputs"),
    ("conduction", '[inputs]\nwall_temperature_k = {}\n[[points]]\n'
     'measured.torque_n_m = { value = 1.0, source = "fixture", location = "Fig. 1", flag = "M" }\n',
     "unknown quantities"),
    ("conduction", '[inputs]\nwall_temperature_k = { value = 78.7, source = "fixture", '
     'location = "Fig. 1", flag = "S" }\n', "not validation truth"),
])
def test_loader_rejects_a_case_it_cannot_trust(tmp_path, mechanism, body, message):
    with pytest.raises(ValueError, match=message):
        load_case(_write(tmp_path, mechanism, body))


def test_nozzle_prediction_is_the_assumed_discharge_coefficient(assumptions):
    comparison = _only("R1", assumptions)
    assert comparison.predicted == assumptions.discharge_coefficient
    assert comparison.measured == 0.9935


def test_windage_matches_taylors_laminar_formula_and_his_scale_pan_torque(assumptions):
    # Taylor 1936, Eq. (1), cgs: G = 2 pi^2 D1^2 D2^2 l mu N / ((D1 + D2)(D1 - D2))
    #   = 2 pi^2 * 8.11^2 * 6.40^2 * 84.2 * 0.0174 * 20.8 / (14.51 * 1.71) = 65 312 dyn cm = 6.5312e-3 N m
    # Measured: (21 - 9) g * 9.80665 m/s^2 * 0.0501 m = 5.8958e-3 N m
    comparison = _only("R2", assumptions)
    assert comparison.predicted == pytest.approx(6.5312e-3, rel=1e-4)
    assert comparison.measured == pytest.approx(5.8958e-3, rel=1e-4)


def test_free_molecular_drag_on_a_sphere_is_the_gauge_theory(assumptions):
    # With full accommodation the model's shear integrated over the ball gives exactly the
    # gauge's ideal decay rate 20 p / (pi c_bar rho d), so sigma_eff = 1; with accommodation
    # 0.9 the model's sigma / (2 - sigma) gives 0.9 / 1.1.
    assert _only("R3a", assumptions).predicted == pytest.approx(1.0, rel=1e-12)
    diffuse = replace(assumptions, accommodation_coefficient=0.9)
    assert _only("R3a", diffuse).predicted == pytest.approx(0.9 / 1.1, rel=1e-12)


def test_turbine_efficiency_matches_the_ideal_gas_fluted_cap(assumptions):
    # Choked jet at the sonic speed V*; ideal work from total to exit static pressure, PR = 3, gamma = 1.4:
    #   V* / V_id = sqrt((1 - 2 / 2.4) / (1 - 3^(-0.4 / 1.4))) = sqrt(0.166667 / 0.269373) = 0.786600
    #   eta = 2 nu (V* / V_id cos(13.39 deg) - nu) = 2 * 0.377 * (0.765217 - 0.377) = 0.292716
    # Real air at 300 K and 1.72 bar shifts it by about 0.1 %.
    eta = _only("R4", assumptions).predicted
    assert eta == pytest.approx(0.292716 * assumptions.turbine_efficiency, rel=2e-3)


def test_radiating_rod_matches_the_sources_own_black_rod_solution(assumptions):
    # Cothran 1970, Table C-4, calculated insulated-end ratio of his black-rod equation (II-18);
    # a calculation, so used here only to check that the model's fin solves the same problem.
    calculated = [0.8960, 0.8473, 0.7865, 0.8433, 0.8408, 0.8396]
    predicted = [c.predicted for c in compare(CASES["R5"], assumptions)]
    assert predicted == pytest.approx(calculated, rel=7e-3)


def test_values_the_source_does_not_report_skip_the_comparison(assumptions):
    comparisons = compare(CASES["R3b"], assumptions)
    assert comparisons and all(c.skipped.startswith("not reported: ball_diameter_mm") for c in comparisons)
    assert all(math.isnan(c.model_form_error) for c in comparisons)


def test_model_form_error_is_reported_per_mechanism(assumptions):
    comparisons = [c for case in CASES.values() for c in compare(case, assumptions)]
    summary = {QUANTITY_MECHANISM[q.quantity]: q for q in model_form_error(comparisons, QUANTITY_MECHANISM)}
    assert set(summary) == set(MECHANISMS)
    assert summary["rarefied_drag"].skipped == 4
    for q in summary.values():
        assert q.compared >= 1 and math.isfinite(q.rms_error)
