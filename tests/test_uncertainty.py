"""Monte Carlo samples: physical inputs, Model-Form Error as a bias correction plus residual spread."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rotor_design import reference
from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import Design, evaluate
from rotor_design.turbine import FLUTED_CAP, RADIAL_INFLOW, TURBINE_TYPES
from rotor_design.uncertainty import _get, mechanism_errors, sample_assumptions

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"
# Enough samples that a few in a thousand going nonphysical would show.
MANY_SAMPLES = 2000


@pytest.fixture(scope="module")
def assumptions():
    return load_assumptions(ASSUMPTIONS_FILE)


@pytest.fixture(scope="module")
def errors(assumptions):
    return mechanism_errors(assumptions)


@pytest.fixture(scope="module")
def samples(assumptions, errors):
    return sample_assumptions(replace(assumptions, uncertainty_samples=MANY_SAMPLES), errors)


def _values(samples, name):
    return np.array([_get(s, name) for s in samples])


def _turbine_efficiencies(samples, turbine):
    """The efficiency each sample gives a Turbine type: the shared knob times that type's correction."""
    return np.array([s.turbine_efficiency * s.turbine_efficiency_correction[turbine] for s in samples])


def _logit(x):
    return math.log(x / (1.0 - x))


def test_no_sample_has_zero_turbine_efficiency_or_a_discharge_coefficient_above_one(samples):
    for turbine in TURBINE_TYPES:
        assert _turbine_efficiencies(samples, turbine).min() > 0.0
    assert _values(samples, "discharge_coefficient").max() <= 1.0


def test_every_sampled_input_lies_in_its_physical_range(assumptions, samples):
    names = set(assumptions.uncertainty_inputs) | set(assumptions.model_form_knobs.values())
    assert names <= set(assumptions.uncertainty_physical_range)
    for name in names:
        low, high = assumptions.uncertainty_physical_range[name]
        values = _values(samples, name)
        assert np.all(values > low) and np.all(values <= high), name
    # The Turbine knob is held to its range as each Turbine type uses it.
    low, high = assumptions.uncertainty_physical_range["turbine_efficiency"]
    for turbine in TURBINE_TYPES:
        values = _turbine_efficiencies(samples, turbine)
        assert np.all(values > low) and np.all(values <= high), turbine


def test_windage_heat_fraction_is_sampled_from_half_to_all(assumptions, samples):
    assert assumptions.uncertainty_inputs["windage_heat_to_rotor_fraction"] == ("uniform", 0.5, 1.0)
    # 1.0, all the windage heat in the Rotor, stays the declared upper bound and the default.
    assert assumptions.windage_heat_to_rotor_fraction == 1.0
    values = _values(samples, "windage_heat_to_rotor_fraction")
    assert values.min() >= 0.5 and values.max() <= 1.0
    assert values.min() < 0.55 and values.max() > 0.95


def test_bias_and_residual_come_from_the_reference_case_comparisons(assumptions, errors):
    comparisons = [c for case in reference.load_cases(assumptions.reference_cases_dir)
                   for c in reference.compare(case, assumptions) if not c.skipped]
    for e in errors:
        rows = [c for c in comparisons if reference.QUANTITY_MECHANISM[c.quantity] == e.mechanism]
        assert len(rows) == e.compared
        if e.scale == "logit":
            # logit(measured) - logit(predicted), both scaled to the knob's physical range.
            low, high = assumptions.uncertainty_physical_range[e.knob]
            shifts = [_logit((c.measured - low) / (high - low)) - _logit((c.predicted - low) / (high - low))
                      for c in rows]
        else:
            shifts = [math.log(c.measured / c.predicted) for c in rows]  # ln(measured / predicted)
        assert e.bias == pytest.approx(np.mean(shifts))
        if len(shifts) > 1:
            assert e.residual_spread == pytest.approx(np.std(shifts, ddof=1))
        else:
            # One comparison cannot tell the bias from scatter: the spread is the whole error.
            assert e.residual_spread == pytest.approx(abs(shifts[0]))
    by_mechanism = {e.mechanism: e for e in errors}
    # The signed biases the Reference Cases show: the model under-predicts the discharge
    # coefficient by about 19% and the Fluted Cap Turbine efficiency by about 52%.
    assert 1.0 / by_mechanism["nozzle"].bias_correction - 1.0 == pytest.approx(-0.19, abs=0.01)
    assert 1.0 / by_mechanism["turbine"].bias_correction - 1.0 == pytest.approx(-0.52, abs=0.02)
    # Only the residual about that bias is left as spread, far below the rms error.
    assert by_mechanism["turbine"].residual_spread < 0.5 * by_mechanism["turbine"].rms_error


def _fixed_inputs(assumptions, **changes):
    """Every input held at its nominal value, so a knob moves by the Model-Form Error alone."""
    fixed = {name: ("uniform", _get(assumptions, name), _get(assumptions, name))
             for name in assumptions.uncertainty_inputs}
    return replace(assumptions, uncertainty_inputs=fixed, uncertainty_samples=MANY_SAMPLES, **changes)


def test_signed_bias_is_a_mean_correction_with_only_the_residual_as_spread(assumptions, errors):
    a = _fixed_inputs(assumptions)
    fixed_samples = sample_assumptions(a, errors)
    for e in errors:
        nominal, high = _get(a, e.knob), a.uncertainty_physical_range[e.knob][1]
        if e.knob == "turbine_efficiency":
            values = _turbine_efficiencies(fixed_samples, FLUTED_CAP)
        else:
            values = _values(fixed_samples, e.knob)
        # The median is the corrected knob, unless that lies past its physical bound.
        assert np.median(values) / nominal == pytest.approx(min(e.bias_correction, high / nominal),
                                                            rel=0.02), e.mechanism
        if values.max() < high:
            shifts = np.log(values / nominal) if e.scale == "log" else [_logit(v) for v in values]
            assert np.std(shifts) == pytest.approx(e.residual_spread, rel=0.1), e.mechanism


def test_discharge_coefficient_is_corrected_on_a_bounded_scale_and_not_piled_at_one(assumptions, errors, samples):
    nozzle = next(e for e in errors if e.mechanism == "nozzle")
    assert nozzle.scale == "logit"
    # The correction still lifts the model's 0.8 to the measured ~0.99 (a -19% bias) ...
    assert 1.0 / nozzle.bias_correction - 1.0 == pytest.approx(-0.19, abs=0.01)
    # ... but on the logit scale it can never reach 1, where a multiplier capped 39% of samples.
    values = _values(samples, "discharge_coefficient")
    assert values.max() < 1.0
    assert np.mean(values >= 0.999) < 0.02


def test_each_turbine_type_takes_its_own_named_bias_correction(assumptions, errors):
    turbine = next(e for e in errors if e.mechanism == "turbine")
    # By default both Turbine types share the Reference Case value, the Fluted Cap's -52%.
    assert assumptions.turbine_bias_correction == {FLUTED_CAP: "reference_cases", RADIAL_INFLOW: "reference_cases"}
    # The model as built applies no correction outside the Monte Carlo.
    assert assumptions.turbine_efficiency_correction == {FLUTED_CAP: 1.0, RADIAL_INFLOW: 1.0}
    shared = sample_assumptions(_fixed_inputs(assumptions), errors)
    high = assumptions.uncertainty_physical_range["turbine_efficiency"][1]
    for t in TURBINE_TYPES:
        assert np.median(_turbine_efficiencies(shared, t)) == pytest.approx(
            min(turbine.bias_correction, high), rel=0.02)
    # The Radial-Inflow value can be set on its own, leaving the Fluted Cap's as it was.
    own = sample_assumptions(_fixed_inputs(assumptions, turbine_bias_correction={
        FLUTED_CAP: "reference_cases", RADIAL_INFLOW: 1.2}), errors)
    assert np.median(_turbine_efficiencies(own, RADIAL_INFLOW)) == pytest.approx(1.2, rel=0.02)
    assert np.array_equal(_turbine_efficiencies(own, FLUTED_CAP), _turbine_efficiencies(shared, FLUTED_CAP))


def test_turbine_efficiency_correction_scales_only_its_turbine_type(assumptions):
    corrected = replace(assumptions, turbine_efficiency_correction={FLUTED_CAP: 1.0, RADIAL_INFLOW: 2.0})
    for turbine, changes in ((FLUTED_CAP, False), (RADIAL_INFLOW, True)):
        design = Design(rotor_length_mm=20.0, turbine=turbine)
        before = evaluate(design, assumptions).required_mass_flow_kg_s
        after = evaluate(design, corrected).required_mass_flow_kg_s
        assert (after < before) if changes else (after == before), turbine


def test_an_input_range_outside_its_physical_range_is_refused(assumptions, errors):
    inputs = {**assumptions.uncertainty_inputs, "discharge_coefficient": ("uniform", 0.6, 1.2)}
    with pytest.raises(ValueError, match="discharge_coefficient"):
        sample_assumptions(replace(assumptions, uncertainty_inputs=inputs), errors)
