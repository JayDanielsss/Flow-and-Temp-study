"""Monte Carlo samples: physical inputs, Model-Form Error as a bias correction plus residual spread."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rotor_design import reference
from rotor_design.assumptions import load_assumptions
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


def test_no_sample_has_zero_turbine_efficiency_or_a_discharge_coefficient_above_one(samples):
    assert _values(samples, "turbine_efficiency").min() > 0.0
    assert _values(samples, "discharge_coefficient").max() <= 1.0


def test_every_sampled_input_lies_in_its_physical_range(assumptions, samples):
    names = set(assumptions.uncertainty_inputs) | set(assumptions.model_form_knobs.values())
    assert names <= set(assumptions.uncertainty_physical_range)
    for name in names:
        low, high = assumptions.uncertainty_physical_range[name]
        values = _values(samples, name)
        assert np.all(values > low) and np.all(values <= high), name


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
        # ln(measured / predicted) of each comparison of the mechanism.
        logs = [-math.log1p(c.model_form_error) for c in comparisons
                if reference.QUANTITY_MECHANISM[c.quantity] == e.mechanism]
        assert len(logs) == e.compared
        assert e.bias_correction == pytest.approx(math.exp(np.mean(logs)))
        if len(logs) > 1:
            assert e.residual_spread == pytest.approx(np.std(logs, ddof=1))
        else:
            # One comparison cannot tell the bias from scatter: the spread is the whole error.
            assert e.residual_spread == pytest.approx(abs(logs[0]))
    by_mechanism = {e.mechanism: e for e in errors}
    # The signed biases the Reference Cases show: the model under-predicts the discharge
    # coefficient by about 19% and the Fluted Cap Turbine efficiency by about 52%.
    assert 1.0 / by_mechanism["nozzle"].bias_correction - 1.0 == pytest.approx(-0.19, abs=0.01)
    assert 1.0 / by_mechanism["turbine"].bias_correction - 1.0 == pytest.approx(-0.52, abs=0.02)
    # Only the residual about that bias is left as spread, far below the rms error.
    assert by_mechanism["turbine"].residual_spread < 0.5 * by_mechanism["turbine"].rms_error


def test_signed_bias_is_a_mean_correction_with_only_the_residual_as_spread(assumptions, errors, samples):
    # Hold every input at its nominal value so the knob moves by the Model-Form Error alone.
    fixed = {name: ("uniform", _get(assumptions, name), _get(assumptions, name))
             for name in assumptions.uncertainty_inputs}
    a = replace(assumptions, uncertainty_inputs=fixed, uncertainty_samples=MANY_SAMPLES)
    fixed_samples = sample_assumptions(a, errors)
    for e in errors:
        nominal, high = _get(a, e.knob), a.uncertainty_physical_range[e.knob][1]
        ratio = _values(fixed_samples, e.knob) / nominal
        # The median is the corrected knob, unless that lies past its physical bound.
        assert np.median(ratio) == pytest.approx(min(e.bias_correction, high / nominal), rel=0.02), e.mechanism
        if _values(fixed_samples, e.knob).max() < high:
            assert np.std(np.log(ratio)) == pytest.approx(e.residual_spread, rel=0.1), e.mechanism
    # Over the input range, the corrected discharge coefficient goes past 1 and is capped there.
    nozzle = next(e for e in errors if e.mechanism == "nozzle")
    assert assumptions.uncertainty_inputs["discharge_coefficient"][2] * nozzle.bias_correction > 1.0
    assert _values(samples, "discharge_coefficient").max() == 1.0


def test_an_input_range_outside_its_physical_range_is_refused(assumptions, errors):
    inputs = {**assumptions.uncertainty_inputs, "discharge_coefficient": ("uniform", 0.6, 1.2)}
    with pytest.raises(ValueError, match="discharge_coefficient"):
        sample_assumptions(replace(assumptions, uncertainty_inputs=inputs), errors)
