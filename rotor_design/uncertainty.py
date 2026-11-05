"""Monte Carlo over the Input Uncertainty, with the Model-Form Error on top.

Each sample is one copy of the assumptions: every uncertain input drawn from its
distribution, then the knob of each Reference Case mechanism multiplied by a lognormal
factor exp(b + s z), z standard normal. Over the mechanism's comparisons, b is the mean of
ln(measured / predicted), the signed bias, so the factor's median corrects the model to the
measurements; s is the standard deviation of the same logs about b, the residual, so only
the scatter the bias leaves widens the bands. A factor is always positive, and the knob is
then capped at the top of its physical range (a discharge coefficient at 1). Benchmark Case
comparisons are not used here: a whole-system error would double count the mechanism
errors it already contains.

Every Design is evaluated on the same samples, so Designs are compared under the same draws.
Percentiles take actual sample values (inverted CDF), so a band edge can be inf when
some samples never reach the Target Spin Rate.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from rotor_design import reference
from rotor_design.benchmark import model_form_error
from rotor_design.bearing import MECHANICAL, TWO_CENTER_PIVOT

# Results fields given a band.
BANDED = ("required_supply_pressure_bar", "required_mass_flow_kg_s", "sample_heat_load_w",
          "sample_hottest_temperature_k", "sample_zone_mm", "bath_heat_load_w")
# Uncertain inputs held in a Bearing model rather than on the assumptions themselves.
_BEARING_INPUTS = {"mechanical_friction_coefficient": (MECHANICAL, "friction_coefficient"),
                   "pivot_friction_coefficient": (TWO_CENTER_PIVOT, "friction_coefficient")}


@dataclass(frozen=True)
class MechanismError:
    mechanism: str
    knob: str            # the input its Model-Form Error spreads
    compared: int
    mean_error: float
    rms_error: float
    bias_correction: float  # median of the knob's factor: geometric mean of measured / predicted
    residual_spread: float  # log standard deviation of the factor, about the bias


def _bias_and_residual(errors):
    """Bias correction and residual spread from Model-Form Errors (predicted - measured) / measured.

    With one comparison the bias cannot be told from scatter, so the whole error is also kept
    as spread.
    """
    logs = [-math.log1p(e) for e in errors]
    bias = sum(logs) / len(logs)
    if len(logs) == 1:
        return math.exp(bias), abs(bias)
    return math.exp(bias), math.sqrt(sum((x - bias) ** 2 for x in logs) / (len(logs) - 1))


def mechanism_errors(assumptions):
    """Model-Form Error of every Reference Case mechanism that has at least one comparison."""
    a = assumptions
    comparisons = [c for case in reference.load_cases(a.reference_cases_dir) for c in reference.compare(case, a)]
    errors = []
    for q in model_form_error(comparisons, reference.QUANTITY_MECHANISM):
        mechanism = reference.QUANTITY_MECHANISM[q.quantity]
        if q.compared == 0:
            continue
        if mechanism not in a.model_form_knobs:
            raise ValueError(f"Reference Case mechanism {mechanism!r} has no knob in [uncertainty.model_form_knobs]")
        bias, residual = _bias_and_residual([c.model_form_error for c in comparisons
                                             if c.quantity == q.quantity and not c.skipped])
        errors.append(MechanismError(mechanism, a.model_form_knobs[mechanism], q.compared, q.mean_error,
                                     q.rms_error, bias, residual))
    return errors


def _get(a, name):
    if name in _BEARING_INPUTS:
        bearing, field = _BEARING_INPUTS[name]
        return getattr(a.bearings[bearing], field)
    return getattr(a, name)


def _set(a, name, value):
    if name in _BEARING_INPUTS:
        bearing, field = _BEARING_INPUTS[name]
        return replace(a, bearings={**a.bearings, bearing: replace(a.bearings[bearing], **{field: value})})
    if not hasattr(a, name):
        raise ValueError(f"Uncertain input {name!r} is not an assumption")
    return replace(a, **{name: value})


def _draw(rng, distribution, low, high):
    if distribution == "uniform":
        return float(rng.uniform(low, high))
    if distribution == "log_uniform":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    raise ValueError(f"Unknown distribution {distribution!r}; expected 'uniform' or 'log_uniform'")


def _check_physical_ranges(assumptions, errors):
    a = assumptions
    for name in [*a.uncertainty_inputs, *(e.knob for e in errors)]:
        if name not in a.uncertainty_physical_range:
            raise ValueError(f"Uncertain input {name!r} has no range in [uncertainty.physical_range]")
    for name, (_, low, high) in a.uncertainty_inputs.items():
        physical_low, physical_high = a.uncertainty_physical_range[name]
        if not physical_low < low <= high <= physical_high:
            raise ValueError(f"Uncertain input {name!r} range [{low}, {high}] leaves its physical range "
                             f"({physical_low}, {physical_high}]")


def sample_assumptions(assumptions, errors):
    """The Monte Carlo samples: one copy of the assumptions per sample, from the fixed seed.

    Every sampled input and knob lies in its physical range (low, high].
    """
    a = assumptions
    _check_physical_ranges(a, errors)
    rng = np.random.default_rng(a.uncertainty_seed)
    samples = []
    for _ in range(a.uncertainty_samples):
        s = a
        for name, (distribution, low, high) in a.uncertainty_inputs.items():
            s = _set(s, name, _draw(rng, distribution, low, high))
        for e in errors:
            value = _get(s, e.knob) * e.bias_correction * math.exp(e.residual_spread * rng.standard_normal())
            low, high = a.uncertainty_physical_range[e.knob]
            s = _set(s, e.knob, min(max(value, math.nextafter(low, math.inf)), high))
        samples.append(s)
    return samples


def bands(results_by_sample, assumptions):
    """One Design's band: lower percentile, median and upper percentile of each BANDED field,
    and the share of samples that are Feasible Designs.

    results_by_sample holds the Design's Results, one per sample.
    """
    lower, upper = assumptions.uncertainty_percentiles
    row = {"samples": len(results_by_sample),
           "feasible_probability": sum(r.feasible for r in results_by_sample) / len(results_by_sample)}
    for field in BANDED:
        values = np.array([getattr(r, field) for r in results_by_sample], dtype=float)
        p_lo, p_mid, p_hi = np.percentile(values, (lower, 50.0, upper), method="inverted_cdf")
        row.update({f"{field}_p{lower:g}": float(p_lo), f"{field}_median": float(p_mid),
                    f"{field}_p{upper:g}": float(p_hi)})
    return row
