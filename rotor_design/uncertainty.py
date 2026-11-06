"""Monte Carlo over the Input Uncertainty, with the Model-Form Error on top.

Each sample is one copy of the assumptions: every uncertain input drawn from its
distribution, then the knob of each Reference Case mechanism shifted by b + s z, z standard
normal, on the mechanism's scale. On the "log" scale (the default) the shift is in ln(knob),
a lognormal factor exp(b + s z) on it; on the "logit" scale it is in logit(x), x the knob
scaled to its physical range, so the knob never reaches either bound. Over the mechanism's
comparisons, b is the mean of the same shift from predicted to measured, the signed bias, so
the median sample corrects the model to the measurements; s is the standard deviation about
b, the residual, so only the scatter the bias leaves widens the bands. The knob is then held
to its physical range, capped at its top.

The Turbine knob takes only the residual; its bias correction is set per Turbine type, as
turbine_efficiency_correction, from [uncertainty.turbine_bias_correction]. Benchmark Case
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
from rotor_design.turbine import TURBINE_TYPES

# Results fields given a band.
BANDED = ("required_supply_pressure_bar", "required_mass_flow_kg_s", "sample_heat_load_w",
          "sample_hottest_temperature_k", "sample_zone_mm", "bath_heat_load_w")
# Uncertain inputs held in a Bearing model rather than on the assumptions themselves.
_BEARING_INPUTS = {"mechanical_friction_coefficient": (MECHANICAL, "friction_coefficient"),
                   "pivot_friction_coefficient": (TWO_CENTER_PIVOT, "friction_coefficient")}
# The knob whose bias correction is set per Turbine type rather than on the knob itself.
_TURBINE_KNOB = "turbine_efficiency"
# A Turbine type's bias correction taken from the Reference Cases.
_FROM_REFERENCE_CASES = "reference_cases"
SCALES = ("log", "logit")


@dataclass(frozen=True)
class MechanismError:
    mechanism: str
    knob: str            # the input its Model-Form Error spreads
    compared: int
    mean_error: float
    rms_error: float
    scale: str              # "log" or "logit": where the bias and residual are taken and applied
    bias: float             # mean shift from predicted to measured, on that scale
    residual_spread: float  # standard deviation of the shift about the bias, on that scale
    bias_correction: float  # the bias as a factor on the knob's nominal value


def _to_scale(value, scale, low, high):
    if scale == "log":
        return math.log(value)
    x = (value - low) / (high - low)
    return math.log(x / (1.0 - x))


def _from_scale(y, scale, low, high):
    if scale == "log":
        return math.exp(y)
    return low + (high - low) / (1.0 + math.exp(-y))


def _bias_and_residual(comparisons, scale, low, high):
    """Mean and residual standard deviation of the shift from predicted to measured.

    With one comparison the bias cannot be told from scatter, so the whole shift is also kept
    as spread.
    """
    shifts = [_to_scale(c.measured, scale, low, high) - _to_scale(c.predicted, scale, low, high)
              for c in comparisons]
    bias = sum(shifts) / len(shifts)
    if len(shifts) == 1:
        return bias, abs(bias)
    return bias, math.sqrt(sum((x - bias) ** 2 for x in shifts) / (len(shifts) - 1))


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
        knob = a.model_form_knobs[mechanism]
        scale = a.model_form_scale.get(mechanism, "log")
        if scale not in SCALES:
            raise ValueError(f"Reference Case mechanism {mechanism!r}: scale {scale!r}; expected one of {SCALES}")
        low, high = a.uncertainty_physical_range.get(knob, (0.0, math.inf))
        bias, residual = _bias_and_residual([c for c in comparisons if c.quantity == q.quantity and not c.skipped],
                                            scale, low, high)
        nominal = _get(a, knob)
        correction = _from_scale(_to_scale(nominal, scale, low, high) + bias, scale, low, high) / nominal
        errors.append(MechanismError(mechanism, knob, q.compared, q.mean_error, q.rms_error, scale, bias, residual,
                                     correction))
    return errors


def _turbine_corrections(assumptions, errors):
    """Turbine type -> the bias correction its efficiency takes in the Monte Carlo."""
    a = assumptions
    derived = next((e.bias_correction for e in errors if e.knob == _TURBINE_KNOB), 1.0)
    if set(a.turbine_bias_correction) != set(TURBINE_TYPES):
        raise ValueError(f"[uncertainty.turbine_bias_correction] must list every Turbine type {TURBINE_TYPES}")
    return {t: derived if c == _FROM_REFERENCE_CASES else float(c) for t, c in a.turbine_bias_correction.items()}


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
    turbine_corrections = _turbine_corrections(a, errors)
    rng = np.random.default_rng(a.uncertainty_seed)
    samples = []
    for _ in range(a.uncertainty_samples):
        s = a
        for name, (distribution, low, high) in a.uncertainty_inputs.items():
            s = _set(s, name, _draw(rng, distribution, low, high))
        for e in errors:
            low, high = a.uncertainty_physical_range[e.knob]
            # The Turbine's bias goes on per Turbine type below, so its knob takes the residual alone.
            bias = 0.0 if e.knob == _TURBINE_KNOB else e.bias
            y = _to_scale(_get(s, e.knob), e.scale, low, high) + bias + e.residual_spread * rng.standard_normal()
            value = min(max(_from_scale(y, e.scale, low, high), math.nextafter(low, math.inf)), high)
            s = _set(s, e.knob, value)
            if e.knob == _TURBINE_KNOB:
                # Each type's efficiency, knob times correction, is capped at the knob's top.
                s = replace(s, turbine_efficiency_correction={t: min(c, high / value)
                                                              for t, c in turbine_corrections.items()})
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
