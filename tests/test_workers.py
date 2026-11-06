"""Worker processes change only how fast the study runs, never its Results."""

from dataclasses import replace
from pathlib import Path

from rotor_design.assumptions import load_assumptions
from rotor_design.study import evaluate_all, evaluate_samples, study_designs
from rotor_design.uncertainty import mechanism_errors, sample_assumptions

ASSUMPTIONS_FILE = Path(__file__).resolve().parent.parent / "assumptions.toml"


def _same(x, y):
    # repr compares floats exactly and treats the nan fields (not applicable) as equal.
    return repr(x) == repr(y)


def test_monte_carlo_results_with_one_worker_equal_results_with_several_for_the_same_seed():
    a = replace(load_assumptions(ASSUMPTIONS_FILE), uncertainty_samples=3)
    designs = study_designs(a)[:2]
    samples = sample_assumptions(a, mechanism_errors(a))
    assert _same(evaluate_samples(designs, samples, workers=1), evaluate_samples(designs, samples, workers=2))
    assert _same(evaluate_all(designs, a, workers=1), evaluate_all(designs, a, workers=2))
