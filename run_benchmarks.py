"""Compare the model with every Benchmark Case and Reference Case; prints the Model-Form Error per quantity and
per mechanism and writes Plots/benchmark_comparisons.csv and Plots/reference_comparisons.csv."""

import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path

from rotor_design import reference
from rotor_design.assumptions import load_assumptions
from rotor_design.benchmark import Comparison, compare, load_cases, model_form_error

HERE = Path(__file__).resolve().parent


def report(comparisons, out, label, quantities=None, names=None):
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Comparison)])
        writer.writeheader()
        writer.writerows(asdict(c) for c in comparisons)
    summary = model_form_error(comparisons) if quantities is None else model_form_error(comparisons, quantities)
    print(f"{label:<28}{'compared':>9}{'skipped':>9}{'mean':>10}{'rms':>10}{'worst':>10}")
    for q in summary:
        name = names[q.quantity] if names else q.quantity
        print(f"{name:<28}{q.compared:>9}{q.skipped:>9}{q.mean_error:>10.3g}{q.rms_error:>10.3g}"
              f"{q.worst_error:>10.3g}")
    print(f"Comparisons, with the reason for each skip, in {out}\n")


if __name__ == "__main__":
    assumptions = load_assumptions(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "assumptions.toml")
    report([c for case in load_cases(HERE / "benchmarks") for c in compare(case, assumptions)],
           HERE / "Plots" / "benchmark_comparisons.csv", "quantity")
    report([c for case in reference.load_cases(HERE / "references") for c in reference.compare(case, assumptions)],
           HERE / "Plots" / "reference_comparisons.csv", "mechanism", reference.QUANTITY_MECHANISM,
           reference.QUANTITY_MECHANISM)
