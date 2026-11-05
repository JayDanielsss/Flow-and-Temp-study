"""Design Sheet for one chosen Design: the hand-off an engineer draws the Rotor, caps, Turbine,
Stator, Nozzles and gaps from.

The Design is chosen by rank from the latest study ranked CSV (Plots/ranked_designs.csv by
default) or from a Design file (TOML of Design fields, at the top level or under [design]);
--set field=value overrides single Design fields on top. Writes one long-format CSV (a '#'
header with the validation status and ranked placeholders, one row per quantity, then the
results block) and a T(z) plot with the Sample marked, both in Plots/.

Each feasible range takes a few dozen evaluations; --ranges limits them to the named row keys
(Design fields such as nozzle_throat_mm, or e.g. rotor_stator_gap_mm), or 'none'.

  python run_design_sheet.py --rank 1 --set nozzle_count=8
  python run_design_sheet.py --design my_design.toml --ranges nozzle_throat_mm,sample_length_mm
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from rotor_design.assumptions import load_assumptions
from rotor_design.design_sheet import (apply_overrides, default_workers, design_from_file, design_from_ranked_csv,
                                       feasible_probability, make_sheet, plot_temperature, write_design_sheet)

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "Plots"


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Write a Design Sheet for one chosen Design.")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--rank", type=int, help="rank in the study ranked CSV (default 1)")
    choice.add_argument("--design", type=Path, help="Design file (TOML of Design fields)")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override one Design field; repeatable")
    parser.add_argument("--assumptions", type=Path, default=HERE / "assumptions.toml")
    parser.add_argument("--ranked-csv", type=Path, default=PLOTS / "ranked_designs.csv")
    parser.add_argument("--uncertainty-csv", type=Path, default=PLOTS / "uncertainty_top_designs.csv",
                        help="study Monte Carlo bands, read for P(feasible)")
    parser.add_argument("--ranges", default="all",
                        help="'all', 'none', or comma-separated row keys to give feasible ranges")
    parser.add_argument("--out", type=Path, help="CSV path (default Plots/design_sheet_<name>.csv)")
    parser.add_argument("--plot", type=Path, help="T(z) plot path (default Plots/design_sheet_<name>_temperature.png)")
    parser.add_argument("--workers", type=int, default=default_workers())
    return parser.parse_args(argv)


def main(argv=None):
    start = time.time()
    args = parse_args(argv)
    a = load_assumptions(args.assumptions)
    if args.design:
        design, provenance = design_from_file(args.design)
        name, label = args.design.stem, f"from Design file {args.design.name}"
    else:
        rank = args.rank or 1
        design, provenance = design_from_ranked_csv(args.ranked_csv, rank, a)
        name, label = f"rank{rank}", f"rank {rank} of {args.ranked_csv.name}"
    design, provenance = apply_overrides(design, provenance, args.set)
    if args.set:
        name, label = name + "_override", f"{label}, overrides {', '.join(args.set)}"

    sheet = make_sheet(design, a, provenance)
    sheet = replace(sheet, p_feasible=feasible_probability(sheet.results, args.uncertainty_csv))
    range_keys = {"all": None, "none": ()}.get(args.ranges, {k.strip() for k in args.ranges.split(",") if k.strip()})
    out = args.out or PLOTS / f"design_sheet_{name}.csv"
    plot = args.plot or PLOTS / f"design_sheet_{name}_temperature.png"
    rows, effects = write_design_sheet(sheet, out, label, range_keys, workers=args.workers)
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot_temperature(sheet, plot)

    r = sheet.results
    print(f"Design {label}: {'feasible' if r.feasible else 'not feasible'}, Binding Constraint "
          f"{r.binding_constraint}")
    print(f"Sample Heat Load {r.sample_heat_load_w * 1e3:.4f} mW, hottest Sample point "
          f"{r.sample_hottest_temperature_k:.4f} K, Bath Heat Load {r.bath_heat_load_w * 1e3:.3f} mW")
    print("Placeholders by shift in the hottest Sample point:")
    for e in effects[:5]:
        print(f"  {e.name}: {e.shift_k * 1e3:.2f} mK")
    print(f"{len(rows)} rows. Outputs:\n  {out}\n  {plot}")
    print(f"Run time {time.time() - start:.0f} s")
    return out, plot


if __name__ == "__main__":
    main(sys.argv[1:])
