"""Run the study: sweep the assumptions file's grids, write CSVs and plots."""

import csv
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import (COLD_SUPPORT, CONSTRAINTS, DUCTED, VAPOUR_ONLY, Design, configurations,
                                   evaluate, largest_rotor_diameter_mm)
from rotor_design.turbine import FLUTED_CAP, RADIAL_INFLOW, TURBINE_TYPES
from rotor_design.uncertainty import bands, mechanism_errors, sample_assumptions

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# Categorical slots 1-3 of the reference palette, in fixed order.
RANK_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
TOP_DESIGNS = len(RANK_COLOURS)
# The full reference categorical order; constraint maps take one slot per constraint, in build order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CONSTRAINT_COLOURS = dict(zip(CONSTRAINTS, CATEGORICAL))
_CONSTRAINT_LABELS = {"Probe Bore": "Bore", "Supply Pressure": "Supply", "Stress": "Stress",
                      "Critical Speed": "Critical", "Whirl": "Whirl", "Placement Window": "Window",
                      "Sample Temperature": "Temp", "Bath Heat Load": "Bath"}
# The Configuration's discrete choices, as Design fields.
CONFIGURATION_FIELDS = ("bearing", "pivot_to_bath", "cooling_path", "turbine", "exhaust_path")
# Design fields that set the drive, not the Rotor. Nozzle setups of one Rotor tie on Sample Heat Load.
NOZZLE_FIELDS = ("nozzle_count", "nozzle_throat_mm", "nozzle_angle_deg")
# Grid fields that apply only to some Configurations.
_APPLIES_WHEN = {
    "cold_support_gap_mm": lambda c: c["cooling_path"] == COLD_SUPPORT,
    "hub_exit_diameter_mm": lambda c: c["turbine"] == RADIAL_INFLOW,
    "back_pressure_bar": lambda c: c["exhaust_path"] == DUCTED,
}
# Slack for a swept diameter that sits exactly on the bore limit (mm, or fraction of the largest).
_DIAMETER_TOLERANCE = 1e-9


def configuration_name(configuration):
    """A short, file-safe name for a Configuration (a dict of its Design fields)."""
    bearing = configuration["bearing"] + ("_to_bath" if configuration["pivot_to_bath"] else "")
    return "-".join((bearing, configuration["cooling_path"], configuration["turbine"],
                     configuration["exhaust_path"]))


def study_designs(assumptions, grid=None):
    """Every allowed Configuration crossed with the full factorial of the grid's values.

    Rotor diameter is swept as rotor_diameter_mm or as rotor_diameter_fraction (of the largest the
    Probe Bore allows), not both; diameters above the bore limit for the Design's Rotor Length and
    Bearing type are skipped, as are Sample lengths that do not fit inside the Rotor body.
    """
    a = assumptions
    grid = a.study_grid if grid is None else grid
    design_fields = {f.name for f in fields(Design)}
    unknown = set(grid) - design_fields | set(grid) & set(CONFIGURATION_FIELDS)
    if unknown:
        raise ValueError(f"Study grid keys must be continuous Design fields, not {sorted(unknown)}")
    if {"rotor_diameter_mm", "rotor_diameter_fraction"} <= set(grid):
        raise ValueError("Sweep the Rotor diameter or its fraction of the largest allowed, not both")
    designs = []
    for c in configurations():
        swept = {f: v for f, v in grid.items() if _APPLIES_WHEN.get(f, lambda _: True)(c)}
        for values in itertools.product(*swept.values()):
            design = Design(**c, **dict(zip(swept, values)))
            sample_mm = _or(design.sample_length_mm, a.baseline_sample_length_mm)
            if sample_mm > design.rotor_length_mm - a.turbine_cap_height_mm - a.bottom_cap_thickness_mm:
                continue
            if _above_bore_limit(design, a):
                continue
            designs.append(design)
    return designs


def _above_bore_limit(design, assumptions):
    if design.rotor_diameter_fraction is not None:
        return design.rotor_diameter_fraction > 1.0 + _DIAMETER_TOLERANCE
    if design.rotor_diameter_mm is None:
        return False
    largest = largest_rotor_diameter_mm(design.rotor_length_mm, design.bearing, assumptions)
    return design.rotor_diameter_mm > largest + _DIAMETER_TOLERANCE


def _or(value, default):
    return default if value is None else value


# Worker processes used when the caller gives none. Worker processes re-import the caller's main
# module, which fails (BrokenProcessPool) when that module is stdin or an interactive session, so
# the default is one (this process) and script entry points such as run_study.py ask for more.
DEFAULT_WORKERS = 1


def _workers(workers):
    return max(1, workers or DEFAULT_WORKERS)


def evaluate_all(designs, assumptions, workers=None):
    """Evaluate Designs in order, over worker processes when workers > 1 (default: one, in this process)."""
    designs = list(designs)
    workers = _workers(workers)
    if workers == 1 or len(designs) < 2:
        return [evaluate(d, assumptions) for d in designs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunk = max(1, len(designs) // (8 * workers))
        return list(pool.map(partial(evaluate, assumptions=assumptions), designs, chunksize=chunk))


def _evaluate_pair(pair):
    return evaluate(*pair)


def evaluate_samples(designs, samples, workers=None):
    """Each Design evaluated on every Monte Carlo sample of the assumptions: one list of Results per Design.

    Over worker processes when workers > 1 (default: one, in this process).
    """
    pairs = [(d, s) for d in designs for s in samples]
    workers = _workers(workers)
    if workers == 1 or len(pairs) < 2:
        flat = [evaluate(d, s) for d, s in pairs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            flat = list(pool.map(_evaluate_pair, pairs, chunksize=max(1, len(pairs) // (8 * workers))))
    return [flat[i:i + len(samples)] for i in range(0, len(flat), len(samples))]


def run_study(assumptions_path, plots_dir, rotor_lengths_mm=None, nozzle_sweeps=None, cold_support_gaps_mm=None,
              grid=None, workers=None, uncertainty_samples=None, uncertainty_designs=None):
    """Run the full study and the one-variable sweeps; return the full study's Results.

    The full study evaluates every allowed Configuration over the full factorial of the grid
    (the assumptions file's [study.grid] unless grid is given, a dict of Design field to values).
    Its Feasible Designs are ranked by lowest Sample Heat Load, ties by lowest Drive Gas flow, into
    ranked_designs.csv; it also writes a constraint map per Configuration (constraint_maps.csv and one
    plot each), Sample Heat Load against Sample volume, and T(z) profiles and Sample Heat Load
    breakdowns of the top Designs. The top Designs are distinct Rotors (Feasible first), each at its
    best-ranked Nozzle setup; they are then run through the Monte Carlo over the Input
    Uncertainty and Model-Form Error: uncertainty_top_designs.csv and its plot give each one's
    median and percentile band and its probability of being a Feasible Design, and
    uncertainty_model_form_error.csv the bias correction and residual spread taken from each
    Reference Case mechanism.
    uncertainty_samples and uncertainty_designs override the assumptions file's counts.

    nozzle_sweeps maps a Design field (nozzle_count, nozzle_throat_mm,
    nozzle_angle_deg) to the values tried, one field at a time about the baseline.
    Cold Support gaps are tried at the longest Rotor and compared with Vapour Cooling alone.
    Every allowed Configuration (Bearing type, pivot-to-Bath option, cooling path, Turbine type)
    is evaluated at the longest Rotor; excluded combinations are never evaluated.
    Both Turbine types are compared on torque per unit Drive Gas flow at each Rotor Length.
    """
    a = load_assumptions(assumptions_path)
    a = replace(a, uncertainty_samples=uncertainty_samples or a.uncertainty_samples,
                uncertainty_designs=uncertainty_designs or a.uncertainty_designs)
    run = partial(evaluate_all, assumptions=a, workers=workers)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    designs = study_designs(a, grid)
    study = run(designs)
    # Feasible Designs first, each group by lowest Sample Heat Load.
    order = sorted(range(len(study)), key=lambda i: (not study[i].feasible, study[i].sample_heat_load_w,
                                                     study[i].required_mass_flow_kg_s))
    ranked = [study[i] for i in order]
    best_per_rotor = {}
    for i in order:
        best_per_rotor.setdefault(_rotor(study[i]), i)
    top = list(best_per_rotor.values())
    feasible = [r for r in ranked if r.feasible]
    _write_rows([{"rank": i + 1, "configuration": configuration_name(_configuration(r)), **asdict(r)}
                 for i, r in enumerate(feasible)], plots_dir / "ranked_designs.csv",
                fieldnames=["rank", "configuration", *_flatten(asdict(study[0]))])
    _plot_profiles([study[i] for i in top[:TOP_DESIGNS]], a, plots_dir / "temperature_profiles_top_designs.png")
    _write_constraint_maps(study, a, plots_dir)
    _plot_trade_off(feasible, a, plots_dir / "sample_heat_load_vs_sample_volume.png")
    _plot_breakdown([study[i] for i in top[:a.study_top_designs]], plots_dir / "sample_heat_load_breakdown_top_designs.png")

    errors = mechanism_errors(a)
    _write_rows([asdict(e) for e in errors], plots_dir / "uncertainty_model_form_error.csv")
    top = top[:a.uncertainty_designs]
    by_design = evaluate_samples([designs[i] for i in top], sample_assumptions(a, errors), workers)
    band_rows = [{"rank": rank + 1, "configuration": configuration_name(_configuration(study[i])),
                  **bands(results, a), **asdict(study[i])}
                 for rank, (i, results) in enumerate(zip(top, by_design))]
    _write_rows(band_rows, plots_dir / "uncertainty_top_designs.csv")
    _plot_uncertainty(band_rows, a, plots_dir / "uncertainty_top_designs.png")

    if rotor_lengths_mm is None:
        rotor_lengths_mm = np.arange(
            a.rotor_length_sweep_min_mm,
            a.max_rotor_length_mm + a.rotor_length_sweep_step_mm / 2,
            a.rotor_length_sweep_step_mm,
        )
    results = run(Design(rotor_length_mm=float(length)) for length in rotor_lengths_mm)
    _write_csv(results, plots_dir / "probe_bore_fit.csv")
    _plot_max_diameter(results, a, plots_dir / "max_rotor_diameter_vs_rotor_length.png")

    if nozzle_sweeps is None:
        nozzle_sweeps = {
            "nozzle_count": a.nozzle_count_sweep,
            "nozzle_throat_mm": a.nozzle_throat_sweep_mm,
            "nozzle_angle_deg": a.nozzle_angle_sweep_deg,
        }
    swept = [(field, value) for field, values in nozzle_sweeps.items() for value in values]
    nozzle_results = run(Design(rotor_length_mm=a.max_rotor_length_mm, **{field: value}) for field, value in swept)
    nozzle_rows = [{"swept_variable": field, **asdict(r)} for (field, _), r in zip(swept, nozzle_results)]
    _write_rows(nozzle_rows, plots_dir / "nozzle_geometry_sweep.csv")
    _plot_nozzle_sweep(nozzle_rows, a, plots_dir / "required_flow_vs_nozzle_geometry.png")

    if cold_support_gaps_mm is None:
        cold_support_gaps_mm = a.cold_support_gap_sweep_mm
    vapour, *cold = run([Design(rotor_length_mm=a.max_rotor_length_mm, cooling_path=VAPOUR_ONLY)]
                        + [Design(rotor_length_mm=a.max_rotor_length_mm, cooling_path=COLD_SUPPORT,
                                  cold_support_gap_mm=float(gap)) for gap in cold_support_gaps_mm])
    _write_csv([vapour, *cold], plots_dir / "cold_support_gap_sweep.csv")
    _plot_cold_support_gap(vapour, cold, a, plots_dir / "sample_heat_load_vs_cold_support_gap.png")

    _write_csv(run(Design(rotor_length_mm=a.max_rotor_length_mm, **c) for c in configurations()),
               plots_dir / "bearing_configurations.csv")

    turbine_results = run(Design(rotor_length_mm=float(length), turbine=turbine)
                          for turbine in TURBINE_TYPES for length in rotor_lengths_mm)
    by_turbine = {turbine: [r for r in turbine_results if r.turbine == turbine] for turbine in TURBINE_TYPES}
    _write_csv(turbine_results, plots_dir / "turbine_comparison.csv")
    _plot_turbine_comparison(by_turbine, a, plots_dir / "torque_per_flow_by_turbine_type.png")
    return study


def _configuration(result):
    return {f: getattr(result, f) for f in CONFIGURATION_FIELDS}


def _rotor(result):
    """The Rotor a Design builds: every Design field but the Nozzle geometry (nan read as unset)."""
    values = (getattr(result, f.name) for f in fields(Design) if f.name not in NOZZLE_FIELDS)
    return tuple(None if isinstance(v, float) and math.isnan(v) else v for v in values)


def _cell_best(results):
    """The most promising Design in a map cell: the Feasible one with the lowest Sample Heat Load,
    or else the one ruled out latest in build order (it got furthest), then by Sample Heat Load."""
    return min(results, key=lambda r: (not r.feasible, -CONSTRAINTS.index(r.binding_constraint),
                                       r.sample_heat_load_w))


def _write_constraint_maps(study, assumptions, plots_dir):
    """Per Configuration, the Binding Constraint of each map cell's most promising Design."""
    x_field, y_field = assumptions.constraint_map_axes
    cells = {}
    for r in study:
        key = (configuration_name(_configuration(r)), getattr(r, x_field), getattr(r, y_field))
        cells.setdefault(key, []).append(r)
    rows = []
    for (name, x, y), results in sorted(cells.items()):
        best = _cell_best(results)
        # The best Design's diameter is shown even when diameter is not an axis.
        rows.append({"configuration": name, "rotor_diameter_mm": best.rotor_diameter_mm, x_field: x, y_field: y,
                     "binding_constraint": best.binding_constraint, "probe_bore_margin": best.probe_bore_margin,
                     "feasible": best.feasible, "feasible_designs": sum(r.feasible for r in results),
                     "designs": len(results), "best_sample_heat_load_w": best.sample_heat_load_w})
    _write_rows(rows, plots_dir / "constraint_maps.csv")
    for name in dict.fromkeys(r["configuration"] for r in rows):
        _plot_constraint_map([r for r in rows if r["configuration"] == name], name, x_field, y_field,
                             assumptions, plots_dir / f"constraint_map_{name}.png")


_AXIS_LABELS = {
    "rotor_length_mm": "Rotor Length (mm)",
    "rotor_diameter_mm": "Rotor diameter (mm)",
    "rotor_diameter_fraction": "Rotor diameter, fraction of the largest allowed",
    "wall_thickness_mm": "Rotor wall thickness (mm)",
    "sample_length_mm": "Sample length (mm)",
    "drive_gas_temperature_k": "Drive Gas temperature (K)",
    "cold_support_gap_mm": "Cold Support gap (mm)",
    "hub_exit_diameter_mm": "Hub Exit diameter (mm)",
    "back_pressure_bar": "Back Pressure (bar)",
}


def _plot_constraint_map(rows, name, x_field, y_field, assumptions, path):
    xs = sorted({r[x_field] for r in rows})
    ys = sorted({r[y_field] for r in rows})
    fig, ax = plt.subplots(figsize=(max(5.0, 1.0 * len(xs) + 2.5), max(3.2, 0.7 * len(ys) + 2.2)),
                           facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for r in rows:
        i, j = xs.index(r[x_field]), ys.index(r[y_field])
        # A 2 px surface gap between neighbouring cells.
        ax.add_patch(matplotlib.patches.Rectangle((i - 0.5, j - 0.5), 1.0, 1.0,
                                                  facecolor=CONSTRAINT_COLOURS[r["binding_constraint"]],
                                                  edgecolor=SURFACE, linewidth=2))
        label = _CONSTRAINT_LABELS[r["binding_constraint"]]
        if r["feasible"]:
            label = f"\u2713 {label}\n{r['feasible_designs']}/{r['designs']}"
        ax.text(i, j, label, ha="center", va="center", color=INK, fontsize=8,
                fontweight="bold" if r["feasible"] else "normal")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-0.5, len(ys) - 0.5)
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    ax.set_xlabel(_AXIS_LABELS.get(x_field, _SWEEP_LABELS.get(x_field, x_field)), color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(_AXIS_LABELS.get(y_field, _SWEEP_LABELS.get(y_field, y_field)), color=INK_SECONDARY, fontsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    present = [c for c in CONSTRAINTS if any(r["binding_constraint"] == c for r in rows)]
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=CONSTRAINT_COLOURS[c], label=c) for c in present],
              frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title(f"Binding Constraint, {name.replace('-', ', ')}\n"
                 f"best Design per cell; \u2713 feasible, with feasible/total Designs",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _write_csv(results, path):
    _write_rows([asdict(r) for r in results], path)


def _flatten(row):
    """One CSV row: profile arrays dropped, the heat-load breakdown spread into columns."""
    flat = {}
    for key, value in row.items():
        if key == "sample_heat_load_breakdown_w":
            flat.update({f"heat_{source}_w": watts for source, watts in value.items()})
        elif not isinstance(value, tuple):
            flat[key] = value
    return flat


def _write_rows(rows, path, fieldnames=None):
    rows = [_flatten(r) for r in rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_max_diameter(results, assumptions, path):
    lengths = [r.rotor_length_mm for r in results]
    diameters = [r.max_rotor_diameter_mm for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(lengths, diameters, color=SERIES, linewidth=2)
    ax.plot(lengths[-1], diameters[-1], "o", color=SERIES, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{diameters[-1]:.2f} mm at {lengths[-1]:g} mm",
                (lengths[-1], diameters[-1]), xytext=(-12, -6), textcoords="offset points",
                ha="right", va="top", color=INK, fontsize=9)

    ax.set_title(f"Largest Rotor diameter that fits the {assumptions.bore_diameter_mm:g} mm Probe Bore",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Rotor Length (mm)", color=INK_SECONDARY)
    ax.set_ylabel("Largest Rotor diameter (mm)", color=INK_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


_SWEEP_LABELS = {
    "nozzle_count": "Nozzle count",
    "nozzle_throat_mm": "Nozzle throat diameter (mm)",
    "nozzle_angle_deg": "Nozzle angle from rim tangent (deg)",
}


def _plot_nozzle_sweep(rows, assumptions, path):
    fields = list(dict.fromkeys(r["swept_variable"] for r in rows))
    fig, axes = plt.subplots(2, len(fields), figsize=(3.6 * len(fields), 6), sharey="row",
                             facecolor=SURFACE, squeeze=False)
    limit = assumptions.max_supply_pressure_bar
    for col, field in enumerate(fields):
        sub = [r for r in rows if r["swept_variable"] == field]
        x = [r[field] for r in sub]
        flow = [r["required_mass_flow_kg_s"] * 1e6 for r in sub]  # mg/s
        pressure = [r["required_supply_pressure_bar"] for r in sub]
        baseline = getattr(assumptions, "baseline_" + field)
        for row, (ax, y) in enumerate(zip(axes[:, col], (flow, pressure))):
            ax.set_facecolor(SURFACE)
            finite = [(xi, yi) for xi, yi in zip(x, y) if math.isfinite(yi)]
            if finite:
                ax.plot(*zip(*finite), color=SERIES, linewidth=2, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.axvline(baseline, color=INK_SECONDARY, linewidth=1, linestyle="--", alpha=0.6, zorder=1)
            if row == 1:
                ax.axhline(limit, color=INK_SECONDARY, linewidth=1, linestyle=":")
                ax.set_xlabel(_SWEEP_LABELS.get(field, field), color=INK_SECONDARY, fontsize=9)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    axes[0, 0].set_ylabel("Required Drive Gas flow (mg/s)", color=INK_SECONDARY, fontsize=9)
    axes[1, 0].set_ylabel("Required Supply Pressure (bar)", color=INK_SECONDARY, fontsize=9)
    axes[1, 0].set_yscale("log")
    axes[1, -1].annotate(f"{limit:g} bar limit", (1.0, limit), xycoords=("axes fraction", "data"),
                         xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                         color=INK_SECONDARY, fontsize=8)
    fig.suptitle(
        f"Drive Gas needed at {assumptions.target_spin_rate_hz:g} Hz, "
        f"{assumptions.max_rotor_length_mm:g} mm Rotor (dashed: baseline)",
        color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_cold_support_gap(vapour, cold, assumptions, path):
    gaps = [r.cold_support_gap_mm for r in cold]
    bath = assumptions.bath_temperature_k
    panels = (
        ("Sample Heat Load (mW)", lambda r: r.sample_heat_load_w * 1e3),
        ("Sample hottest-point rise above Bath (mK)", lambda r: (r.sample_hottest_temperature_k - bath) * 1e3),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (label, value) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(gaps, [value(r) for r in cold], color=SERIES, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(value(vapour), color=INK_SECONDARY, linewidth=1, linestyle=":")
        ax.annotate("Vapour Cooling alone", (1.0, value(vapour)), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                    color=INK_SECONDARY, fontsize=8)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.4g}"))
        ax.set_xlabel("Cold Support gap (mm)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(label, color=INK_SECONDARY, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    fig.suptitle(
        f"Vapour Cooling plus Cold Support, {assumptions.max_rotor_length_mm:g} mm Rotor "
        f"at {assumptions.target_spin_rate_hz:g} Hz (dotted: Vapour Cooling alone)",
        color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


_TURBINE_LABELS = {FLUTED_CAP: "Fluted Cap", RADIAL_INFLOW: "Radial-Inflow"}


def _plot_turbine_comparison(by_turbine, assumptions, path):
    hub = assumptions.turbines[RADIAL_INFLOW].hub_exit_diameter_mm
    panels = (
        ("Torque per unit Drive Gas flow (mN m per g/s)", lambda r: r.turbine_torque_per_flow_n_m_s_kg),
        ("Required Drive Gas flow (mg/s)", lambda r: r.required_mass_flow_kg_s * 1e6),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (label, value) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for (turbine, rows), colour in zip(by_turbine.items(), RANK_COLOURS):
            points = [(r.rotor_length_mm, value(r)) for r in rows if math.isfinite(value(r))]
            if not points:
                continue
            ax.plot(*zip(*points), color=colour, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=_TURBINE_LABELS[turbine])
        ax.set_xlabel("Rotor Length (mm)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(label, color=INK_SECONDARY, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        f"Turbine types at {assumptions.target_spin_rate_hz:g} Hz, largest Rotor that fits "
        f"(Radial-Inflow hub exit {hub:g} mm)",
        color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


_BEARING_LABELS = {"gas": "Gas bearing", "mechanical": "Mechanical bearing",
                   "two_center_pivot": "Two-center pivots"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=INK_SECONDARY, labelsize=8)


def _plot_trade_off(feasible, assumptions, path):
    """Sample Heat Load against Sample volume for every Feasible Design, by Bearing type,
    with the frontier of the lowest load at each volume or more."""
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    for (bearing, label), colour in zip(_BEARING_LABELS.items(), RANK_COLOURS):
        points = [(r.sample_volume_mm3, r.sample_heat_load_w * 1e3) for r in feasible if r.bearing == bearing]
        if points:
            ax.plot(*zip(*points), "o", color=colour, markersize=6, alpha=0.8,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    frontier, lowest = [], math.inf
    for r in sorted(feasible, key=lambda r: -r.sample_volume_mm3):
        if r.sample_heat_load_w < lowest:
            lowest = r.sample_heat_load_w
            frontier.append((r.sample_volume_mm3, lowest * 1e3))
    if frontier:
        ax.step(*zip(*frontier[::-1]), where="post", color=INK_SECONDARY, linewidth=1.2,
                label="Lowest load at this volume or more")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    else:
        ax.text(0.5, 0.5, "No Feasible Designs", transform=ax.transAxes, ha="center", va="center",
                color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title(f"Sample Heat Load against Sample volume, Feasible Designs at "
                 f"{assumptions.target_spin_rate_hz:g} Hz", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Sample volume (mm\u00b3)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Sample Heat Load (mW)", color=INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


_SOURCE_LABELS = {
    "turbine_cap_conduction": "Turbine Cap conduction",
    "windage": "Windage",
    "bearing_friction": "Bearing friction",
    "convection": "Convection from warmer gas",
    "radiation": "Radiation",
    "internal_load": "Sample internal load",
    "cold_support_shear": "Cold Support gap shear",
}


def _plot_breakdown(results, path):
    """Each top Design's Sample Heat Load as shares by source, total at the bar end."""
    sources = list(_SOURCE_LABELS)
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(results) + 2.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(results):
        heat = {k: max(r.sample_heat_load_breakdown_w.get(k, 0.0), 0.0) for k in sources}
        total = sum(heat.values())
        left = 0.0
        for source, colour in zip(sources, CATEGORICAL):
            share = 100.0 * heat[source] / total if total > 0 else 0.0
            ax.barh(i, share, left=left, height=0.7, color=colour, edgecolor=SURFACE, linewidth=2,
                    label=_SOURCE_LABELS[source] if i == 0 else None)
            left += share
        status = "" if r.feasible else ", infeasible"
        ax.text(101, i, _format_watts(r.sample_heat_load_w) + status, va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(results)),
                  [f"#{i + 1} {configuration_name(_configuration(r)).replace('-', ', ')}\n"
                   f"{r.rotor_length_mm:g} x {r.rotor_diameter_mm:.3g} mm Rotor, {r.wall_thickness_mm:g} mm wall, "
                   f"{r.sample_length_mm:g} mm Sample, {r.nozzle_count} x {r.nozzle_throat_mm:g} mm "
                   f"Nozzles at {r.nozzle_angle_deg:g} deg" for i, r in enumerate(results)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of Sample Heat Load (%)", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.08))
    ax.set_title("Sample Heat Load by source, top-ranked Rotors at their best Nozzle setup", color=INK,
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _format_watts(watts):
    return f"{watts * 1e3:.2f} mW" if watts >= 1e-4 else f"{watts * 1e6:.1f} uW"


def _plot_profiles(results, assumptions, path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bath = assumptions.bath_temperature_k
    limit = assumptions.sample_limit_temperature_k
    for i, (r, colour) in enumerate(zip(results, RANK_COLOURS)):
        status = "" if r.feasible else ", infeasible"
        rise = np.maximum(np.array(r.temperature_k) - bath, 1e-6)
        ax.plot(r.z_mm, rise, color=colour, linewidth=2,
                label=f"#{i + 1}: {r.rotor_length_mm:g} x {r.rotor_diameter_mm:.3g} mm Rotor, "
                      f"{_format_watts(r.sample_heat_load_w)}{status}")
    ax.axhline(limit - bath, color=INK_SECONDARY, linewidth=1, linestyle=":",
               label=f"Sample Limit Temperature, {limit:g} K")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("Temperature rise above the Bath along the Rotor body, top-ranked Rotors",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Distance from Bottom Cap (mm); Turbine Cap at the right end", color=INK_SECONDARY)
    ax.set_ylabel(f"T - {bath:g} K (K)", color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_uncertainty(rows, assumptions, path):
    """Median and percentile band of the key results for each top-ranked Design, with its feasibility probability."""
    lower, upper = assumptions.uncertainty_percentiles
    panels = (
        ("Sample Heat Load (mW)", "sample_heat_load_w", 1e3, None),
        ("Required Supply Pressure (bar)", "required_supply_pressure_bar", 1.0, assumptions.max_supply_pressure_bar),
        ("Bath Heat Load (mW)", "bath_heat_load_w", 1e3, assumptions.bath_cooling_power_w * 1e3),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 0.45 * len(rows) + 2.2), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, (label, field, scale, limit) in zip(axes[0], panels):
        _style(ax)
        ax.grid(axis="y", visible=False)
        edges = [(r[f"{field}_p{lower:g}"] * scale, r[f"{field}_median"] * scale, r[f"{field}_p{upper:g}"] * scale)
                 for r in rows]
        finite = [v for e in edges for v in e if math.isfinite(v) and v > 0.0]
        if limit is not None:
            finite.append(limit)
        # A band edge that never reaches the Target Spin Rate is drawn to the axis end with an arrow.
        right = max(finite) * 1.5 if finite else 1.0
        for i, (lo, mid, hi) in enumerate(edges):
            ax.plot([lo, min(hi, right)], [i, i], color=SERIES, linewidth=2, solid_capstyle="round")
            if not math.isfinite(hi):
                ax.plot(right, i, ">", color=SERIES, markersize=6)
            if math.isfinite(mid):
                ax.plot(mid, i, "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        if limit is not None:
            ax.axvline(limit, color=INK_SECONDARY, linewidth=1, linestyle=":")
            ax.annotate("limit", (limit, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", ha="left", va="top", color=INK_SECONDARY, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(min(finite) / 1.5 if finite else None, right)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(label, color=INK_SECONDARY, fontsize=9)
    last = axes[0, -1]
    for i, r in enumerate(rows):
        last.annotate(f"P(feasible) {r['feasible_probability']:.0%}", (1.02, i), xycoords=("axes fraction", "data"),
                      ha="left", va="center", color=INK, fontsize=8)
    axes[0, 0].set_yticks(range(len(rows)),
                          [f"#{r['rank']} {r['configuration'].replace('-', ', ')}\n"
                           f"{r['rotor_length_mm']:g} x {r['rotor_diameter_mm']:.3g} mm Rotor, "
                           f"{r['wall_thickness_mm']:g} mm wall, "
                           f"{r['sample_length_mm']:g} mm Sample"
                           + (f", {r['cold_support_gap_mm']:g} mm gap" if math.isfinite(r['cold_support_gap_mm']) else "")
                           for r in rows], fontsize=7)
    axes[0, 0].invert_yaxis()
    fig.suptitle(f"Top-ranked Rotors under uncertainty: median (dot) and {lower:g}th-{upper:g}th percentile band, "
                 f"{rows[0]['samples']} samples", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
