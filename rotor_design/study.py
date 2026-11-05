"""Run the study: sweep the assumptions file's grids, write CSVs and plots."""

import csv
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import Design, evaluate

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def run_study(assumptions_path, plots_dir, rotor_lengths_mm=None, nozzle_sweeps=None):
    """Evaluate each Rotor Length and Nozzle variant; return the Rotor Length Results.

    nozzle_sweeps maps a Design field (nozzle_count, nozzle_throat_mm,
    nozzle_angle_deg) to the values tried, one field at a time about the baseline.
    """
    a = load_assumptions(assumptions_path)
    if rotor_lengths_mm is None:
        rotor_lengths_mm = np.arange(
            a.rotor_length_sweep_min_mm,
            a.max_rotor_length_mm + a.rotor_length_sweep_step_mm / 2,
            a.rotor_length_sweep_step_mm,
        )
    results = [evaluate(Design(rotor_length_mm=float(length)), a) for length in rotor_lengths_mm]

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results, plots_dir / "probe_bore_fit.csv")
    _plot_max_diameter(results, a, plots_dir / "max_rotor_diameter_vs_rotor_length.png")

    if nozzle_sweeps is None:
        nozzle_sweeps = {
            "nozzle_count": a.nozzle_count_sweep,
            "nozzle_throat_mm": a.nozzle_throat_sweep_mm,
            "nozzle_angle_deg": a.nozzle_angle_sweep_deg,
        }
    nozzle_rows = [
        {"swept_variable": field, **asdict(evaluate(Design(rotor_length_mm=a.max_rotor_length_mm,
                                                            **{field: value}), a))}
        for field, values in nozzle_sweeps.items()
        for value in values
    ]
    _write_rows(nozzle_rows, plots_dir / "nozzle_geometry_sweep.csv")
    _plot_nozzle_sweep(nozzle_rows, a, plots_dir / "required_flow_vs_nozzle_geometry.png")
    return results


def _write_csv(results, path):
    _write_rows([asdict(r) for r in results], path)


def _write_rows(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
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
