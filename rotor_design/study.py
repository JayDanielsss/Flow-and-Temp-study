"""Run the study: sweep Rotor Length from the assumptions file, write CSV and plots."""

import csv
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


def run_study(assumptions_path, plots_dir, rotor_lengths_mm=None):
    """Evaluate each Rotor Length; return the Results and write them to plots_dir."""
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
    return results


def _write_csv(results, path):
    rows = [asdict(r) for r in results]
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
