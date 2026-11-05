"""Turbine diameter study: reachable Spin Rate and Sample Heat Load against Turbine diameter and Spin Rate.

The Turbine sits on the Turbine Cap at the Rotor rim, so Turbine diameter is the Rotor diameter.
Every Design is the assumptions file's baseline Configuration at the longest Rotor, with the
diameter swept up to the largest the Probe Bore allows. Both Turbine types are compared.

The model holds the Spin Rate at the Target Spin Rate and solves for the Supply Pressure, so the
reachable Spin Rate is found here by bisection: the highest Spin Rate whose required Supply
Pressure stays within the Supply Pressure limit. Other constraints (stress, critical speeds,
Sample Zone) do not cap it; the stress limit is drawn alongside for reference.

Outputs go to Plots/ (turbine_study_*.png and turbine_study.csv). Evaluates in at most 2 processes.
"""

import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import Design, evaluate
from rotor_design.gas import GASES
from rotor_design.study import GRID, INK, INK_SECONDARY, RANK_COLOURS, SURFACE, _style
from rotor_design.turbine import FLUTED_CAP, RADIAL_INFLOW

HERE = Path(__file__).resolve().parent
WORKERS = 2
DIAMETER_COUNT = 12
SMALLEST_DIAMETER_MM = 2.0
# Bisection bracket on Spin Rate (Hz) and its relative tolerance.
SPIN_FLOOR_HZ = 10.0
SPIN_CEILING_HZ = 50_000.0
SPIN_REL_TOLERANCE = 0.01
HEAT_SWEEP_HZ = np.geomspace(100.0, 2000.0, 14)
TURBINE_LABELS = {FLUTED_CAP: "Fluted Cap", RADIAL_INFLOW: "Radial-Inflow"}
TURBINE_COLOURS = dict(zip(TURBINE_LABELS, RANK_COLOURS))
SOURCE_LABELS = {
    "windage": "Windage",
    "bearing_friction": "Bearing friction",
    "turbine_cap_conduction": "Turbine Cap conduction",
    "convection": "Convection from warmer gas",
    "radiation": "Radiation",
    "cold_support_shear": "Cold Support gap shear",
    "internal_load": "Sample internal load",
}
CAVEAT = ("Caveat: the Fluted Cap model gives about half the measured impulse-turbine efficiency (issue #25),\n"
          "so reachable Spin Rate is likely pessimistic and required Supply Pressure and flow overstated.")


def at_spin(assumptions, spin_hz):
    return replace(assumptions, target_spin_rate_hz=float(spin_hz))


def reaches(design, assumptions, spin_hz):
    r = evaluate(design, at_spin(assumptions, spin_hz))
    return r.required_supply_pressure_bar <= assumptions.max_supply_pressure_bar


def reachable_spin_rate(args):
    """Highest Spin Rate (Hz) within the Supply Pressure limit: 0 if not even the floor, the
    ceiling if that is reached. Geometric bisection, assuming one crossing."""
    design, assumptions = args
    if not reaches(design, assumptions, SPIN_FLOOR_HZ):
        return 0.0
    if reaches(design, assumptions, SPIN_CEILING_HZ):
        return SPIN_CEILING_HZ
    lo, hi = SPIN_FLOOR_HZ, SPIN_CEILING_HZ
    while hi / lo > 1.0 + SPIN_REL_TOLERANCE:
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if reaches(design, assumptions, mid) else (lo, mid)
    return lo


def evaluate_pair(args):
    return evaluate(*args)


def radial_inflow_energy_bound_hz(diameter_mm, assumptions):
    """Rough upper bound on Radial-Inflow Spin Rate from rothalpy: gas entering at the rim must climb
    the centrifugal field to the Hub Exit, so U_rim^2 (1 - (r_e/r_1)^2) / 2 <= cp T0 (dilute cp, the
    Drive Gas's whole stagnation enthalpy above 0 K). The Radial-Inflow torque model does not apply it."""
    r1 = diameter_mm / 2e3
    re = assumptions.turbines[RADIAL_INFLOW].hub_exit_diameter_mm / 2e3
    if re >= r1:
        return math.nan
    gas = GASES[assumptions.drive_gas]
    rim_speed = math.sqrt(2.0 * gas.cp_dilute * assumptions.drive_gas_temperature_k / (1.0 - (re / r1) ** 2))
    return rim_speed / (2.0 * math.pi * r1)


def specific_work_ratio(r):
    """Turbine work per kg of Drive Gas over the jet's kinetic energy per kg. Above 1 breaks
    energy conservation for the jet as modelled."""
    if not math.isfinite(r.required_mass_flow_kg_s) or not r.required_mass_flow_kg_s > 0:
        return math.nan
    work = r.turbine_torque_n_m * 2.0 * math.pi * r.spin_rate_hz / r.required_mass_flow_kg_s
    return work / (0.5 * r.jet_velocity_m_s**2)


def log_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0) & np.isfinite(y)
    if keep.sum() < 2:
        return math.nan
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def _log_axis(axis):
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _finish(fig, ax, title, path, note=None):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if note:
        fig.text(0.01, 0.01, note, color=INK_SECONDARY, fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035 * (note.count(chr(10)) + 1) + 0.01 if note else 0, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spin_vs_diameter(diameters, reach, stress_limit, a, path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    _style(ax)
    for turbine, label in TURBINE_LABELS.items():
        d = [x for x, s in zip(diameters, reach[turbine]) if s > 0]
        s = [s for s in reach[turbine] if s > 0]
        ax.plot(d, s, color=TURBINE_COLOURS[turbine], linewidth=2, marker="o", markersize=5, label=label)
    ax.plot(diameters, stress_limit, color=INK_SECONDARY, linewidth=1.2, linestyle="--",
            label=f"Stress limit (safety factor {a.required_safety_factor:g})")
    bound = [(d, radial_inflow_energy_bound_hz(d, a)) for d in diameters]
    bound = [(d, f) for d, f in bound if math.isfinite(f)]
    ax.plot(*zip(*bound), color=TURBINE_COLOURS[RADIAL_INFLOW], linewidth=1.2, linestyle=":",
            label="Radial-Inflow energy bound (rough, not in model)")
    ax.axhline(a.target_spin_rate_hz, color=INK_SECONDARY, linewidth=1, linestyle=":")
    ax.annotate(f"Target Spin Rate {a.target_spin_rate_hz:g} Hz", (0.01, a.target_spin_rate_hz),
                xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, 4),
                color=INK_SECONDARY, fontsize=8)
    ax.axhline(SPIN_CEILING_HZ, color=GRID, linewidth=1)
    ax.annotate("search ceiling", (0.99, SPIN_CEILING_HZ), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, -11), ha="right", color=INK_SECONDARY, fontsize=8)
    ax.set_yscale("log")
    _log_axis(ax.yaxis)
    ax.set_xlabel("Turbine diameter = Rotor diameter (mm)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Highest reachable Spin Rate (Hz)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(bottom=500)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish(fig, ax, f"Highest Spin Rate within the {a.max_supply_pressure_bar:g} bar Supply Pressure limit",
            path, "Spin Rate limited by Supply Pressure only; other constraints not applied. "
                  "Radial-Inflow has no blades below its Hub Exit diameter.\n"
                  "Radial-Inflow points above its energy bound are not physical: the model omits the "
                  "centrifugal head across the blade passages.\n" + CAVEAT)


def plot_drive_at_target(diameters, at_target, a, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    panels = [("Required Supply Pressure (bar)", lambda r: r.required_supply_pressure_bar),
              ("Required Drive Gas flow (mg/s)", lambda r: r.required_mass_flow_kg_s * 1e6)]
    for ax, (label, value) in zip(axes, panels):
        _style(ax)
        for turbine, name in TURBINE_LABELS.items():
            pts = [(d, value(r)) for d, r in zip(diameters, at_target[turbine]) if math.isfinite(value(r))]
            if pts:
                ax.plot(*zip(*pts), color=TURBINE_COLOURS[turbine], linewidth=2, marker="o", markersize=5,
                        label=name)
        ax.set_xlabel("Turbine diameter (mm)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(label, color=INK_SECONDARY, fontsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.4g}"))
    back = at_target[FLUTED_CAP][0].back_pressure_bar
    axes[0].axhline(back, color=INK_SECONDARY, linewidth=1, linestyle=":")
    axes[0].annotate(f"Back Pressure {back:g} bar", (0.99, back), xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 4), ha="right", color=INK_SECONDARY, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.suptitle(f"Drive needed at the {a.target_spin_rate_hz:g} Hz Target Spin Rate", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.text(0.01, 0.01, CAVEAT, color=INK_SECONDARY, fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _heat_lines(ax, x, results, a):
    """Total Sample Heat Load (Feasible filled, not Feasible hollow) and its main sources, in mW."""
    total = [r.sample_heat_load_w * 1e3 for r in results]
    ax.plot(x, total, color=RANK_COLOURS[0], linewidth=2, label="Sample Heat Load (total)", zorder=3)
    feas = [(xi, t) for xi, t, r in zip(x, total, results) if r.feasible]
    infeas = [(xi, t) for xi, t, r in zip(x, total, results) if not r.feasible]
    if feas:
        ax.plot(*zip(*feas), "o", color=RANK_COLOURS[0], markersize=6, label="Feasible Design", zorder=4)
    if infeas:
        ax.plot(*zip(*infeas), "o", color=SURFACE, markeredgecolor=RANK_COLOURS[0], markersize=6,
                markeredgewidth=1.5, label="Not Feasible", zorder=4)
    for source, colour, style in (("windage", RANK_COLOURS[1], "--"), ("bearing_friction", RANK_COLOURS[2], "--"),
                                  ("turbine_cap_conduction", INK_SECONDARY, ":")):
        ax.plot(x, [max(r.sample_heat_load_breakdown_w[source], 0.0) * 1e3 for r in results], color=colour,
                linewidth=1.2, linestyle=style, label=SOURCE_LABELS[source])
    ax.set_yscale("log")
    _log_axis(ax.yaxis)
    ax.set_ylabel("Heat into the Sample Zone (mW)", color=INK_SECONDARY, fontsize=9)


def plot_heat_vs_spin(spins, results, diameter, slope, a, path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    _style(ax)
    kept = [(s, r) for s, r in zip(spins, results) if r.sample_heat_load_w > 0]
    _heat_lines(ax, [s for s, _ in kept], [r for _, r in kept], a)
    lost = [s for s, r in zip(spins, results) if r.sample_zone_mm < a.min_sample_zone_mm]
    if lost:
        ax.axvspan(min(lost), max(spins), color=GRID, alpha=0.6, zorder=0)
        ax.annotate(f"Sample Zone\nunder {a.min_sample_zone_mm:g} mm", (min(lost), 0.03),
                    xycoords=("data", "axes fraction"), textcoords="offset points", xytext=(4, 0),
                    color=INK_SECONDARY, fontsize=8)
    kept_full = [s for s, r in zip(spins, results) if r.sample_zone_mm >= a.min_sample_zone_mm]
    ax.axvline(a.target_spin_rate_hz, color=INK_SECONDARY, linewidth=1, linestyle=":")
    ax.set_xscale("log")
    _log_axis(ax.xaxis)
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xlabel("Spin Rate (Hz)", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish(fig, ax, f"Sample Heat Load against Spin Rate, {diameter:.2f} mm Rotor, "
                     f"{a.max_rotor_length_mm:g} mm long",
            path, f"Fitted Sample Heat Load ~ Spin Rate^{slope:.2f} over points with a full Sample Zone. "
                  f"Fluted Cap;\nthe Turbine type does not change the Sample Heat Load here. "
                  "Points with no Sample Zone left carry zero load by definition and are not drawn."
                  + (f"\nThe Sample Zone falls under {a.min_sample_zone_mm:g} mm between {max(kept_full):.0f} and "
                     f"{min(lost):.0f} Hz at this diameter." if lost and kept_full else ""))


def plot_heat_vs_diameter(diameters, results, slope, a, path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    _style(ax)
    _heat_lines(ax, diameters, results, a)
    ax.set_xlabel("Turbine diameter = Rotor diameter (mm)", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish(fig, ax, f"Sample Heat Load against Turbine diameter at {a.target_spin_rate_hz:g} Hz",
            path, f"Fitted Sample Heat Load ~ diameter^{slope:.2f}. Fluted Cap, "
                  f"{a.max_rotor_length_mm:g} mm Rotor;\nthe Turbine type does not change the Sample Heat Load here.")


def main(assumptions_path):
    start = time.time()
    a = load_assumptions(assumptions_path)
    plots = HERE / "Plots"
    plots.mkdir(exist_ok=True)
    length = a.max_rotor_length_mm
    largest = evaluate(Design(length), a).max_rotor_diameter_mm
    diameters = list(np.linspace(SMALLEST_DIAMETER_MM, largest, DIAMETER_COUNT))
    designs = {t: [Design(length, d, turbine=t) for d in diameters] for t in TURBINE_LABELS}
    heat_design = Design(length, largest, turbine=FLUTED_CAP)

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        reach_jobs = {t: [pool.submit(reachable_spin_rate, (d, a)) for d in ds] for t, ds in designs.items()}
        target_jobs = {t: [pool.submit(evaluate_pair, (d, a)) for d in ds] for t, ds in designs.items()}
        heat_jobs = [pool.submit(evaluate_pair, (heat_design, at_spin(a, s))) for s in HEAT_SWEEP_HZ]
        reach = {t: [j.result() for j in js] for t, js in reach_jobs.items()}
        at_target = {t: [j.result() for j in js] for t, js in target_jobs.items()}
        heat_spin = [j.result() for j in heat_jobs]
        reached = {t: [pool.submit(evaluate_pair, (d, at_spin(a, s))) if 0 < s < SPIN_CEILING_HZ else None
                       for d, s in zip(designs[t], reach[t])] for t in TURBINE_LABELS}
        reached = {t: [j.result() if j else None for j in js] for t, js in reached.items()}

    fluted = at_target[FLUTED_CAP]
    stress_limit = [a.target_spin_rate_hz * math.sqrt(r.stress_safety_factor / a.required_safety_factor)
                    for r in fluted]
    full_zone = [(s, r.sample_heat_load_w) for s, r in zip(HEAT_SWEEP_HZ, heat_spin)
                 if r.sample_zone_mm >= a.min_sample_zone_mm]
    spin_slope = log_slope(*zip(*full_zone)) if full_zone else math.nan
    diameter_slope = log_slope(diameters, [r.sample_heat_load_w for r in fluted])

    paths = {
        "spin": plots / "turbine_study_spin_rate_vs_diameter.png",
        "drive": plots / "turbine_study_drive_at_target_vs_diameter.png",
        "heat_spin": plots / "turbine_study_heat_load_vs_spin_rate.png",
        "heat_d": plots / "turbine_study_heat_load_vs_diameter.png",
    }
    plot_spin_vs_diameter(diameters, reach, stress_limit, a, paths["spin"])
    plot_drive_at_target(diameters, at_target, a, paths["drive"])
    plot_heat_vs_spin(HEAT_SWEEP_HZ, heat_spin, largest, spin_slope, a, paths["heat_spin"])
    plot_heat_vs_diameter(diameters, fluted, diameter_slope, a, paths["heat_d"])

    rows = []
    for t in TURBINE_LABELS:
        for d, s, r, rr in zip(diameters, reach[t], at_target[t], reached[t]):
            rows.append({"sweep": "diameter", "turbine": t, "rotor_diameter_mm": d, "spin_rate_hz": a.target_spin_rate_hz,
                         "reachable_spin_rate_hz": s,
                         "supply_pressure_bar_at_reachable": rr.required_supply_pressure_bar if rr else math.nan,
                         "work_over_jet_energy_at_reachable": specific_work_ratio(rr) if rr else math.nan,
                         "required_supply_pressure_bar": r.required_supply_pressure_bar,
                         "required_mass_flow_kg_s": r.required_mass_flow_kg_s, "sample_heat_load_w": r.sample_heat_load_w,
                         "sample_zone_mm": r.sample_zone_mm, "feasible": r.feasible,
                         "binding_constraint": r.binding_constraint,
                         **{f"heat_{k}_w": v for k, v in r.sample_heat_load_breakdown_w.items()}})
    for s, r in zip(HEAT_SWEEP_HZ, heat_spin):
        rows.append({"sweep": "spin_rate", "turbine": FLUTED_CAP, "rotor_diameter_mm": largest, "spin_rate_hz": s,
                     "required_supply_pressure_bar": r.required_supply_pressure_bar,
                     "required_mass_flow_kg_s": r.required_mass_flow_kg_s, "sample_heat_load_w": r.sample_heat_load_w,
                     "sample_zone_mm": r.sample_zone_mm, "feasible": r.feasible,
                     "binding_constraint": r.binding_constraint,
                     **{f"heat_{k}_w": v for k, v in r.sample_heat_load_breakdown_w.items()}})
    csv_path = plots / "turbine_study.csv"
    with open(csv_path, "w", newline="") as f:
        names = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Diameters {diameters[0]:.2f} to {largest:.2f} mm at {length:g} mm Rotor Length; "
          f"Supply Pressure limit {a.max_supply_pressure_bar:g} bar")
    print("d (mm) | reachable Spin Rate (Hz): Fluted Cap, Radial-Inflow | stress limit (Hz) | "
          "at target: Supply (bar) FC, RI | flow (mg/s) FC, RI | Sample Heat Load (mW) | Feasible FC | work/jet energy RI")
    for i, d in enumerate(diameters):
        fc, ri = at_target[FLUTED_CAP][i], at_target[RADIAL_INFLOW][i]
        rr = reached[RADIAL_INFLOW][i]
        print(f"{d:6.2f} | {reach[FLUTED_CAP][i]:8.0f} {reach[RADIAL_INFLOW][i]:8.0f} | {stress_limit[i]:8.0f} | "
              f"{fc.required_supply_pressure_bar:7.4f} {ri.required_supply_pressure_bar:7.4f} | "
              f"{fc.required_mass_flow_kg_s * 1e6:7.4f} {ri.required_mass_flow_kg_s * 1e6:7.4f} | "
              f"{fc.sample_heat_load_w * 1e3:8.4f} | {fc.feasible!s:5} {fc.binding_constraint:18} | "
              f"{specific_work_ratio(rr) if rr else math.nan:6.2f} | RI energy bound "
              f"{radial_inflow_energy_bound_hz(d, a):6.0f} Hz")
    print(f"\nSpin Rate sweep at {largest:.2f} mm (Fluted Cap):")
    for s, r in zip(HEAT_SWEEP_HZ, heat_spin):
        top = max(r.sample_heat_load_breakdown_w, key=r.sample_heat_load_breakdown_w.get)
        print(f"{s:7.0f} Hz | Sample Heat Load {r.sample_heat_load_w * 1e3:8.4f} mW | zone {r.sample_zone_mm:5.2f} mm | "
              f"largest source {top:22} | Feasible {r.feasible!s:5} ({r.binding_constraint})")
    print(f"\nSample Heat Load ~ Spin Rate^{spin_slope:.2f} (full Sample Zone), ~ diameter^{diameter_slope:.2f} "
          f"(at {a.target_spin_rate_hz:g} Hz)")
    print(CAVEAT)
    print("Outputs:", *paths.values(), csv_path, sep="\n  ")
    print(f"Run time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "assumptions.toml")
