"""Design Sheet: one chosen Design to a long-format CSV an engineer can draw from.

One row per quantity, with columns part, quantity, operating, as_machined, feasible_min,
feasible_max, unit, source and note. Values are Operating Dimensions; as_machined stays
empty until thermal contraction is modelled. A row's feasible range is found one quantity at
a time, every other value held at nominal (the Rotor diameter pinned at its nominal value
too, so a sized-to-bore Rotor does not grow with its length). Feasibility is the model's
own Feasible Design test.

The file starts with a header of '#' lines: the validation status and the placeholders
ranked by how far their low and high cases move the Sample's hottest point. A results
block (part "Results") follows the dimension rows.

Adding rows: write a function sheet -> list of Row and append it to ROW_BUILDERS (dimensions)
or RESULT_BUILDERS (results block). A Row given a Knob gets a feasible range; a Knob says how
to set the quantity on a Design or on the assumptions. A Placeholder appended to PLACEHOLDERS
is ranked in the header.
"""

import csv
import math
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rotor_design.bearing import GAS, MECHANICAL, TWO_CENTER_PIVOT
from rotor_design.evaluate import COLD_SUPPORT, DUCTED, Design, evaluate
from rotor_design.study import (CONFIGURATION_FIELDS, GRID, INK, INK_SECONDARY, SERIES, SURFACE, _style,
                                configuration_name)
from rotor_design.turbine import FLUTED_CAP, RADIAL_INFLOW

VALIDATION_STATUS = "verified against analytical limits and conservation; not validated against a built system"
COLUMNS = ("part", "quantity", "operating", "as_machined", "feasible_min", "feasible_max", "unit", "source", "note")
# Where a value comes from.
MODEL_DERIVED = "model-derived"
PLACEHOLDER = "placeholder"
ENGINEERS_CHOICE = "engineer's choice"
# Feasible-range edges are found to within this fraction of the nominal value (one step for integers).
RANGE_REL_TOLERANCE = 0.01
# First outward step as a fraction of the nominal value; it doubles until the Design fails.
_FIRST_STEP = 0.05
_MAX_EXPANSIONS = 12
# A Design whose evaluation fails (no room, no convergence) counts as not feasible.
_EVALUATION_ERRORS = (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError)


@dataclass(frozen=True)
class Knob:
    """How to set one quantity: apply(design, assumptions, value) -> (design, assumptions).

    The range search stays within [lower, upper]; step is the first outward step when the
    nominal value is zero.
    """

    apply: Callable
    lower: float = 0.0
    upper: float = math.inf
    integer: bool = False
    step: float | None = None


@dataclass(frozen=True)
class Row:
    part: str
    quantity: str
    operating: object
    unit: str
    source: str
    note: str = ""
    as_machined: object = None      # empty until thermal contraction is modelled
    feasible_min: object = None
    feasible_max: object = None
    key: str = ""                   # short name used to pick ranges on the command line
    knob: Knob | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Placeholder:
    """A placeholder input tried at a low and a high case; apply(design, assumptions, value)."""

    name: str
    low: float
    high: float
    apply: Callable
    note: str = ""
    applies: Callable = lambda sheet: True


@dataclass(frozen=True)
class PlaceholderEffect:
    name: str
    low: float
    high: float
    hottest_low_k: float            # nan when the case cannot be evaluated
    hottest_high_k: float
    shift_k: float                  # largest |change| in the Sample's hottest point
    note: str


@dataclass(frozen=True)
class Sheet:
    """What every row builder sees: the Design, its assumptions and Results, and where each
    Design field came from (field -> (source, note))."""

    design: Design
    assumptions: object
    results: object
    provenance: dict
    p_feasible: float | None = None


# --- Knob setters (module-level so range searches can run in worker processes) ---

def _set_design(name, design, assumptions, value):
    return replace(design, **{name: value}), assumptions


def _set_assumption(name, design, assumptions, value):
    return design, replace(assumptions, **{name: value})


def _set_allowance(name, design, assumptions, value):
    return design, replace(assumptions, stator_allowance=replace(assumptions.stator_allowance, **{name: value}))


def _set_bearing(bearing, name, design, assumptions, value):
    model = replace(assumptions.bearings[bearing], **{name: value})
    return design, replace(assumptions, bearings={**assumptions.bearings, bearing: model})


def design_knob(name, **limits):
    return Knob(partial(_set_design, name), **limits)


def assumption_knob(name, **limits):
    return Knob(partial(_set_assumption, name), **limits)


def allowance_knob(name, **limits):
    return Knob(partial(_set_allowance, name), **limits)


def bearing_knob(bearing, name, **limits):
    return Knob(partial(_set_bearing, bearing, name), **limits)


# Positive lengths are searched down to this (mm), not to zero.
_SMALLEST_MM = 1e-3


# --- Choosing the Design ---

def _parse_value(name, text):
    """A command-line or CSV string as the type of Design field name; 'none' or empty is None."""
    kind = {f.name: f.type for f in fields(Design)}.get(name)
    if kind is None:
        raise ValueError(f"Unknown Design field {name!r}")
    if text is None or str(text).strip().lower() in ("", "none", "nan"):
        return None
    text = str(text).strip()
    kind = str(kind)
    if "bool" in kind:
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{name} must be true or false, not {text!r}")
        return text.lower() == "true"
    if "int" in kind:
        return int(float(text))
    if "float" in kind:
        return float(text)
    return text


def design_from_ranked_csv(path, rank, assumptions):
    """The Design at the given rank of a study ranked CSV, and where each field came from.

    The study sets the Configuration and its grid fields; every other field took the
    assumptions file's value, so it is left unset here. A diameter equal to the largest the
    Probe Bore allows was not set either.
    """
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["rank"]) == int(rank)]
    if not rows:
        raise ValueError(f"No Design at rank {rank} in {path}")
    row = rows[0]
    chosen = ["rotor_length_mm", *CONFIGURATION_FIELDS, *assumptions.study_grid]
    values = {name: _parse_value(name, row.get(name)) for name in dict.fromkeys(chosen) if name in row}
    diameter, largest = row.get("rotor_diameter_mm"), row.get("max_rotor_diameter_mm")
    if diameter and largest and not math.isclose(float(diameter), float(largest), rel_tol=1e-9):
        values["rotor_diameter_mm"] = float(diameter)
    values = {k: v for k, v in values.items() if v is not None}
    note = f"study rank {rank}, {Path(path).name}"
    return Design(**values), {name: (MODEL_DERIVED, note) for name in values}


def design_from_file(path):
    """A Design from a TOML file of Design fields, and where each field came from."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    raw = raw.get("design", raw)
    unknown = set(raw) - {f.name for f in fields(Design)}
    if unknown:
        raise ValueError(f"Unknown Design fields in {path}: {sorted(unknown)}")
    note = f"Design file {Path(path).name}"
    return Design(**raw), {name: (ENGINEERS_CHOICE, note) for name in raw}


def apply_overrides(design, provenance, overrides):
    """overrides: 'field=value' strings, applied on top of the chosen Design."""
    values = {}
    for item in overrides or ():
        name, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"Override {item!r} must be field=value")
        values[name.strip()] = _parse_value(name.strip(), text)
    provenance = {**provenance, **{name: (ENGINEERS_CHOICE, "command-line override") for name in values}}
    return replace(design, **values), provenance


def _same(text, value):
    if isinstance(value, (bool, str)):
        return str(text) == str(value)
    try:
        x, y = float(text), float(value)
    except (TypeError, ValueError):
        return False
    return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def feasible_probability(results, uncertainty_csv):
    """P(feasible) from a study's uncertainty CSV, if it holds this Design; else None."""
    path = Path(uncertainty_csv)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if all(name in row and _same(row[name], getattr(results, name)) for name in (f.name for f in fields(Design))):
                return float(row["feasible_probability"])
    return None


# --- Feasibility and one-at-a-time ranges ---

def is_feasible(design, assumptions):
    try:
        return bool(evaluate(design, assumptions).feasible)
    except _EVALUATION_ERRORS:
        return False


def _edge(design, assumptions, knob, nominal, direction):
    """Furthest value from nominal, in one direction, before the Design stops being feasible.

    Returns (edge, reached_limit): reached_limit is True when the search stopped at the
    knob's bounds or its last expansion while still feasible.
    """
    def ok(value):
        return is_feasible(*knob.apply(design, assumptions, value))

    def clamp(value):
        value = min(max(value, knob.lower), knob.upper)
        return round(value) if knob.integer else value

    step = 1 if knob.integer else (abs(nominal) * _FIRST_STEP if nominal else (knob.step or 1.0))
    tolerance = 1 if knob.integer else RANGE_REL_TOLERANCE * (abs(nominal) if nominal else (knob.step or 1.0))
    good, bad = nominal, None
    for _ in range(_MAX_EXPANSIONS):
        trial = clamp(nominal + direction * step)
        if trial == good:
            return good, True
        if not ok(trial):
            bad = trial
            break
        good = trial
        step *= 2
    if bad is None:
        return good, True
    while abs(bad - good) > tolerance:
        mid = (good + bad) / 2
        mid = int(math.floor(mid) if direction > 0 else math.ceil(mid)) if knob.integer else mid
        if mid in (good, bad):
            break
        if ok(mid):
            good = mid
        else:
            bad = mid
    return good, False


def feasible_range(design, assumptions, knob, nominal):
    """(min, max, min_at_limit, max_at_limit) over which the Design stays feasible, the quantity
    varied alone; None if the Design is not feasible at nominal."""
    if not is_feasible(*knob.apply(design, assumptions, nominal)):
        return None
    lo, lo_limit = _edge(design, assumptions, knob, nominal, -1)
    hi, hi_limit = _edge(design, assumptions, knob, nominal, +1)
    return lo, hi, lo_limit, hi_limit


def _range_job(job):
    return feasible_range(*job)


def with_ranges(rows, sheet, keys=None, workers=1):
    """Rows with feasible_min/max filled for each row that has a Knob (and whose key is in keys,
    when given)."""
    pinned = replace(sheet.design, rotor_diameter_mm=sheet.results.rotor_diameter_mm)
    todo = [i for i, r in enumerate(rows) if r.knob is not None and (keys is None or r.key in keys)]
    jobs = [(pinned, sheet.assumptions, rows[i].knob, rows[i].operating) for i in todo]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            found = list(pool.map(_range_job, jobs))
    else:
        found = [_range_job(job) for job in jobs]
    rows = list(rows)
    for i, edges in zip(todo, found):
        row = rows[i]
        if edges is None:
            rows[i] = replace(row, note=_join(row.note, "not feasible at nominal, so no range"))
            continue
        lo, hi, lo_limit, hi_limit = edges
        limits = [side for side, hit in (("min", lo_limit), ("max", hi_limit)) if hit]
        note = _join(row.note, f"range {' and '.join(limits)} at the edge searched" if limits else "")
        rows[i] = replace(row, feasible_min=lo, feasible_max=hi, note=note)
    return rows


def _join(*parts):
    return "; ".join(p for p in parts if p)


# --- Rows ---

def _field_row(sheet, part, quantity, name, unit, note="", knob=None):
    """A row for a Design field: its resolved value, from the Results, and where it came from."""
    source, origin = sheet.provenance.get(name, (PLACEHOLDER, "assumptions-file default"))
    return Row(part, quantity, getattr(sheet.results, name), unit, source, _join(note, origin), key=name, knob=knob)


def rotor_rows(sheet):
    r, a, d = sheet.results, sheet.assumptions, sheet.design
    if d.rotor_diameter_mm is None:
        diameter = Row("Rotor", "outer diameter", r.rotor_diameter_mm, "mm", MODEL_DERIVED,
                       f"largest the Probe Bore allows at this Rotor Length; also the Turbine diameter",
                       key="rotor_diameter_mm", knob=design_knob("rotor_diameter_mm", lower=_SMALLEST_MM))
    else:
        diameter = _field_row(sheet, "Rotor", "outer diameter", "rotor_diameter_mm", "mm",
                              "also the Turbine diameter", design_knob("rotor_diameter_mm", lower=_SMALLEST_MM))
    return [
        _field_row(sheet, "Rotor", "Rotor Length", "rotor_length_mm", "mm",
                   f"everything that spins, Turbine Cap included; limit {a.max_rotor_length_mm:g} mm",
                   design_knob("rotor_length_mm", lower=_SMALLEST_MM, upper=a.max_rotor_length_mm)),
        diameter,
        Row("Rotor", "largest outer diameter the Probe Bore allows", r.max_rotor_diameter_mm, "mm", MODEL_DERIVED,
            "at this Rotor Length, Tilt Angle and Stator Allowance"),
        _field_row(sheet, "Rotor", "wall thickness", "wall_thickness_mm", "mm", "zirconia tube",
                   design_knob("wall_thickness_mm", lower=_SMALLEST_MM)),
        Row("Rotor", "bore diameter", r.rotor_diameter_mm - 2.0 * r.wall_thickness_mm, "mm", MODEL_DERIVED,
            "outer diameter less two walls"),
        Row("Rotor", "body length below the Turbine Cap", r.rotor_length_mm - a.turbine_cap_height_mm, "mm",
            MODEL_DERIVED, "Rotor Length less the Turbine Cap height"),
        Row("Turbine Cap", "height", a.turbine_cap_height_mm, "mm", PLACEHOLDER,
            "part of the Rotor Length; held at the Drive Gas temperature",
            key="turbine_cap_height_mm", knob=assumption_knob("turbine_cap_height_mm", lower=_SMALLEST_MM)),
        Row("Bottom Cap", "thickness", a.bottom_cap_thickness_mm, "mm", PLACEHOLDER, "solid zirconia",
            key="bottom_cap_thickness_mm", knob=assumption_knob("bottom_cap_thickness_mm", lower=_SMALLEST_MM)),
    ]


def sample_rows(sheet):
    r = sheet.results
    return [
        _field_row(sheet, "Sample", "length", "sample_length_mm", "mm",
                   "packed from the Bottom Cap face upward", design_knob("sample_length_mm", lower=_SMALLEST_MM)),
        Row("Sample", "volume", r.sample_volume_mm3, "mm^3", MODEL_DERIVED, "Rotor bore over the Sample length"),
        _field_row(sheet, "Sample", "thermal conductivity", "sample_conductivity_w_m_k", "W/(m K)",
                   "no source", design_knob("sample_conductivity_w_m_k", lower=1e-6)),
        _field_row(sheet, "Sample", "internal heat load", "sample_internal_heat_load_w", "W", "",
                   design_knob("sample_internal_heat_load_w", lower=0.0, step=1e-4)),
    ]


def drive_rows(sheet):
    r, a, d = sheet.results, sheet.assumptions, sheet.design
    rows = [
        _field_row(sheet, "Turbine", "Turbine type", "turbine", ""),
        Row("Turbine", "outer diameter", r.rotor_diameter_mm, "mm", MODEL_DERIVED,
            "the Turbine sits on the Turbine Cap at the Rotor rim"),
        Row("Turbine", "Turbine Cap face area left to the blades or flutes", r.turbine_blade_area_mm2, "mm^2",
            MODEL_DERIVED),
    ]
    if r.turbine == RADIAL_INFLOW:
        rows.append(_field_row(sheet, "Turbine", "Hub Exit diameter", "hub_exit_diameter_mm", "mm",
                               "cut from the Turbine Cap", design_knob("hub_exit_diameter_mm", lower=_SMALLEST_MM)))
    rows += [
        _field_row(sheet, "Nozzle", "count", "nozzle_count", "",
                   knob=design_knob("nozzle_count", lower=1, integer=True)),
        _field_row(sheet, "Nozzle", "throat diameter", "nozzle_throat_mm", "mm",
                   knob=design_knob("nozzle_throat_mm", lower=_SMALLEST_MM)),
        _field_row(sheet, "Nozzle", "angle from the rim tangent", "nozzle_angle_deg", "deg",
                   knob=design_knob("nozzle_angle_deg", lower=0.0, upper=90.0, step=5.0)),
        _field_row(sheet, "Drive Gas", "gas", "drive_gas", "", "also feeds the gas Bearings"),
        _field_row(sheet, "Drive Gas", "supply temperature", "drive_gas_temperature_k", "K", "",
                   design_knob("drive_gas_temperature_k", lower=0.1)),
        Row("Drive Gas", "required Supply Pressure", r.required_supply_pressure_bar, "bar", MODEL_DERIVED,
            f"absolute, to reach the Target Spin Rate; limit {a.max_supply_pressure_bar:g} bar",
            feasible_max=a.max_supply_pressure_bar),
        Row("Drive Gas", "required mass flow", r.required_mass_flow_kg_s, "kg/s", MODEL_DERIVED, "all Nozzles"),
        _field_row(sheet, "Drive Gas", "Exhaust Path", "exhaust_path", ""),
    ]
    if r.exhaust_path == DUCTED:
        rows.append(_field_row(sheet, "Drive Gas", "Back Pressure", "back_pressure_bar", "bar", "absolute",
                               design_knob("back_pressure_bar", lower=1e-6)))
    else:
        rows.append(Row("Drive Gas", "Back Pressure", r.back_pressure_bar, "bar", PLACEHOLDER,
                        "Vented: the environment pressure"))
    return rows


def bearing_rows(sheet):
    r, a = sheet.results, sheet.assumptions
    rows = [_field_row(sheet, "Bearing", "Bearing type", "bearing", "", r.warning),
            _field_row(sheet, "Bearing", "lower pivot tied to the Bath", "pivot_to_bath", "")]
    model = a.bearings[r.bearing]
    if r.bearing == TWO_CENTER_PIVOT:
        rows += [Row("Bearing", "pivot tip radius", model.tip_radius_mm, "mm", PLACEHOLDER, "spherical tip",
                     key="pivot_tip_radius_mm",
                     knob=bearing_knob(TWO_CENTER_PIVOT, "tip_radius_mm", lower=_SMALLEST_MM)),
                 Row("Bearing", "pivot preload", model.preload_n, "N", PLACEHOLDER, "axial, per pivot",
                     key="pivot_preload_n", knob=bearing_knob(TWO_CENTER_PIVOT, "preload_n", lower=1e-6))]
    elif r.bearing == GAS:
        rows += [Row("Bearing", "journal length", model.journal_length_mm, "mm", PLACEHOLDER, "per journal; two journals",
                     key="gas_journal_length_mm", knob=bearing_knob(GAS, "journal_length_mm", lower=_SMALLEST_MM)),
                 Row("Bearing", "radial film clearance", model.clearance_mm, "mm", PLACEHOLDER, "",
                     key="gas_clearance_mm", knob=bearing_knob(GAS, "clearance_mm", lower=1e-5))]
    elif r.bearing == MECHANICAL:
        rows.append(Row("Bearing", "preload", model.preload_n, "N", PLACEHOLDER, "per bearing",
                        key="mechanical_preload_n", knob=bearing_knob(MECHANICAL, "preload_n", lower=1e-6)))
    return rows


def cooling_rows(sheet):
    r = sheet.results
    rows = [_field_row(sheet, "Cold Support", "cooling path", "cooling_path", "")]
    if r.cooling_path == COLD_SUPPORT:
        rows.append(_field_row(sheet, "Cold Support", "gap to the Bottom Cap face", "cold_support_gap_mm", "mm",
                               knob=design_knob("cold_support_gap_mm", lower=1e-5)))
    return rows


def stator_rows(sheet):
    r, a = sheet.results, sheet.assumptions
    s = a.stator_allowance
    allowance = [("wall", "wall", "radial, each side"),
                 ("nozzle_plenum", "nozzle plenum", "radial, each side"),
                 ("housing_above_turbine_cap", "housing above the Turbine Cap", "axial"),
                 ("cold_support_plate", "Cold Support plate", "axial"),
                 ("probe_wall_clearance", "probe-wall clearance", "kept free each side of the Probe Bore")]
    rows = [Row("Stator", f"Stator Allowance: {label}", getattr(s, name), "mm", PLACEHOLDER, note,
                key=f"stator_{name}_mm", knob=allowance_knob(name, lower=0.0, step=0.1))
            for name, label, note in allowance]
    rows += [
        Row("Stator", "Stator Allowance: Bearing radial space", r.bearing_radial_space_mm, "mm", PLACEHOLDER,
            "radial, each side; set by the Bearing type"),
        Row("Stator", "length", r.stator_length_mm, "mm", MODEL_DERIVED, "Rotor Length plus axial allowances"),
        Row("Stator", "outer diameter", r.stator_diameter_mm, "mm", MODEL_DERIVED,
            "Rotor diameter plus radial allowances"),
        Row("Stator", "projected width at the Tilt Angle", r.projected_width_mm, "mm", MODEL_DERIVED,
            "must not exceed the usable bore", feasible_max=r.usable_bore_mm),
        Row("Probe Bore", "inner diameter", a.bore_diameter_mm, "mm", PLACEHOLDER, "requirement"),
        Row("Probe Bore", "usable bore", r.usable_bore_mm, "mm", MODEL_DERIVED, "less the probe-wall clearance"),
        Row("Probe Bore", "Tilt Angle", a.tilt_angle_deg, "deg", PLACEHOLDER, "requirement"),
        Row("Gaps", "Rotor-Stator radial gap", a.rotor_stator_gap_mm, "mm", PLACEHOLDER,
            "sets windage and Vapour Cooling", key="rotor_stator_gap_mm",
            knob=assumption_knob("rotor_stator_gap_mm", lower=1e-5)),
    ]
    return rows


def engineers_choice_rows(sheet):
    """Dimensions the model does not compute, each with what the model assumed about it."""
    r = sheet.results
    if r.turbine == FLUTED_CAP:
        turbine = [("flute count", "not modelled; torque depends only on the Nozzle jets and rim speed"),
                   ("flute depth and axial length", "not modelled; the flutes are taken to fill the Turbine Cap face"),
                   ("flute profile", "straight flutes assumed: gas leaves with no swirl relative to the rim, "
                                     "T = m r (c_theta - U), scaled by the turbine efficiency placeholder")]
    else:
        turbine = [("blade count", "not modelled; blade passages fill the ring outside the Hub Exit"),
                   ("blade angle and passage height", "not modelled; gas leaves the Hub Exit with no swirl "
                                                      "relative to the Rotor, scaled by the turbine efficiency "
                                                      "placeholder; centrifugal head across the passages omitted")]
    closures = [
        ("Turbine Cap", "joint to the Rotor tube", "not modelled; Rotor, caps and Turbine are one zirconia piece "
                                                   "in perfect thermal contact"),
        ("Bottom Cap", "joint to the Rotor tube", "not modelled; solid zirconia in perfect thermal contact with the tube"),
        ("Sample", "fill and closure", "not modelled; the Sample is packed from the Bottom Cap face with no closure "
                                       "or void"),
        ("Nozzle", "length and profile", "not modelled; flow is set by the throat with a discharge coefficient"),
    ]
    rows = [Row("Turbine", q, None, "", ENGINEERS_CHOICE, n) for q, n in turbine]
    rows += [Row(part, q, None, "", ENGINEERS_CHOICE, n) for part, q, n in closures]
    if r.bearing == TWO_CENTER_PIVOT:
        rows.append(Row("Bearing", "pivot seat geometry", None, "", ENGINEERS_CHOICE,
                        "not modelled; a sphere on a flat seat (Hertz contact) is assumed"))
    return rows


# Dimension rows, in sheet order. A later model adds its own builder here.
ROW_BUILDERS = [rotor_rows, sample_rows, drive_rows, bearing_rows, cooling_rows, stator_rows, engineers_choice_rows]


def sample_limit_k(assumptions):
    """The Sample Limit Temperature as the model applies it."""
    return assumptions.environment_temperature_k + assumptions.sample_zone_delta_t_k


def results_rows(sheet):
    r, a = sheet.results, sheet.assumptions
    res = partial(Row, "Results", source=MODEL_DERIVED)
    rows = [res("Sample Heat Load", r.sample_heat_load_w, "W", note="heat crossing into the Sample Zone")]
    rows += [res(f"Sample Heat Load: {source.replace('_', ' ')}", watts, "W")
             for source, watts in r.sample_heat_load_breakdown_w.items()]
    p_note = ("Monte Carlo over the Input Uncertainty and Model-Form Error, from the study"
              if sheet.p_feasible is not None else "not available: this Design is not in the study's uncertainty CSV")
    rows += [
        res("Sample hottest point", r.sample_hottest_temperature_k, "K",
            note="Sample Limit Temperature in feasible_max", feasible_max=sample_limit_k(a)),
        res("Sample Zone", r.sample_zone_mm, "mm", note=f"at least {a.min_sample_zone_mm:g} mm required",
            feasible_min=a.min_sample_zone_mm),
        res("Bath Heat Load", r.bath_heat_load_w, "W", note="Bath Cooling Power in feasible_max",
            feasible_max=a.bath_cooling_power_w),
        res("P(feasible)", sheet.p_feasible, "", note=p_note),
        res("Feasible Design", r.feasible, ""),
        res("Binding Constraint", r.binding_constraint, ""),
        res("stress safety factor", r.stress_safety_factor, "", feasible_min=a.required_safety_factor),
        res("nearest critical speed separation", r.critical_speed_separation, "",
            note="fraction of the Target Spin Rate", feasible_min=a.critical_speed_separation),
        res("Target Spin Rate", r.spin_rate_hz, "Hz"),
        res("Rotor mass", r.rotor_mass_kg, "kg", note="zirconia only; the Sample is not included"),
    ]
    return rows


# Results-block rows, after the dimensions. A later model adds its own builder here.
RESULT_BUILDERS = [results_rows]


# --- Placeholders ranked by their effect on the Sample's hottest point ---

def _scale_bearing(bearing, name):
    return partial(_set_bearing, bearing, name)


def _bearing_is(bearing):
    return lambda sheet: sheet.results.bearing == bearing


PLACEHOLDERS = [
    Placeholder("zirconia conductivity, 1.7-25 K bridge (unmeasured)", 0.3, 3.0,
                partial(_set_assumption, "zirconia_conductivity_multiplier"),
                "multiplier on the whole curve; no measured data between 1.7 and 25 K"),
    Placeholder("Sample conductivity (W/(m K), no source)", 1e-3, 1.0, partial(_set_design, "sample_conductivity_w_m_k")),
    Placeholder("Stator temperature (K)", 1.2, 1.3, partial(_set_assumption, "stator_temperature_k"),
                "Stator held at a fixed temperature"),
    Placeholder("windage heat fraction into the Rotor", 0.5, 1.0,
                partial(_set_assumption, "windage_heat_to_rotor_fraction")),
    Placeholder("Rotor-Stator radial gap (mm)", 0.05, 0.5, partial(_set_assumption, "rotor_stator_gap_mm")),
    Placeholder("accommodation coefficient", 0.5, 1.0, partial(_set_assumption, "accommodation_coefficient")),
    Placeholder("Rotor emissivity", 0.1, 1.0, partial(_set_assumption, "rotor_emissivity")),
    Placeholder("Turbine Cap height (mm)", 2.0, 4.0, partial(_set_assumption, "turbine_cap_height_mm")),
    Placeholder("Bottom Cap thickness (mm)", 0.5, 2.0, partial(_set_assumption, "bottom_cap_thickness_mm")),
    Placeholder("pivot friction coefficient", 0.03, 0.3, _scale_bearing(TWO_CENTER_PIVOT, "friction_coefficient"),
                applies=_bearing_is(TWO_CENTER_PIVOT)),
    Placeholder("pivot preload (N)", 0.025, 0.1, _scale_bearing(TWO_CENTER_PIVOT, "preload_n"),
                applies=_bearing_is(TWO_CENTER_PIVOT)),
    Placeholder("gas-bearing mass flow (kg/s)", 1e-8, 1e-6, _scale_bearing(GAS, "mass_flow_kg_s"),
                applies=_bearing_is(GAS)),
    Placeholder("gas-bearing clearance (mm)", 0.005, 0.02, _scale_bearing(GAS, "clearance_mm"),
                applies=_bearing_is(GAS)),
    Placeholder("mechanical-bearing friction coefficient", 0.003, 0.03,
                _scale_bearing(MECHANICAL, "friction_coefficient"), applies=_bearing_is(MECHANICAL)),
    Placeholder("discharge coefficient", 0.6, 0.95, partial(_set_assumption, "discharge_coefficient"),
                "sets the drive only"),
    Placeholder("turbine efficiency", 0.5, 1.0, partial(_set_assumption, "turbine_efficiency"),
                "sets the drive only"),
]


def _hottest(design, assumptions):
    try:
        return evaluate(design, assumptions).sample_hottest_temperature_k
    except _EVALUATION_ERRORS:
        return math.nan


def _effect_job(job):
    name, low, high, apply, note, design, assumptions, nominal = job
    hot_low, hot_high = (_hottest(*apply(design, assumptions, v)) for v in (low, high))
    shifts = [abs(t - nominal) for t in (hot_low, hot_high) if math.isfinite(t)]
    return PlaceholderEffect(name, low, high, hot_low, hot_high, max(shifts, default=math.nan), note)


def rank_placeholders(sheet, placeholders=None, workers=1):
    """Each applicable placeholder's effect on the Sample's hottest point, largest first."""
    placeholders = PLACEHOLDERS if placeholders is None else placeholders
    pinned = replace(sheet.design, rotor_diameter_mm=sheet.results.rotor_diameter_mm)
    # Fields only: a Placeholder's applies test need not be picklable for the workers.
    jobs = [(p.name, p.low, p.high, p.apply, p.note, pinned, sheet.assumptions,
             sheet.results.sample_hottest_temperature_k) for p in placeholders if p.applies(sheet)]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            effects = list(pool.map(_effect_job, jobs))
    else:
        effects = [_effect_job(job) for job in jobs]
    return sorted(effects, key=lambda e: -e.shift_k if math.isfinite(e.shift_k) else math.inf)


# --- Building and writing the sheet ---

def make_sheet(design, assumptions, provenance=None, p_feasible=None):
    return Sheet(design, assumptions, evaluate(design, assumptions), provenance or {}, p_feasible)


def build_rows(sheet, range_keys=None, workers=1):
    """Every dimension row (with feasible ranges) then the results block."""
    rows = [row for build in ROW_BUILDERS for row in build(sheet)]
    rows = with_ranges(rows, sheet, keys=range_keys, workers=workers)
    return rows + [row for build in RESULT_BUILDERS for row in build(sheet)]


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool) or isinstance(value, (int, np.integer)) or isinstance(value, str):
        return str(value)
    value = float(value)
    return "" if math.isnan(value) else f"{value:.6g}"


def _inward(value, up):
    """A range edge rounded to 6 significant figures towards the inside of the range, so the
    written edge is itself feasible."""
    if not isinstance(value, float) or not math.isfinite(value) or value == 0.0:
        return value
    scale = 10.0 ** (math.floor(math.log10(abs(value))) - 5)
    return (math.ceil(value / scale) if up else math.floor(value / scale)) * scale


def _cells(row):
    """A row's CSV cells; range edges from a search are rounded inward."""
    ranged = row.knob is not None
    values = {c: getattr(row, c) for c in COLUMNS}
    if ranged:
        values["feasible_min"] = _inward(row.feasible_min, up=True)
        values["feasible_max"] = _inward(row.feasible_max, up=False)
    return [_format(values[c]) for c in COLUMNS]


def header_lines(sheet, effects, label):
    r = sheet.results
    lines = [
        "Design Sheet",
        f"Design: {label}; Configuration {configuration_name({f: getattr(r, f) for f in CONFIGURATION_FIELDS})}",
        f"Validation status: {VALIDATION_STATUS}",
        "Values are Operating Dimensions at the Bath temperature; as_machined stays empty until thermal "
        "contraction is modelled.",
        f"Feasible ranges: one quantity at a time, all others at nominal, edges to within "
        f"{RANGE_REL_TOLERANCE:.0%} of nominal.",
        f"Placeholders ranked by how far they move the Sample's hottest point "
        f"(nominal {r.sample_hottest_temperature_k:.4f} K, limit {sample_limit_k(sheet.assumptions):.4f} K):",
    ]
    for i, e in enumerate(effects, 1):
        shift = f"{e.shift_k * 1e3:.2f} mK" if math.isfinite(e.shift_k) else "not evaluable"
        lines.append(f"  {i}. {e.name}: {_format(e.low)} -> {_format(e.hottest_low_k)} K, "
                     f"{_format(e.high)} -> {_format(e.hottest_high_k)} K; shift {shift}"
                     + (f" ({e.note})" if e.note else ""))
    return lines


def write_design_sheet(sheet, path, label="", range_keys=None, placeholders=None, workers=1):
    """Write the Design Sheet CSV; return its rows and the ranked placeholder effects."""
    rows = build_rows(sheet, range_keys, workers)
    effects = rank_placeholders(sheet, placeholders, workers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        for line in header_lines(sheet, effects, label):
            f.write(f"# {line}\n")
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(_cells(row))
    return rows, effects


def read_design_sheet(path):
    """(header lines, rows as dicts) of a written Design Sheet."""
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    header = [line[2:] for line in lines if line.startswith("# ")]
    body = [line for line in lines if not line.startswith("#")]
    return header, list(csv.DictReader(body))


def plot_temperature(sheet, path):
    """T(z) along the Rotor body with the Sample, its hottest point and the Sample Limit marked."""
    r, a = sheet.results, sheet.assumptions
    z, t = np.array(r.z_mm), np.array(r.temperature_k)
    bottom, top = a.bottom_cap_thickness_mm, a.bottom_cap_thickness_mm + r.sample_length_mm
    limit = sample_limit_k(a)
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.axvspan(bottom, top, color=GRID, alpha=0.8, zorder=0, linewidth=0)
    ax.axvspan(0.0, bottom, color=GRID, alpha=0.35, zorder=0, linewidth=0)
    ax.plot(z, t, color=SERIES, linewidth=2, zorder=3)
    in_sample = (z >= bottom) & (z <= top)
    if in_sample.any():
        i = int(np.argmax(np.where(in_sample, t, -np.inf)))
        ax.plot(z[i], t[i], "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"hottest Sample point {r.sample_hottest_temperature_k:.4f} K", (z[i], t[i]),
                    textcoords="offset points", xytext=(8, -14), color=INK, fontsize=8)
    ax.axhline(limit, color=INK_SECONDARY, linewidth=1, linestyle=":")
    ax.annotate(f"Sample Limit Temperature {limit:g} K", (0.99, limit), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4), ha="right", color=INK_SECONDARY, fontsize=8)
    ax.annotate("Sample", ((bottom + top) / 2, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top",
                color=INK, fontsize=9)
    ax.annotate("Bottom\nCap", (bottom / 2, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top",
                color=INK_SECONDARY, fontsize=7)
    ax.annotate(f"Turbine Cap\n({r.drive_gas_temperature_k:g} K) above", (z[-1], 0.03),
                xycoords=("data", "axes fraction"), ha="right", va="bottom", color=INK_SECONDARY, fontsize=8)
    ax.set_xlim(0.0, z[-1])
    ax.set_ylim(min(t.min(), limit) - 0.02, min(t.max(), limit + 0.3) + 0.02)
    ax.set_xlabel("Distance from the Bottom Cap face (mm)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Rotor temperature (K)", color=INK_SECONDARY, fontsize=9)
    status = "feasible" if r.feasible else f"not feasible ({r.binding_constraint})"
    ax.set_title(f"Temperature along the Rotor body, Design Sheet Design, {status}", color=INK, fontsize=11,
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def default_workers():
    return min(2, os.cpu_count() or 1)
