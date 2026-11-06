"""Benchmark Cases: published, already-built rotor systems the model is asked to predict.

Each case is one TOML file in benchmarks/ describing one published system as a Design.
Every value carries the source it came from, where in that source, and how it was read
(the flags of the Benchmark Case research). A value the source does not report stays
empty, and every comparison that needs it is skipped, never filled with a guess.

Each operating point runs through evaluate() at its measured Spin Rate, with the
case's environment in place of the Bath. Model details no source reports (cap
heights, gaps, Bearing film data, discharge coefficient, turbine efficiency, Sample
fill) keep their values from the assumptions file: they are part of the model whose
Model-Form Error is being measured. The Turbine exit is taken as Ducted at the
case's Back Pressure, the Stator at the environment temperature, and the Rotor as
turning in its own Drive Gas.
"""

import math
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from rotor_design.bearing import BEARING_TYPES
from rotor_design.evaluate import DUCTED, PA_PER_BAR, Design, evaluate
from rotor_design.gas import GASES
from rotor_design.turbine import FLUTED_CAP, TURBINE_TYPES

# How each value was read. S (simulation) and 2nd (second-hand) are never validation truth.
FLAGS = ("M", "F", "A", "U")
REJECTED_FLAGS = ("S", "2nd")
# Published Turbine types and the model Turbine type that stands in for them.
# A Pelton cap takes rim jets on blades at the cap rim and exhausts around the rim.
MODELLED_TURBINE = {FLUTED_CAP: FLUTED_CAP, "pelton": FLUTED_CAP}
MODELLED_MATERIAL = "zirconia"

# Inputs every case file lists, filled or empty.
SYSTEM_INPUTS = ("drive_gas", "rotor_material", "rotor_length_mm", "rotor_diameter_mm", "wall_thickness_mm",
                 "bearing", "drive_gas_temperature_k", "environment_temperature_k", "environment_pressure_bar")
DRIVE_INPUTS = ("turbine", "nozzle_count", "nozzle_throat_mm", "nozzle_angle_deg", "back_pressure_bar")
INPUTS = SYSTEM_INPUTS + DRIVE_INPUTS
POINT_INPUTS = ("spin_rate_hz",)


def _drag_power_w(results):
    return results.drag_torque_n_m * 2.0 * math.pi * results.spin_rate_hz


# Measured quantity -> (Results to compare against, inputs it needs).
# At steady Spin Rate the Turbine torque balances the drag, so measured torque is compared with the drag torque.
QUANTITIES = {
    "supply_pressure_bar": (lambda r: r.required_supply_pressure_bar, SYSTEM_INPUTS + DRIVE_INPUTS),
    "supply_pressure_bar_gauge": (lambda r: r.required_supply_pressure_bar - r.back_pressure_bar,
                                  SYSTEM_INPUTS + DRIVE_INPUTS),
    "mass_flow_kg_s": (lambda r: r.required_mass_flow_kg_s, SYSTEM_INPUTS + DRIVE_INPUTS),
    "torque_n_m": (lambda r: r.drag_torque_n_m, SYSTEM_INPUTS),
    "drag_power_w": (_drag_power_w, SYSTEM_INPUTS),
    "sample_temperature_k": (lambda r: r.sample_hottest_temperature_k, SYSTEM_INPUTS),
    "sample_heat_load_w": (lambda r: r.sample_heat_load_w, SYSTEM_INPUTS),
    "bath_heat_load_w": (lambda r: r.bath_heat_load_w, SYSTEM_INPUTS),
}


@dataclass(frozen=True)
class Value:
    """One cited value; value is None when the source does not report it."""
    value: object = None
    source: str = ""
    location: str = ""
    flag: str = ""
    note: str = ""


@dataclass(frozen=True)
class Point:
    inputs: dict     # point-level inputs, over the case inputs: name -> Value
    measured: dict   # quantity -> Value


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    name: str
    sources: dict    # key -> full citation
    inputs: dict     # name -> Value
    points: tuple


@dataclass(frozen=True)
class Comparison:
    case_id: str
    point: int
    quantity: str
    measured: float
    predicted: float
    model_form_error: float  # (predicted - measured) / measured; nan when skipped
    flag: str
    skipped: str             # why the comparison could not be made; empty when it was


def cited_value(where, name, entry, sources):
    """One Value from a case file entry, checked for a usable citation."""
    v = Value(**entry)
    if v.value is None:
        return v
    if v.source not in sources:
        raise ValueError(f"{where}: {name} = {v.value!r} cites unknown source {v.source!r}")
    if not v.location:
        raise ValueError(f"{where}: {name} = {v.value!r} gives no location in its source")
    if v.flag in REJECTED_FLAGS:
        raise ValueError(f"{where}: {name} is flagged {v.flag}, which is not validation truth")
    if not v.flag or any(f.strip() not in FLAGS for f in v.flag.split(",")):
        raise ValueError(f"{where}: {name} has flag {v.flag!r}; expected some of {FLAGS}")
    return v


def load_case(path):
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sources = raw.get("sources", {})
    where = path.name

    def value(name, entry):
        return cited_value(where, name, entry, sources)

    raw_inputs = raw.get("inputs", {})
    missing = [name for name in INPUTS if name not in raw_inputs]
    if missing:
        raise ValueError(f"{where}: inputs not listed (leave them empty, not out): {missing}")
    unknown = set(raw_inputs) - set(INPUTS)
    if unknown:
        raise ValueError(f"{where}: unknown inputs {sorted(unknown)}")
    points = []
    for i, raw_point in enumerate(raw.get("points", [])):
        point_inputs = {k: value(f"points[{i}].{k}", e) for k, e in raw_point.get("inputs", {}).items()}
        if set(point_inputs) - set(INPUTS + POINT_INPUTS):
            raise ValueError(f"{where}: points[{i}] has unknown inputs")
        if set(raw_point.get("measured", {})) - set(QUANTITIES):
            raise ValueError(f"{where}: points[{i}] measures unknown quantities; expected some of {tuple(QUANTITIES)}")
        measured = {k: value(f"points[{i}].{k}", e) for k, e in raw_point.get("measured", {}).items()}
        points.append(Point(point_inputs, measured))
    return BenchmarkCase(
        id=raw["id"],
        name=raw["name"],
        sources=sources,
        inputs={name: value(name, entry) for name, entry in raw_inputs.items()},
        points=tuple(points),
    )


def load_cases(directory):
    return [load_case(p) for p in sorted(Path(directory).glob("*.toml"))]


def _unmodelled(inputs):
    """Why the model cannot describe this system at all; empty if it can."""
    gas, material = inputs["drive_gas"].value, inputs["rotor_material"].value
    turbine, bearing = inputs["turbine"].value, inputs["bearing"].value
    if gas is not None and gas not in GASES:
        return f"Drive Gas {gas!r}: the model handles {', '.join(GASES)} only"
    if material is not None and material != MODELLED_MATERIAL:
        return f"Rotor material {material!r}: the model handles {MODELLED_MATERIAL} only"
    if turbine is not None and turbine not in MODELLED_TURBINE:
        return f"Turbine {turbine!r} has no model Turbine type"
    if bearing is not None and bearing not in BEARING_TYPES:
        return f"Bearing {bearing!r} has no model Bearing type"
    return ""


def _design_and_assumptions(inputs, assumptions):
    v = {name: entry.value for name, entry in inputs.items()}
    turbine = MODELLED_TURBINE.get(v["turbine"])
    design = Design(
        rotor_length_mm=v["rotor_length_mm"],
        rotor_diameter_mm=v["rotor_diameter_mm"],
        wall_thickness_mm=v["wall_thickness_mm"],
        nozzle_count=v["nozzle_count"],
        nozzle_throat_mm=v["nozzle_throat_mm"],
        nozzle_angle_deg=v["nozzle_angle_deg"],
        drive_gas=v["drive_gas"],
        drive_gas_temperature_k=v["drive_gas_temperature_k"],
        bearing=v["bearing"],
        pivot_to_bath=False,
        turbine=turbine,
        exhaust_path=DUCTED,
        back_pressure_bar=v["back_pressure_bar"],
    )
    environment_k = v["environment_temperature_k"]
    case_assumptions = replace(
        assumptions,
        target_spin_rate_hz=v["spin_rate_hz"],
        # The case's environment stands in for the Bath, with the same headroom to the Sample Limit.
        bath_temperature_k=environment_k,
        sample_limit_temperature_k=environment_k + (assumptions.sample_limit_temperature_k
                                                    - assumptions.bath_temperature_k),
        environment_gas=v["drive_gas"],
        cold_space_pressure_pa=v["environment_pressure_bar"] * PA_PER_BAR,
        max_rotor_length_mm=max(assumptions.max_rotor_length_mm, v["rotor_length_mm"]),
    )
    return design, case_assumptions


def compare(case, assumptions):
    """Every measured value of the case against the model, one Comparison each."""
    comparisons = []
    for i, point in enumerate(case.points):
        inputs = {**case.inputs, **point.inputs}
        inputs.setdefault("spin_rate_hz", Value())
        reason = _unmodelled(inputs)
        results = None
        for quantity, measured in point.measured.items():
            if measured.value is None:
                continue
            predict, needs = QUANTITIES[quantity]
            missing = [name for name in needs + POINT_INPUTS if inputs[name].value is None]
            skipped = reason or (f"not reported: {', '.join(missing)}" if missing else "")
            if skipped:
                comparisons.append(Comparison(case.id, i, quantity, measured.value, math.nan, math.nan,
                                              measured.flag, skipped))
                continue
            if results is None:
                results = evaluate(*_design_and_assumptions(inputs, assumptions))
            predicted = predict(results)
            comparisons.append(Comparison(case.id, i, quantity, measured.value, predicted,
                                          (predicted - measured.value) / measured.value, measured.flag, ""))
    return comparisons


@dataclass(frozen=True)
class QuantityError:
    quantity: str
    compared: int
    skipped: int
    mean_error: float   # signed: positive means the model over-predicts
    rms_error: float
    worst_error: float  # largest in magnitude, with its sign


def model_form_error(comparisons, quantities=QUANTITIES):
    """Model-Form Error per quantity, over every comparison that was made."""
    summary = []
    for quantity in quantities:
        rows = [c for c in comparisons if c.quantity == quantity]
        if not rows:
            continue
        errors = [c.model_form_error for c in rows if not c.skipped]
        if errors:
            mean = sum(errors) / len(errors)
            rms = math.sqrt(sum(e * e for e in errors) / len(errors))
            worst = max(errors, key=abs)
        else:
            mean = rms = worst = math.nan
        summary.append(QuantityError(quantity, len(errors), len(rows) - len(errors), mean, rms, worst))
    return summary
