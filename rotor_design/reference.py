"""Reference Cases: classic, fully specified experiments that each test one mechanism of the model.

Each case is one TOML file in references/ with the same citation rules as a Benchmark Case:
every value carries its source, location and flag, and a value the source does not report
stays empty and skips every comparison that needs it. Unlike a Benchmark Case, a Reference
Case is not a Design: it calls the model function for its one mechanism directly. Model
details the experiment cannot supply (discharge coefficient, accommodation coefficient,
turbine efficiency) keep their values from the assumptions file, so the Model-Form Error
is that of the model as it is used. Values keep the units the source prints them in.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rotor_design.benchmark import Comparison, Value, cited_value
from rotor_design.drag import couette_torque, free_molecular_shear
from rotor_design.evaluate import STANDARD_GRAVITY
from rotor_design.gas import GASES, isentropic_state, stagnation_state
from rotor_design.nozzle import nozzle_flow
from rotor_design.thermal import fin_profile
from rotor_design.turbine import FLUTED_CAP, FlutedCap

PA_PER_PSI = 6894.757293168


@dataclass(frozen=True)
class Quantity:
    name: str            # the compared quantity, as reported
    predict: object      # (input values, assumptions) -> predicted value
    needs: tuple         # inputs the prediction uses
    measured_from: object = None  # (cited value, input values) -> measured value; None: as cited


@dataclass(frozen=True)
class Mechanism:
    inputs: tuple        # every case file lists these, filled or empty
    point_inputs: tuple  # inputs that may also vary per point
    quantities: dict     # measured key in the file -> Quantity


def _discharge_coefficient(v, a):
    # The model's Nozzle flow is real-gas isentropic flow times one constant discharge
    # coefficient, so its prediction is that constant at every Reynolds number.
    return a.discharge_coefficient


def _couette_torque(v, a):
    return couette_torque(v["inner_diameter_cm"] / 200.0, v["outer_diameter_cm"] / 200.0, v["length_cm"] / 100.0,
                          v["spin_rate_hz"], v["viscosity_poise"] / 10.0)


def _torque_from_scale_pan(load_g, v):
    # The weight that holds the still cylinder, less the same with the apparatus empty, on the torque pulley.
    return (load_g - v["empty_load_g"]) / 1e3 * STANDARD_GRAVITY * v["pulley_radius_cm"] / 100.0


def _effective_accommodation(v, a):
    """Decay rate of the model's free-molecular drag on the ball over the gauge's ideal rate.

    The model's shear is linear in wall speed, so on a sphere of radius R turning at omega
    the torque is tau(1 m/s) omega 8 pi R^4 / 3 and the decay rate -omega_dot / omega is that
    over I omega, I = 8 pi rho R^5 / 15. The gauge's ideal rate, with full accommodation,
    is 20 p / (pi c_bar rho d) with c_bar = sqrt(8 R T / pi).
    """
    gas = GASES[v["gas"]]
    radius = v["ball_diameter_mm"] / 2e3
    density = v["ball_density_g_cm3"] * 1e3
    t, p = v["gas_temperature_k"], v["pressure_pa"]
    omega = 2.0 * math.pi * v["spin_rate_hz"]
    torque = (free_molecular_shear(1.0, t, p, a.accommodation_coefficient, gas)
              * omega * 8.0 * math.pi * radius**4 / 3.0)
    decay_rate = torque / (8.0 * math.pi * density * radius**5 / 15.0 * omega)
    mean_speed = math.sqrt(8.0 * gas.specific_gas_constant * t / math.pi)
    ideal_decay_rate = 20.0 * p / (math.pi * mean_speed * density * 2.0 * radius)
    return decay_rate / ideal_decay_rate


def _total_to_static_efficiency(v, a):
    """Model Turbine work over the ideal work of expanding from inlet total to exit static pressure.

    The jet is the model's Nozzle jet; the rim speed is the blade-jet speed ratio times the
    ideal jet speed sqrt(2 dh_ideal). Efficiency is per unit flow, so the Nozzle is
    evaluated at a unit throat diameter.
    """
    gas = GASES[v["gas"]]
    p0, t0 = v["supply_pressure_psia"] * PA_PER_PSI, v["supply_temperature_k"]
    p_exit = p0 / v["pressure_ratio"]
    flow = nozzle_flow(p0, t0, p_exit, 1.0, a.discharge_coefficient, gas)
    h0, s0 = stagnation_state(t0, p0, gas)
    ideal_work = h0 - isentropic_state(p_exit, s0, gas)[0]
    rim_speed = v["blade_jet_speed_ratio"] * math.sqrt(2.0 * ideal_work)
    radius = v["mean_diameter_cm"] / 200.0
    spin_rate = rim_speed / (2.0 * math.pi * radius)
    torque = FlutedCap().torque(flow.mass_flow_kg_s, flow.jet_velocity_m_s, v["nozzle_angle_deg"], radius,
                                spin_rate, a.turbine_efficiency * a.turbine_efficiency_correction[FLUTED_CAP])
    return torque * 2.0 * math.pi * spin_rate / (flow.mass_flow_kg_s * ideal_work)


def _tip_temperature_ratio(v, a):
    """Insulated-end over root temperature of a black rod radiating to a cold wall, by the model's fin.

    With length and axial conductance k A taken as 1, the net radiation exchange the model
    uses, sigma (T^2 + T_w^2)(T + T_w) P (T - T_w), becomes N_c / T0^3 (T^2 + T_w^2)(T + T_w) (T - T_w),
    N_c = 2 sigma L^2 T0^3 / (r k) being the rod's conduction parameter.
    """
    t0, t_wall, n_c = v["root_temperature_k"], v["wall_temperature_k"], v["conduction_parameter"]
    fin = fin_profile(1.0, lambda z, t: 1.0,
                      lambda z, t: n_c / t0**3 * (t**2 + t_wall**2) * (t + t_wall), t_wall, t0)
    return float(fin.temperature_k[0] / t0)


MECHANISMS = {
    "nozzle": Mechanism(
        inputs=("gas", "throat_diameter_mm"),
        point_inputs=("inverse_sqrt_reynolds",),
        quantities={"discharge_coefficient": Quantity("discharge_coefficient", _discharge_coefficient, ("gas",))},
    ),
    "windage": Mechanism(
        inputs=("inner_diameter_cm", "outer_diameter_cm", "length_cm", "viscosity_poise", "pulley_radius_cm"),
        point_inputs=("spin_rate_hz", "empty_load_g"),
        quantities={"scale_pan_load_g": Quantity(
            "couette_torque_n_m", _couette_torque,
            ("inner_diameter_cm", "outer_diameter_cm", "length_cm", "viscosity_poise", "pulley_radius_cm",
             "spin_rate_hz", "empty_load_g"),
            _torque_from_scale_pan)},
    ),
    "rarefied_drag": Mechanism(
        inputs=("gas", "ball_diameter_mm", "ball_density_g_cm3", "gas_temperature_k", "pressure_pa"),
        point_inputs=("spin_rate_hz",),
        quantities={"effective_accommodation": Quantity(
            "effective_accommodation", _effective_accommodation,
            ("gas", "ball_diameter_mm", "ball_density_g_cm3", "gas_temperature_k", "pressure_pa", "spin_rate_hz"))},
    ),
    "turbine": Mechanism(
        inputs=("gas", "supply_pressure_psia", "supply_temperature_k", "pressure_ratio", "nozzle_angle_deg",
                "mean_diameter_cm"),
        point_inputs=("blade_jet_speed_ratio", "admission_percent"),
        quantities={"total_to_static_efficiency": Quantity(
            "total_to_static_efficiency", _total_to_static_efficiency,
            ("gas", "supply_pressure_psia", "supply_temperature_k", "pressure_ratio", "nozzle_angle_deg",
             "mean_diameter_cm", "blade_jet_speed_ratio"))},
    ),
    "conduction": Mechanism(
        inputs=("wall_temperature_k",),
        point_inputs=("root_temperature_k", "conduction_parameter"),
        quantities={"tip_temperature_ratio": Quantity(
            "tip_temperature_ratio", _tip_temperature_ratio,
            ("wall_temperature_k", "root_temperature_k", "conduction_parameter"))},
    ),
}
# Compared quantity -> its mechanism, in the order the Model-Form Error is reported.
QUANTITY_MECHANISM = {q.name: name for name, m in MECHANISMS.items() for q in m.quantities.values()}


@dataclass(frozen=True)
class Point:
    inputs: dict     # point-level inputs, over the case inputs: name -> Value
    measured: dict   # measured key -> Value


@dataclass(frozen=True)
class ReferenceCase:
    id: str
    name: str
    mechanism: str
    sources: dict
    inputs: dict
    points: tuple


def load_case(path):
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sources = raw.get("sources", {})
    where = path.name
    mechanism = MECHANISMS.get(raw.get("mechanism"))
    if mechanism is None:
        raise ValueError(f"{where}: mechanism {raw.get('mechanism')!r}; expected one of {tuple(MECHANISMS)}")

    raw_inputs = raw.get("inputs", {})
    missing = [name for name in mechanism.inputs if name not in raw_inputs]
    if missing:
        raise ValueError(f"{where}: inputs not listed (leave them empty, not out): {missing}")
    unknown = set(raw_inputs) - set(mechanism.inputs)
    if unknown:
        raise ValueError(f"{where}: unknown inputs {sorted(unknown)}")
    points = []
    for i, raw_point in enumerate(raw.get("points", [])):
        if set(raw_point.get("inputs", {})) - set(mechanism.inputs + mechanism.point_inputs):
            raise ValueError(f"{where}: points[{i}] has unknown inputs")
        if set(raw_point.get("measured", {})) - set(mechanism.quantities):
            raise ValueError(f"{where}: points[{i}] measures unknown quantities; "
                             f"expected some of {tuple(mechanism.quantities)}")
        points.append(Point(
            {k: cited_value(where, f"points[{i}].{k}", e, sources) for k, e in raw_point.get("inputs", {}).items()},
            {k: cited_value(where, f"points[{i}].{k}", e, sources) for k, e in raw_point.get("measured", {}).items()},
        ))
    return ReferenceCase(
        id=raw["id"],
        name=raw["name"],
        mechanism=raw["mechanism"],
        sources=sources,
        inputs={name: cited_value(where, name, entry, sources) for name, entry in raw_inputs.items()},
        points=tuple(points),
    )


def load_cases(directory):
    return [load_case(p) for p in sorted(Path(directory).glob("*.toml"))]


def compare(case, assumptions):
    """Every measured value of the case against the model, one Comparison each."""
    mechanism = MECHANISMS[case.mechanism]
    comparisons = []
    for i, point in enumerate(case.points):
        inputs = {**case.inputs, **point.inputs}
        for name in mechanism.point_inputs:
            inputs.setdefault(name, Value())
        values = {name: entry.value for name, entry in inputs.items()}
        for key, measured in point.measured.items():
            if measured.value is None:
                continue
            quantity = mechanism.quantities[key]
            missing = [name for name in quantity.needs if values[name] is None]
            gas = values.get("gas")
            if missing:
                skipped = f"not reported: {', '.join(missing)}"
            elif gas is not None and gas not in GASES:
                skipped = f"gas {gas!r}: the model handles {', '.join(GASES)} only"
            else:
                skipped = ""
            measured_value = measured.value
            if quantity.measured_from is not None:
                # A cited reading that cannot be converted has no value in the compared units.
                measured_value = math.nan if skipped else quantity.measured_from(measured.value, values)
            if skipped:
                comparisons.append(Comparison(case.id, i, quantity.name, measured_value, math.nan, math.nan,
                                              measured.flag, skipped))
                continue
            predicted = quantity.predict(values, assumptions)
            comparisons.append(Comparison(case.id, i, quantity.name, measured_value, predicted,
                                          (predicted - measured_value) / measured_value, measured.flag, ""))
    return comparisons
