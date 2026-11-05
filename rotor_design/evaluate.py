"""Evaluate one Design: Design plus assumptions in, Results out."""

import math
from dataclasses import dataclass

from rotor_design.drag import windage_torque
from rotor_design.geometry import max_rotor_diameter, projected_width
from rotor_design.nozzle import nozzle_flow
from rotor_design.turbine import fluted_cap_torque

# Slack for a Design that sits exactly on the bore limit (mm).
_FIT_TOLERANCE_MM = 1e-9
PA_PER_TORR = 133.322
PA_PER_BAR = 1e5
# Required Supply Pressure is searched up to this multiple of the limit; beyond it, reported as inf.
_PRESSURE_SEARCH_CEILING = 10.0
_PRESSURE_REL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Design:
    rotor_length_mm: float
    # None means the largest diameter the Probe Bore allows.
    rotor_diameter_mm: float | None = None
    # None means the value from the assumptions file.
    nozzle_count: int | None = None
    nozzle_throat_mm: float | None = None
    nozzle_angle_deg: float | None = None
    drive_gas_temperature_k: float | None = None


@dataclass(frozen=True)
class Results:
    rotor_length_mm: float
    rotor_diameter_mm: float
    max_rotor_diameter_mm: float
    stator_length_mm: float
    stator_diameter_mm: float
    projected_width_mm: float
    usable_bore_mm: float
    fits_bore: bool
    nozzle_count: int
    nozzle_throat_mm: float
    nozzle_angle_deg: float
    drive_gas_temperature_k: float
    spin_rate_hz: float
    back_pressure_bar: float
    required_supply_pressure_bar: float
    required_mass_flow_kg_s: float
    nozzle_choked: bool
    jet_velocity_m_s: float
    turbine_torque_n_m: float
    drag_torque_n_m: float
    windage_regime: str
    knudsen_number: float
    feasible: bool
    binding_constraint: str


def _or(value, default):
    return default if value is None else value


def evaluate(design, assumptions):
    a = assumptions
    if design.rotor_length_mm > a.max_rotor_length_mm:
        raise ValueError(
            f"Rotor Length {design.rotor_length_mm} mm exceeds the {a.max_rotor_length_mm} mm limit"
        )
    allowance = a.stator_allowance
    largest = max_rotor_diameter(design.rotor_length_mm, allowance, a.bore_diameter_mm, a.tilt_angle_deg)
    diameter = _or(design.rotor_diameter_mm, largest)

    stator_length = allowance.stator_length(design.rotor_length_mm)
    stator_diameter = allowance.stator_diameter(diameter)
    width = projected_width(stator_length, stator_diameter, a.tilt_angle_deg)
    usable_bore = allowance.usable_bore(a.bore_diameter_mm)
    fits = diameter > 0.0 and width <= usable_bore + _FIT_TOLERANCE_MM

    nozzle_count = _or(design.nozzle_count, a.baseline_nozzle_count)
    throat_mm = _or(design.nozzle_throat_mm, a.baseline_nozzle_throat_mm)
    angle_deg = _or(design.nozzle_angle_deg, a.baseline_nozzle_angle_deg)
    gas_t = _or(design.drive_gas_temperature_k, a.drive_gas_temperature_k)
    spin = a.target_spin_rate_hz
    # Vented exhaust: the turbine exits into the probe space.
    back_p = a.environment_pressure_torr * PA_PER_TORR
    radius_m = max(diameter, 0.0) / 2e3

    windage = windage_torque(radius_m, design.rotor_length_mm / 1e3, a.rotor_stator_gap_mm / 1e3, spin,
                             a.environment_temperature_k, back_p, a.accommodation_coefficient)

    def drive(supply_p):
        flow = nozzle_flow(supply_p, gas_t, back_p, throat_mm / 1e3, a.discharge_coefficient)
        torque = nozzle_count * fluted_cap_torque(flow.mass_flow_kg_s, flow.jet_velocity_m_s, angle_deg,
                                                  radius_m, spin, a.turbine_efficiency)
        return flow, torque

    supply_p = _required_supply_pressure(drive, windage.torque_n_m, back_p,
                                         _PRESSURE_SEARCH_CEILING * a.max_supply_pressure_bar * PA_PER_BAR)
    if math.isfinite(supply_p):
        flow, turbine_torque = drive(supply_p)
        mass_flow, choked, jet_v = nozzle_count * flow.mass_flow_kg_s, flow.choked, flow.jet_velocity_m_s
    else:
        mass_flow, choked, jet_v, turbine_torque = math.inf, True, math.nan, math.nan
    supply_bar = supply_p / PA_PER_BAR
    reaches_spin = supply_bar <= a.max_supply_pressure_bar

    # Each constraint's fractional margin; the smallest is the Binding Constraint.
    margins = {
        "Probe Bore": (usable_bore - width) / usable_bore if diameter > 0.0 else -1.0,
        "Supply Pressure": (a.max_supply_pressure_bar - supply_bar) / a.max_supply_pressure_bar,
    }
    binding = min(margins, key=margins.get)

    return Results(
        rotor_length_mm=design.rotor_length_mm,
        rotor_diameter_mm=diameter,
        max_rotor_diameter_mm=largest,
        stator_length_mm=stator_length,
        stator_diameter_mm=stator_diameter,
        projected_width_mm=width,
        usable_bore_mm=usable_bore,
        fits_bore=fits,
        nozzle_count=nozzle_count,
        nozzle_throat_mm=throat_mm,
        nozzle_angle_deg=angle_deg,
        drive_gas_temperature_k=gas_t,
        spin_rate_hz=spin,
        back_pressure_bar=back_p / PA_PER_BAR,
        required_supply_pressure_bar=supply_bar,
        required_mass_flow_kg_s=mass_flow,
        nozzle_choked=choked,
        jet_velocity_m_s=jet_v,
        turbine_torque_n_m=turbine_torque,
        drag_torque_n_m=windage.torque_n_m,
        windage_regime=windage.regime,
        knudsen_number=windage.knudsen_number,
        feasible=fits and reaches_spin,
        binding_constraint=binding,
    )


def _required_supply_pressure(drive, drag_torque, back_pressure, ceiling):
    """Lowest Supply Pressure whose turbine torque balances the drag; inf if none up to ceiling."""
    if drive(ceiling)[1] < drag_torque:
        return math.inf
    lo, hi = back_pressure, ceiling
    while hi / lo > 1.0 + _PRESSURE_REL_TOLERANCE:
        mid = math.sqrt(lo * hi)  # geometric: the answer can lie decades above back pressure
        if drive(mid)[1] < drag_torque:
            lo = mid
        else:
            hi = mid
    return hi
