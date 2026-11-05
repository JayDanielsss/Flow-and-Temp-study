"""Evaluate one Design: Design plus assumptions in, Results out."""

import math
from dataclasses import dataclass, replace

from rotor_design.bearing import BEARING_TYPES, GAS, TWO_CENTER_PIVOT, pivot_contact_conductance
from rotor_design.drag import cold_support_shear_torque, windage_torque
from rotor_design.gas import R_HELIUM, gap_heat_transfer_coefficient, stagnation_state, vapour_enthalpy
from rotor_design.geometry import max_rotor_diameter, projected_width
from rotor_design.materials import ZIRCONIA
from rotor_design.nozzle import nozzle_flow
from rotor_design.rotordynamics import (first_bending_mode_hz, half_frequency_whirl_threshold_hz,
                                        rigid_body_critical_speeds_hz, rotor_mass_properties)
from rotor_design.stress import rotating_tube_stress
from rotor_design.thermal import rotor_thermal
from rotor_design.turbine import RADIAL_INFLOW, TURBINE_TYPES

# Slack for a Design that sits exactly on the bore limit (mm).
_FIT_TOLERANCE_MM = 1e-9
PA_PER_TORR = 133.322
PA_PER_BAR = 1e5
# Required Supply Pressure is searched up to this multiple of the limit; beyond it, reported as inf.
_PRESSURE_SEARCH_CEILING = 10.0
_PRESSURE_REL_TOLERANCE = 1e-6
_MARGIN_TOLERANCE = 1e-9

# Cooling paths, part of the Configuration.
VAPOUR_ONLY = "vapour"
COLD_SUPPORT = "vapour_and_cold_support"
COOLING_PATHS = (VAPOUR_ONLY, COLD_SUPPORT)
# Exhaust Paths, part of the Configuration.
VENTED = "vented"
DUCTED = "ducted"
EXHAUST_PATHS = (VENTED, DUCTED)
STANDARD_GRAVITY = 9.80665
# Every constraint, in build order; a Design that breaks several is ruled out by the first.
CONSTRAINTS = ("Probe Bore", "Supply Pressure", "Stress", "Critical Speed", "Whirl", "Sample Zone",
               "Sample Temperature", "Bath Heat Load")


def is_excluded(bearing, pivot_to_bath):
    """Over-constrained Configurations: a contact tie to the Bath on anything but two-center pivots."""
    return pivot_to_bath and bearing != TWO_CENTER_PIVOT


def configurations():
    """Every allowed combination of Bearing type, pivot-to-Bath option, cooling path, Turbine type
    and Exhaust Path, as Design fields."""
    return [
        {"bearing": bearing, "pivot_to_bath": tied, "cooling_path": path, "turbine": turbine,
         "exhaust_path": exhaust}
        for bearing in BEARING_TYPES
        for tied in (False, True)
        for path in COOLING_PATHS
        for turbine in TURBINE_TYPES
        for exhaust in EXHAUST_PATHS
        if not is_excluded(bearing, tied)
    ]


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
    wall_thickness_mm: float | None = None
    sample_length_mm: float | None = None
    sample_conductivity_w_m_k: float | None = None
    sample_internal_heat_load_w: float | None = None
    cooling_path: str | None = None
    cold_support_gap_mm: float | None = None
    bearing: str | None = None
    pivot_to_bath: bool | None = None
    turbine: str | None = None
    hub_exit_diameter_mm: float | None = None   # Radial-Inflow only
    exhaust_path: str | None = None
    back_pressure_bar: float | None = None      # Ducted only; Vented takes the environment pressure


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
    exhaust_path: str
    back_pressure_bar: float
    required_supply_pressure_bar: float
    required_mass_flow_kg_s: float
    nozzle_choked: bool
    jet_velocity_m_s: float
    turbine: str
    hub_exit_diameter_mm: float     # nan for a Fluted Cap
    turbine_blade_area_mm2: float   # Turbine Cap face left to the blades or flutes
    turbine_torque_n_m: float
    turbine_torque_per_flow_n_m_s_kg: float
    drag_torque_n_m: float          # windage, Cold Support gap shear and Bearing friction
    windage_regime: str
    knudsen_number: float
    wall_thickness_mm: float
    sample_length_mm: float
    sample_conductivity_w_m_k: float
    sample_internal_heat_load_w: float
    sample_volume_mm3: float        # the Rotor bore over the Sample length
    cooling_path: str
    cold_support_gap_mm: float      # nan without a Cold Support
    cold_support_drag_torque_n_m: float
    cold_support_heat_to_bath_w: float
    bearing: str
    pivot_to_bath: bool
    bearing_friction_torque_n_m: float
    bearing_stiffness_n_m: float
    bearing_radial_space_mm: float
    pivot_heat_to_bath_w: float
    warning: str
    z_mm: tuple
    temperature_k: tuple
    sample_zone_mm: float
    sample_hottest_temperature_k: float
    sample_heat_load_w: float
    sample_heat_load_breakdown_w: dict
    exhaust_heat_to_bath_w: float   # Vented Drive Gas cooled to Bath vapour; 0 when Ducted
    pumping_speed_m3_s: float       # to hold the environment pressure against the Vented gas; nan when Ducted
    bath_heat_load_w: float
    rotor_mass_kg: float            # zirconia only; the Sample is not included
    peak_hoop_stress_mpa: float
    peak_radial_stress_mpa: float
    stress_safety_factor: float     # tensile strength over peak hoop stress
    first_rigid_body_critical_hz: float
    second_rigid_body_critical_hz: float  # nan when the Rotor has no conical critical speed
    bending_critical_hz: float
    whirl_threshold_hz: float       # gas bearings only; nan otherwise
    critical_speed_separation: float  # nearest critical speed's distance from the Target Spin Rate, as a fraction
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
    body_length_mm = design.rotor_length_mm - a.turbine_cap_height_mm
    if body_length_mm <= a.bottom_cap_thickness_mm:
        raise ValueError(f"Rotor Length {design.rotor_length_mm} mm leaves no room below the Turbine Cap")
    bearing = _or(design.bearing, a.bearing)
    pivot_to_bath = _or(design.pivot_to_bath, a.pivot_to_bath)
    if bearing not in BEARING_TYPES:
        raise ValueError(f"Unknown Bearing type {bearing!r}; expected one of {BEARING_TYPES}")
    if is_excluded(bearing, pivot_to_bath):
        raise ValueError(f"Excluded Configuration: pivot-to-Bath needs two-center pivots, not {bearing!r}")
    bearing_model = a.bearings[bearing]
    turbine = _or(design.turbine, a.turbine)
    if turbine not in TURBINE_TYPES:
        raise ValueError(f"Unknown Turbine type {turbine!r}; expected one of {TURBINE_TYPES}")
    turbine_model = a.turbines[turbine]
    if turbine == RADIAL_INFLOW and design.hub_exit_diameter_mm is not None:
        turbine_model = replace(turbine_model, hub_exit_diameter_mm=design.hub_exit_diameter_mm)
    allowance = replace(a.stator_allowance, bearing_radial_space=bearing_model.radial_space_mm)
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
    exhaust_path = _or(design.exhaust_path, a.exhaust_path)
    if exhaust_path not in EXHAUST_PATHS:
        raise ValueError(f"Unknown Exhaust Path {exhaust_path!r}; expected one of {EXHAUST_PATHS}")
    # The Rotor always turns in the Bath vapour; only the turbine exit sees a return line.
    vapour_p = a.environment_pressure_torr * PA_PER_TORR
    if exhaust_path == VENTED:
        back_p = vapour_p
    else:
        back_p = _or(design.back_pressure_bar, a.ducted_back_pressure_bar) * PA_PER_BAR
    radius_m = max(diameter, 0.0) / 2e3

    cooling_path = _or(design.cooling_path, a.cooling_path)
    if cooling_path not in COOLING_PATHS:
        raise ValueError(f"Unknown cooling path {cooling_path!r}; expected one of {COOLING_PATHS}")
    windage = windage_torque(radius_m, design.rotor_length_mm / 1e3, a.rotor_stator_gap_mm / 1e3, spin,
                             a.environment_temperature_k, vapour_p, a.accommodation_coefficient)
    if cooling_path == COLD_SUPPORT:
        cold_gap_mm = _or(design.cold_support_gap_mm, a.cold_support_gap_mm)
        cold_torque = cold_support_shear_torque(radius_m, cold_gap_mm / 1e3, spin, a.environment_temperature_k,
                                                vapour_p, a.accommodation_coefficient).torque_n_m
    else:
        cold_gap_mm, cold_torque = math.nan, 0.0
    wall_mm = _or(design.wall_thickness_mm, a.baseline_wall_thickness_mm)
    length_m, inner_radius_m = design.rotor_length_mm / 1e3, max(radius_m - wall_mm / 1e3, 0.0)
    body = rotor_mass_properties(radius_m, inner_radius_m, length_m, a.bottom_cap_thickness_mm / 1e3,
                                 a.turbine_cap_height_mm / 1e3, ZIRCONIA.density_kg_m3)
    bearing_load = bearing_model.performance(
        rotor_radius_m=radius_m,
        rotor_weight_n=body.mass_kg * STANDARD_GRAVITY,
        spin_rate_hz=spin,
        bath_temperature_k=a.environment_temperature_k,
        ambient_pressure_pa=vapour_p,
        accommodation=a.accommodation_coefficient,
    )
    drag_torque = windage.torque_n_m + cold_torque + bearing_load.friction_torque_n_m

    def drive(supply_p):
        flow = nozzle_flow(supply_p, gas_t, back_p, throat_mm / 1e3, a.discharge_coefficient)
        torque = nozzle_count * turbine_model.torque(flow.mass_flow_kg_s, flow.jet_velocity_m_s, angle_deg,
                                                     radius_m, spin, a.turbine_efficiency)
        return flow, torque

    supply_p = _required_supply_pressure(drive, drag_torque, back_p,
                                         _PRESSURE_SEARCH_CEILING * a.max_supply_pressure_bar * PA_PER_BAR)
    if math.isfinite(supply_p):
        flow, turbine_torque = drive(supply_p)
        mass_flow, choked, jet_v = nozzle_count * flow.mass_flow_kg_s, flow.choked, flow.jet_velocity_m_s
    else:
        mass_flow, choked, jet_v, turbine_torque = math.inf, True, math.nan, math.nan
    supply_bar = supply_p / PA_PER_BAR
    reaches_spin = supply_bar <= a.max_supply_pressure_bar

    sample_mm = _or(design.sample_length_mm, a.baseline_sample_length_mm)
    sample_k = _or(design.sample_conductivity_w_m_k, a.sample_conductivity_w_m_k)
    sample_q = _or(design.sample_internal_heat_load_w, a.sample_internal_heat_load_w)
    windage_heat = (windage.torque_n_m * 2.0 * math.pi * spin * a.windage_heat_to_rotor_fraction
                    / (design.rotor_length_mm / 1e3))
    thermal = rotor_thermal(
        outer_radius_m=radius_m,
        wall_m=wall_mm / 1e3,
        body_length_m=body_length_mm / 1e3,
        bottom_cap_m=a.bottom_cap_thickness_mm / 1e3,
        sample_length_m=sample_mm / 1e3,
        wall_conductivity=lambda t: ZIRCONIA.conductivity(t, a.zirconia_conductivity_multiplier),
        sample_conductivity_w_m_k=sample_k,
        sample_internal_heat_load_w=sample_q,
        # The Turbine Cap sits in the Drive Gas stream; taken at its supply temperature (upper bound).
        turbine_cap_temperature_k=gas_t,
        bath_temperature_k=a.environment_temperature_k,
        stator_temperature_k=a.stator_temperature_k,
        vapour_pressure_pa=vapour_p,
        gap_m=a.rotor_stator_gap_mm / 1e3,
        accommodation=a.accommodation_coefficient,
        emissivity=a.rotor_emissivity,
        windage_heat_per_length_w_m=windage_heat,
        sample_zone_delta_t_k=a.sample_zone_delta_t_k,
        gap_heat_transfer_coefficient=gap_heat_transfer_coefficient,
        cold_support_gap_m=None if cooling_path == VAPOUR_ONLY else cold_gap_mm / 1e3,
        cold_support_shear_heat_w=cold_torque * 2.0 * math.pi * spin * a.windage_heat_to_rotor_fraction,
        bearing_friction_heat_w=bearing_load.bottom_friction_heat_w,
        bearing_gas_heat_w=bearing_load.bottom_gas_heat_w,
        pivot_conductance=(lambda t: pivot_contact_conductance(bearing_load.contact_radius_m, t,
                                                               a.environment_temperature_k,
                                                               a.zirconia_conductivity_multiplier))
        if pivot_to_bath else None,
        vented_gas_temperature_k=gas_t if exhaust_path == VENTED else None,
    )
    sample_zone_mm = thermal.sample_zone_m * 1e3

    # Bath Heat Load, with the Stator at Bath temperature: all drag work ends there as heat.
    drag_power = drag_torque * 2.0 * math.pi * spin
    sources_w = drag_power + bearing_load.bottom_gas_heat_w + sample_q
    if exhaust_path == VENTED:
        # The vented gas is cooled to Bath vapour; this also covers its convection onto the
        # Rotor and the Turbine Cap conduction, which it supplies.
        if math.isfinite(mass_flow):
            h0, _ = stagnation_state(gas_t, supply_p)
            exhaust_heat = mass_flow * (h0 - vapour_enthalpy(a.environment_temperature_k, vapour_p)) - drag_power
        else:
            exhaust_heat = math.inf
        pumping_speed = mass_flow * R_HELIUM * a.environment_temperature_k / vapour_p
        bath_load = sources_w + exhaust_heat
    else:
        exhaust_heat, pumping_speed = 0.0, math.nan
        bath_load = sources_w + thermal.turbine_cap_heat_in_w
    sample_limit_k = a.environment_temperature_k + a.sample_zone_delta_t_k

    stress = rotating_tube_stress(radius_m, inner_radius_m, spin, ZIRCONIA.density_kg_m3, ZIRCONIA.poissons_ratio)
    safety_factor = ZIRCONIA.tensile_strength_pa / stress.peak_hoop_pa if stress.peak_hoop_pa > 0.0 else math.inf
    # Bearing stations at the two ends of the Rotor.
    rigid = rigid_body_critical_speeds_hz(body.mass_kg, body.transverse_inertia_kg_m2, body.polar_inertia_kg_m2,
                                          bearing_load.stiffness_n_m, body.centre_of_mass_m,
                                          length_m - body.centre_of_mass_m)
    bending = (first_bending_mode_hz(length_m, radius_m, inner_radius_m, ZIRCONIA.youngs_modulus_pa,
                                     ZIRCONIA.density_kg_m3) if radius_m > 0.0 else math.nan)
    criticals = [f for f in (*rigid, bending) if math.isfinite(f)]
    separation = min((abs(f - spin) / spin for f in criticals), default=math.inf)
    whirl = half_frequency_whirl_threshold_hz(rigid, a.whirl_onset_ratio) if bearing == GAS and rigid else math.nan
    # Gas-bearing whirl must stay clear above the Target Spin Rate by the same separation.
    whirl_limit_hz = spin * (1.0 + a.critical_speed_separation)

    # Each constraint's fractional margin; the smallest is the Binding Constraint.
    margins = {
        "Probe Bore": (usable_bore - width) / usable_bore if diameter > 0.0 else -1.0,
        "Supply Pressure": (a.max_supply_pressure_bar - supply_bar) / a.max_supply_pressure_bar,
        "Stress": (safety_factor - a.required_safety_factor) / a.required_safety_factor,
        "Critical Speed": (separation - a.critical_speed_separation) / a.critical_speed_separation,
        "Whirl": whirl / whirl_limit_hz - 1.0 if math.isfinite(whirl) else math.inf,
        "Sample Zone": (sample_zone_mm - a.min_sample_zone_mm) / a.min_sample_zone_mm,
        "Sample Temperature": (sample_limit_k - thermal.sample_hottest_temperature_k) / a.sample_zone_delta_t_k,
        "Bath Heat Load": (a.bath_cooling_power_w - bath_load) / a.bath_cooling_power_w,
    }
    assert tuple(margins) == CONSTRAINTS
    # A Design that breaks several constraints is ruled out by the first in build order
    # (it cannot be built if it does not fit); otherwise the smallest margin limits it.
    violated = [name for name, margin in margins.items() if margin < -_MARGIN_TOLERANCE]
    binding = violated[0] if violated else min(margins, key=margins.get)

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
        exhaust_path=exhaust_path,
        back_pressure_bar=back_p / PA_PER_BAR,
        required_supply_pressure_bar=supply_bar,
        required_mass_flow_kg_s=mass_flow,
        nozzle_choked=choked,
        jet_velocity_m_s=jet_v,
        turbine=turbine,
        hub_exit_diameter_mm=turbine_model.hub_exit_diameter_mm,
        turbine_blade_area_mm2=turbine_model.blade_area_m2(radius_m) * 1e6,
        turbine_torque_n_m=turbine_torque,
        turbine_torque_per_flow_n_m_s_kg=turbine_torque / mass_flow if math.isfinite(mass_flow) else math.nan,
        drag_torque_n_m=drag_torque,
        windage_regime=windage.regime,
        knudsen_number=windage.knudsen_number,
        wall_thickness_mm=wall_mm,
        sample_length_mm=sample_mm,
        sample_conductivity_w_m_k=sample_k,
        sample_internal_heat_load_w=sample_q,
        sample_volume_mm3=math.pi * (inner_radius_m * 1e3) ** 2 * sample_mm,
        cooling_path=cooling_path,
        cold_support_gap_mm=cold_gap_mm,
        cold_support_drag_torque_n_m=cold_torque,
        cold_support_heat_to_bath_w=thermal.cold_support_heat_out_w,
        bearing=bearing,
        pivot_to_bath=pivot_to_bath,
        bearing_friction_torque_n_m=bearing_load.friction_torque_n_m,
        bearing_stiffness_n_m=bearing_load.stiffness_n_m,
        bearing_radial_space_mm=bearing_load.radial_space_mm,
        pivot_heat_to_bath_w=thermal.pivot_heat_out_w,
        warning=bearing_load.warning,
        z_mm=tuple(thermal.z_m * 1e3),
        temperature_k=tuple(thermal.temperature_k),
        sample_zone_mm=sample_zone_mm,
        sample_hottest_temperature_k=thermal.sample_hottest_temperature_k,
        sample_heat_load_w=thermal.sample_heat_load_w,
        sample_heat_load_breakdown_w=thermal.breakdown_w,
        exhaust_heat_to_bath_w=exhaust_heat,
        pumping_speed_m3_s=pumping_speed,
        bath_heat_load_w=bath_load,
        rotor_mass_kg=body.mass_kg,
        peak_hoop_stress_mpa=stress.peak_hoop_pa / 1e6,
        peak_radial_stress_mpa=stress.peak_radial_pa / 1e6,
        stress_safety_factor=safety_factor,
        first_rigid_body_critical_hz=rigid[0] if rigid else math.nan,
        second_rigid_body_critical_hz=rigid[1] if len(rigid) > 1 else math.nan,
        bending_critical_hz=bending,
        whirl_threshold_hz=whirl,
        critical_speed_separation=separation,
        feasible=(fits and reaches_spin and sample_zone_mm >= a.min_sample_zone_mm
                  and thermal.sample_hottest_temperature_k <= sample_limit_k
                  and bath_load <= a.bath_cooling_power_w
                  and safety_factor >= a.required_safety_factor
                  and separation >= a.critical_speed_separation
                  and not whirl < whirl_limit_hz),
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
