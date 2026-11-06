"""Evaluate one Design: Design plus assumptions in, Results out."""

import math
from dataclasses import dataclass, replace
from functools import partial

from rotor_design.bearing import BEARING_TYPES, GAS
from rotor_design.drag import cold_support_shear_torque, rotor_windage
from rotor_design.gas import GASES, gap_heat_transfer_coefficient, stagnation_state, vapour_enthalpy
from rotor_design.geometry import max_rotor_diameter, projected_width
from rotor_design.materials import HELIUM4_CURVE_K, ZIRCONIA, helium_vapour_pressure
from rotor_design.nozzle import ThroatSearchRangeError, nozzle_flow
from rotor_design.rotordynamics import (first_bending_mode_hz, half_frequency_whirl_threshold_hz,
                                        rigid_body_critical_speeds_hz, rotor_mass_properties)
from rotor_design.stress import rotating_tube_stress
from rotor_design.thermal import place_sample
from rotor_design.turbine import RADIAL_INFLOW, TURBINE_TYPES

# Slack for a Design that sits exactly on the bore limit (mm).
_FIT_TOLERANCE_MM = 1e-9
PA_PER_TORR = 133.322
PA_PER_BAR = 1e5
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
CONSTRAINTS = ("Probe Bore", "Supply Pressure", "Stress", "Critical Speed", "Whirl", "Placement Window",
               "Sample Temperature", "Bath Heat Load")
# The physics constraints, which name what limits a Design that fits the Probe Bore.
PHYSICS_CONSTRAINTS = CONSTRAINTS[1:]


def configurations():
    """Every combination of Bearing type, cooling path, Turbine type and Exhaust Path, as Design fields."""
    return [
        {"bearing": bearing, "cooling_path": path, "turbine": turbine, "exhaust_path": exhaust}
        for bearing in BEARING_TYPES
        for path in COOLING_PATHS
        for turbine in TURBINE_TYPES
        for exhaust in EXHAUST_PATHS
    ]


def largest_rotor_diameter_mm(rotor_length_mm, bearing, assumptions):
    """Largest Rotor diameter the Probe Bore allows at this Rotor Length with this Bearing type
    (None means the assumptions file's)."""
    a = assumptions
    radial_space = a.bearings[_or(bearing, a.bearing)].radial_space_mm
    allowance = replace(a.stator_allowance, bearing_radial_space=radial_space)
    return max_rotor_diameter(rotor_length_mm, allowance, a.bore_diameter_mm, a.tilt_angle_deg)


@dataclass(frozen=True)
class Design:
    rotor_length_mm: float
    # Give at most one of the two; both None means the largest diameter the Probe Bore allows.
    rotor_diameter_mm: float | None = None
    rotor_diameter_fraction: float | None = None   # of the largest the Probe Bore allows
    # None means the value from the assumptions file.
    nozzle_count: int | None = None
    nozzle_throat_mm: float | None = None
    nozzle_angle_deg: float | None = None
    drive_gas: str | None = None                # also feeds the gas Bearings
    drive_gas_temperature_k: float | None = None
    wall_thickness_mm: float | None = None
    sample_length_mm: float | None = None
    # From the Bottom Cap face to the bottom of the Sample; None places it where the Sample Heat
    # Load is lowest within the Placement Window.
    sample_position_mm: float | None = None
    sample_conductivity_w_m_k: float | None = None
    sample_internal_heat_load_w: float | None = None
    cooling_path: str | None = None
    cold_support_gap_mm: float | None = None
    bearing: str | None = None
    turbine: str | None = None
    hub_exit_diameter_mm: float | None = None   # Radial-Inflow only
    exhaust_path: str | None = None
    back_pressure_bar: float | None = None      # Ducted only; Vented takes the Cold Space pressure


@dataclass(frozen=True)
class Results:
    rotor_length_mm: float
    rotor_diameter_mm: float
    rotor_diameter_fraction: float  # of the largest the Probe Bore allows
    max_rotor_diameter_mm: float
    stator_length_mm: float
    stator_diameter_mm: float
    projected_width_mm: float
    usable_bore_mm: float
    fits_bore: bool
    probe_bore_margin: float        # usable bore left over, as a fraction; reported apart from the Binding Constraint
    nozzle_count: int
    nozzle_throat_mm: float
    nozzle_angle_deg: float
    drive_gas: str
    environment_gas: str            # the gas the Rotor turns in
    bath_temperature_k: float       # every sink: Bath, Cold Support, Vapour Cooling
    sample_limit_temperature_k: float
    cold_space_pressure_torr: float  # the Bath vapour pressure at the Bath Temperature
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
    turbine_cap_windage_torque_n_m: float  # Turbine Cap section, in Drive Gas at the Back Pressure
    turbine_cap_windage_regime: str
    turbine_cap_windage_heat_w: float      # into the Turbine Cap, held at Drive Gas temperature; not the body
    body_windage_torque_n_m: float  # body below the Turbine Cap, in the vapour
    windage_regime: str             # body section
    knudsen_number: float           # body section
    wall_thickness_mm: float
    sample_length_mm: float
    sample_position_mm: float       # from the Bottom Cap face to the bottom of the Sample
    placement_window_low_mm: float  # lowest Sample Position keeping the whole Sample at or below the limit; nan if none
    placement_window_high_mm: float  # highest such Sample Position; nan if none
    bottom_standoff_mm: float       # least distance from the Bottom Cap face to the bottom of the Sample
    turbine_cap_standoff_mm: float  # least distance from the top of the Sample to the Turbine Cap
    sample_conductivity_w_m_k: float
    sample_internal_heat_load_w: float
    sample_volume_mm3: float        # the Rotor bore over the Sample length
    cooling_path: str
    cold_support_gap_mm: float      # nan without a Cold Support
    cold_support_drag_torque_n_m: float
    cold_support_heat_to_bath_w: float
    bearing: str
    bearing_friction_torque_n_m: float
    bearing_stiffness_n_m: float
    bearing_radial_space_mm: float
    balance_grade_um: float         # centre-of-mass offset from the spin axis
    unbalance_force_n: float        # rotating load m e omega^2 at the Balance Grade
    pivot_preload_n: float          # sized Two-Center Pivot Preload; nan for other Bearings
    warning: str
    z_mm: tuple
    temperature_k: tuple
    sample_zone_mm: float           # diagnostic: longest stretch at or below the Sample Limit Temperature
    sample_hottest_temperature_k: float
    sample_heat_load_w: float       # heat crossing the Sample's boundary at its Sample Position
    sample_heat_load_breakdown_w: dict
    exhaust_heat_to_bath_w: float   # Vented Drive Gas cooled to Bath vapour; 0 when Ducted
    pumping_speed_m3_s: float       # to hold the Cold Space pressure against the Vented gas; nan when Ducted
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
    if bearing not in BEARING_TYPES:
        raise ValueError(f"Unknown Bearing type {bearing!r}; expected one of {BEARING_TYPES}")
    bearing_model = a.bearings[bearing]
    turbine = _or(design.turbine, a.turbine)
    if turbine not in TURBINE_TYPES:
        raise ValueError(f"Unknown Turbine type {turbine!r}; expected one of {TURBINE_TYPES}")
    turbine_model = a.turbines[turbine]
    if turbine == RADIAL_INFLOW and design.hub_exit_diameter_mm is not None:
        turbine_model = replace(turbine_model, hub_exit_diameter_mm=design.hub_exit_diameter_mm)
    allowance = replace(a.stator_allowance, bearing_radial_space=bearing_model.radial_space_mm)
    largest = largest_rotor_diameter_mm(design.rotor_length_mm, bearing, a)
    if design.rotor_diameter_mm is not None and design.rotor_diameter_fraction is not None:
        raise ValueError("Give the Rotor diameter or its fraction of the largest allowed, not both")
    if design.rotor_diameter_fraction is not None:
        diameter = design.rotor_diameter_fraction * largest
    else:
        diameter = _or(design.rotor_diameter_mm, largest)
    fraction = _or(design.rotor_diameter_fraction, diameter / largest if largest > 0.0 else math.nan)

    stator_length = allowance.stator_length(design.rotor_length_mm)
    stator_diameter = allowance.stator_diameter(diameter)
    width = projected_width(stator_length, stator_diameter, a.tilt_angle_deg)
    usable_bore = allowance.usable_bore(a.bore_diameter_mm)
    fits = diameter > 0.0 and width <= usable_bore + _FIT_TOLERANCE_MM

    nozzle_count = _or(design.nozzle_count, a.baseline_nozzle_count)
    throat_mm = _or(design.nozzle_throat_mm, a.baseline_nozzle_throat_mm)
    angle_deg = _or(design.nozzle_angle_deg, a.baseline_nozzle_angle_deg)
    gas_t = _or(design.drive_gas_temperature_k, a.drive_gas_temperature_k)
    drive_gas_name = _or(design.drive_gas, a.drive_gas)
    for name in (drive_gas_name, a.environment_gas):
        if name not in GASES:
            raise ValueError(f"Unknown gas {name!r}; expected one of {tuple(GASES)}")
    drive_gas, environment_gas = GASES[drive_gas_name], GASES[a.environment_gas]
    spin = a.target_spin_rate_hz
    exhaust_path = _or(design.exhaust_path, a.exhaust_path)
    if exhaust_path not in EXHAUST_PATHS:
        raise ValueError(f"Unknown Exhaust Path {exhaust_path!r}; expected one of {EXHAUST_PATHS}")
    # The Rotor always turns in the Bath vapour; only the turbine exit sees a return line.
    bath_t, sample_limit_k = a.bath_temperature_k, a.sample_limit_temperature_k
    if sample_limit_k <= bath_t:
        raise ValueError(f"Sample Limit Temperature {sample_limit_k} K must lie above the "
                         f"{bath_t} K Bath Temperature, the coldest sink")
    vapour_p = _cold_space_pressure(environment_gas, bath_t, a.cold_space_pressure_pa)
    stator_t = _or(a.stator_temperature_k, bath_t)
    if exhaust_path == VENTED:
        back_p = vapour_p
    else:
        back_p = _or(design.back_pressure_bar, a.ducted_back_pressure_bar) * PA_PER_BAR
    radius_m = max(diameter, 0.0) / 2e3

    cooling_path = _or(design.cooling_path, a.cooling_path)
    if cooling_path not in COOLING_PATHS:
        raise ValueError(f"Unknown cooling path {cooling_path!r}; expected one of {COOLING_PATHS}")
    # The Turbine Cap turns in Drive Gas at its supply temperature and the Back Pressure,
    # the body below it in the vapour; both across the same gap to the Stator.
    windage = rotor_windage(radius_m, a.rotor_stator_gap_mm / 1e3, spin,
                            turbine_cap_length_m=a.turbine_cap_height_mm / 1e3, drive_gas_temperature_k=gas_t,
                            back_pressure_pa=back_p, body_length_m=body_length_mm / 1e3,
                            vapour_temperature_k=bath_t, vapour_pressure_pa=vapour_p,
                            accommodation=a.accommodation_coefficient, drive_gas=drive_gas,
                            vapour_gas=environment_gas)
    cap_windage_torque = windage.turbine_cap.torque_n_m * a.windage_torque_multiplier
    body_windage_torque = windage.body.torque_n_m * a.windage_torque_multiplier
    windage_torque_n_m = cap_windage_torque + body_windage_torque
    if cooling_path == COLD_SUPPORT:
        cold_gap_mm = _or(design.cold_support_gap_mm, a.cold_support_gap_mm)
        cold_torque = cold_support_shear_torque(radius_m, cold_gap_mm / 1e3, spin, bath_t,
                                                vapour_p, a.accommodation_coefficient,
                                                environment_gas).torque_n_m
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
        bath_temperature_k=bath_t,
        ambient_pressure_pa=vapour_p,
        accommodation=a.accommodation_coefficient,
        gas=drive_gas,
        rotor_mass_kg=body.mass_kg,
        lower_arm_m=body.centre_of_mass_m,
        upper_arm_m=length_m - body.centre_of_mass_m,
        tilt_angle_deg=a.tilt_angle_deg,
        balance_grade_m=a.balance_grade_um / 1e6,
    )
    drag_torque = windage_torque_n_m + cold_torque + bearing_load.friction_torque_n_m

    def drive(supply_p):
        flow = nozzle_flow(supply_p, gas_t, back_p, throat_mm / 1e3, a.discharge_coefficient, drive_gas)
        torque = nozzle_count * turbine_model.torque(flow.mass_flow_kg_s, flow.jet_velocity_m_s, angle_deg,
                                                     radius_m, spin,
                                                     a.turbine_efficiency * a.turbine_efficiency_correction[turbine])
        return flow, torque

    # Searched only up to the Supply Pressure limit; a Design that needs more is reported as inf.
    supply_p = _required_supply_pressure(drive, drag_torque, back_p, a.max_supply_pressure_bar * PA_PER_BAR)
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
    # The body's windage heat is spread over the body alone. The cap's is taken up at the Turbine
    # Cap, which the Drive Gas holds at its temperature, so it does not reach the body.
    windage_heat = (body_windage_torque * 2.0 * math.pi * spin * a.windage_heat_to_rotor_fraction
                    / (body_length_mm / 1e3))
    cap_windage_heat = cap_windage_torque * 2.0 * math.pi * spin * a.windage_heat_to_rotor_fraction
    position_mm = design.sample_position_mm
    placement = place_sample(
        position_step_m=a.sample_position_step_mm / 1e3,
        sample_position_m=None if position_mm is None else position_mm / 1e3,
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
        bath_temperature_k=bath_t,
        stator_temperature_k=stator_t,
        vapour_pressure_pa=vapour_p,
        gap_m=a.rotor_stator_gap_mm / 1e3,
        accommodation=a.accommodation_coefficient,
        emissivity=a.rotor_emissivity,
        windage_heat_per_length_w_m=windage_heat,
        sample_limit_temperature_k=sample_limit_k,
        # The Rotor's surroundings, Vented Drive Gas included, take the environment gas.
        gap_heat_transfer_coefficient=partial(gap_heat_transfer_coefficient, gas=environment_gas),
        cold_support_gap_m=None if cooling_path == VAPOUR_ONLY else cold_gap_mm / 1e3,
        cold_support_shear_heat_w=cold_torque * 2.0 * math.pi * spin * a.windage_heat_to_rotor_fraction,
        bearing_friction_heat_w=bearing_load.bottom_friction_heat_w,
        bearing_gas_heat_w=bearing_load.bottom_gas_heat_w,
        vented_gas_temperature_k=gas_t if exhaust_path == VENTED else None,
    )
    thermal = placement.thermal
    sample_zone_mm = thermal.sample_zone_m * 1e3
    window_lo, window_hi = (edge * 1e3 for edge in placement.window_m)

    # Bath Heat Load, with the Stator at Bath Temperature: all drag work ends there as heat.
    drag_power = drag_torque * 2.0 * math.pi * spin
    sources_w = drag_power + bearing_load.bottom_gas_heat_w + sample_q
    if exhaust_path == VENTED:
        # The vented gas is cooled to Bath vapour; this also covers its convection onto the
        # Rotor and the Turbine Cap conduction, which it supplies.
        if math.isfinite(mass_flow):
            h0, _ = stagnation_state(gas_t, supply_p, drive_gas)
            exhaust_heat = (mass_flow * (h0 - vapour_enthalpy(bath_t, vapour_p, drive_gas))
                            - drag_power)
        else:
            exhaust_heat = math.inf
        pumping_speed = mass_flow * drive_gas.specific_gas_constant * bath_t / vapour_p
        bath_load = sources_w + exhaust_heat
    else:
        exhaust_heat, pumping_speed = 0.0, math.nan
        bath_load = sources_w + thermal.turbine_cap_heat_in_w

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

    # Each constraint's fractional margin.
    bore_margin = (usable_bore - width) / usable_bore if diameter > 0.0 else -1.0
    margins = {
        "Probe Bore": bore_margin,
        "Supply Pressure": (a.max_supply_pressure_bar - supply_bar) / a.max_supply_pressure_bar,
        "Stress": (safety_factor - a.required_safety_factor) / a.required_safety_factor,
        "Critical Speed": (separation - a.critical_speed_separation) / a.critical_speed_separation,
        "Whirl": whirl / whirl_limit_hz - 1.0 if math.isfinite(whirl) else math.inf,
        # Headroom at the coolest Sample Position: negative exactly when the Placement Window is empty.
        "Placement Window": placement.best_headroom_k / (sample_limit_k - bath_t),
        # Fraction of the headroom between the Bath and the Sample Limit Temperature left unused.
        "Sample Temperature": ((sample_limit_k - thermal.sample_hottest_temperature_k)
                               / (sample_limit_k - bath_t)),
        "Bath Heat Load": (a.bath_cooling_power_w - bath_load) / a.bath_cooling_power_w,
    }
    assert tuple(margins) == CONSTRAINTS
    # A Design that breaks several constraints is ruled out by the first in build order
    # (it cannot be built if it does not fit); otherwise the physics constraint with the smallest
    # margin limits it. The Probe Bore margin only says how close the chosen diameter sits to the
    # bore limit, so it is reported on its own.
    violated = [name for name, margin in margins.items() if margin < -_MARGIN_TOLERANCE]
    binding = violated[0] if violated else min(PHYSICS_CONSTRAINTS, key=margins.get)

    return Results(
        rotor_length_mm=design.rotor_length_mm,
        rotor_diameter_mm=diameter,
        rotor_diameter_fraction=fraction,
        max_rotor_diameter_mm=largest,
        stator_length_mm=stator_length,
        stator_diameter_mm=stator_diameter,
        projected_width_mm=width,
        usable_bore_mm=usable_bore,
        fits_bore=fits,
        probe_bore_margin=bore_margin,
        nozzle_count=nozzle_count,
        nozzle_throat_mm=throat_mm,
        nozzle_angle_deg=angle_deg,
        drive_gas=drive_gas.name,
        environment_gas=environment_gas.name,
        bath_temperature_k=bath_t,
        sample_limit_temperature_k=sample_limit_k,
        cold_space_pressure_torr=vapour_p / PA_PER_TORR,
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
        turbine_cap_windage_torque_n_m=cap_windage_torque,
        turbine_cap_windage_regime=windage.turbine_cap.regime,
        turbine_cap_windage_heat_w=cap_windage_heat,
        body_windage_torque_n_m=body_windage_torque,
        windage_regime=windage.body.regime,
        knudsen_number=windage.body.knudsen_number,
        wall_thickness_mm=wall_mm,
        sample_length_mm=sample_mm,
        sample_position_mm=thermal.sample_position_m * 1e3,
        placement_window_low_mm=window_lo,
        placement_window_high_mm=window_hi,
        bottom_standoff_mm=window_lo,
        turbine_cap_standoff_mm=body_length_mm - a.bottom_cap_thickness_mm - sample_mm - window_hi,
        sample_conductivity_w_m_k=sample_k,
        sample_internal_heat_load_w=sample_q,
        sample_volume_mm3=math.pi * (inner_radius_m * 1e3) ** 2 * sample_mm,
        cooling_path=cooling_path,
        cold_support_gap_mm=cold_gap_mm,
        cold_support_drag_torque_n_m=cold_torque,
        cold_support_heat_to_bath_w=thermal.cold_support_heat_out_w,
        bearing=bearing,
        bearing_friction_torque_n_m=bearing_load.friction_torque_n_m,
        bearing_stiffness_n_m=bearing_load.stiffness_n_m,
        bearing_radial_space_mm=bearing_load.radial_space_mm,
        balance_grade_um=a.balance_grade_um,
        unbalance_force_n=bearing_load.unbalance_force_n,
        pivot_preload_n=bearing_load.preload_n,
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
        feasible=(fits and reaches_spin and not placement.empty
                  and thermal.sample_hottest_temperature_k <= sample_limit_k
                  and bath_load <= a.bath_cooling_power_w
                  and safety_factor >= a.required_safety_factor
                  and separation >= a.critical_speed_separation
                  and not whirl < whirl_limit_hz),
        binding_constraint=binding,
    )


def _cold_space_pressure(gas, bath_temperature_k, override_pa):
    """Cold Space pressure (Pa): the helium-4 vapour pressure at the Bath Temperature.

    A warm case not above a helium-4 Bath (a Benchmark Case or toy model) sets its own.
    """
    on_curve = gas.name == "helium" and bath_temperature_k <= HELIUM4_CURVE_K[1]
    if override_pa is None:
        if not on_curve:
            raise ValueError(f"No helium-4 Bath at {bath_temperature_k} K in {gas.name}: "
                             "set the Cold Space pressure for this warm case")
        return helium_vapour_pressure(bath_temperature_k)
    if on_curve:
        raise ValueError(f"The Cold Space pressure follows from the {bath_temperature_k} K Bath Temperature "
                         "on the helium-4 vapour-pressure curve; it cannot also be set")
    return override_pa


def _required_supply_pressure(drive, drag_torque, back_pressure, ceiling):
    """Lowest Supply Pressure whose turbine torque balances the drag; inf if none up to ceiling.

    A pressure whose Nozzle throat falls below the search range (a liquid-like supply) is
    counted as not balancing the drag, so it can only raise the answer, never invent one.
    """
    def balances(supply_p):
        try:
            return drive(supply_p)[1] >= drag_torque
        except ThroatSearchRangeError:
            return False

    if not balances(ceiling):
        return math.inf
    lo, hi = back_pressure, ceiling
    while hi / lo > 1.0 + _PRESSURE_REL_TOLERANCE:
        mid = math.sqrt(lo * hi)  # geometric: the answer can lie decades above back pressure
        if not balances(mid):
            lo = mid
        else:
            hi = mid
    return hi
