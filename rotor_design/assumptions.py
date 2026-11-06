"""Load the assumptions file into one immutable object."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from rotor_design.bearing import GAS, MECHANICAL, TWO_CENTER_PIVOT, GasBearing, MechanicalBearing, TwoCenterPivots
from rotor_design.geometry import StatorAllowance
from rotor_design.turbine import FLUTED_CAP, RADIAL_INFLOW, FlutedCap, RadialInflow


@dataclass(frozen=True)
class Assumptions:
    bore_diameter_mm: float
    tilt_angle_deg: float
    max_rotor_length_mm: float
    rotor_length_sweep_min_mm: float
    rotor_length_sweep_step_mm: float
    nozzle_count_sweep: tuple
    nozzle_throat_sweep_mm: tuple
    nozzle_angle_sweep_deg: tuple
    stator_allowance: StatorAllowance
    turbine_cap_height_mm: float
    bottom_cap_thickness_mm: float
    bath_temperature_k: float       # every sink: Bath, Cold Support, Vapour Cooling
    sample_limit_temperature_k: float  # the hottest the Sample may reach
    environment_gas: str
    drive_gas: str
    drive_gas_temperature_k: float
    max_supply_pressure_bar: float
    target_spin_rate_hz: float
    discharge_coefficient: float
    turbine_efficiency: float
    turbine_efficiency_correction: dict  # Turbine type -> factor on turbine_efficiency
    exhaust_path: str
    ducted_back_pressure_bar: float
    turbine: str
    turbines: dict                  # Turbine type -> its model
    rotor_stator_gap_mm: float
    accommodation_coefficient: float
    windage_torque_multiplier: float
    cooling_path: str
    cold_support_gap_mm: float
    cold_support_gap_sweep_mm: tuple
    bearing: str
    pivot_to_bath: bool
    bearings: dict                  # Bearing type -> its model
    baseline_nozzle_count: int
    baseline_nozzle_throat_mm: float
    baseline_nozzle_angle_deg: float
    baseline_wall_thickness_mm: float
    baseline_sample_length_mm: float
    min_sample_zone_mm: float
    bath_cooling_power_w: float
    sample_internal_heat_load_w: float
    sample_conductivity_w_m_k: float
    rotor_emissivity: float
    windage_heat_to_rotor_fraction: float
    required_safety_factor: float
    critical_speed_separation: float
    whirl_onset_ratio: float
    zirconia_conductivity_multiplier: float
    study_grid: dict                # Design field -> values tried in the full study
    study_top_designs: int
    constraint_map_axes: tuple      # (x, y) Design fields
    benchmark_tolerance: dict       # measured quantity -> allowed |Model-Form Error|
    uncertainty_samples: int
    uncertainty_seed: int
    uncertainty_designs: int        # top-ranked Designs given Monte Carlo bands
    uncertainty_percentiles: tuple  # (lower, upper) band edges, in percent
    uncertainty_inputs: dict        # uncertain input -> (distribution, low, high)
    model_form_knobs: dict          # Reference Case mechanism -> the input its Model-Form Error spreads
    uncertainty_physical_range: dict  # uncertain input or knob -> (low, high], its physical range
    model_form_scale: dict          # Reference Case mechanism -> "log" or "logit"; absent is "log"
    turbine_bias_correction: dict   # Turbine type -> "reference_cases" or a factor, in the Monte Carlo
    reference_cases_dir: Path
    # Warm cases only (a Benchmark Case or toy model not above a helium-4 Bath). None, as in the
    # assumptions file, takes the Cold Space pressure from the Bath Temperature on the helium-4
    # vapour-pressure curve.
    cold_space_pressure_pa: float | None = None
    # None holds the Stator at the Bath Temperature.
    stator_temperature_k: float | None = None


def load_assumptions(path):
    with open(Path(path), "rb") as f:
        raw = tomllib.load(f)
    geometry, sweep, allowance = raw["geometry"], raw["sweep"], raw["stator_allowance"]
    environment, drive, thermal = raw["environment"], raw["drive"], raw["thermal"]
    drag, baseline = raw["drag"], raw["design_baseline"]
    rotor, materials, cold_support = raw["rotor"], raw["materials"], raw["cold_support"]
    bearing, turbine, exhaust = raw["bearing"], raw["turbine"], raw["exhaust"]
    stress, rotordynamics, study = raw["stress"], raw["rotordynamics"], raw["study"]
    benchmark, uncertainty = raw["benchmark"], raw["uncertainty"]
    return Assumptions(
        bore_diameter_mm=geometry["bore_diameter_mm"],
        tilt_angle_deg=geometry["tilt_angle_deg"],
        max_rotor_length_mm=geometry["max_rotor_length_mm"],
        rotor_length_sweep_min_mm=sweep["rotor_length_min_mm"],
        rotor_length_sweep_step_mm=sweep["rotor_length_step_mm"],
        nozzle_count_sweep=tuple(sweep["nozzle_counts"]),
        nozzle_throat_sweep_mm=tuple(sweep["nozzle_throats_mm"]),
        nozzle_angle_sweep_deg=tuple(sweep["nozzle_angles_deg"]),
        stator_allowance=StatorAllowance(
            wall=allowance["wall_mm"],
            nozzle_plenum=allowance["nozzle_plenum_mm"],
            housing_above_turbine_cap=allowance["housing_above_turbine_cap_mm"],
            cold_support_plate=allowance["cold_support_plate_mm"],
            probe_wall_clearance=allowance["probe_wall_clearance_mm"],
        ),
        turbine_cap_height_mm=rotor["turbine_cap_height_mm"],
        bottom_cap_thickness_mm=rotor["bottom_cap_thickness_mm"],
        bath_temperature_k=environment["bath_temperature_k"],
        sample_limit_temperature_k=environment["sample_limit_temperature_k"],
        environment_gas=environment["gas"],
        drive_gas=drive["gas"],
        drive_gas_temperature_k=drive["drive_gas_temperature_k"],
        max_supply_pressure_bar=drive["max_supply_pressure_bar"],
        target_spin_rate_hz=drive["target_spin_rate_hz"],
        discharge_coefficient=drive["discharge_coefficient"],
        turbine_efficiency=drive["turbine_efficiency"],
        turbine_efficiency_correction=dict(drive["turbine_efficiency_correction"]),
        exhaust_path=exhaust["path"],
        ducted_back_pressure_bar=exhaust["ducted_back_pressure_bar"],
        turbine=turbine["type"],
        turbines={
            FLUTED_CAP: FlutedCap(),
            RADIAL_INFLOW: RadialInflow(**turbine["radial_inflow"]),
        },
        rotor_stator_gap_mm=drag["rotor_stator_gap_mm"],
        accommodation_coefficient=drag["accommodation_coefficient"],
        windage_torque_multiplier=drag["windage_torque_multiplier"],
        cooling_path=cold_support["cooling_path"],
        cold_support_gap_mm=cold_support["gap_mm"],
        cold_support_gap_sweep_mm=tuple(cold_support["gap_sweep_mm"]),
        bearing=bearing["type"],
        pivot_to_bath=bearing["pivot_to_bath"],
        bearings={
            GAS: GasBearing(**bearing["gas"]),
            MECHANICAL: MechanicalBearing(**bearing["mechanical"]),
            TWO_CENTER_PIVOT: TwoCenterPivots(**bearing["pivot"]),
        },
        baseline_nozzle_count=baseline["nozzle_count"],
        baseline_nozzle_throat_mm=baseline["nozzle_throat_mm"],
        baseline_nozzle_angle_deg=baseline["nozzle_angle_deg"],
        baseline_wall_thickness_mm=baseline["wall_thickness_mm"],
        baseline_sample_length_mm=baseline["sample_length_mm"],
        min_sample_zone_mm=thermal["min_sample_zone_mm"],
        bath_cooling_power_w=thermal["bath_cooling_power_w"],
        sample_internal_heat_load_w=thermal["sample_internal_heat_load_w"],
        sample_conductivity_w_m_k=thermal["sample_conductivity_w_m_k"],
        rotor_emissivity=thermal["rotor_emissivity"],
        windage_heat_to_rotor_fraction=thermal["windage_heat_to_rotor_fraction"],
        required_safety_factor=stress["required_safety_factor"],
        critical_speed_separation=rotordynamics["critical_speed_separation"],
        whirl_onset_ratio=rotordynamics["whirl_onset_ratio"],
        zirconia_conductivity_multiplier=materials["zirconia_conductivity_multiplier"],
        study_grid={field: tuple(values) for field, values in study["grid"].items()},
        study_top_designs=study["top_designs"],
        constraint_map_axes=(study["constraint_map_x"], study["constraint_map_y"]),
        benchmark_tolerance=dict(benchmark["tolerance"]),
        uncertainty_samples=uncertainty["samples"],
        uncertainty_seed=uncertainty["seed"],
        uncertainty_designs=uncertainty["designs"],
        uncertainty_percentiles=tuple(uncertainty["percentiles"]),
        uncertainty_inputs={name: (d["distribution"], d["low"], d["high"])
                            for name, d in uncertainty["inputs"].items()},
        model_form_knobs=dict(uncertainty["model_form_knobs"]),
        uncertainty_physical_range={name: tuple(r) for name, r in uncertainty["physical_range"].items()},
        model_form_scale=dict(uncertainty.get("model_form_scale", {})),
        turbine_bias_correction=dict(uncertainty["turbine_bias_correction"]),
        reference_cases_dir=Path(path).resolve().parent / uncertainty["reference_cases"],
    )
