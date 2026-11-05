"""Load the assumptions file into one immutable object."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from rotor_design.geometry import StatorAllowance


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
    environment_temperature_k: float
    environment_pressure_torr: float
    drive_gas_temperature_k: float
    max_supply_pressure_bar: float
    target_spin_rate_hz: float
    discharge_coefficient: float
    turbine_efficiency: float
    rotor_stator_gap_mm: float
    accommodation_coefficient: float
    baseline_nozzle_count: int
    baseline_nozzle_throat_mm: float
    baseline_nozzle_angle_deg: float
    baseline_wall_thickness_mm: float
    baseline_sample_length_mm: float
    sample_zone_delta_t_k: float
    min_sample_zone_mm: float
    bath_cooling_power_w: float
    sample_internal_heat_load_w: float
    sample_conductivity_w_m_k: float
    stator_temperature_k: float
    rotor_emissivity: float
    windage_heat_to_rotor_fraction: float
    zirconia_conductivity_multiplier: float


def load_assumptions(path):
    with open(Path(path), "rb") as f:
        raw = tomllib.load(f)
    geometry, sweep, allowance = raw["geometry"], raw["sweep"], raw["stator_allowance"]
    environment, drive, thermal = raw["environment"], raw["drive"], raw["thermal"]
    drag, baseline = raw["drag"], raw["design_baseline"]
    rotor, materials = raw["rotor"], raw["materials"]
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
        environment_temperature_k=environment["temperature_k"],
        environment_pressure_torr=environment["pressure_torr"],
        drive_gas_temperature_k=drive["drive_gas_temperature_k"],
        max_supply_pressure_bar=drive["max_supply_pressure_bar"],
        target_spin_rate_hz=drive["target_spin_rate_hz"],
        discharge_coefficient=drive["discharge_coefficient"],
        turbine_efficiency=drive["turbine_efficiency"],
        rotor_stator_gap_mm=drag["rotor_stator_gap_mm"],
        accommodation_coefficient=drag["accommodation_coefficient"],
        baseline_nozzle_count=baseline["nozzle_count"],
        baseline_nozzle_throat_mm=baseline["nozzle_throat_mm"],
        baseline_nozzle_angle_deg=baseline["nozzle_angle_deg"],
        baseline_wall_thickness_mm=baseline["wall_thickness_mm"],
        baseline_sample_length_mm=baseline["sample_length_mm"],
        sample_zone_delta_t_k=thermal["sample_zone_delta_t_k"],
        min_sample_zone_mm=thermal["min_sample_zone_mm"],
        bath_cooling_power_w=thermal["bath_cooling_power_w"],
        sample_internal_heat_load_w=thermal["sample_internal_heat_load_w"],
        sample_conductivity_w_m_k=thermal["sample_conductivity_w_m_k"],
        stator_temperature_k=thermal["stator_temperature_k"],
        rotor_emissivity=thermal["rotor_emissivity"],
        windage_heat_to_rotor_fraction=thermal["windage_heat_to_rotor_fraction"],
        zirconia_conductivity_multiplier=materials["zirconia_conductivity_multiplier"],
    )
