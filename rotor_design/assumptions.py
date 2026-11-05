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
    stator_allowance: StatorAllowance
    environment_temperature_k: float
    environment_pressure_torr: float
    drive_gas_temperature_k: float
    max_supply_pressure_bar: float
    target_spin_rate_hz: float
    sample_zone_delta_t_k: float
    min_sample_zone_mm: float
    max_sample_temperature_k: float
    bath_cooling_power_w: float
    sample_internal_heat_load_w: float


def load_assumptions(path):
    with open(Path(path), "rb") as f:
        raw = tomllib.load(f)
    geometry, sweep, allowance = raw["geometry"], raw["sweep"], raw["stator_allowance"]
    environment, drive, thermal = raw["environment"], raw["drive"], raw["thermal"]
    return Assumptions(
        bore_diameter_mm=geometry["bore_diameter_mm"],
        tilt_angle_deg=geometry["tilt_angle_deg"],
        max_rotor_length_mm=geometry["max_rotor_length_mm"],
        rotor_length_sweep_min_mm=sweep["rotor_length_min_mm"],
        rotor_length_sweep_step_mm=sweep["rotor_length_step_mm"],
        stator_allowance=StatorAllowance(
            wall=allowance["wall_mm"],
            nozzle_plenum=allowance["nozzle_plenum_mm"],
            housing_above_turbine_cap=allowance["housing_above_turbine_cap_mm"],
            cold_support_plate=allowance["cold_support_plate_mm"],
            probe_wall_clearance=allowance["probe_wall_clearance_mm"],
        ),
        environment_temperature_k=environment["temperature_k"],
        environment_pressure_torr=environment["pressure_torr"],
        drive_gas_temperature_k=drive["drive_gas_temperature_k"],
        max_supply_pressure_bar=drive["max_supply_pressure_bar"],
        target_spin_rate_hz=drive["target_spin_rate_hz"],
        sample_zone_delta_t_k=thermal["sample_zone_delta_t_k"],
        min_sample_zone_mm=thermal["min_sample_zone_mm"],
        max_sample_temperature_k=thermal["max_sample_temperature_k"],
        bath_cooling_power_w=thermal["bath_cooling_power_w"],
        sample_internal_heat_load_w=thermal["sample_internal_heat_load_w"],
    )
