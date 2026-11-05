"""Probe Bore fit at the Tilt Angle. All lengths in mm, angles in degrees."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StatorAllowance:
    """Thicknesses that turn a Rotor size into a Stator size.

    Radial (each side): wall, nozzle_plenum, and the Bearing's radial space (set by Bearing type).
    Axial (one end each): housing_above_turbine_cap, cold_support_plate.
    Probe-wall clearance is kept free on each side of the Probe Bore.
    """

    wall: float
    nozzle_plenum: float
    housing_above_turbine_cap: float
    cold_support_plate: float
    probe_wall_clearance: float
    bearing_radial_space: float = 0.0

    def radial_each_side(self):
        return self.wall + self.nozzle_plenum + self.bearing_radial_space

    def stator_length(self, rotor_length):
        return rotor_length + self.housing_above_turbine_cap + self.cold_support_plate

    def stator_diameter(self, rotor_diameter):
        return rotor_diameter + 2.0 * self.radial_each_side()

    def usable_bore(self, bore_diameter):
        return bore_diameter - 2.0 * self.probe_wall_clearance


def projected_width(stator_length, stator_diameter, tilt_angle_deg):
    """Width the tilted Stator occupies across the probe axis."""
    theta = math.radians(tilt_angle_deg)
    return stator_length * math.sin(theta) + stator_diameter * math.cos(theta)


def max_rotor_diameter(rotor_length, allowance, bore_diameter, tilt_angle_deg):
    """Largest Rotor diameter whose Stator still fits the Probe Bore.

    Negative when the Stator length alone overfills the bore.
    """
    theta = math.radians(tilt_angle_deg)
    room = allowance.usable_bore(bore_diameter) - allowance.stator_length(rotor_length) * math.sin(theta)
    max_stator_diameter = room / math.cos(theta)
    return max_stator_diameter - 2.0 * allowance.radial_each_side()
