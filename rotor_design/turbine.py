"""Turbine types behind one interface: torque from the Euler turbine equation. SI units; Spin Rate in Hz.

Both types take Nozzle jets at the Turbine Cap rim; nozzle_angle_deg is measured
from the rim tangent, so the inlet swirl is c_theta1 = V cos(angle). They differ in
where the gas leaves, and so in the angular momentum it carries away.
"""

import math
from dataclasses import dataclass

FLUTED_CAP = "fluted_cap"
RADIAL_INFLOW = "radial_inflow"
TURBINE_TYPES = (FLUTED_CAP, RADIAL_INFLOW)


@dataclass(frozen=True)
class FlutedCap:
    """Flutes at the cap rim; exhaust leaves around the rim.

    Straight flutes discharge the gas with no swirl relative to the rim, so it
    leaves with the rim speed U: T = m r1 (c_theta1 - U).
    """

    hub_exit_diameter_mm = math.nan   # no hub exit

    def torque(self, mass_flow_kg_s, jet_velocity_m_s, nozzle_angle_deg, rim_radius_m, spin_rate_hz,
               efficiency=1.0):
        """Torque for the given flow; efficiency scales the ideal Euler torque."""
        c_theta_in = jet_velocity_m_s * math.cos(math.radians(nozzle_angle_deg))
        rim_speed = 2.0 * math.pi * rim_radius_m * spin_rate_hz
        return efficiency * mass_flow_kg_s * rim_radius_m * (c_theta_in - rim_speed)

    def blade_area_m2(self, rim_radius_m):
        """Turbine Cap face area available to the flutes: the whole cap."""
        return math.pi * rim_radius_m**2


@dataclass(frozen=True)
class RadialInflow:
    """Gas flows inward through blade passages and exits axially through a central hub exit.

    The gas leaves the hub exit with no swirl relative to the Rotor; for uniform
    axial flow over the exit disc of radius r_e its mean angular momentum is
    omega r_e^2 / 2, so T = m (r1 c_theta1 - omega r_e^2 / 2). The blade passages
    fill the ring between the hub exit and the rim.
    """

    hub_exit_diameter_mm: float

    def torque(self, mass_flow_kg_s, jet_velocity_m_s, nozzle_angle_deg, rim_radius_m, spin_rate_hz,
               efficiency=1.0):
        """Torque for the given flow; efficiency scales the ideal Euler torque. No blades, no torque."""
        exit_radius = self.hub_exit_diameter_mm / 2e3
        if exit_radius >= rim_radius_m:
            return 0.0
        c_theta_in = jet_velocity_m_s * math.cos(math.radians(nozzle_angle_deg))
        exit_momentum = 2.0 * math.pi * spin_rate_hz * exit_radius**2 / 2.0
        return efficiency * mass_flow_kg_s * (rim_radius_m * c_theta_in - exit_momentum)

    def blade_area_m2(self, rim_radius_m):
        """Turbine Cap face area left to the blade passages once the hub exit is cut out."""
        exit_radius = self.hub_exit_diameter_mm / 2e3
        return math.pi * max(rim_radius_m**2 - exit_radius**2, 0.0)
