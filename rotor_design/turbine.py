"""Turbine torque from the Euler turbine equation. SI units; Spin Rate in Hz."""

import math


def fluted_cap_torque(mass_flow_kg_s, jet_velocity_m_s, nozzle_angle_deg, rim_radius_m, spin_rate_hz,
                      efficiency=1.0):
    """Torque on a Fluted Cap Turbine.

    Jets strike flutes at the cap rim; nozzle_angle_deg is measured from the
    rim tangent. Straight flutes discharge the gas with no swirl relative to the
    rim, so it leaves with the rim speed U. efficiency scales the ideal Euler torque.
    """
    c_theta_in = jet_velocity_m_s * math.cos(math.radians(nozzle_angle_deg))
    rim_speed = 2.0 * math.pi * rim_radius_m * spin_rate_hz
    return efficiency * mass_flow_kg_s * rim_radius_m * (c_theta_in - rim_speed)
