"""Helium properties: real-gas Drive Gas (CoolProp) and dilute vapour. SI units."""

import math

import numpy as np
from CoolProp.CoolProp import PropsSI

FLUID = "Helium"
R_HELIUM = 8.314462618 / 0.004002602  # J/(kg K)

# Dilute-gas viscosity and thermal conductivity of helium-4 from ab initio calculation.
# Hurly & Moldover (2000), J. Res. NIST 105, 667, Table A1; viscosity u_r about 0.4 % at 1 K.
# CoolProp's helium model stops at 2.1768 K, above the 1.2 K vapour, so the table is used to 10 K.
_TABLE_T_K = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0])
_TABLE_MU_UPA_S = np.array([0.3279, 0.3395, 0.3573, 0.3834, 0.4171, 0.4573, 0.5146, 0.5775,
                            0.6440, 0.7123, 0.8492, 0.9815, 1.106, 1.222, 1.433, 1.620, 1.791,
                            1.951, 2.102])
_TABLE_K_MW_M_K = np.array([2.624, 2.713, 2.839, 3.026, 3.276, 3.581, 4.022, 4.510, 5.028, 5.560,
                            6.628, 7.658, 8.630, 9.537, 11.180, 12.650, 14.000, 15.250, 16.440])
GAMMA = 5.0 / 3.0
CP_DILUTE = 2.5 * R_HELIUM
_COOLPROP_MIN_T_K = 2.2  # just above CoolProp's helium limit of 2.1768 K
_DILUTE_PRESSURE_PA = 100.0  # for CoolProp above the table


def vapour_viscosity(temperature_k):
    """Dilute helium viscosity (Pa s); valid from 1 K up."""
    if temperature_k < _TABLE_T_K[0]:
        raise ValueError(f"No helium viscosity data below {_TABLE_T_K[0]} K")
    if temperature_k <= _TABLE_T_K[-1]:
        return float(np.interp(temperature_k, _TABLE_T_K, _TABLE_MU_UPA_S)) * 1e-6
    return PropsSI("V", "T", temperature_k, "P", _DILUTE_PRESSURE_PA, FLUID)


def vapour_thermal_conductivity(temperature_k):
    """Dilute helium thermal conductivity (W/(m K)); valid from 1 K up."""
    if temperature_k < _TABLE_T_K[0]:
        raise ValueError(f"No helium conductivity data below {_TABLE_T_K[0]} K")
    if temperature_k <= _TABLE_T_K[-1]:
        return float(np.interp(temperature_k, _TABLE_T_K, _TABLE_K_MW_M_K)) * 1e-3
    return PropsSI("L", "T", temperature_k, "P", _DILUTE_PRESSURE_PA, FLUID)


def gap_heat_transfer_coefficient(temperature_k, pressure_pa, gap_m, accommodation=1.0):
    """Gas conduction across a narrow gap between parallel walls (W/(m^2 K)).

    Continuum k/gap and the free-molecular kinetic-theory limit, joined
    harmonically (the temperature-jump result in between).
    """
    h_continuum = vapour_thermal_conductivity(temperature_k) / gap_m
    h_free_molecular = (accommodation / (2.0 - accommodation) * (GAMMA + 1.0) / (GAMMA - 1.0)
                        * pressure_pa * math.sqrt(R_HELIUM / (8.0 * math.pi * temperature_k)))
    return 1.0 / (1.0 / h_continuum + 1.0 / h_free_molecular)


def mean_free_path(temperature_k, pressure_pa):
    """Hard-sphere mean free path from viscosity (m)."""
    mu = vapour_viscosity(temperature_k)
    return mu / pressure_pa * math.sqrt(math.pi * R_HELIUM * temperature_k / 2.0)


def knudsen_number(temperature_k, pressure_pa, length_m):
    return mean_free_path(temperature_k, pressure_pa) / length_m


def vapour_density(temperature_k, pressure_pa):
    """Ideal-gas density of the dilute vapour (kg/m^3)."""
    return pressure_pa / (R_HELIUM * temperature_k)


def vapour_enthalpy(temperature_k, pressure_pa):
    """Specific enthalpy of helium gas (J/kg) on the same datum as stagnation_state.

    Real gas (CoolProp) down to its lower limit; below it, as for the 1.2 K vapour,
    extended with the dilute monatomic cp = 5/2 R at the same pressure.
    """
    t_ref = max(temperature_k, _COOLPROP_MIN_T_K)
    return PropsSI("H", "T", t_ref, "P", pressure_pa, FLUID) - CP_DILUTE * (t_ref - temperature_k)


def stagnation_state(temperature_k, pressure_pa):
    """Real-gas specific enthalpy (J/kg) and entropy (J/(kg K)) of the Drive Gas."""
    h = PropsSI("H", "T", temperature_k, "P", pressure_pa, FLUID)
    s = PropsSI("S", "T", temperature_k, "P", pressure_pa, FLUID)
    return h, s


def isentropic_state(pressure_pa, entropy):
    """Real-gas enthalpy (J/kg) and density (kg/m^3) at a pressure on an isentrope."""
    h = PropsSI("H", "P", pressure_pa, "S", entropy, FLUID)
    rho = PropsSI("D", "P", pressure_pa, "S", entropy, FLUID)
    return h, rho
