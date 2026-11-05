"""Helium properties: real-gas Drive Gas (CoolProp) and dilute vapour. SI units."""

import math

import numpy as np
from CoolProp.CoolProp import PropsSI

FLUID = "Helium"
R_HELIUM = 8.314462618 / 0.004002602  # J/(kg K)

# Dilute-gas viscosity of helium-4 from ab initio calculation.
# Hurly & Moldover (2000), J. Res. NIST 105, 667, Table A1; u_r about 0.4 % at 1 K.
# CoolProp's helium model stops at 2.1768 K, above the 1.2 K vapour, so the table is used to 10 K.
_TABLE_T_K = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0])
_TABLE_MU_UPA_S = np.array([0.3279, 0.3395, 0.3573, 0.3834, 0.4171, 0.4573, 0.5146, 0.5775,
                            0.6440, 0.7123, 0.8492, 0.9815, 1.106, 1.222, 1.433, 1.620, 1.791,
                            1.951, 2.102])
_DILUTE_PRESSURE_PA = 100.0  # for CoolProp above the table


def vapour_viscosity(temperature_k):
    """Dilute helium viscosity (Pa s); valid from 1 K up."""
    if temperature_k < _TABLE_T_K[0]:
        raise ValueError(f"No helium viscosity data below {_TABLE_T_K[0]} K")
    if temperature_k <= _TABLE_T_K[-1]:
        return float(np.interp(temperature_k, _TABLE_T_K, _TABLE_MU_UPA_S)) * 1e-6
    return PropsSI("V", "T", temperature_k, "P", _DILUTE_PRESSURE_PA, FLUID)


def mean_free_path(temperature_k, pressure_pa):
    """Hard-sphere mean free path from viscosity (m)."""
    mu = vapour_viscosity(temperature_k)
    return mu / pressure_pa * math.sqrt(math.pi * R_HELIUM * temperature_k / 2.0)


def knudsen_number(temperature_k, pressure_pa, length_m):
    return mean_free_path(temperature_k, pressure_pa) / length_m


def vapour_density(temperature_k, pressure_pa):
    """Ideal-gas density of the dilute vapour (kg/m^3)."""
    return pressure_pa / (R_HELIUM * temperature_k)


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
