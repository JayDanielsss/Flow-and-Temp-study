"""Gas properties: real-gas Drive Gas (CoolProp) and dilute vapour. SI units.

Each Gas carries its molar mass, dilute ratio of specific heats and the lowest
temperature it has property data for; calls below that raise instead of extrapolating.
Every function takes the gas as a keyword, helium by default.
"""

import math
from dataclasses import dataclass

import numpy as np
from CoolProp.CoolProp import PropsSI

R_UNIVERSAL = 8.314462618  # J/(mol K)
_DILUTE_PRESSURE_PA = 100.0  # for CoolProp transport properties in the dilute limit


@dataclass(frozen=True)
class Gas:
    name: str
    coolprop_name: str
    molar_mass_kg_mol: float
    gamma: float                  # dilute ideal gas
    coolprop_min_temperature_k: float
    # Dilute transport table (T K, viscosity uPa s, conductivity mW/(m K)) used up to its last
    # temperature, below CoolProp's range; None to use CoolProp everywhere.
    transport_table: tuple | None = None

    @property
    def specific_gas_constant(self):
        """J/(kg K)."""
        return R_UNIVERSAL / self.molar_mass_kg_mol

    @property
    def cp_dilute(self):
        """Ideal-gas specific heat at constant pressure, J/(kg K)."""
        return self.gamma / (self.gamma - 1.0) * self.specific_gas_constant

    @property
    def min_temperature_k(self):
        """Lowest temperature with property data."""
        return self.transport_table[0][0] if self.transport_table else self.coolprop_min_temperature_k

    def check_temperature(self, temperature_k):
        if temperature_k < self.min_temperature_k:
            raise ValueError(f"No {self.name} property data below {self.min_temperature_k} K "
                             f"(asked for {temperature_k} K)")


# Dilute-gas viscosity and thermal conductivity of helium-4 from ab initio calculation.
# Hurly & Moldover (2000), J. Res. NIST 105, 667, Table A1; viscosity u_r about 0.4 % at 1 K.
# CoolProp's helium model stops at 2.1768 K, above the 1.2 K vapour, so the table is used to 10 K.
_HELIUM_TABLE = (
    (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    (0.3279, 0.3395, 0.3573, 0.3834, 0.4171, 0.4573, 0.5146, 0.5775, 0.6440, 0.7123, 0.8492, 0.9815,
     1.106, 1.222, 1.433, 1.620, 1.791, 1.951, 2.102),
    (2.624, 2.713, 2.839, 3.026, 3.276, 3.581, 4.022, 4.510, 5.028, 5.560, 6.628, 7.658, 8.630, 9.537,
     11.180, 12.650, 14.000, 15.250, 16.440),
)
# Lower limits are CoolProp's: just above the helium lambda line (2.1768 K), and the
# nitrogen (63.151 K) and air (59.75 K) triple points.
HELIUM = Gas("helium", "Helium", 0.004002602, 5.0 / 3.0, 2.2, _HELIUM_TABLE)
NITROGEN = Gas("nitrogen", "Nitrogen", 0.02801348, 1.4, 63.151)
AIR = Gas("air", "Air", 0.02896546, 1.4, 59.75)
GASES = {gas.name: gas for gas in (HELIUM, NITROGEN, AIR)}

# Helium constants, kept for the formulas and tests written in them.
R_HELIUM = HELIUM.specific_gas_constant
GAMMA = HELIUM.gamma
CP_DILUTE = HELIUM.cp_dilute


def _transport(temperature_k, gas, column, coolprop_key, scale):
    gas.check_temperature(temperature_k)
    table = gas.transport_table
    if table and temperature_k <= table[0][-1]:
        return float(np.interp(temperature_k, table[0], table[column])) * scale
    return PropsSI(coolprop_key, "T", temperature_k, "P", _DILUTE_PRESSURE_PA, gas.coolprop_name)


def vapour_viscosity(temperature_k, gas=HELIUM):
    """Dilute viscosity (Pa s)."""
    return _transport(temperature_k, gas, 1, "V", 1e-6)


def vapour_thermal_conductivity(temperature_k, gas=HELIUM):
    """Dilute thermal conductivity (W/(m K))."""
    return _transport(temperature_k, gas, 2, "L", 1e-3)


def gap_heat_transfer_coefficient(temperature_k, pressure_pa, gap_m, accommodation=1.0, gas=HELIUM):
    """Gas conduction across a narrow gap between parallel walls (W/(m^2 K)).

    Continuum k/gap and the free-molecular kinetic-theory limit, joined
    harmonically (the temperature-jump result in between).
    """
    h_continuum = vapour_thermal_conductivity(temperature_k, gas) / gap_m
    h_free_molecular = (accommodation / (2.0 - accommodation) * (gas.gamma + 1.0) / (gas.gamma - 1.0)
                        * pressure_pa * math.sqrt(gas.specific_gas_constant / (8.0 * math.pi * temperature_k)))
    return 1.0 / (1.0 / h_continuum + 1.0 / h_free_molecular)


def mean_free_path(temperature_k, pressure_pa, gas=HELIUM):
    """Hard-sphere mean free path from viscosity (m)."""
    mu = vapour_viscosity(temperature_k, gas)
    return mu / pressure_pa * math.sqrt(math.pi * gas.specific_gas_constant * temperature_k / 2.0)


def knudsen_number(temperature_k, pressure_pa, length_m, gas=HELIUM):
    return mean_free_path(temperature_k, pressure_pa, gas) / length_m


def vapour_density(temperature_k, pressure_pa, gas=HELIUM):
    """Ideal-gas density of the dilute vapour (kg/m^3)."""
    return pressure_pa / (gas.specific_gas_constant * temperature_k)


def vapour_enthalpy(temperature_k, pressure_pa, gas=HELIUM):
    """Specific enthalpy (J/kg) on the same datum as stagnation_state.

    Real gas (CoolProp) down to its lower limit; below it, as for the 1.2 K helium vapour,
    extended with the dilute cp at the same pressure, down to the gas's data limit.
    """
    gas.check_temperature(temperature_k)
    t_ref = max(temperature_k, gas.coolprop_min_temperature_k)
    return (PropsSI("H", "T", t_ref, "P", pressure_pa, gas.coolprop_name)
            - gas.cp_dilute * (t_ref - temperature_k))


def stagnation_state(temperature_k, pressure_pa, gas=HELIUM):
    """Real-gas specific enthalpy (J/kg) and entropy (J/(kg K)) of the Drive Gas."""
    gas.check_temperature(temperature_k)
    h = PropsSI("H", "T", temperature_k, "P", pressure_pa, gas.coolprop_name)
    s = PropsSI("S", "T", temperature_k, "P", pressure_pa, gas.coolprop_name)
    return h, s


def isentropic_state(pressure_pa, entropy, gas=HELIUM):
    """Real-gas enthalpy (J/kg) and density (kg/m^3) at a pressure on an isentrope."""
    h = PropsSI("H", "P", pressure_pa, "S", entropy, gas.coolprop_name)
    rho = PropsSI("D", "P", pressure_pa, "S", entropy, gas.coolprop_name)
    return h, rho
