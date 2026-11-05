"""Windage drag on the Rotor in the gas gap to the Stator. SI units; Spin Rate in Hz."""

import math
from dataclasses import dataclass

from rotor_design.gas import R_HELIUM, knudsen_number, vapour_viscosity

CONTINUUM_MAX_KN = 0.01
FREE_MOLECULAR_MIN_KN = 10.0


@dataclass(frozen=True)
class Windage:
    torque_n_m: float
    knudsen_number: float
    regime: str


def windage_torque(radius_m, length_m, gap_m, spin_rate_hz, temperature_k, pressure_pa,
                   accommodation=1.0):
    """Drag torque on the cylindrical surface of the Rotor turning inside the Stator.

    Laminar Taylor-Couette shear in the continuum, kinetic-theory shear in the
    free-molecular limit, joined harmonically in between (for flat plates this is
    exactly the slip-flow result mu U / (gap + 2 (2 - sigma) / sigma * lambda)).
    End faces are not included.
    """
    omega = 2.0 * math.pi * spin_rate_hz
    r1, r2 = radius_m, radius_m + gap_m
    area = 2.0 * math.pi * r1 * length_m

    mu = vapour_viscosity(temperature_k)
    torque_continuum = 4.0 * math.pi * mu * length_m * omega * r1**2 * r2**2 / (r2**2 - r1**2)
    tau_free_molecular = (accommodation / (2.0 - accommodation) * pressure_pa * omega * r1
                          / math.sqrt(2.0 * math.pi * R_HELIUM * temperature_k))
    torque_free_molecular = tau_free_molecular * area * r1
    torque = 1.0 / (1.0 / torque_continuum + 1.0 / torque_free_molecular)

    kn = knudsen_number(temperature_k, pressure_pa, gap_m)
    if kn < CONTINUUM_MAX_KN:
        regime = "continuum"
    elif kn > FREE_MOLECULAR_MIN_KN:
        regime = "free-molecular"
    else:
        regime = "transition"
    return Windage(torque, kn, regime)
