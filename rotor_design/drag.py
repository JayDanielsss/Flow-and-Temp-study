"""Gas drag on the Rotor: windage in the gap to the Stator and shear in the Cold Support gap.

SI units; Spin Rate in Hz.
"""

import math
from dataclasses import dataclass

from rotor_design.gas import HELIUM, knudsen_number, vapour_viscosity

CONTINUUM_MAX_KN = 0.01
FREE_MOLECULAR_MIN_KN = 10.0


@dataclass(frozen=True)
class Windage:
    torque_n_m: float
    knudsen_number: float
    regime: str


def windage_torque(radius_m, length_m, gap_m, spin_rate_hz, temperature_k, pressure_pa,
                   accommodation=1.0, gas=HELIUM):
    """Drag torque on the cylindrical surface of the Rotor turning inside the Stator.

    Laminar Taylor-Couette shear in the continuum, kinetic-theory shear in the
    free-molecular limit, joined harmonically in between (for flat plates this is
    exactly the slip-flow result mu U / (gap + 2 (2 - sigma) / sigma * lambda)).
    End faces are not included.
    """
    omega = 2.0 * math.pi * spin_rate_hz
    r1 = radius_m
    area = 2.0 * math.pi * r1 * length_m

    torque_continuum = couette_torque(r1, r1 + gap_m, length_m, spin_rate_hz,
                                      vapour_viscosity(temperature_k, gas))
    tau_free_molecular = free_molecular_shear(omega * r1, temperature_k, pressure_pa, accommodation, gas)
    torque_free_molecular = tau_free_molecular * area * r1
    torque = 1.0 / (1.0 / torque_continuum + 1.0 / torque_free_molecular)

    return _in_regime(torque, temperature_k, pressure_pa, gap_m, gas)


def cold_support_shear_torque(radius_m, gap_m, spin_rate_hz, temperature_k, pressure_pa, accommodation=1.0,
                              gas=HELIUM):
    """Drag torque on the Bottom Cap face turning a gap from the still Cold Support plate.

    Narrow-gap Couette shear tau = mu omega r / gap in the continuum, kinetic-theory
    shear in the free-molecular limit, joined harmonically as for windage_torque.
    The same torque times omega is the heat generated in the gap.
    """
    omega = 2.0 * math.pi * spin_rate_hz
    # Both limits are tau = c r; integrating c r * r * 2 pi r dr over the disc gives pi c R^4 / 2.
    disc = math.pi * omega * radius_m**4 / 2.0
    torque_continuum = vapour_viscosity(temperature_k, gas) / gap_m * disc
    torque_free_molecular = free_molecular_shear(1.0, temperature_k, pressure_pa, accommodation, gas) * disc
    torque = 1.0 / (1.0 / torque_continuum + 1.0 / torque_free_molecular)
    return _in_regime(torque, temperature_k, pressure_pa, gap_m, gas)


def couette_torque(inner_radius_m, outer_radius_m, length_m, spin_rate_hz, viscosity_pa_s):
    """Laminar Taylor-Couette torque between concentric cylinders, one turning at spin_rate_hz relative to the other."""
    omega = 2.0 * math.pi * spin_rate_hz
    r1, r2 = inner_radius_m, outer_radius_m
    return 4.0 * math.pi * viscosity_pa_s * length_m * omega * r1**2 * r2**2 / (r2**2 - r1**2)


def free_molecular_shear(speed_m_s, temperature_k, pressure_pa, accommodation=1.0, gas=HELIUM):
    """Kinetic-theory shear stress on a wall sliding at speed_m_s through still gas in the free-molecular limit."""
    return (accommodation / (2.0 - accommodation) * pressure_pa * speed_m_s
            / math.sqrt(2.0 * math.pi * gas.specific_gas_constant * temperature_k))


def _in_regime(torque, temperature_k, pressure_pa, gap_m, gas):
    kn = knudsen_number(temperature_k, pressure_pa, gap_m, gas)
    if kn < CONTINUUM_MAX_KN:
        regime = "continuum"
    elif kn > FREE_MOLECULAR_MIN_KN:
        regime = "free-molecular"
    else:
        regime = "transition"
    return Windage(torque, kn, regime)
