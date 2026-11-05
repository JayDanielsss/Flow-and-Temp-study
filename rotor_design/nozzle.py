"""Real-gas isentropic Nozzle flow, choked or unchoked, with a discharge coefficient. SI units."""

import math
from dataclasses import dataclass

from rotor_design.gas import HELIUM, isentropic_state, stagnation_state

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_SEARCH_ITERATIONS = 40  # brackets the peak to 1e-8 of p0
_MIN_THROAT_RATIO = 0.2


@dataclass(frozen=True)
class NozzleFlow:
    mass_flow_kg_s: float
    jet_velocity_m_s: float
    throat_pressure_pa: float
    choked: bool


def nozzle_flow(supply_pressure_pa, supply_temperature_k, back_pressure_pa, throat_diameter_m,
                discharge_coefficient, gas=HELIUM):
    """Flow through one converging Nozzle from stagnation supply to back pressure.

    The throat sits at whichever pressure between back and supply maximises the
    isentropic mass flux; it is choked when that maximum lies above the back pressure.
    """
    if supply_pressure_pa <= back_pressure_pa:
        return NozzleFlow(0.0, 0.0, back_pressure_pa, False)
    h0, s0 = stagnation_state(supply_temperature_k, supply_pressure_pa, gas)

    def flux_and_velocity(p):
        h, rho = isentropic_state(p, s0, gas)
        v = math.sqrt(max(2.0 * (h0 - h), 0.0))
        return rho * v, v

    # The mass-flux peak lies near 0.49 p0 for helium and 0.53 p0 for nitrogen and air; searching no lower than
    # _MIN_THROAT_RATIO keeps a choked expansion clear of the property range limit.
    lo, hi = max(back_pressure_pa, _MIN_THROAT_RATIO * supply_pressure_pa), supply_pressure_pa
    throat_floor = lo
    a, b = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    fa, fb = flux_and_velocity(a)[0], flux_and_velocity(b)[0]
    for _ in range(_SEARCH_ITERATIONS):
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + _GOLDEN * (hi - lo)
            fb = flux_and_velocity(b)[0]
        else:
            hi, b, fb = b, a, fa
            a = hi - _GOLDEN * (hi - lo)
            fa = flux_and_velocity(a)[0]
    p_peak = (lo + hi) / 2.0
    if back_pressure_pa < throat_floor and p_peak - throat_floor < 1e-6 * supply_pressure_pa:
        raise ValueError("Mass-flux peak below the throat search range; widen _MIN_THROAT_RATIO")

    choked = p_peak - back_pressure_pa > 1e-6 * (supply_pressure_pa - back_pressure_pa)
    throat_p = p_peak if choked else back_pressure_pa
    flux, v = flux_and_velocity(throat_p)

    area = math.pi * throat_diameter_m**2 / 4.0
    return NozzleFlow(discharge_coefficient * area * flux, v, throat_p, choked)
