"""Segmented 1-D fin along the Rotor axis. SI units.

z = 0 is the Bottom Cap end (adiabatic unless a Cold Support cools it);
z = length is the fin base, held at a fixed temperature (the Turbine Cap side).
"""

from dataclasses import dataclass

import numpy as np

_DEFAULT_NODES = 401
_MAX_ITERATIONS = 200
_TOLERANCE_K = 1e-9


@dataclass(frozen=True)
class FinProfile:
    z_m: np.ndarray
    temperature_k: np.ndarray
    cell_length_m: np.ndarray   # axial length each node represents
    base_heat_w: float          # heat entering through the base at z = length
    tip_heat_out_w: float       # heat leaving through the tip conductance at z = 0


def fin_profile(length_m, conductance, cooling, sink_temperature_k, base_temperature_k,
                heat_per_length=None, n_nodes=_DEFAULT_NODES, tip_conductance=None,
                tip_sink_temperature_k=None, tip_heat_w=0.0):
    """Steady temperature along a fin with temperature-dependent properties.

    conductance(z, T) -> axial k*A (W m/K); cooling(z, T) -> h*P to the sink (W/(m K));
    heat_per_length(z) -> distributed heat input (W/m); tip_conductance(T) -> W/K from
    the z = 0 end to tip_sink_temperature_k (default the sink); tip_heat_w enters at z = 0.
    Solved by finite volumes with Picard iteration on the temperature-dependent properties.
    """
    tip_sink = sink_temperature_k if tip_sink_temperature_k is None else tip_sink_temperature_k

    def g_tip(t):
        return 0.0 if tip_conductance is None else tip_conductance(t)

    z = np.linspace(0.0, length_m, n_nodes)
    dz = z[1] - z[0]
    cell = np.full(n_nodes, dz)
    cell[0] = cell[-1] = dz / 2.0
    source = np.zeros(n_nodes) if heat_per_length is None else np.array([heat_per_length(zi) for zi in z])
    z_face = (z[:-1] + z[1:]) / 2.0

    t = np.full(n_nodes, float(base_temperature_k))
    for _ in range(_MAX_ITERATIONS):
        t_face = (t[:-1] + t[1:]) / 2.0
        g = np.array([conductance(zf, tf) for zf, tf in zip(z_face, t_face)]) / dz  # W/K per face
        hp = np.array([cooling(zi, ti) for zi, ti in zip(z, t)]) * cell              # W/K per node

        # Each node couples only to its neighbours, so the system is tridiagonal.
        b = source * cell + hp * sink_temperature_k
        b[0] += tip_heat_w + g_tip(t[0]) * tip_sink
        diag = hp.copy()
        diag[:-1] += g
        diag[1:] += g
        diag[0] += g_tip(t[0])
        lower, upper = -g.copy(), -g.copy()
        lower[-1] = 0.0   # the base row holds the base temperature
        diag[-1] = 1.0
        b[-1] = base_temperature_k
        t_new = solve_tridiagonal(lower, diag, upper, b)
        converged = np.max(np.abs(t_new - t)) < _TOLERANCE_K
        t = t_new
        if converged:
            break
    else:
        raise RuntimeError("Fin temperature did not converge")

    hp = np.array([cooling(zi, ti) for zi, ti in zip(z, t)]) * cell
    tip_out = float(g_tip(t[0]) * (t[0] - tip_sink))
    base_heat = float(np.sum(hp * (t - sink_temperature_k)) - np.sum(source * cell) + tip_out - tip_heat_w)
    return FinProfile(z, t, cell, base_heat, tip_out)


def solve_tridiagonal(lower, diag, upper, b):
    """x with lower[i-1] x[i-1] + diag[i] x[i] + upper[i] x[i+1] = b[i] (Thomas algorithm;
    stable for the diagonally dominant fin matrix)."""
    n = len(diag)
    c, d = np.empty(n - 1), np.empty(n)
    c[0], d[0] = upper[0] / diag[0], b[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / denom
        d[i] = (b[i] - lower[i - 1] * d[i - 1]) / denom
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


STEFAN_BOLTZMANN = 5.670374419e-8

HEAT_SOURCES = ("turbine_cap_conduction", "windage", "bearing_friction", "convection", "radiation",
                "internal_load", "cold_support_shear")


@dataclass(frozen=True)
class RotorThermal:
    z_m: np.ndarray
    temperature_k: np.ndarray
    sample_zone_m: float
    sample_hottest_temperature_k: float
    sample_heat_load_w: float
    breakdown_w: dict
    cold_support_heat_out_w: float   # heat crossing the Cold Support gap to the Bath
    pivot_heat_out_w: float          # heat conducted through the lower pivot contact to the Bath
    turbine_cap_heat_in_w: float     # heat conducted into the body from the Turbine Cap


def rotor_thermal(*, outer_radius_m, wall_m, body_length_m, bottom_cap_m, sample_length_m,
                  wall_conductivity, sample_conductivity_w_m_k, sample_internal_heat_load_w,
                  turbine_cap_temperature_k, bath_temperature_k, stator_temperature_k,
                  vapour_pressure_pa, gap_m, accommodation, emissivity, windage_heat_per_length_w_m,
                  sample_limit_temperature_k, gap_heat_transfer_coefficient, cold_support_gap_m=None,
                  cold_support_shear_heat_w=0.0, bearing_friction_heat_w=0.0, bearing_gas_heat_w=0.0,
                  pivot_conductance=None, vented_gas_temperature_k=None):
    """Temperature along the Rotor body below the Turbine Cap, and the Sample Zone heat budget.

    Layout from z = 0: solid Bottom Cap, then the Sample inside the tube, then empty bore
    up to the Turbine Cap at z = body_length_m. The Sample Zone is the longest stretch at or
    below sample_limit_temperature_k. Vapour Cooling and radiation exchange
    with the Stator across the gap; the radiation is the net exchange
    eps sigma (T^4 - T_stator^4), so it heats the Rotor only where the Stator is warmer.
    The Sample's centre runs hotter than its wall by Q / (4 pi k L) under internal load.
    With cold_support_gap_m set, the Bottom Cap face conducts across that gap to a Cold
    Support at Bath temperature and receives cold_support_shear_heat_w from the gap shear.
    The lower Bearing sits at the Bottom Cap: its friction heat and warm-gas heat enter there,
    and pivot_conductance(T) (W/K), when set, conducts from there to the Bath in parallel.
    With vented_gas_temperature_k set, Vented Drive Gas at that temperature washes the whole
    body (an upper bound) and convects onto it through the same gap coefficient, in parallel
    with Vapour Cooling to the Stator.
    """
    r_o = outer_radius_m
    r_i = r_o - wall_m
    perimeter = 2.0 * np.pi * r_o
    annulus = np.pi * (r_o**2 - r_i**2)
    sample_top = bottom_cap_m + sample_length_m
    sample_area = np.pi * r_i**2

    def in_sample(z):
        return bottom_cap_m <= z <= sample_top

    # Spread the internal load over the grid cells that lie in the Sample, so exactly Q is injected.
    z_grid = np.linspace(0.0, body_length_m, _DEFAULT_NODES)
    cell_grid = np.full(_DEFAULT_NODES, z_grid[1] - z_grid[0])
    cell_grid[0] = cell_grid[-1] = cell_grid[0] / 2.0
    sample_cells = np.sum(cell_grid[[in_sample(zi) for zi in z_grid]])
    internal_per_length = sample_internal_heat_load_w / sample_cells if sample_cells > 0 else 0.0

    def conductance(z, t):
        k = wall_conductivity(t)
        if z < bottom_cap_m:
            return k * np.pi * r_o**2
        return k * annulus + (sample_conductivity_w_m_k * sample_area if in_sample(z) else 0.0)

    if vented_gas_temperature_k is None:
        vented_hp = 0.0
    else:
        # Film taken at a fixed temperature so the gas exchange stays linear in T.
        vented_hp = gap_heat_transfer_coefficient((vented_gas_temperature_k + bath_temperature_k) / 2.0,
                                                  vapour_pressure_pa, gap_m, accommodation) * perimeter
    # Exchange with the vented gas, vented_hp (T_gas - T), split into a cooling term to the
    # Stator temperature and a fixed source, so the fin keeps one sink.
    vented_source = 0.0 if vented_gas_temperature_k is None else (
        vented_hp * (vented_gas_temperature_k - stator_temperature_k))

    def radiation(t):
        """h_rad with h_rad (T - T_stator) = eps sigma (T^4 - T_stator^4) exactly."""
        return emissivity * STEFAN_BOLTZMANN * (t**2 + stator_temperature_k**2) * (t + stator_temperature_k)

    def cooling(z, t):
        t_film = (t + stator_temperature_k) / 2.0
        h_gas = gap_heat_transfer_coefficient(t_film, vapour_pressure_pa, gap_m, accommodation)
        return (h_gas + radiation(t)) * perimeter + vented_hp

    def sources(z):
        return (windage_heat_per_length_w_m + vented_source
                + (internal_per_length if in_sample(z) else 0.0))

    if cold_support_gap_m is None:
        cold_support_shear_heat_w = 0.0

        def cold_support(t):
            return 0.0
    else:
        def cold_support(t):
            t_film = (t + bath_temperature_k) / 2.0
            return (gap_heat_transfer_coefficient(t_film, vapour_pressure_pa, cold_support_gap_m, accommodation)
                    * np.pi * r_o**2)
    pivot = pivot_conductance or (lambda t: 0.0)
    tip_heat_w = cold_support_shear_heat_w + bearing_friction_heat_w + bearing_gas_heat_w
    has_tip_sink = cold_support_gap_m is not None or pivot_conductance is not None

    # Cooling is referenced to the Stator; with the Stator at Bath temperature these coincide.
    fin = fin_profile(body_length_m, conductance, cooling, stator_temperature_k, turbine_cap_temperature_k,
                      sources, n_nodes=_DEFAULT_NODES,
                      tip_conductance=(lambda t: cold_support(t) + pivot(t)) if has_tip_sink else None,
                      tip_sink_temperature_k=bath_temperature_k, tip_heat_w=tip_heat_w)
    z, t, cell = fin.z_m, fin.temperature_k, fin.cell_length_m

    first, last, zone = _longest_cold_stretch(z, t, sample_limit_temperature_k)

    in_zone = np.zeros(len(z), dtype=bool)
    if zone > 0.0:
        in_zone[first:last + 1] = True
    zone_cell = cell * in_zone
    cooling_out = np.array([cooling(zi, ti) for zi, ti in zip(z, t)]) * (t - stator_temperature_k)
    source_in = np.array([sources(zi) for zi in z])
    # Net heat the zone sheds sideways beyond what is generated in it; conduction and the
    # Bottom Cap end sources make up the difference.
    shed = float(np.sum((cooling_out - source_in) * zone_cell))
    point_heat_w = {"bearing_friction": bearing_friction_heat_w, "convection": bearing_gas_heat_w,
                    "cold_support_shear": cold_support_shear_heat_w}
    if zone == 0.0:
        delivered, conducted_in = dict.fromkeys(point_heat_w, 0.0), 0.0
    elif first == 0:
        # The zone reaches the Bottom Cap face: its sources enter directly, the tip sink draws out.
        delivered = point_heat_w
        conducted_in = shed + fin.tip_heat_out_w - tip_heat_w
    else:
        # A warm Bottom Cap end lies below the zone: what crosses the lower edge comes from its sources.
        below = slice(0, first)
        from_bottom = (tip_heat_w - fin.tip_heat_out_w
                       + float(np.sum((source_in - cooling_out)[below] * cell[below])))
        if tip_heat_w > 0.0:
            delivered = {k: from_bottom * q / tip_heat_w for k, q in point_heat_w.items()}
            conducted_in = shed - from_bottom
        else:
            delivered, conducted_in = dict.fromkeys(point_heat_w, 0.0), shed
    sample_mask = np.array([in_sample(zi) for zi in z])
    vented_in = float(np.sum(vented_hp * (vented_gas_temperature_k - t) * zone_cell)) if vented_hp else 0.0
    # Net radiation from the Stator into the zone, only where the Stator is the warmer surface.
    radiation_in = float(np.sum(np.maximum(radiation(t) * (stator_temperature_k - t), 0.0) * perimeter * zone_cell))
    breakdown = {
        "turbine_cap_conduction": conducted_in,
        "windage": float(windage_heat_per_length_w_m * np.sum(zone_cell)),
        "bearing_friction": float(delivered["bearing_friction"]),
        # Bearing gas warmer than the Rotor, and Vented Drive Gas.
        "convection": float(delivered["convection"]) + vented_in,
        "radiation": radiation_in,
        "internal_load": float(internal_per_length * np.sum(zone_cell * sample_mask)),
        "cold_support_shear": float(delivered["cold_support_shear"]),
    }

    radial_rise = (sample_internal_heat_load_w / (4.0 * np.pi * sample_conductivity_w_m_k * sample_length_m)
                   if sample_length_m > 0 else 0.0)
    hottest = float(np.max(t[sample_mask])) + radial_rise if sample_mask.any() else float("nan")
    tip_rise = t[0] - bath_temperature_k
    return RotorThermal(z, t, zone, hottest, float(sum(breakdown.values())), breakdown,
                        float(cold_support(t[0]) * tip_rise), float(pivot(t[0]) * tip_rise), fin.base_heat_w)


def _longest_cold_stretch(z, t, limit):
    """First and last node of the longest contiguous run with t <= limit, and its interpolated length."""
    best = (0, 0, 0.0)
    n = len(z)
    i = 0
    while i < n:
        if t[i] > limit:
            i += 1
            continue
        j = i
        while j + 1 < n and t[j + 1] <= limit:
            j += 1
        lo = z[0] if i == 0 else _crossing(z[i - 1], z[i], t[i - 1], t[i], limit)
        hi = z[-1] if j == n - 1 else _crossing(z[j], z[j + 1], t[j], t[j + 1], limit)
        if hi - lo > best[2]:
            best = (i, j, float(hi - lo))
        i = j + 1
    return best


def _crossing(z0, z1, t0, t1, limit):
    return z0 + (limit - t0) * (z1 - z0) / (t1 - t0)
