"""Segmented 1-D fin along the Rotor axis. SI units.

z = 0 is the Bottom Cap end (adiabatic unless a Cold Support is added);
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


def fin_profile(length_m, conductance, cooling, sink_temperature_k, base_temperature_k,
                heat_per_length=None, n_nodes=_DEFAULT_NODES):
    """Steady temperature along a fin with temperature-dependent properties.

    conductance(z, T) -> axial k*A (W m/K); cooling(z, T) -> h*P to the sink (W/(m K));
    heat_per_length(z) -> distributed heat input (W/m). Solved by finite volumes
    with Picard iteration on the temperature-dependent properties.
    """
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

        a = np.zeros((n_nodes, n_nodes))
        b = source * cell + hp * sink_temperature_k
        for i in range(n_nodes - 1):
            a[i, i] += g[i]
            a[i + 1, i + 1] += g[i]
            a[i, i + 1] -= g[i]
            a[i + 1, i] -= g[i]
        a[np.diag_indices(n_nodes)] += hp
        a[-1, :] = 0.0
        a[-1, -1] = 1.0
        b[-1] = base_temperature_k
        t_new = np.linalg.solve(a, b)
        converged = np.max(np.abs(t_new - t)) < _TOLERANCE_K
        t = t_new
        if converged:
            break
    else:
        raise RuntimeError("Fin temperature did not converge")

    hp = np.array([cooling(zi, ti) for zi, ti in zip(z, t)]) * cell
    base_heat = float(np.sum(hp * (t - sink_temperature_k)) - np.sum(source * cell))
    return FinProfile(z, t, cell, base_heat)


STEFAN_BOLTZMANN = 5.670374419e-8

HEAT_SOURCES = ("turbine_cap_conduction", "windage", "bearing_friction", "convection", "radiation",
                "internal_load")


@dataclass(frozen=True)
class RotorThermal:
    z_m: np.ndarray
    temperature_k: np.ndarray
    sample_zone_m: float
    sample_hottest_temperature_k: float
    sample_heat_load_w: float
    breakdown_w: dict


def rotor_thermal(*, outer_radius_m, wall_m, body_length_m, bottom_cap_m, sample_length_m,
                  wall_conductivity, sample_conductivity_w_m_k, sample_internal_heat_load_w,
                  turbine_cap_temperature_k, bath_temperature_k, stator_temperature_k,
                  vapour_pressure_pa, gap_m, accommodation, emissivity, windage_heat_per_length_w_m,
                  sample_zone_delta_t_k, gap_heat_transfer_coefficient):
    """Temperature along the Rotor body below the Turbine Cap, and the Sample Zone heat budget.

    Layout from z = 0: solid Bottom Cap, then the Sample inside the tube, then empty bore
    up to the Turbine Cap at z = body_length_m. Vapour Cooling and radiation exchange
    with the Stator across the gap; Stator radiation is absorbed as a source.
    The Sample's centre runs hotter than its wall by Q / (4 pi k L) under internal load.
    """
    r_o = outer_radius_m
    r_i = r_o - wall_m
    perimeter = 2.0 * np.pi * r_o
    annulus = np.pi * (r_o**2 - r_i**2)
    sample_top = bottom_cap_m + sample_length_m
    sample_area = np.pi * r_i**2
    radiation_in_per_area = emissivity * STEFAN_BOLTZMANN * stator_temperature_k**4

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

    def cooling(z, t):
        t_film = (t + stator_temperature_k) / 2.0
        h_gas = gap_heat_transfer_coefficient(t_film, vapour_pressure_pa, gap_m, accommodation)
        # Emission linearised about the local temperature.
        h_rad = emissivity * STEFAN_BOLTZMANN * (t**2 + stator_temperature_k**2) * (t + stator_temperature_k)
        return (h_gas + h_rad) * perimeter

    def sources(z):
        return (windage_heat_per_length_w_m + radiation_in_per_area * perimeter
                + (internal_per_length if in_sample(z) else 0.0))

    # Cooling is referenced to the Stator; with the Stator at Bath temperature these coincide.
    fin = fin_profile(body_length_m, conductance, cooling, stator_temperature_k, turbine_cap_temperature_k,
                      sources, n_nodes=_DEFAULT_NODES)
    z, t, cell = fin.z_m, fin.temperature_k, fin.cell_length_m

    limit = bath_temperature_k + sample_zone_delta_t_k
    if t[0] > limit:
        zone = 0.0
    elif t[-1] <= limit:
        zone = body_length_m
    else:
        i = int(np.argmax(t > limit))
        zone = float(z[i - 1] + (limit - t[i - 1]) * (z[i] - z[i - 1]) / (t[i] - t[i - 1]))

    in_zone = z <= zone
    zone_cell = cell * in_zone
    cooling_out = np.array([cooling(zi, ti) for zi, ti in zip(z, t)]) * (t - stator_temperature_k)
    source_in = np.array([sources(zi) for zi in z])
    conducted_in = float(np.sum((cooling_out - source_in) * zone_cell))
    sample_mask = np.array([in_sample(zi) for zi in z])
    breakdown = {
        "turbine_cap_conduction": conducted_in,
        "windage": float(windage_heat_per_length_w_m * np.sum(zone_cell)),
        "bearing_friction": 0.0,   # no Bearing model yet
        "convection": 0.0,         # no gas warmer than the Rotor reaches the Sample Zone yet
        "radiation": float(radiation_in_per_area * perimeter * np.sum(zone_cell)),
        "internal_load": float(internal_per_length * np.sum(zone_cell * sample_mask)),
    }

    radial_rise = (sample_internal_heat_load_w / (4.0 * np.pi * sample_conductivity_w_m_k * sample_length_m)
                   if sample_length_m > 0 else 0.0)
    hottest = float(np.max(t[sample_mask])) + radial_rise if sample_mask.any() else float("nan")
    return RotorThermal(z, t, zone, hottest, float(sum(breakdown.values())), breakdown)
