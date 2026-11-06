"""Segmented 1-D fin along the Rotor axis. SI units.

z = 0 is the Bottom Cap end (adiabatic unless a Cold Support cools it);
z = length is the fin base, held at a fixed temperature (the Turbine Cap side).
"""

from dataclasses import dataclass

import numpy as np

_DEFAULT_NODES = 401
_MAX_ITERATIONS = 1000
_MIN_RELAXATION = 1.0 / 64.0
_TOLERANCE_K = 1e-9


@dataclass(frozen=True)
class FinProfile:
    z_m: np.ndarray
    temperature_k: np.ndarray
    cell_length_m: np.ndarray   # axial length each node represents
    base_heat_w: float          # heat entering through the base at z = length
    tip_heat_out_w: float       # heat leaving through the tip conductance at z = 0
    face_conductance_w_k: np.ndarray  # W/K between node i and i + 1, at the converged temperatures


def fin_profile(length_m, conductance, cooling, sink_temperature_k, base_temperature_k,
                heat_per_length=None, n_nodes=_DEFAULT_NODES, tip_conductance=None,
                tip_sink_temperature_k=None, tip_heat_w=0.0, vectorized=False):
    """Steady temperature along a fin with temperature-dependent properties.

    conductance(z, T) -> axial k*A (W m/K); cooling(z, T) -> h*P to the sink (W/(m K));
    heat_per_length(z) -> distributed heat input (W/m); tip_conductance(T) -> W/K from
    the z = 0 end to tip_sink_temperature_k (default the sink); tip_heat_w enters at z = 0.
    With vectorized, conductance, cooling and heat_per_length take and return arrays.
    Solved by finite volumes with Picard iteration on the temperature-dependent properties.
    """
    if not vectorized:
        conductance, cooling = _elementwise(conductance), _elementwise(cooling)
        if heat_per_length is not None:
            heat_per_length = _elementwise_source(heat_per_length)
    tip_sink = sink_temperature_k if tip_sink_temperature_k is None else tip_sink_temperature_k

    def g_tip(t):
        return 0.0 if tip_conductance is None else tip_conductance(t)

    z = np.linspace(0.0, length_m, n_nodes)
    dz = z[1] - z[0]
    cell = np.full(n_nodes, dz)
    cell[0] = cell[-1] = dz / 2.0
    source = np.zeros(n_nodes) if heat_per_length is None else np.broadcast_to(heat_per_length(z), z.shape)
    z_face = (z[:-1] + z[1:]) / 2.0

    def face_conductance(t):
        return np.broadcast_to(conductance(z_face, (t[:-1] + t[1:]) / 2.0), z_face.shape) / dz  # W/K per face

    def node_cooling(t):
        return np.broadcast_to(cooling(z, t), z.shape) * cell                                    # W/K per node

    t = np.full(n_nodes, float(base_temperature_k))
    relaxation, previous_change = 1.0, np.inf
    for _ in range(_MAX_ITERATIONS):
        g = face_conductance(t)
        hp = node_cooling(t)

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
        change = np.max(np.abs(t_new - t))
        converged = change < _TOLERANCE_K
        # Plain Picard while it contracts; once a step grows (strongly temperature-dependent
        # conductivity under a large point load can make it oscillate), damp the steps.
        if change > previous_change:
            relaxation = max(relaxation / 2.0, _MIN_RELAXATION)
        previous_change = change
        t = t_new if relaxation == 1.0 else t + relaxation * (t_new - t)
        if converged:
            break
    else:
        raise RuntimeError("Fin temperature did not converge")

    hp = node_cooling(t)
    tip_out = float(g_tip(t[0]) * (t[0] - tip_sink))
    base_heat = float(np.sum(hp * (t - sink_temperature_k)) - np.sum(source * cell) + tip_out - tip_heat_w)
    return FinProfile(z, t, cell, base_heat, tip_out, np.array(face_conductance(t)))


def _elementwise(f):
    """A scalar f(z, T) as one over arrays."""
    return lambda z, t: np.array([f(zi, ti) for zi, ti in zip(z, t)])


def _elementwise_source(f):
    """A scalar f(z) as one over an array."""
    return lambda z: np.array([f(zi) for zi in z])


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
# A Sample whose top lies this close to the Turbine Cap face is taken to touch it (m).
_FIT_TOLERANCE_M = 1e-12
# Placement Window edges are bisected to this fraction of the scan step.
_EDGE_TOLERANCE = 1.0 / 256.0


@dataclass(frozen=True)
class RotorThermal:
    z_m: np.ndarray
    temperature_k: np.ndarray
    sample_position_m: float         # Sample bottom, from the Bottom Cap face
    sample_zone_m: float             # diagnostic: longest stretch at or below the Sample Limit Temperature
    sample_hottest_temperature_k: float
    sample_heat_load_w: float        # heat crossing the Sample's boundary
    breakdown_w: dict
    cold_support_heat_out_w: float   # heat crossing the Cold Support gap to the Bath
    turbine_cap_heat_in_w: float     # heat conducted into the body from the Turbine Cap


def rotor_thermal(*, outer_radius_m, wall_m, body_length_m, bottom_cap_m, sample_length_m,
                  wall_conductivity, sample_conductivity_w_m_k, sample_internal_heat_load_w,
                  turbine_cap_temperature_k, bath_temperature_k, stator_temperature_k,
                  vapour_pressure_pa, gap_m, accommodation, emissivity, windage_heat_per_length_w_m,
                  sample_limit_temperature_k, gap_heat_transfer_coefficient, sample_position_m=0.0,
                  cold_support_gap_m=None, cold_support_shear_heat_w=0.0, bearing_friction_heat_w=0.0,
                  bearing_gas_heat_w=0.0, vented_gas_temperature_k=None):
    """Temperature along the Rotor body below the Turbine Cap, and the Sample's heat budget.

    Layout from z = 0: solid Bottom Cap, then empty bore for sample_position_m, then the Sample
    inside the tube, then empty bore up to the Turbine Cap at z = body_length_m. The Sample
    conducts in parallel with the wall, so where it sits changes the profile. Vapour Cooling and
    radiation exchange with the Stator across the gap; the radiation is the net exchange
    eps sigma (T^4 - T_stator^4), so it heats the Rotor only where the Stator is warmer.
    The Sample's centre runs hotter than its wall by Q / (4 pi k L) under internal load.
    With cold_support_gap_m set, the Bottom Cap face conducts across that gap to a Cold
    Support at Bath temperature and receives cold_support_shear_heat_w from the gap shear.
    The lower Bearing sits at the Bottom Cap: its friction heat and warm-gas heat enter there.
    With vented_gas_temperature_k set, Vented Drive Gas at that temperature washes the whole
    body (an upper bound) and convects onto it through the same gap coefficient, in parallel
    with Vapour Cooling to the Stator.

    The Sample Heat Load is the heat crossing the Sample's boundary: the heat generated or
    received across the side within the Sample's length, and the heat conducted in across its
    two ends where that flows inward. Heat conducted in across an end is shared among the
    sources on that side of the Sample in proportion to the heat each puts in there.
    """
    r_o = outer_radius_m
    r_i = r_o - wall_m
    perimeter = 2.0 * np.pi * r_o
    annulus = np.pi * (r_o**2 - r_i**2)
    sample_bottom = bottom_cap_m + sample_position_m
    sample_top = sample_bottom + sample_length_m
    if sample_position_m < 0.0 or sample_top > body_length_m + _FIT_TOLERANCE_M:
        raise ValueError(f"A {sample_length_m * 1e3:g} mm Sample at {sample_position_m * 1e3:g} mm from the "
                         f"Bottom Cap does not fit the {body_length_m * 1e3:g} mm Rotor body")
    sample_area = np.pi * r_i**2

    def in_sample(z):
        return (z >= sample_bottom) & (z <= sample_top)

    # Spread the internal load over the grid cells that lie in the Sample, so exactly Q is injected.
    z_grid = np.linspace(0.0, body_length_m, _DEFAULT_NODES)
    cell_grid = np.full(_DEFAULT_NODES, z_grid[1] - z_grid[0])
    cell_grid[0] = cell_grid[-1] = cell_grid[0] / 2.0
    sample_cells = np.sum(cell_grid[in_sample(z_grid)])
    internal_per_length = sample_internal_heat_load_w / sample_cells if sample_cells > 0 else 0.0

    def conductance(z, t):
        k = wall_conductivity(t)
        return np.where(z < bottom_cap_m, k * np.pi * r_o**2,
                        k * annulus + np.where(in_sample(z), sample_conductivity_w_m_k * sample_area, 0.0))

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
        return windage_heat_per_length_w_m + vented_source + np.where(in_sample(z), internal_per_length, 0.0)

    if cold_support_gap_m is None:
        cold_support_shear_heat_w = 0.0

        def cold_support(t):
            return 0.0
    else:
        def cold_support(t):
            t_film = (t + bath_temperature_k) / 2.0
            return (gap_heat_transfer_coefficient(t_film, vapour_pressure_pa, cold_support_gap_m, accommodation)
                    * np.pi * r_o**2)
    tip_heat_w = cold_support_shear_heat_w + bearing_friction_heat_w + bearing_gas_heat_w

    # Cooling is referenced to the Stator; with the Stator at Bath temperature these coincide.
    fin = fin_profile(body_length_m, conductance, cooling, stator_temperature_k, turbine_cap_temperature_k,
                      sources, n_nodes=_DEFAULT_NODES,
                      tip_conductance=cold_support if cold_support_gap_m is not None else None,
                      tip_sink_temperature_k=bath_temperature_k, tip_heat_w=tip_heat_w, vectorized=True)
    z, t, cell, g = fin.z_m, fin.temperature_k, fin.cell_length_m, fin.face_conductance_w_k
    n = len(z)
    _, _, zone = _longest_cold_stretch(z, t, sample_limit_temperature_k)

    # Heat each node takes in from each source (W): side exchanges count only where they heat the Rotor.
    windage_in = windage_heat_per_length_w_m * cell
    vented_in = vented_hp * np.maximum(vented_gas_temperature_k - t, 0.0) * cell if vented_hp else np.zeros(n)
    # Net radiation from the Stator, only where the Stator is the warmer surface.
    radiation_in = np.maximum(radiation(t) * (stator_temperature_k - t), 0.0) * perimeter * cell
    internal_in = np.where(in_sample(z), internal_per_length, 0.0) * cell
    sample_nodes = np.flatnonzero(in_sample(z))
    if sample_nodes.size == 0:
        raise ValueError("The Sample is shorter than one grid cell")
    first, last = int(sample_nodes[0]), int(sample_nodes[-1])
    last = min(last, n - 2)   # the base node is the Turbine Cap face itself
    inside = slice(first, last + 1)

    def side(region, extra):
        """Heat put in on one side of the Sample, by source."""
        return {"windage": float(np.sum(windage_in[region])), "convection": float(np.sum(vented_in[region])),
                "radiation": float(np.sum(radiation_in[region])),
                "internal_load": float(np.sum(internal_in[region])), **extra}

    def shared(conducted_w, inputs):
        """Heat conducted in across an end, shared among that side's sources by the heat each puts in."""
        total = sum(inputs.values())
        if conducted_w <= 0.0 or total <= 0.0:
            return dict.fromkeys(inputs, 0.0)
        return {k: conducted_w * q / total for k, q in inputs.items()}

    from_above = float(g[last] * (t[last + 1] - t[last]))
    # The Turbine Cap's input is what it conducts into the body across the face next to it.
    above = side(slice(last + 1, n - 1),
                 {"turbine_cap_conduction": max(float(g[n - 2] * (t[n - 1] - t[n - 2])), 0.0)})
    if first > 0:
        from_below = float(g[first - 1] * (t[first - 1] - t[first]))
        below = side(slice(0, first), {"bearing_friction": bearing_friction_heat_w,
                                       "convection_point": bearing_gas_heat_w,
                                       "cold_support_shear": cold_support_shear_heat_w})
        tip_direct = {}
    else:
        # The Sample reaches the Bottom Cap face: the lower Bearing's heat enters it directly.
        from_below, below = 0.0, {}
        tip_direct = {"bearing_friction": bearing_friction_heat_w, "convection": bearing_gas_heat_w,
                      "cold_support_shear": cold_support_shear_heat_w}
    breakdown = dict.fromkeys(HEAT_SOURCES, 0.0)
    for parts in (side(inside, {}), tip_direct, shared(from_above, above), shared(from_below, below)):
        for key, watts in parts.items():
            breakdown["convection" if key == "convection_point" else key] += watts

    radial_rise = (sample_internal_heat_load_w / (4.0 * np.pi * sample_conductivity_w_m_k * sample_length_m)
                   if sample_length_m > 0 else 0.0)
    hottest = float(np.max(t[sample_nodes])) + radial_rise
    tip_rise = t[0] - bath_temperature_k
    return RotorThermal(z, t, sample_position_m, zone, hottest, float(sum(breakdown.values())), breakdown,
                        float(cold_support(t[0]) * tip_rise), fin.base_heat_w)


@dataclass(frozen=True)
class Placement:
    """Where the Sample goes, and the Placement Window it may go in. Positions are measured from
    the Bottom Cap face to the bottom of the Sample (m)."""

    thermal: RotorThermal            # at the chosen Sample Position
    window_m: tuple                  # (lowest, highest) Sample Position in the window; nan when empty
    best_headroom_k: float           # largest Sample Limit Temperature less hottest point over the scan
    positions_m: np.ndarray          # the scan
    hottest_k: np.ndarray
    heat_load_w: np.ndarray

    @property
    def empty(self):
        return not np.isfinite(self.window_m[0])


def place_sample(*, position_step_m, sample_position_m=None, **thermal_inputs):
    """Scan Sample Positions from the Bottom Cap face to the Turbine Cap, every position_step_m.

    The Placement Window is the run of scanned positions whose hottest point stays at or below
    the Sample Limit Temperature, with its edges bisected between scan points; where several
    runs exist, the one holding the lowest Sample Heat Load. The Sample goes at the scanned
    position in the window with the lowest Sample Heat Load, or at sample_position_m when given
    (where it need not be in the window). With an empty window it goes where it runs coolest.
    thermal_inputs are rotor_thermal's.
    """
    room = thermal_inputs["body_length_m"] - thermal_inputs["bottom_cap_m"] - thermal_inputs["sample_length_m"]
    if room < -_FIT_TOLERANCE_M:
        raise ValueError(f"A {thermal_inputs['sample_length_m'] * 1e3:g} mm Sample does not fit the "
                         f"{thermal_inputs['body_length_m'] * 1e3:g} mm Rotor body")
    room = max(room, 0.0)
    positions = np.linspace(0.0, room, max(2, int(np.ceil(room / position_step_m - 1e-9)) + 1)) if room > 0 else np.zeros(1)
    scan = [rotor_thermal(sample_position_m=float(p), **thermal_inputs) for p in positions]
    limit = thermal_inputs["sample_limit_temperature_k"]
    hottest = np.array([s.sample_hottest_temperature_k for s in scan])
    load = np.array([s.sample_heat_load_w for s in scan])
    headroom = limit - hottest
    ok = headroom >= 0.0
    if ok.any():
        best = int(np.argmin(np.where(ok, load, np.inf)))
        lo, hi = best, best
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(positions) - 1 and ok[hi + 1]:
            hi += 1
        def edge(inside, outside):
            """Bisect between a scanned position in the window and its neighbour outside."""
            while abs(outside - inside) > position_step_m * _EDGE_TOLERANCE:
                mid = (inside + outside) / 2.0
                fits = rotor_thermal(sample_position_m=mid, **thermal_inputs).sample_hottest_temperature_k <= limit
                inside, outside = (mid, outside) if fits else (inside, mid)
            return inside

        low_edge = positions[lo] if lo == 0 else edge(positions[lo], positions[lo - 1])
        high_edge = positions[hi] if hi == len(positions) - 1 else edge(positions[hi], positions[hi + 1])
        window = (float(low_edge), float(high_edge))
    else:
        best, window = int(np.argmax(headroom)), (float("nan"), float("nan"))
    if sample_position_m is None:
        chosen = scan[best]
    else:
        match = np.flatnonzero(np.isclose(positions, sample_position_m, rtol=0.0, atol=1e-12))
        chosen = (scan[int(match[0])] if match.size
                  else rotor_thermal(sample_position_m=sample_position_m, **thermal_inputs))
    return Placement(chosen, window, float(np.max(headroom)), positions, hottest, load)


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
