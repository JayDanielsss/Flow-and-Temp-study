"""Bearing types behind one interface. SI units; Spin Rate in Hz.

Each Bearing set locates the Rotor at two stations: one at the Turbine Cap and one at
the Bottom Cap end. Both stations carry equal loads, so each takes half the friction.
Heat at the Turbine Cap station is lost into the Turbine Cap (held at a fixed temperature);
heat at the Bottom Cap station enters the Rotor there. Bearings are heat sources only.
"""

import math
from dataclasses import dataclass

from rotor_design.gas import HELIUM, gap_heat_transfer_coefficient, vapour_viscosity
from rotor_design.materials import ZIRCONIA

GAS = "gas"
MECHANICAL = "mechanical"
TWO_CENTER_PIVOT = "two_center_pivot"
BEARING_TYPES = (GAS, MECHANICAL, TWO_CENTER_PIVOT)

_STATIONS = 2

MECHANICAL_WARNING = ("Mechanical bearing: unlubricated operation at 1.2 K and 60,000 rpm is untested; "
                      "friction coefficient and life are placeholders")


@dataclass(frozen=True)
class BearingPerformance:
    friction_torque_n_m: float     # both stations
    bottom_friction_heat_w: float  # friction heat entering the Rotor at the Bottom Cap station
    bottom_gas_heat_w: float       # heat from bearing gas warmer than the Rotor, Bottom Cap station; >= 0
    stiffness_n_m: float           # radial, both stations together
    radial_space_mm: float         # added to the Stator radius, each side
    contact_radius_m: float        # pivot contact ring radius; nan for other types
    warning: str
    preload_n: float = math.nan    # sized Two-Center Pivot Preload; nan for other types
    unbalance_force_n: float = math.nan  # rotating load at the Balance Grade; nan where not modelled


@dataclass(frozen=True)
class GasBearing:
    """Externally pressurised gas journals with Petroff friction in the film; gas is the bearing gas."""

    journal_length_mm: float
    clearance_mm: float
    gas_temperature_k: float
    supply_pressure_bar: float
    mass_flow_kg_s: float          # both journals together
    stiffness_coefficient: float   # K = coefficient (p_s - p_a) L D / clearance, per journal
    radial_space_mm: float

    def performance(self, *, rotor_radius_m, rotor_weight_n, spin_rate_hz, bath_temperature_k,
                    ambient_pressure_pa, accommodation, gas=HELIUM, **_):
        omega = 2.0 * math.pi * spin_rate_hz
        length, clearance = self.journal_length_mm / 1e3, self.clearance_mm / 1e3
        # Petroff: tau = mu omega R / c over 2 pi R L, at radius R.
        torque = 2.0 * math.pi * vapour_viscosity(self.gas_temperature_k, gas) * rotor_radius_m**3 * length * omega / clearance
        supply_pa = self.supply_pressure_bar * 1e5
        stiffness = (self.stiffness_coefficient * (supply_pa - ambient_pressure_pa) * length
                     * 2.0 * rotor_radius_m / clearance)
        return BearingPerformance(
            friction_torque_n_m=_STATIONS * torque,
            bottom_friction_heat_w=torque * omega,
            bottom_gas_heat_w=self._gas_heat(rotor_radius_m, length, clearance, bath_temperature_k,
                                             (supply_pa + ambient_pressure_pa) / 2.0, accommodation, gas),
            stiffness_n_m=_STATIONS * stiffness,
            radial_space_mm=self.radial_space_mm,
            contact_radius_m=math.nan,
            warning="",
        )

    def _gas_heat(self, radius_m, length_m, clearance_m, rotor_temperature_k, film_pressure_pa, accommodation,
                  gas):
        """Heat one journal's gas gives the Rotor: m cp dT (1 - exp(-NTU)), zero if the gas is not warmer.

        The Rotor is taken at the Bath temperature, the coldest it can be (upper bound).
        """
        excess = self.gas_temperature_k - rotor_temperature_k
        if excess <= 0.0 or self.mass_flow_kg_s <= 0.0:
            return 0.0
        capacity = self.mass_flow_kg_s / _STATIONS * gas.cp_dilute
        h = gap_heat_transfer_coefficient((self.gas_temperature_k + rotor_temperature_k) / 2.0,
                                          film_pressure_pa, clearance_m, accommodation, gas)
        ntu = h * 2.0 * math.pi * radius_m * length_m / capacity
        return capacity * excess * -math.expm1(-ntu)


@dataclass(frozen=True)
class MechanicalBearing:
    """Unlubricated ball bearings around the Rotor; friction torque mu P d_m / 2 per bearing."""

    friction_coefficient: float
    preload_n: float
    stiffness_n_m: float           # radial, per bearing
    radial_space_mm: float

    def performance(self, *, rotor_radius_m, rotor_weight_n, spin_rate_hz, bath_temperature_k,
                    ambient_pressure_pa, accommodation, gas=HELIUM, **_):
        omega = 2.0 * math.pi * spin_rate_hz
        load = self.preload_n + rotor_weight_n / _STATIONS
        # Ball pitch circle in the middle of the radial space.
        pitch_diameter = 2.0 * rotor_radius_m + self.radial_space_mm / 1e3
        torque = self.friction_coefficient * load * pitch_diameter / 2.0
        return BearingPerformance(
            friction_torque_n_m=_STATIONS * torque,
            bottom_friction_heat_w=torque * omega,
            bottom_gas_heat_w=0.0,
            stiffness_n_m=_STATIONS * self.stiffness_n_m,
            radial_space_mm=self.radial_space_mm,
            contact_radius_m=math.nan,
            warning=MECHANICAL_WARNING,
        )


def unbalance_force_n(mass_kg, balance_grade_m, spin_rate_hz):
    """Rotating load m e omega^2 of a Rotor whose centre of mass sits balance_grade_m off its spin axis (N)."""
    return mass_kg * balance_grade_m * (2.0 * math.pi * spin_rate_hz) ** 2


def cone_seat_radial_limit_n(axial_load_n, cone_half_angle_deg):
    """Largest radial load a ball held in a cone seat by axial_load_n reacts before the ring unloads (N).

    The ball touches the cone on a ring. The axial load spreads normal force P / (2 pi sin alpha) per
    radian evenly round it; a radial load F adds F cos(phi) / (pi cos alpha). That side of the ring
    unloads when F reaches P / (2 tan alpha).
    """
    return axial_load_n / (2.0 * math.tan(math.radians(cone_half_angle_deg)))


def line_contact_stiffness_n_m2(load_per_length_n_m, ball_radius_m, e_star_pa):
    """Normal stiffness per unit length of a Hertz line contact, a cylinder of radius R on a flat (N/m^2).

    Approach delta = (q / (pi E*)) (2 ln(4 R / b) - 1), b = sqrt(4 q R / (pi E*)) the contact
    half-width (the cylinder-on-half-space form, Johnson, Contact Mechanics (1985), ch. 4); its slope
    dq/d(delta) = pi E* / (2 ln(4 R / b) - 2). Approximate: a line contact's approach depends on where
    it is measured from, only logarithmically.
    """
    b = math.sqrt(4.0 * load_per_length_n_m * ball_radius_m / (math.pi * e_star_pa))
    return math.pi * e_star_pa / (2.0 * math.log(4.0 * ball_radius_m / b) - 2.0)


@dataclass(frozen=True)
class TwoCenterPivots:
    """A zirconia ball on each end cap axis, running in a cone seat in the Stator, held by an axial Preload.

    The ball touches its cone on a ring of radius R cos(alpha) (alpha the cone half-angle), so the
    seat locates the Rotor radially. The Preload is sized, not chosen: each seat must react, with the
    margin, its share of the rotating unbalance load m e omega^2 and of the weight component across
    the spin axis at the Tilt Angle; the weight component along the axis bears on the lower seat.
    The ring slides at the Spin Rate, so each seat's friction torque is mu N R cos(alpha), with
    N = (axial load) / sin(alpha) the total normal force. Radial stiffness of a seat:
    pi rho k' cos(alpha)^2 from the ring, k' the line-contact stiffness per unit length.
    They need no radial space.
    """

    friction_coefficient: float
    ball_radius_mm: float
    cone_half_angle_deg: float
    preload_margin: float          # sized Preload over the least that keeps both seats loaded
    radial_space_mm = 0.0          # pivots sit on the axis

    def seat_depth_mm(self):
        """Depth of the cone from its apex to the contact ring."""
        alpha = math.radians(self.cone_half_angle_deg)
        return self.ball_radius_mm * math.cos(alpha) / math.tan(alpha)

    def performance(self, *, rotor_radius_m, rotor_weight_n, spin_rate_hz, bath_temperature_k,
                    ambient_pressure_pa, accommodation, gas=HELIUM, rotor_mass_kg=0.0, lower_arm_m=1.0,
                    upper_arm_m=1.0, tilt_angle_deg=0.0, balance_grade_m=0.0):
        omega = 2.0 * math.pi * spin_rate_hz
        alpha = math.radians(self.cone_half_angle_deg)
        ball_m = self.ball_radius_mm / 1e3
        ring_radius = ball_m * math.cos(alpha)
        tilt = math.radians(tilt_angle_deg)
        unbalance = unbalance_force_n(rotor_mass_kg, balance_grade_m, spin_rate_hz)
        # Rotating and steady radial loads at their worst alignment, shared by the two seats in
        # inverse proportion to their distance from the centre of mass.
        radial = unbalance + rotor_weight_n * math.sin(tilt)
        span = lower_arm_m + upper_arm_m
        lower_radial, upper_radial = radial * upper_arm_m / span, radial * lower_arm_m / span
        axial_weight = rotor_weight_n * math.cos(tilt)
        # The upper seat carries the Preload; the lower one the Preload plus the axial weight.
        needed = max(2.0 * math.tan(alpha) * upper_radial,
                     2.0 * math.tan(alpha) * lower_radial - axial_weight, 0.0)
        preload = self.preload_margin * needed
        upper_axial, lower_axial = preload, preload + axial_weight
        e_star = ZIRCONIA.youngs_modulus_pa / (2.0 * (1.0 - ZIRCONIA.poissons_ratio**2))

        def seat_torque(axial):
            return self.friction_coefficient * axial / math.sin(alpha) * ring_radius

        def seat_stiffness(axial):
            if axial <= 0.0:
                return 0.0
            per_length = axial / math.sin(alpha) / (2.0 * math.pi * ring_radius)
            return (math.pi * ring_radius * math.cos(alpha) ** 2
                    * line_contact_stiffness_n_m2(per_length, ball_m, e_star))

        lower_torque = seat_torque(lower_axial)
        return BearingPerformance(
            friction_torque_n_m=seat_torque(upper_axial) + lower_torque,
            bottom_friction_heat_w=lower_torque * omega,
            bottom_gas_heat_w=0.0,
            stiffness_n_m=seat_stiffness(upper_axial) + seat_stiffness(lower_axial),
            radial_space_mm=self.radial_space_mm,
            contact_radius_m=ring_radius,
            warning="",
            preload_n=preload,
            unbalance_force_n=unbalance,
        )
