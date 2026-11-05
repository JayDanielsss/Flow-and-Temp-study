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
    contact_radius_m: float        # pivot contact radius; nan for other types
    warning: str


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
                    ambient_pressure_pa, accommodation, gas=HELIUM):
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
                    ambient_pressure_pa, accommodation, gas=HELIUM):
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


@dataclass(frozen=True)
class TwoCenterPivots:
    """Spherical zirconia pivot tips on flat zirconia seats, one on each end cap axis.

    Hertz contact radius a = (3 W R / (4 E*))^(1/3); spinning friction torque
    (3 pi / 16) mu W a; normal contact stiffness 2 E* a. They need no radial space.
    """

    friction_coefficient: float
    preload_n: float
    tip_radius_mm: float
    radial_space_mm = 0.0          # pivots sit on the axis

    def performance(self, *, rotor_radius_m, rotor_weight_n, spin_rate_hz, bath_temperature_k,
                    ambient_pressure_pa, accommodation, gas=HELIUM):
        omega = 2.0 * math.pi * spin_rate_hz
        load = self.preload_n + rotor_weight_n / _STATIONS
        e_star = ZIRCONIA.youngs_modulus_pa / (2.0 * (1.0 - ZIRCONIA.poissons_ratio**2))
        a = (3.0 * load * self.tip_radius_mm / 1e3 / (4.0 * e_star)) ** (1.0 / 3.0)
        torque = 3.0 * math.pi / 16.0 * self.friction_coefficient * load * a
        return BearingPerformance(
            friction_torque_n_m=_STATIONS * torque,
            bottom_friction_heat_w=torque * omega,
            bottom_gas_heat_w=0.0,
            stiffness_n_m=_STATIONS * 2.0 * e_star * a,
            radial_space_mm=self.radial_space_mm,
            contact_radius_m=a,
            warning="",
        )


def pivot_contact_conductance(contact_radius_m, temperature_k, bath_temperature_k, conductivity_multiplier=1.0):
    """Constriction conductance 2 k a of a circular contact between two zirconia half-spaces (W/K).

    k is taken at the mean of the Rotor-side and Bath-side temperatures.
    """
    k = ZIRCONIA.conductivity((temperature_k + bath_temperature_k) / 2.0, conductivity_multiplier)
    return 2.0 * k * contact_radius_m
