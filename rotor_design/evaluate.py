"""Evaluate one Design: Design plus assumptions in, Results out."""

from dataclasses import dataclass

from rotor_design.geometry import max_rotor_diameter, projected_width

# Slack for a Design that sits exactly on the bore limit (mm).
_FIT_TOLERANCE_MM = 1e-9


@dataclass(frozen=True)
class Design:
    rotor_length_mm: float
    # None means the largest diameter the Probe Bore allows.
    rotor_diameter_mm: float | None = None


@dataclass(frozen=True)
class Results:
    rotor_length_mm: float
    rotor_diameter_mm: float
    max_rotor_diameter_mm: float
    stator_length_mm: float
    stator_diameter_mm: float
    projected_width_mm: float
    usable_bore_mm: float
    fits_bore: bool
    feasible: bool
    binding_constraint: str


def evaluate(design, assumptions):
    a = assumptions
    if design.rotor_length_mm > a.max_rotor_length_mm:
        raise ValueError(
            f"Rotor Length {design.rotor_length_mm} mm exceeds the {a.max_rotor_length_mm} mm limit"
        )
    allowance = a.stator_allowance
    largest = max_rotor_diameter(design.rotor_length_mm, allowance, a.bore_diameter_mm, a.tilt_angle_deg)
    diameter = largest if design.rotor_diameter_mm is None else design.rotor_diameter_mm

    stator_length = allowance.stator_length(design.rotor_length_mm)
    stator_diameter = allowance.stator_diameter(diameter)
    width = projected_width(stator_length, stator_diameter, a.tilt_angle_deg)
    usable_bore = allowance.usable_bore(a.bore_diameter_mm)
    fits = diameter > 0.0 and width <= usable_bore + _FIT_TOLERANCE_MM

    return Results(
        rotor_length_mm=design.rotor_length_mm,
        rotor_diameter_mm=diameter,
        max_rotor_diameter_mm=largest,
        stator_length_mm=stator_length,
        stator_diameter_mm=stator_diameter,
        projected_width_mm=width,
        usable_bore_mm=usable_bore,
        fits_bore=fits,
        feasible=fits,
        # The Probe Bore is the only constraint so far: it rules a Design out
        # or, for a Feasible Design, is what limits it first.
        binding_constraint="Probe Bore",
    )
