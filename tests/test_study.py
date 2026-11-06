"""Run the study: assumptions in, results CSV and plots out."""

import csv
from dataclasses import fields
from pathlib import Path

import pytest

from rotor_design.assumptions import load_assumptions
from rotor_design.evaluate import Design, configurations, largest_rotor_diameter_mm
from rotor_design.study import configuration_name, run_study, study_designs

# Every test here runs the study end to end: excluded from the default run (see README.md).
pytestmark = pytest.mark.slow

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"
# Every Configuration at two Rotor Lengths: the 10 mm Rotor leaves its 5 mm Sample no Placement Window.
TINY_GRID = {"rotor_length_mm": [10.0, 20.0], "nozzle_throat_mm": [0.3], "cold_support_gap_mm": [0.1]}
# The shared study also tries two Nozzle counts: each Rotor then has two Nozzle setups, which tie
# on Sample Heat Load because Nozzle geometry only sets the drive. Of its two Rotor diameters,
# 9 mm is above the bore limit at 20 mm (7.87 mm with pivots) and is skipped there.
SHARED_GRID = {**TINY_GRID, "nozzle_count": [2, 4], "rotor_diameter_mm": [3.5, 9.0]}
# Design fields that set the drive, not the Rotor.
NOZZLE_FIELDS = ("nozzle_count", "nozzle_throat_mm", "nozzle_angle_deg")
# A few Monte Carlo samples on the top Designs keep the study test quick.
TINY_UNCERTAINTY = {"uncertainty_samples": 6, "uncertainty_designs": 2}


def _tiny_run(plots):
    run_study(ASSUMPTIONS_FILE, plots, rotor_lengths_mm=[20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.1], grid=SHARED_GRID, **TINY_UNCERTAINTY)


@pytest.fixture(scope="module")
def tiny_study(tmp_path_factory):
    """Plots folder of one full study over SHARED_GRID, shared by the tests that only read it."""
    plots = tmp_path_factory.mktemp("Plots")
    _tiny_run(plots)
    return plots


def test_study_on_tiny_grid_writes_csv_and_plot(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[10.0, 20.0], grid=TINY_GRID)

    with open(tmp_path / "probe_bore_fit.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [float(r["rotor_length_mm"]) for r in rows] == [10.0, 20.0]
    for column in ("max_rotor_diameter_mm", "projected_width_mm", "fits_bore", "feasible", "binding_constraint"):
        assert column in rows[0]
    # 20 mm hand value from test_geometry; a shorter Rotor allows a wider one.
    assert float(rows[1]["max_rotor_diameter_mm"]) == pytest.approx(7.874, abs=1e-3)
    assert float(rows[0]["max_rotor_diameter_mm"]) > float(rows[1]["max_rotor_diameter_mm"])

    assert (tmp_path / "max_rotor_diameter_vs_rotor_length.png").stat().st_size > 0


def test_study_writes_required_flow_versus_nozzle_geometry(tmp_path):
    sweeps = {"nozzle_count": [2, 4], "nozzle_throat_mm": [0.2, 0.3], "nozzle_angle_deg": [10.0, 20.0]}
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps=sweeps, grid=TINY_GRID)

    with open(tmp_path / "nozzle_geometry_sweep.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert {r["swept_variable"] for r in rows} == set(sweeps)
    for r in rows:
        assert float(r["required_mass_flow_kg_s"]) > 0.0
        assert float(r["spin_rate_hz"]) == 1000.0

    assert (tmp_path / "required_flow_vs_nozzle_geometry.png").stat().st_size > 0


def test_study_plots_temperature_profiles_of_top_designs(tiny_study):
    assert (tiny_study / "temperature_profiles_top_designs.png").stat().st_size > 0


def test_study_compares_cooling_paths_and_plots_sample_heat_load_against_cold_support_gap(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.01, 0.1, 1.0], grid=TINY_GRID)

    with open(tmp_path / "cold_support_gap_sweep.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    # One Vapour Cooling row, then one row per Cold Support gap.
    assert [r["cooling_path"] for r in rows] == ["vapour"] + ["vapour_and_cold_support"] * 3
    assert [float(r["cold_support_gap_mm"]) for r in rows[1:]] == [0.01, 0.1, 1.0]
    # Narrower gaps shear harder. (Little of that heat reaches the Sample now that it stands off the
    # Bottom Cap: the Cold Support takes it straight back.)
    shear = [float(r["cold_support_drag_torque_n_m"]) for r in rows[1:]]
    assert shear == sorted(shear, reverse=True) and shear[-1] > 0.0
    assert all(float(r["heat_cold_support_shear_w"]) >= 0.0 for r in rows[1:])

    assert (tmp_path / "sample_heat_load_vs_cold_support_gap.png").stat().st_size > 0


def test_study_compares_bearing_configurations(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.1], grid=TINY_GRID)

    with open(tmp_path / "bearing_configurations.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["bearing"] for r in rows} == {"gas", "mechanical", "two_center_pivot"}
    assert {r["cooling_path"] for r in rows} == {"vapour", "vapour_and_cold_support"}
    assert {r["exhaust_path"] for r in rows} == {"vented", "ducted"}
    # One row per Configuration: no duplicate pivot rows now the pivot-to-Bath option is gone.
    assert len(rows) == len({(r["bearing"], r["cooling_path"], r["turbine"], r["exhaust_path"]) for r in rows})
    for column in ("bearing_friction_torque_n_m", "bearing_stiffness_n_m", "bearing_radial_space_mm",
                   "heat_bearing_friction_w", "warning", "bath_heat_load_w", "pumping_speed_m3_s",
                   "peak_hoop_stress_mpa", "stress_safety_factor", "first_rigid_body_critical_hz",
                   "second_rigid_body_critical_hz", "bending_critical_hz", "whirl_threshold_hz"):
        assert column in rows[0]
    assert all(r["whirl_threshold_hz"] != "nan" for r in rows if r["bearing"] == "gas")
    assert all(r["warning"] for r in rows if r["bearing"] == "mechanical")


def test_study_compares_turbine_types_on_torque_per_unit_flow(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[15.0, 20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.1], grid=TINY_GRID)

    with open(tmp_path / "turbine_comparison.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert sorted((r["turbine"], float(r["rotor_length_mm"])) for r in rows) == [
        ("fluted_cap", 15.0), ("fluted_cap", 20.0), ("radial_inflow", 15.0), ("radial_inflow", 20.0)]
    for r in rows:
        assert float(r["turbine_torque_per_flow_n_m_s_kg"]) > 0.0

    assert (tmp_path / "torque_per_flow_by_turbine_type.png").stat().st_size > 0


def test_full_study_ranks_only_feasible_designs_with_every_input_and_result(tiny_study):
    with open(tiny_study / "ranked_designs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "the 20 mm Rotor has Feasible Designs"
    assert all(r["feasible"] == "True" for r in rows)
    assert {float(r["rotor_length_mm"]) for r in rows} == {20.0}
    assert [int(r["rank"]) for r in rows] == list(range(1, len(rows) + 1))
    loads = [float(r["sample_heat_load_w"]) for r in rows]
    assert loads == sorted(loads)
    # Tied Designs, such as Nozzle setups of one Rotor, are ranked by lowest Drive Gas flow.
    order = [(float(r["sample_heat_load_w"]), float(r["required_mass_flow_kg_s"])) for r in rows]
    assert order == sorted(order)
    for column in [f.name for f in fields(Design)] + ["configuration", "sample_volume_mm3",
                                                     "sample_heat_load_w", "bath_heat_load_w",
                                                     "binding_constraint", "heat_windage_w",
                                                     "max_rotor_diameter_mm", "probe_bore_margin"]:
        assert column in rows[0]
    assert {r["configuration"] for r in rows} <= {configuration_name(c) for c in configurations()}
    assert {float(r["rotor_diameter_mm"]) for r in rows} == {3.5}
    # The Binding Constraint of a Feasible Design is a physics constraint, never the Probe Bore.
    assert all(r["binding_constraint"] != "Probe Bore" for r in rows)
    assert all(float(r["probe_bore_margin"]) > 0.0 for r in rows)


def test_full_study_maps_the_binding_constraint_for_each_configuration(tiny_study):
    with open(tiny_study / "constraint_maps.csv", newline="") as f:
        cells = list(csv.DictReader(f))
    names = {configuration_name(c) for c in configurations()}
    # Rotor diameter is the map's second axis: one cell per Configuration, Rotor Length and
    # diameter, with no cell for the 9 mm Rotor above the 20 mm bore limit.
    assert sorted((c["configuration"], float(c["rotor_length_mm"]), float(c["rotor_diameter_mm"]))
                  for c in cells) == sorted(
        (n, length, d) for n in names for length, d in ((10.0, 3.5), (10.0, 9.0), (20.0, 3.5)))
    assert all("probe_bore_margin" in c for c in cells)
    # The 10 mm Rotor is ruled out by an empty Placement Window in every Configuration.
    assert {c["binding_constraint"] for c in cells if float(c["rotor_length_mm"]) == 10.0} == {"Placement Window"}
    assert all(c["feasible"] == "False" for c in cells if float(c["rotor_length_mm"]) == 10.0)

    for name in names:
        assert (tiny_study / f"constraint_map_{name}.png").stat().st_size > 0


def test_study_grid_sweeps_rotor_diameter_up_to_the_bore_limit_for_each_rotor_length():
    a = load_assumptions(ASSUMPTIONS_FILE)
    designs = study_designs(a, {"rotor_length_mm": [10.0, 20.0], "rotor_diameter_mm": [3.5, 5.0, 9.0]})
    assert {d.rotor_diameter_mm for d in designs} == {3.5, 5.0, 9.0}
    for d in designs:
        assert d.rotor_diameter_mm <= largest_rotor_diameter_mm(d.rotor_length_mm, d.bearing, a)
    # Two-center pivots allow 7.87 mm at 20 mm; a mechanical bearing's races only 3.87 mm.
    at_20mm = {b: {d.rotor_diameter_mm for d in designs if d.rotor_length_mm == 20.0 and d.bearing == b}
               for b in ("two_center_pivot", "mechanical")}
    assert at_20mm == {"two_center_pivot": {3.5, 5.0}, "mechanical": {3.5}}


def test_study_grid_sweeps_rotor_diameter_as_fractions_of_the_largest_allowed():
    a = load_assumptions(ASSUMPTIONS_FILE)
    designs = study_designs(a, {"rotor_length_mm": [20.0], "rotor_diameter_fraction": [0.5, 1.0, 1.2]})
    assert {d.rotor_diameter_fraction for d in designs} == {0.5, 1.0}
    with pytest.raises(ValueError, match="diameter"):
        study_designs(a, {"rotor_length_mm": [20.0], "rotor_diameter_mm": [5.0], "rotor_diameter_fraction": [0.5]})


def test_full_study_plots_sample_heat_load_trade_off_and_breakdown(tiny_study):
    for plot in ("sample_heat_load_vs_sample_volume.png", "sample_heat_load_breakdown_top_designs.png"):
        assert (tiny_study / plot).stat().st_size > 0


# Quantities given a Monte Carlo band, as Results fields.
BANDED = ("required_supply_pressure_bar", "required_mass_flow_kg_s", "sample_heat_load_w",
          "sample_hottest_temperature_k", "sample_zone_mm", "bath_heat_load_w")


def _rotor(row):
    """A CSV row's Rotor: every Design field except the Nozzle geometry."""
    return tuple(row[f.name] for f in fields(Design) if f.name not in NOZZLE_FIELDS)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_monte_carlo_bands_the_top_designs_and_reports_feasibility_as_a_probability(tiny_study):
    rows = _read(tiny_study / "uncertainty_top_designs.csv")
    ranked = _read(tiny_study / "ranked_designs.csv")
    assert [int(r["rank"]) for r in rows] == [1, 2]
    # The bands go to distinct Rotors, each at its best-ranked Nozzle setup.
    best = {}
    for r in ranked:
        best.setdefault(_rotor(r), r)
    distinct = list(best.values())
    assert [_rotor(r) for r in rows] == [_rotor(r) for r in distinct[:2]]
    assert [r["nozzle_count"] for r in rows] == [r["nozzle_count"] for r in distinct[:2]]
    for r in rows:
        assert int(r["samples"]) == TINY_UNCERTAINTY["uncertainty_samples"]
        assert 0.0 <= float(r["feasible_probability"]) <= 1.0
        for q in BANDED:
            p5, median, p95 = (float(r[f"{q}_{s}"]) for s in ("p5", "median", "p95"))
            assert p5 <= median <= p95, q
        # The conductivity range alone spreads the Sample Heat Load.
        assert float(r["sample_heat_load_w_p5"]) < float(r["sample_heat_load_w_p95"])

    errors = _read(tiny_study / "uncertainty_model_form_error.csv")
    # Every Reference Case mechanism with a comparison feeds a spread into the bands.
    assert {e["mechanism"] for e in errors} == {"nozzle", "windage", "rarefied_drag", "turbine", "conduction"}
    assert all(float(e["rms_error"]) > 0.0 for e in errors)
    # Each is a signed bias applied as a correction and the residual about it as spread.
    assert all(float(e["bias_correction"]) > 0.0 and float(e["residual_spread"]) > 0.0 for e in errors)

    assert (tiny_study / "uncertainty_top_designs.png").stat().st_size > 0


def test_monte_carlo_is_reproducible_with_a_fixed_seed(tiny_study, tmp_path):
    _tiny_run(tmp_path)
    assert _read(tmp_path / "uncertainty_top_designs.csv") == _read(tiny_study / "uncertainty_top_designs.csv")
