"""Run the study: assumptions in, results CSV and plots out."""

import csv
from dataclasses import fields
from pathlib import Path

import pytest

from rotor_design.evaluate import Design, configurations
from rotor_design.study import configuration_name, run_study

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"
# Every Configuration at two Rotor Lengths: the 10 mm Rotor is too short for a 5 mm Sample Zone.
TINY_GRID = {"rotor_length_mm": [10.0, 20.0], "nozzle_throat_mm": [0.3], "cold_support_gap_mm": [0.1]}


@pytest.fixture(scope="module")
def tiny_study(tmp_path_factory):
    """Plots folder of one full study over TINY_GRID, shared by the tests that only read it."""
    plots = tmp_path_factory.mktemp("Plots")
    run_study(ASSUMPTIONS_FILE, plots, rotor_lengths_mm=[20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.1], grid=TINY_GRID)
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
    shear = [float(r["heat_cold_support_shear_w"]) for r in rows[1:]]
    assert shear == sorted(shear, reverse=True)

    assert (tmp_path / "sample_heat_load_vs_cold_support_gap.png").stat().st_size > 0


def test_study_compares_bearing_configurations_without_excluded_combinations(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.1], grid=TINY_GRID)

    with open(tmp_path / "bearing_configurations.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["bearing"] for r in rows} == {"gas", "mechanical", "two_center_pivot"}
    assert {r["cooling_path"] for r in rows} == {"vapour", "vapour_and_cold_support"}
    assert {r["exhaust_path"] for r in rows} == {"vented", "ducted"}
    assert all(r["bearing"] == "two_center_pivot" for r in rows if r["pivot_to_bath"] == "True")
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
    for column in [f.name for f in fields(Design)] + ["configuration", "sample_volume_mm3",
                                                     "sample_heat_load_w", "bath_heat_load_w",
                                                     "binding_constraint", "heat_windage_w"]:
        assert column in rows[0]
    assert {r["configuration"] for r in rows} <= {configuration_name(c) for c in configurations()}


def test_full_study_maps_the_binding_constraint_for_each_configuration(tiny_study):
    with open(tiny_study / "constraint_maps.csv", newline="") as f:
        cells = list(csv.DictReader(f))
    names = {configuration_name(c) for c in configurations()}
    # One cell per Configuration, Rotor Length and Nozzle throat.
    assert sorted((c["configuration"], float(c["rotor_length_mm"])) for c in cells) == sorted(
        (n, length) for n in names for length in (10.0, 20.0))
    assert all(float(c["nozzle_throat_mm"]) == 0.3 for c in cells)
    # The 10 mm Rotor is ruled out by too short a Sample Zone in every Configuration.
    assert {c["binding_constraint"] for c in cells if float(c["rotor_length_mm"]) == 10.0} == {"Sample Zone"}
    assert all(c["feasible"] == "False" for c in cells if float(c["rotor_length_mm"]) == 10.0)

    for name in names:
        assert (tiny_study / f"constraint_map_{name}.png").stat().st_size > 0


def test_full_study_plots_sample_heat_load_trade_off_and_breakdown(tiny_study):
    for plot in ("sample_heat_load_vs_sample_volume.png", "sample_heat_load_breakdown_top_designs.png"):
        assert (tiny_study / plot).stat().st_size > 0
