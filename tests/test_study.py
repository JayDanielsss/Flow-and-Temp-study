"""Run the study: assumptions in, results CSV and plots out."""

import csv
from pathlib import Path

import pytest

from rotor_design.study import run_study

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"


def test_study_on_tiny_grid_writes_csv_and_plot(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[10.0, 20.0])

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
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps=sweeps)

    with open(tmp_path / "nozzle_geometry_sweep.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert {r["swept_variable"] for r in rows} == set(sweeps)
    for r in rows:
        assert float(r["required_mass_flow_kg_s"]) > 0.0
        assert float(r["spin_rate_hz"]) == 1000.0

    assert (tmp_path / "required_flow_vs_nozzle_geometry.png").stat().st_size > 0


def test_study_ranks_feasible_designs_by_sample_heat_load_and_plots_profiles(tmp_path):
    tiny = {"nozzle_count": [4]}
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[8.0, 14.0, 17.0, 20.0], nozzle_sweeps=tiny)

    with open(tmp_path / "ranked_designs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["rank"]) for r in rows] == [1, 2, 3, 4]
    feasible = [r for r in rows if r["feasible"] == "True"]
    loads = [float(r["sample_heat_load_w"]) for r in feasible]
    assert loads == sorted(loads)
    # Infeasible Designs (the 8 mm Rotor has too short a Sample Zone) come after every feasible one.
    flags = [r["feasible"] == "True" for r in rows]
    assert flags == sorted(flags, reverse=True)
    assert "8.0" in {r["rotor_length_mm"] for r in rows if r["feasible"] == "False"}
    assert "heat_turbine_cap_conduction_w" in rows[0]

    assert (tmp_path / "temperature_profiles_top_designs.png").stat().st_size > 0


def test_study_compares_cooling_paths_and_plots_sample_heat_load_against_cold_support_gap(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.01, 0.1, 1.0])

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
              cold_support_gaps_mm=[0.1])

    with open(tmp_path / "bearing_configurations.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["bearing"] for r in rows} == {"gas", "mechanical", "two_center_pivot"}
    assert {r["cooling_path"] for r in rows} == {"vapour", "vapour_and_cold_support"}
    assert all(r["bearing"] == "two_center_pivot" for r in rows if r["pivot_to_bath"] == "True")
    for column in ("bearing_friction_torque_n_m", "bearing_stiffness_n_m", "bearing_radial_space_mm",
                   "heat_bearing_friction_w", "warning"):
        assert column in rows[0]
    assert all(r["warning"] for r in rows if r["bearing"] == "mechanical")


def test_study_compares_turbine_types_on_torque_per_unit_flow(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[15.0, 20.0], nozzle_sweeps={"nozzle_count": [4]},
              cold_support_gaps_mm=[0.1])

    with open(tmp_path / "turbine_comparison.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert sorted((r["turbine"], float(r["rotor_length_mm"])) for r in rows) == [
        ("fluted_cap", 15.0), ("fluted_cap", 20.0), ("radial_inflow", 15.0), ("radial_inflow", 20.0)]
    for r in rows:
        assert float(r["turbine_torque_per_flow_n_m_s_kg"]) > 0.0

    assert (tmp_path / "torque_per_flow_by_turbine_type.png").stat().st_size > 0
