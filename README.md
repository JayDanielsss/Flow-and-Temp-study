This is a study to determine the limits of a spining rotor at high speeds in a cryogenic environment. 

The Bath is pumped to a Bath Temperature of 1.0 K; its vapour pressure (about 0.12 Torr) follows from the helium-4 vapour-pressure curve. The Sample's hottest point must stay at or below the Sample Limit Temperature of 1.2 K. 
The length of the rotor is limited to 20 mm. The turbine that drives the rotation is limited to a radial turbine to conserve space. 
A high flow gas stream of helium is cooled close to liquid temperature, 10K. The flow rate is varible. 

The turbine sits on top of the rotor. A gas bearing using cooled gas or a mechanical bearing is being explored or simple support between two very small centers is being explored. 

The goal is to reach an angular rotational speed of 1 kHz with minimal temperature gradiant. A portion of the rotor must remain 1.2K. 

Goals to determine:
Maximum diameter of turbine and rotor needed to reach 1kHz.
limits on turbine geometery. 
Max mass flow rate of the helium to achieve 1kHz.
limits on Nozzle geometery.
The max heat load produced from the high speed rotation along the rotor.
The amount of space along the rotor that can maintain 1.2K.
The amount of torque produced and stress along the 20 mm rotor. 

The goal is make the right assumptions to understand the limits for design purposes. 

The following will use approximations and assumptions to get a best start. 


## Running the study and the tests

`python run_study.py` runs the whole study on every core; outputs go to `Plots/`. Called from
Python, `run_study`, `evaluate_all` and `evaluate_samples` use one process unless given
`workers=N` (worker processes fail when called from stdin or an interactive session).

`pytest` runs the fast tests (a few minutes). The tests marked `slow` run the study end to end
and take much longer: run them alone with `pytest -m slow`, or the full suite with `pytest -m ""`.
