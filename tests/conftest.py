import os

from rotor_design import study


def pytest_configure(config):
    # pytest's main module is a file, so worker processes are safe here: the slow study tests
    # use every core, as run_study.py does.
    study.DEFAULT_WORKERS = os.cpu_count() or 1
