import pytest

import sire as sr


@pytest.fixture(scope="session")
def water_box():
    return sr.load_test_files("water_box.prm7", "water_box.rst7")


@pytest.fixture(scope="session")
def bpti():
    return sr.load_test_files("bpti.prm7", "bpti.rst7")
