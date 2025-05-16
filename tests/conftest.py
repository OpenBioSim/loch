import pytest

import sire as sr


@pytest.fixture(scope="session")
def water_box():
    """Bulk water box."""
    return (
        sr.load_test_files("water_box.prm7", "water_box.rst7"),
        None,
    )


@pytest.fixture(scope="session")
def bpti():
    """Bovine pancreatic trypsin inhibitor."""
    return (
        sr.load_test_files("bpti.prm7", "bpti.rst7"),
        "(resnum 10 and atomname CA) or (resnum 43 and atomname CA)",
    )


@pytest.fixture(scope="session")
def sd12():
    """Scytalone dehydratase system with pertubation between ligands 1 and 2."""
    mols = sr.load_test_files("sd12.bss")
    return (
        sr.morph.link_to_reference(mols),
        "(residx 22 or residx 42) and (atomname OH)",
    )
