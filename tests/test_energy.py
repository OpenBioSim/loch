import math
import openmm
import pytest
import socket

from loch import GCMCSampler


@pytest.mark.skipif(
    socket.gethostname() != "porridge",
    reason="Local test requiring CUDA enabled GPU.",
)
def test_energy(water_box):
    """
    Test that the RF energy difference agrees with OpenMM.
    """
    # Create a GCMC sampler.
    sampler = GCMCSampler(water_box, reference=None, log_level="debug")

    # Create a dynamics object using the modified GCMC system.
    d = sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
    )

    # Get the context.
    context = d.context()

    # Store the inital energy in kcal/mol.
    initial_energy = (
        d.context()
        .getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(openmm.unit.kilocalories_per_mole)
    )

    # Loop until we accept an insertion move.
    is_accepted = False
    while not is_accepted:
        # Perform a GCMC move.
        context, move, is_accepted = sampler.insertion_move(d.context())

    # Store the final energy in kcal/mol.
    final_energy = (
        d.context()
        .getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(openmm.unit.kilocalories_per_mole)
    )

    # Get the debugging information.
    sampler_energy = sampler._debug["energy_coul"] + sampler._debug["energy_lj"]

    # Calculate the energy difference.
    energy_difference = final_energy - initial_energy

    # Check that the energy difference is close to the calculated energy change.
    assert math.isclose(energy_difference, sampler_energy, abs_tol=1e-2)

    # Re-set the initial energy.
    initial_energy = final_energy

    # Loop until we accept a deletion move.
    is_accepted = False
    while not is_accepted:
        # Perform a GCMC move.
        context, move, is_accepted = sampler.deletion_move(d.context())

    # Store the final energy in kcal/mol.
    final_energy = (
        d.context()
        .getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(openmm.unit.kilocalories_per_mole)
    )

    # Get the debugging information.
    sampler_energy = sampler._debug["energy_coul"] + sampler._debug["energy_lj"]

    # Calculate the energy difference.
    energy_difference = final_energy - initial_energy

    # Check that the energy difference is close to the calculated energy change.
    assert math.isclose(energy_difference, sampler_energy, abs_tol=1e-2)
