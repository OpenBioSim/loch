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
    sampler = GCMCSampler(
        water_box,
        cutoff_type="rf",
        cutoff="10 A",
        reference=None,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
    )

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

    # Loop until we accept an insertion move.
    is_accepted = False
    while not is_accepted:
        # Store the initial energy in kcal/mol.
        initial_energy = (
            d.context()
            .getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(openmm.unit.kilocalories_per_mole)
        )

        # Perform a GCMC move.
        moves = sampler.move(d.context())

        # No moves were made.
        if len(moves) == 0:
            is_accepted = False
        else:
            # Deletion move.
            if moves[0] != 0:
                is_accepted = False
            # Insertion move.
            else:
                is_accepted = True

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

    # Loop until we accept a deletion move.
    is_accepted = False
    while not is_accepted:
        # Store the initial energy in kcal/mol.
        initial_energy = (
            d.context()
            .getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(openmm.unit.kilocalories_per_mole)
        )

        # Perform a GCMC move.
        moves = sampler.move(d.context())

        # No moves were made.
        if len(moves) == 0:
            is_accepted = False
        else:
            # Insertion move.
            if moves[0] != 1:
                is_accepted = False
            # Deletion move.
            else:
                is_accepted = True

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
