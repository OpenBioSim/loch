import math
import openmm
import os
import pytest

import sire as sr

from loch import GCMCSampler


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
@pytest.mark.parametrize("fixture", ["water_box", "bpti", "sd12"])
def test_energy(fixture, platform, request):
    """
    Test that the RF energy difference agrees with OpenMM.
    """

    # Get the fixture.
    mols, reference = request.getfixturevalue(fixture)

    # Standard lambda schedule.
    schedule = sr.cas.LambdaSchedule.standard_morph()

    # Set the lambda value.
    lambda_value = 0.5

    # Create a GCMC sampler.
    sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        lambda_schedule=schedule,
        lambda_value=lambda_value,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
    )

    # Create a dynamics object using the modified GCMC system.
    d = sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
        schedule=schedule,
        lambda_value=lambda_value,
        coulomb_power=sampler._coulomb_power,
        shift_coulomb=str(sampler._shift_coulomb),
        shift_delta=str(sampler._shift_delta),
        platform=platform,
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


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("fixture", ["water_box", "bpti", "sd12"])
def test_platform_consistency(fixture, request):
    """
    Test that CUDA and OpenCL platforms produce consistent energy calculations.
    """

    # Get the fixture.
    mols, reference = request.getfixturevalue(fixture)

    # Standard lambda schedule.
    schedule = sr.cas.LambdaSchedule.standard_morph()

    # Set the lambda value.
    lambda_value = 0.5

    # Use a fixed seed for reproducibility
    seed = 42

    # Create CUDA sampler.
    cuda_sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        lambda_schedule=schedule,
        lambda_value=lambda_value,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform="cuda",
        seed=seed,
    )

    # Create OpenCL sampler with same configuration.
    opencl_sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        lambda_schedule=schedule,
        lambda_value=lambda_value,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform="opencl",
        seed=seed,
    )

    # Perform insertion moves on both samplers.
    # With same seed, both platforms should generate identical random numbers
    # and thus identical water positions, allowing direct energy comparison.

    # Create dynamics objects for both.
    cuda_d = cuda_sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
        schedule=schedule,
        lambda_value=lambda_value,
        coulomb_power=cuda_sampler._coulomb_power,
        shift_coulomb=str(cuda_sampler._shift_coulomb),
        shift_delta=str(cuda_sampler._shift_delta),
        platform="cuda",
    )

    opencl_d = opencl_sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
        schedule=schedule,
        lambda_value=lambda_value,
        coulomb_power=opencl_sampler._coulomb_power,
        shift_coulomb=str(opencl_sampler._shift_coulomb),
        shift_delta=str(opencl_sampler._shift_delta),
        platform="opencl",
    )

    # Perform moves until we get an accepted insertion on CUDA.
    is_accepted = False
    while not is_accepted:
        moves = cuda_sampler.move(cuda_d.context())
        if len(moves) > 0 and moves[0] == 0:
            is_accepted = True

    # Get CUDA energy calculation.
    cuda_energy = cuda_sampler._debug["energy_coul"] + cuda_sampler._debug["energy_lj"]

    # Perform moves until we get an accepted insertion on OpenCL.
    is_accepted = False
    while not is_accepted:
        moves = opencl_sampler.move(opencl_d.context())
        if len(moves) > 0 and moves[0] == 0:
            is_accepted = True

    # Get OpenCL energy calculation.
    opencl_energy = (
        opencl_sampler._debug["energy_coul"] + opencl_sampler._debug["energy_lj"]
    )

    # With same seed, the water positions should be identical, so energies
    # should match closely. Allow small tolerance for floating point differences.
    assert math.isfinite(cuda_energy), "CUDA energy is not finite"
    assert math.isfinite(opencl_energy), "OpenCL energy is not finite"

    # Energy calculations should be very close (within 0.1%)
    relative_diff = abs(cuda_energy - opencl_energy) / max(
        abs(cuda_energy), abs(opencl_energy), 1.0
    )
    assert (
        relative_diff < 0.001
    ), f"Platform energies differ: CUDA={cuda_energy:.6f}, OpenCL={opencl_energy:.6f}, relative_diff={relative_diff:.6f}"
