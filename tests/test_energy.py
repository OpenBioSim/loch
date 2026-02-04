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


# Reference energy values captured with seed=42 on the original kernel implementation.
# These anchor the kernel output to exact values so that refactors (e.g. moving from
# __device__ static arrays to buffer arguments) can be validated.
_REFERENCE_ENERGIES = {
    "water_box": {
        "energy_coul": -9.45853172201302,
        "energy_lj": 3.2191088,
    },
    "bpti": {
        "energy_coul": -15.377882774621897,
        "energy_lj": -0.58867246,
    },
}


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
@pytest.mark.parametrize("fixture", ["water_box", "bpti"])
def test_energy_regression(fixture, platform, request):
    """
    Test that kernel energy values are unchanged for a fixed random seed.

    This catches silent numerical changes introduced by kernel refactors.
    """

    # Get the fixture.
    mols, reference = request.getfixturevalue(fixture)

    # Standard lambda schedule.
    schedule = sr.cas.LambdaSchedule.standard_morph()

    # Set the lambda value.
    lambda_value = 0.5

    # Create a GCMC sampler with a fixed seed.
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
        seed=42,
    )

    # Create a dynamics object.
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

    # Loop until we get an accepted insertion move.
    is_accepted = False
    while not is_accepted:
        moves = sampler.move(d.context())
        if len(moves) > 0 and moves[0] == 0:
            is_accepted = True

    # Get the energy components.
    energy_coul = sampler._debug["energy_coul"]
    energy_lj = sampler._debug["energy_lj"]

    # Check against reference values.
    ref = _REFERENCE_ENERGIES[fixture]
    assert math.isclose(
        energy_coul, ref["energy_coul"], abs_tol=1e-4
    ), f"Coulomb energy changed: {energy_coul!r} != {ref['energy_coul']!r}"
    assert math.isclose(
        energy_lj, ref["energy_lj"], abs_tol=1e-4
    ), f"LJ energy changed: {energy_lj!r} != {ref['energy_lj']!r}"


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_cached_kernel_correctness(platform, water_box):
    """
    A second sampler using cached kernels must produce the same energies
    as the first.
    """

    mols, reference = water_box

    schedule = sr.cas.LambdaSchedule.standard_morph()

    def _create_and_run(seed):
        sampler = GCMCSampler(
            mols,
            cutoff_type="rf",
            cutoff="10 A",
            reference=reference,
            lambda_schedule=schedule,
            lambda_value=0.5,
            log_level="debug",
            ghost_file=None,
            log_file=None,
            test=True,
            platform=platform,
            seed=seed,
        )

        d = sampler.system().dynamics(
            cutoff_type="rf",
            cutoff="10 A",
            temperature="298 K",
            pressure=None,
            constraint="h_bonds",
            timestep="2 fs",
            schedule=schedule,
            lambda_value=0.5,
            coulomb_power=sampler._coulomb_power,
            shift_coulomb=str(sampler._shift_coulomb),
            shift_delta=str(sampler._shift_delta),
            platform=platform,
        )

        is_accepted = False
        while not is_accepted:
            moves = sampler.move(d.context())
            if len(moves) > 0 and moves[0] == 0:
                is_accepted = True

        return sampler

    # Clear the cache so the first sampler compiles from source.
    if platform == "cuda":
        from loch._platforms._cuda import CUDAPlatform

        CUDAPlatform.clear_cache()
    else:
        from loch._platforms._opencl import OpenCLPlatform

        OpenCLPlatform.clear_cache()

    # First sampler compiles kernels, second uses the cache.
    # Both use the same seed so random water positions are identical.
    sampler1 = _create_and_run(seed=42)
    sampler2 = _create_and_run(seed=42)

    # Verify cache behaviour.
    assert not sampler1._kernel_cache_hit, "First sampler should compile from source"
    assert sampler2._kernel_cache_hit, "Second sampler should use cached kernels"

    # Verify energy consistency.
    energy1_coul = sampler1._debug["energy_coul"]
    energy1_lj = sampler1._debug["energy_lj"]
    energy2_coul = sampler2._debug["energy_coul"]
    energy2_lj = sampler2._debug["energy_lj"]

    assert math.isclose(
        energy1_coul, energy2_coul, abs_tol=1e-4
    ), f"Coulomb energy mismatch: {energy1_coul!r} vs {energy2_coul!r}"
    assert math.isclose(
        energy1_lj, energy2_lj, abs_tol=1e-4
    ), f"LJ energy mismatch: {energy1_lj!r} vs {energy2_lj!r}"
