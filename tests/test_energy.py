import math
import openmm
import openmm.unit as omm_unit
import os
import pytest

import numpy as np
import sire as sr

from loch import GCMCSampler, SoftcoreForm


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
@pytest.mark.parametrize(
    "fixture,softcore_form",
    [
        ("water_box", "zacharias"),
        ("bpti", "zacharias"),
        ("sd12", "zacharias"),
        ("sd12", "taylor"),
        ("sd12", "beutler"),
    ],
)
def test_energy(fixture, softcore_form, platform, request):
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
        softcore_form=softcore_form,
        # Sample within the region when there is one. Without a reference
        # every move is a bulk move regardless.
        bulk_sampling_probability=0.0 if reference is not None else 0.1,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
    )

    # Build map of extra options for the dynamics object.
    dyn_map = {}
    if sampler._softcore_form == SoftcoreForm.TAYLOR:
        dyn_map["use_taylor_softening"] = True
    elif sampler._softcore_form == SoftcoreForm.BEUTLER:
        dyn_map["use_beutler_softening"] = True
        dyn_map["beutler_alpha"] = sampler._beutler_alpha

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
        shift_coulomb=str(sampler._shift_coulomb),
        shift_delta=str(sampler._shift_delta),
        platform=platform,
        map=dyn_map,
    )

    # Empty the region, since at equilibrium it is full and insertions into it
    # are rejected. A no-op when there is no region.
    sampler.delete_waters(d.context())

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
    assert relative_diff < 0.001, (
        f"Platform energies differ: CUDA={cuda_energy:.6f}, OpenCL={opencl_energy:.6f}, relative_diff={relative_diff:.6f}"
    )


# Reference energy values captured with seed=42.
# These anchor the kernel output to exact values so that refactors (e.g. moving from
# __device__ static arrays to buffer arguments) can be validated.
_REFERENCE_ENERGIES = {
    "water_box": {
        "energy_coul": -4.674884,
        "energy_lj": 0.82380486,
    },
    "bpti": {
        "energy_coul": -13.205343,
        "energy_lj": 5.061536,
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
    assert math.isclose(energy_coul, ref["energy_coul"], abs_tol=1e-4), (
        f"Coulomb energy changed: {energy_coul!r} != {ref['energy_coul']!r}"
    )
    assert math.isclose(energy_lj, ref["energy_lj"], abs_tol=1e-4), (
        f"LJ energy changed: {energy_lj!r} != {ref['energy_lj']!r}"
    )


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

    assert math.isclose(energy1_coul, energy2_coul, abs_tol=1e-4), (
        f"Coulomb energy mismatch: {energy1_coul!r} vs {energy2_coul!r}"
    )
    assert math.isclose(energy1_lj, energy2_lj, abs_tol=1e-4), (
        f"LJ energy mismatch: {energy1_lj!r} vs {energy2_lj!r}"
    )


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_update_system_insertion(platform, water_box):
    """
    After an accepted insertion of a ghost buffer water, update_system() must:
      - restore real LJ/charge parameters on the inserted water, and
      - reflect the current OpenMM coordinates for that water.
    """
    mols, reference = water_box

    sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
    )

    d = sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
        platform=platform,
    )

    ctx = d.context()
    first_ghost_buffer = sampler._num_waters - sampler._num_ghost_waters

    # Loop until a ghost buffer water is inserted.
    inserted_idx = None
    while inserted_idx is None:
        state_before = sampler.water_state()
        moves = sampler.move(ctx)
        if len(moves) == 0 or moves[0] != 0:
            continue
        state_after = sampler.water_state()
        # Find a buffer-slot water that changed from 0 → 1.
        for i in range(first_ghost_buffer, sampler._num_waters):
            if state_before[i] == 0 and state_after[i] == 1:
                inserted_idx = i
                break

    assert inserted_idx is not None, "No buffer water was inserted"

    # Call update_system and get the returned system.
    system = sampler.update_system(ctx)

    # Get the sire atom index (no vsite offset) of the inserted water's oxygen.
    o_idx = int(sampler._water_indices_sire[inserted_idx])
    inserted_mol = system[system.atoms()[o_idx].molecule()]

    # Check parameters are real (non-zero).
    for j, atom in enumerate(inserted_mol.atoms()):
        charge = atom.property("charge").value()
        lj = atom.property("LJ")
        epsilon = lj.epsilon().value()

        assert charge == pytest.approx(sampler._water_charge[j], abs=1e-6), (
            f"Atom {j} charge not restored: got {charge}, expected {sampler._water_charge[j]}"
        )
        assert epsilon == pytest.approx(sampler._water_epsilon[j], abs=1e-6), (
            f"Atom {j} epsilon not restored: got {epsilon}, expected {sampler._water_epsilon[j]}"
        )

    # Check coordinates match the OpenMM context.
    omm_positions = (
        ctx.getState(getPositions=True, enforcePeriodicBox=False)
        .getPositions(asNumpy=True)
        .value_in_unit(omm_unit.angstrom)
    )

    # _water_indices uses the vsite-offset index into the OpenMM atom list.
    omm_o_idx = int(sampler._water_indices[inserted_idx])

    for j, atom in enumerate(inserted_mol.atoms()):
        sire_coord = atom.property("coordinates")
        sire_xyz = np.array(
            [sire_coord.x().value(), sire_coord.y().value(), sire_coord.z().value()]
        )
        omm_xyz = omm_positions[omm_o_idx + j]
        assert np.allclose(sire_xyz, omm_xyz, atol=1e-3), (
            f"Atom {j} coordinates mismatch: sire={sire_xyz}, omm={omm_xyz}"
        )


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_finalise_system(platform, water_box):
    """
    After an accepted insertion of a ghost buffer water, finalise_system() must
    return a system whose water count equals the number of real (non-ghost)
    waters in the sampler.
    """
    mols, reference = water_box

    sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
    )

    d = sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
        platform=platform,
    )

    ctx = d.context()
    first_ghost_buffer = sampler._num_waters - sampler._num_ghost_waters

    # Loop until a ghost buffer water is inserted.
    inserted = False
    while not inserted:
        state_before = sampler.water_state()
        moves = sampler.move(ctx)
        if len(moves) == 0 or moves[0] != 0:
            continue
        state_after = sampler.water_state()
        for i in range(first_ghost_buffer, sampler._num_waters):
            if state_before[i] == 0 and state_after[i] == 1:
                inserted = True
                break

    system = sampler.finalise_system(ctx)

    expected_waters = int(sampler._get_non_ghost_waters().size)
    actual_waters = system["water"].num_molecules()

    assert actual_waters == expected_waters, (
        f"Water count mismatch: got {actual_waters}, expected {expected_waters}"
    )


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_set_lambda_uploads_parameters(sd12, platform):
    """
    Test that set_lambda() uploads the parameters for the new lambda value.

    A sampler may be re-used across lambda values, re-uploading the lambda
    dependent non-bonded parameters rather than rebuilding a context. If the
    upload were skipped, or the wrong entry taken from the cache, the sampler
    would evaluate insertion and deletion energies against the parameters of
    the lambda value it was built at, while reporting the new one.
    """

    mols, reference = sd12

    # The end states, so that the parameters differ as much as they can.
    build_lambda = 0.0
    run_lambda = 1.0

    sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        lambda_schedule=sr.cas.LambdaSchedule.standard_morph(),
        lambda_value=build_lambda,
        lambda_values=[run_lambda],
        log_level="error",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
    )

    assert sampler._is_fep, "the test system must be perturbable"

    names = ("_gpu_charge", "_gpu_sigma", "_gpu_epsilon", "_gpu_alpha")
    built = sampler._lambda_params[(build_lambda, sampler._rest2_scale)]
    wanted = sampler._lambda_params[(run_lambda, sampler._rest2_scale)]

    # The two end points must differ, otherwise the test cannot tell whether
    # the upload happened.
    assert any(not np.allclose(a, b) for a, b in zip(built, wanted)), (
        "the parameters are the same at both lambda values"
    )

    sampler.set_lambda(run_lambda)
    assert sampler._lambda_value == run_lambda

    sampler.push()
    try:
        for name, expected in zip(names, wanted):
            on_gpu = np.asarray(sampler._backend.from_gpu(getattr(sampler, name)))
            assert np.allclose(on_gpu, expected), (
                f"{name} does not hold the parameters for lambda {run_lambda}"
            )
    finally:
        sampler.pop()


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_energy_after_set_lambda(sd12, platform):
    """
    Test that the RF energy difference agrees with OpenMM after the sampler has
    been switched to a different lambda value.

    This checks the uploaded parameters through the physics rather than by
    inspecting them, so it also covers the kernel using them correctly.

    The move has to happen where the perturbation is. Bulk sampling would
    place the water anywhere in the box, typically tens of Angstrom from the
    perturbable molecule, where the lambda dependent parameters contribute
    nothing and the comparison holds however wrong they are. The sphere is
    emptied first, since at equilibrium it is full and insertions into it are
    rejected.
    """

    mols, reference = sd12

    schedule = sr.cas.LambdaSchedule.standard_morph()

    # The end states, so that the parameters differ as much as they can.
    build_lambda = 0.0
    run_lambda = 1.0

    sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        reference=reference,
        lambda_schedule=schedule,
        lambda_value=build_lambda,
        lambda_values=[run_lambda],
        bulk_sampling_probability=0.0,
        log_level="debug",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
    )

    sampler.set_lambda(run_lambda)

    d = sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure=None,
        constraint="h_bonds",
        timestep="2 fs",
        schedule=schedule,
        lambda_value=run_lambda,
        shift_coulomb=str(sampler._shift_coulomb),
        shift_delta=str(sampler._shift_delta),
        platform=platform,
    )

    def potential_energy():
        return (
            d.context()
            .getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(openmm.unit.kilocalories_per_mole)
        )

    # Empty the sphere so that the next accepted move is an insertion into it.
    sampler.delete_waters(d.context())

    for _ in range(50):
        initial_energy = potential_energy()
        moves = sampler.move(d.context())
        if moves:
            break
    else:
        pytest.fail("no GCMC move was accepted")

    energy_difference = potential_energy() - initial_energy
    sampler_energy = sampler._debug["energy_coul"] + sampler._debug["energy_lj"]

    # The move must be near the atoms whose parameters perturb, otherwise the
    # comparison cannot see them.
    charges0 = np.asarray(sampler._lambda_params[(build_lambda, 1.0)][0])
    charges1 = np.asarray(sampler._lambda_params[(run_lambda, 1.0)][0])
    changed = np.where(np.abs(charges0 - charges1) > 1e-9)[0]
    positions = (
        d.context().getState(getPositions=True).getPositions(asNumpy=True)
        / omm_unit.angstrom
    )
    oxygen = positions[sampler._water_indices[sampler._debug["idx"]]]
    distances = np.linalg.norm(positions[changed] - oxygen, axis=1)
    assert (distances <= 10).any(), (
        "no perturbing atom within the cutoff of the move, so the lambda "
        "dependent parameters are not being tested"
    )

    assert math.isclose(energy_difference, sampler_energy, abs_tol=1e-2)
