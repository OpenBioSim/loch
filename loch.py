import argparse
import numpy as np
import pickle
import time

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import BioSimSpace as BSS
import sire as sr

from _kernels import code


def get_atom_properties(system, search="all"):
    """
    Get the required atomic properties for the atoms in the system.

    Parameters
    ----------

    system: sire.system.System
        The molecular system.

    search: str
        A selection string for the atoms of interest.

    Returns
    -------

    charges: numpy.ndarray
        The charges on the atoms.

    sigmas: numpy.ndarray
        The LJ sigma parameters for the atoms.

    epsilons: numpy.ndarray
        The LJ epsilon parameters for the atoms.

    positions: numpy.ndarray
        The positions of the atoms.
    """

    # Get the charges on all the atoms.
    try:
        charges = []
        for mol in system:
            charges_mol = [charge.value() for charge in mol.property("charge")]
            charges.extend(charges_mol)

        # Convert to a NumPy array.
        charges = np.array(charges)

    except Exception as e:
        raise ValueError(f"Could not get the charges on the atoms: {e}")

    # Try to get the sigma and epsilon for the atoms.
    try:
        sigmas = []
        epsilons = []
        for mol in system:
            for lj in mol.property("LJ"):
                sigmas.append(lj.sigma().value())
                epsilons.append(lj.epsilon().value())

        # Convert to a NumPy array.
        sigmas = np.array(sigmas)
        epsilons = np.array(epsilons)

    except Exception as e:
        raise ValueError(f"Could not get the LJ parameters: {e}")

    # Get the positions of all the atoms.
    try:
        positions = sr.io.get_coords_array(system)

    except Exception as e:
        raise ValueError(f"Could not get the positions of the atoms: {e}")

    return charges, sigmas, epsilons, positions


def create_gpu_memory(
    charges,
    charge_water,
    sigmas,
    sigma_water,
    epsilons,
    epsilon_water,
    positions,
    threads_per_block,
):
    """
    Create the GPU memory for the atomic properties.

    Parameters
    ----------

    charges: numpy.ndarray
        The charges on the atoms.

    charge_water: numpy.ndarray
        The charge on the water atoms.

    sigmas: numpy.ndarray
        The LJ sigma parameters for the atoms.

    sigma_water: numpy.ndarray
        The LJ sigma parameters for the water atoms.

    epsilons: numpy.ndarray
        The LJ epsilon parameters for the atoms.

    epsilon_water: numpy.ndarray
        The LJ epsilon parameters for the water atoms.

    positions: numpy.ndarray
        The positions of the atoms.

    threads_per_block: int
        The number of threads per block.

    Returns
    -------

    charges_gpu: pycuda.gpuarray.GPUArray
        The charges on the atoms.

    charge_water_gpu: pycuda.gpuarray.GPUArray
        The charge on the water atoms.

    sigmas_gpu: pycuda.gpuarray.GPUArray
        The LJ sigma parameters for the atoms.

    sigma_water_gpu: pycuda.gpuarray.GPUArray
        The LJ sigma parameters for the water atoms.

    epsilons_gpu: pycuda.gpuarray.GPUArray
        The LJ epsilon parameters for the atoms.

    epsilon_water_gpu: pycuda.gpuarray.GPUArray
        The LJ epsilon parameters for the water atoms.

    positions_gpu: pycuda.gpuarray.GPUArray
        The positions of the atoms.

    energy_coul: pycuda.gpuarray.GPUArray
        The Coulomb energy.

    energy_lj: pycuda.gpuarray.GPUArray
        The LJ energy.

    probability: pycuda.gpuarray.GPUArray
        The acceptance probability.

    num_atoms: int
        The number of atoms.
    """

    # Place the arrays on the GPU.
    charges_gpu = gpuarray.to_gpu(charges.astype(np.float32))
    charge_water_gpu = gpuarray.to_gpu(np.array(charge_water).astype(np.float32))
    sigmas_gpu = gpuarray.to_gpu(sigmas.astype(np.float32))
    sigma_water_gpu = gpuarray.to_gpu(np.array(sigma_water).astype(np.float32))
    epsilons_gpu = gpuarray.to_gpu(epsilons.astype(np.float32))
    epsilon_water_gpu = gpuarray.to_gpu(np.array(epsilon_water).astype(np.float32))
    positions_gpu = gpuarray.to_gpu(positions.flatten().astype(np.float32))

    # Store the number of atoms.
    num_atoms = len(charges)

    # Create an array to hold the results.
    result_array = np.zeros((1, num_insertions * num_atoms)).astype(np.float32)
    energy_coul = gpuarray.to_gpu(result_array)
    energy_lj = gpuarray.to_gpu(result_array)
    probability = gpuarray.to_gpu(np.zeros((1, num_insertions)).astype(np.float32))

    return (
        charges_gpu,
        charge_water_gpu,
        sigmas_gpu,
        sigma_water_gpu,
        epsilons_gpu,
        epsilon_water_gpu,
        positions_gpu,
        energy_coul,
        energy_lj,
        probability,
        num_atoms,
    )


def evaluate_candidate(system, candidate_position, cutoff, context=None):
    """
    Evaluate the energy of candidate water insertion into the system.

    Parameters
    ----------

    system: sire.system.System
        The molecular system.

    candidate_position: numpy.ndarray
        The coordinates of the water molecule to insert.

    cutoff: float
        The cut-off distance for the non-bonded interactions.

    context: openmm.Context
        The OpenMM context to use.

    Returns
    -------

    energy: float
        The energy of the water insertion, in kcal/mol.

    context: openmm.Context
        The OpenMM context used for the evaluation.
    """

    if context is None:
        # Find the first water in the system.
        water = system["water"][0]

        # Convert the system and water to BioSimSpace objects.
        system_bss = BSS._SireWrappers.System(system._system)
        water_bss = BSS._SireWrappers.Molecule(water)

        # Renumber the water.
        water_bss = water_bss.copy()

        # Update the water coordinates.
        cursor = water_bss._sire_object.cursor()
        for i, atom in enumerate(cursor.atoms()):
            atom["coordinates"] = candidate_position[i].tolist()
        water_bss._sire_object = cursor.commit()

        # Add the water to the system and convert to a Sire object.
        system_bss += water_bss
        system_sire = sr.system.System(system_bss._sire_object)

        # Create a dynamics object.
        d = system_sire.dynamics(
            cutoff=f"{cutoff} A",
            cutoff_type="rf",
        )

        # Get the energy of the system.
        energy = d.current_potential_energy().value()

        # Get the OpenMM context.
        context = d._d._omm_mols

    else:
        from openmm.unit import kilocalorie_per_mole, nanometer, Quantity

        # Get the current positions.
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        # Replace the water coordinates.
        for i in range(3):
            positions[-3 + i] = Quantity(0.1 * candidate_position[i], nanometer)

        # Update the positions.
        context.setPositions(positions)

        # Get the energy.
        energy = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(kilocalorie_per_mole)
        )

    return energy, context


def trial_move(generator, probability, num_insertions):
    """
    Choose a trial move according to the probabilities.

    Parameters
    ----------

    probability: numpy.ndarray
        The probabilities of each move.

    num_insertions: int
        The number of insertions.

    Returns
    -------

    state: int
        The state to move to.
    """

    # Compute the total probability.
    total_probability = np.sum(probability)

    # Add the probability of staying in the same state.
    if total_probability < 1.0:
        probability = np.append(probability, 1.0 - total_probability)
    else:
        probability = np.append(probability, 0.0)

    # Choose a state according to its probability.
    return generator.choice(
        np.arange(num_insertions + 1), p=probability / np.sum(probability)
    )


def random_choice_numba(arr, prob):
    """
    Perform a random choice from an array with a given probability.

    Parameters
    ----------

    arr: numpy.ndarray
        The array of values to sample from.

    prob: numpy.ndarray
        The probabilities for the given samples.

    Returns
    -------

    index: int
        The index of the chosen value.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


def print_energy_components(context):
    """
    Print the energy components of the OpenMM system.

    Parameters
    ----------

    context : openmm.Context
        The current OpenMM context.
    """

    from copy import deepcopy
    import openmm

    # Get the current context and system.
    system = deepcopy(context.getSystem())

    # Add each force to a unique group.
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)

    # Create a new context.
    new_context = openmm.Context(system, deepcopy(context.getIntegrator()))
    new_context.setPositions(context.getState(getPositions=True).getPositions())

    # Process the records.
    print("Energy components:")
    for i, f in enumerate(system.getForces()):
        state = new_context.getState(getEnergy=True, groups={i})
        print(
            f"{f.getName()}: {state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalories_per_mole):.2f} kcal/mol"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCMC benchmarking")
    parser.add_argument("input", help="Input file(s)", type=str, nargs="+")
    parser.add_argument(
        "reference", help="A search string for the reference atoms.", type=str
    )
    parser.add_argument(
        "--num-insertions",
        help="The number of insertions to attempt.",
        type=int,
        default=1000,
        required=False,
    )
    parser.add_argument(
        "--num-batches",
        help="The number of batches to run.",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--num-threads",
        help="The number of threads per block. Must be a multiple of 32.",
        type=int,
        default=1024,
        required=False,
    )
    parser.add_argument(
        "--radius",
        help="Radius of the GCMC region, in Angstrom",
        type=float,
        default=4.0,
        required=False,
    )
    parser.add_argument(
        "--cut-off",
        help="The cut-off distance for the non-bonded interaction, in Angstrom.",
        type=float,
        default=10.0,
        required=False,
    )
    parser.add_argument(
        "--excess-chemical-potential",
        help="The excess chemical potential in kcal/mol.",
        type=float,
        default=-6.09,
        required=False,
    )
    parser.add_argument(
        "--standard-volume",
        help="The standard volume of water in Angstrom^3.",
        type=float,
        default=30.345,
        required=False,
    )
    parser.add_argument(
        "--seed",
        help="The seed for the random number generator.",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--adams-shift",
        help="The Adams shift.",
        type=float,
        default=0.0,
        required=False,
    )

    args = parser.parse_args()

    # Seed the random number generator.
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create a random number generator.
    generator = np.random.Generator(np.random.PCG64())

    # Set the max threads per block.
    threads_per_block = args.num_threads

    # Energy conversion factor.
    beta = 1 / (sr.units.k_boltz.to("kcal/(mol*kelvin)") * 298)

    # Work out the volume of the GCMC sphere.
    volume = (4.0 * np.pi * args.radius**3) / 3.0

    # Work out the Adams value.
    B = (
        beta * args.excess_chemical_potential + np.log(volume / args.standard_volume)
    ) + args.adams_shift

    # Store the exponential of the Adams value.
    exp_B = np.exp(B)

    # Make sure the number of threads per block is a multiple of 32.
    if threads_per_block % 32 != 0:
        raise ValueError("The number of threads per block must be a multiple of 32.")

    # Try to load the system.
    try:
        system = sr.load(*args.input)
    except Exception as e:
        raise IOError(f"Could not load system: {e}")

    # Make sure there are waters.
    try:
        waters = system["water"]
    except Exception as e:
        raise ValuError(f"System does not contain any water molecules!")

    # Get the box.
    try:
        space = system.property("space")
        cell_matrix = space.box_matrix()
        cell_matrix_inverse = cell_matrix.inverse()
        M = cell_matrix.transpose() * cell_matrix

        # Convert to NumPy.
        row0 = [x.value() for x in cell_matrix.row0()]
        row1 = [x.value() for x in cell_matrix.row1()]
        row2 = [x.value() for x in cell_matrix.row2()]
        cell_matrix = np.array([row0, row1, row2])
        row0 = [x.value() for x in cell_matrix_inverse.row0()]
        row1 = [x.value() for x in cell_matrix_inverse.row1()]
        row2 = [x.value() for x in cell_matrix_inverse.row2()]
        cell_matrix_inverse = np.array([row0, row1, row2])
        row0 = [x.value() for x in M.row0()]
        row1 = [x.value() for x in M.row1()]
        row2 = [x.value() for x in M.row2()]
        M = np.array([row0, row1, row2])

        # Convert to GPU memory.
        cell_matrix = gpuarray.to_gpu(cell_matrix.flatten().astype(np.float32))
        cell_matrix_inverse = gpuarray.to_gpu(
            cell_matrix_inverse.flatten().astype(np.float32)
        )
        M = gpuarray.to_gpu(M.flatten().astype(np.float32))
    except Exception as e:
        raise ValuError(f"System does not contain a periodic box information!")

    # Get the initial energy in kT.
    try:
        d = system.dynamics(
            cutoff=f"{args.cut_off} A",
            cutoff_type="rf",
        )
        original_energy = beta * d.current_potential_energy().value()
    except Exception as e:
        raise ValueError(f"Could not get the initial energy: {e}")

    # Get the number of insertions.
    num_insertions = args.num_insertions

    # Use the first water as a template.
    water = waters[0]

    # Get the positions as a numpy array.
    template = gpuarray.to_gpu(
        sr.io.get_coords_array(water).flatten().astype(np.float32)
    )

    # Get the water properties.
    try:
        charge_water = []
        sigma_water = []
        epsilon_water = []
        for atom in water.atoms():
            charge_water.append(atom.charge().value())
            lj = atom.property("LJ")
            sigma_water.append(lj.sigma().value())
            epsilon_water.append(lj.epsilon().value())

    except Exception as e:
        raise ValueError(f"Could not get the atomic properties of the water: {e}")

    # Try to locate the target position.
    try:
        target = np.array(
            [x.value() for x in system[args.reference].atoms()[0].coordinates()]
        )
        target = gpuarray.to_gpu(target.astype(np.float32))
        print(f"Target position: {target}")
    except Exception as e:
        raise ValueError(f"Could not locate the target position: {e}")

    # Create the search string.
    search = (
        f"atoms within {args.cut_off + args.radius} of "
        f"{target[0]}, {target[1]}, {target[2]}"
    )

    # Get the atomic properties.
    start = time.time()
    charges, sigmas, epsilons, positions = get_atom_properties(system, search=search)
    end = time.time()
    print(f"Time taken to get atomic properties: {1000*(end - start):.2f} ms")

    start = time.time()
    # Create the GPU memory.
    (
        charges_gpu,
        charge_water_gpu,
        sigmas_gpu,
        sigma_water_gpu,
        epsilons_gpu,
        epsilon_water_gpu,
        positions_gpu,
        energy_coul,
        energy_lj,
        probability,
        num_atoms,
    ) = create_gpu_memory(
        charges,
        charge_water,
        sigmas,
        sigma_water,
        epsilons,
        epsilon_water,
        positions,
        threads_per_block,
    )
    end = time.time()
    print(f"Time taken to create GPU memory: {1000*(end - start):.2f} ms")

    # Set the dielectric constant.
    dielectric = 78.3

    # Set a null context.
    context = None

    mod = SourceModule(
        code
        % {
            "NUM_POINTS": water.num_atoms(),
            "NUM_WATERS": num_insertions,
            "NUM_ATOMS": num_atoms,
        },
        no_extern_c=True,
    )
    cell_kernel = mod.get_function("setCellMatrix")
    rng_kernel = mod.get_function("initialiseRNG")
    rf_kernel = mod.get_function("setReactionField")
    atom_properties_kernel = mod.get_function("setAtomProperties")
    atom_positions_kernel = mod.get_function("setAtomPositions")
    water_properties_kernel = mod.get_function("setWaterProperties")
    water_kernel = mod.get_function("generateWater")
    energy_kernel = mod.get_function("computeEnergy")
    probability_kernel = mod.get_function("computeAcceptanceProbability")

    # Initialise the cell.
    cell_kernel(cell_matrix, cell_matrix_inverse, M, block=(1, 1, 1), grid=(1, 1, 1))

    # Initialise the random number generator.
    water_blocks = num_insertions // threads_per_block + 1
    rng_kernel(
        gpuarray.to_gpu(
            np.random.randint(np.iinfo(np.int32).max, size=(1, num_insertions)).astype(
                np.int32
            )
        ),
        block=(threads_per_block, 1, 1),
        grid=(water_blocks, 1, 1),
    )

    # Initialise the reaction field.
    rf_kernel(
        np.float32(args.cut_off),
        np.float32(dielectric),
        block=(1, 1, 1),
        grid=(1, 1, 1),
    )

    # Work out the number of blocks to process the atoms.
    atom_blocks = num_atoms // threads_per_block + 1

    # Set the atomic properties.
    atom_properties_kernel(
        charges_gpu,
        sigmas_gpu,
        epsilons_gpu,
        block=(threads_per_block, 1, 1),
        grid=(atom_blocks, 1, 1),
    )

    # Set the atomic positions.
    atom_positions_kernel(
        positions_gpu,
        np.float32(1.0),
        block=(threads_per_block, 1, 1),
        grid=(atom_blocks, 1, 1),
    )

    # Set the water properties.
    water_properties_kernel(
        charge_water_gpu,
        sigma_water_gpu,
        epsilon_water_gpu,
        block=(1, 1, 1),
        grid=(1, 1, 1),
    )

    # Initialise the memory to store the water positions.
    water_positions = gpuarray.empty((num_insertions, 9), np.float32)

    # Loop over the batches.
    for i in range(args.num_batches):
        # Print the batch number.
        print(f"Batch {i+1}")

        # Initialise the water position array.
        start = time.time()
        water_kernel(
            template,
            target,
            np.float32(args.radius),
            water_positions,
            block=(threads_per_block, 1, 1),
            grid=(water_blocks, 1, 1),
        )
        end = time.time()
        print(f"Time taken to generate water positions: {1000*(end - start):.2f} ms")

        # Run the energy calculation.
        start = time.time()
        energy_kernel(
            water_positions,
            energy_coul,
            energy_lj,
            block=(threads_per_block, 1, 1),
            grid=(atom_blocks, num_insertions, 1),
        )
        end = time.time()

        # Print the timing for the insertion calculation.
        print(f"Time taken to evaluate interactions: {1000*(end - start):.2f} ms")

        # Copy the results back to the CPU.
        energy_coul_cpu = energy_coul.get().reshape(num_insertions, num_atoms)
        energy_lj_cpu = energy_lj.get().reshape(num_insertions, num_atoms)

        prefactor = 1.0 / (4.0 * np.pi * sr.units.epsilon0.value())

        # Calculate sum of the Couloumb energy for each water in kT.
        result_coul = beta * prefactor * np.sum(energy_coul_cpu, axis=1)

        # Calculate the sum of the LJ energy for each water in kT.
        result_lj = beta * np.sum(energy_lj_cpu, axis=1)

        # Sum the energies.
        energies = result_coul + result_lj

        # Get the indices of the sorted energies.
        idxs = np.argsort(energies)

        # Get the minumum energy.
        min_energy = energies[idxs[0]]

        # Evaluate the energy for the best candidate insertion.
        try:
            waters = water_positions.get().reshape(num_insertions, 3, 3)
            new_energy, context = evaluate_candidate(
                system, waters[idxs[0]], cutoff=args.cut_off, context=context
            )
            new_energy *= beta
        except Exception as e:
            raise RuntimeError(f"Could not evaluate the candidate insertion: {e}")

        # Print the energies.
        print(
            f"Energies: before {original_energy:.3f} kT, "
            f"after {new_energy:.3f} kT, "
            f"change {new_energy - original_energy:.3f} kT, "
            f"estimated {min_energy:.3f} kT, "
            f"difference {min_energy - (new_energy - original_energy):.3f} kT"
        )

        start = time.time()

        # Compute the acceptance probabilities.
        probability_kernel(
            np.int32(0),
            np.float32(exp_B),
            np.float32(beta),
            energy_coul,
            energy_lj,
            probability,
            block=(threads_per_block, 1, 1),
            grid=(water_blocks, 1, 1),
        )

        end = time.time()

        print(f"Time taken to compute probabilities: {1000*(end - start):.2f} ms")

        start = time.time()

        # Copy the probabilities back to the CPU.
        probability_cpu = probability.get().flatten()

        # Get the new state.
        state = trial_move(generator, probability_cpu, num_insertions)

        end = time.time()

        print(f"Time taken to choose move: {1000*(end - start):.2f} ms")

        if state == num_insertions:
            print("Insertion rejected.")
        else:
            print(f"Insertion accepted: {state}, position {waters[idxs[state]]}")
