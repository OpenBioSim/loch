import argparse
import numpy as np
import pickle
import time

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import sire as sr


def uniform_random_rotation(x):
    """
    Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Adapted from:
    https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/

    Parameters
    ----------

        x:  numpy.ndarray
            Vector or set of vectors with dimension (n, 3), where n is the
            number of vectors

    Returns
    -------

        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.

    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([np.cos(x2) * np.sqrt(x3), np.sin(x2) * np.sqrt(x3), np.sqrt(1 - x3)])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    return ((x - mean_coord) @ M) + mean_coord @ M


def generate_waters(template, cell, num_waters, target=None, distance=1.0):
    """
    Generate a set of water positions in a box.

    Parameters
    ----------

    template: numpy.ndarray
        The position of the template water molecule.

    dimensions: numpy.ndarray
        The box cell dimensions.

    num_waters: int
        The number of water molecules to generate.

    target: numpy.ndarray
        The target position to generate the water molecules around.

    distance: float
        The distance in Angstrom around the target position to generate
        the water molecules.

    Returns
    -------

    waters: numpy.ndarray
        The positions of the generated water molecules.
    """

    # Initialize the array to store the water positions.
    waters = np.zeros((num_waters, 3, 3))

    for i in range(num_waters):
        # Water index.
        j = i % 3

        # Copy the template.
        water = template.copy()

        # Translate to the origin.
        water -= template[0]

        # Rotate the water randomly.
        water = uniform_random_rotation(water)

        # Calculate the distance between the oxygen and the hydrogens.
        dh1 = water[1] - water[0]
        dh2 = water[2] - water[0]

        # Generate a random position around the target.
        if target is not None:
            xyz = np.random.randn(3)
            xyz /= np.linalg.norm(xyz)
            r = distance * np.power(np.random.rand(), 1.0 / 3.0)
            xyz = target + r * xyz
        # Generate a random position in the cell.
        else:
            xyz = np.random.rand(3) * dimensions

        # Place the oxygen (first atom) at the random position.
        water[0] = xyz

        # Shift the hydrogens by the appropriate amount.
        water[1] = xyz + dh1
        water[2] = xyz + dh2

        # Store the water in the array.
        waters[i] = water

    return waters


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
        if search == "all":
            charges = []
            for mol in system:
                charges_mol = [charge.value() for charge in mol.property("charge")]
                charges.extend(charges_mol)
        # Loop over the selection to get the charges.
        else:
            charges = [atom.charge().value() for atom in system[search].atoms()]

        # Convert to a NumPy array.
        charges = np.array(charges)

    except Exception as e:
        raise ValueError(f"Could not get the charges on the atoms: {e}")

    # Try to get the sigma and epsilon for the atoms.
    try:
        # Loop over the molecules and get the sigma and epsilon.
        if search == "all":
            sigmas = []
            epsilons = []
            for mol in system:
                for lj in mol.property("LJ"):
                    sigmas.append(lj.sigma().value())
                    epsilons.append(lj.epsilon().value())
        # Loop over the selection to get the sigma and epsilon.
        else:
            sigmas = []
            epsilons = []
            for atom in system[search].atoms():
                lj = atom.property("LJ")
                sigmas.append(lj.sigma().value())
                epsilons.append(lj.epsilon().value())

        # Convert to a NumPy array.
        sigmas = np.array(sigmas)
        epsilons = np.array(epsilons)

    except Exception as e:
        raise ValueError(f"Could not get the LJ parameters: {e}")

    # Get the positions of all the atoms.
    try:
        if search == "all":
            positions = sr.io.get_coords_array(system)
        else:
            positions = sr.io.get_coords_array(system[search])

    except Exception as e:
        raise ValueError(f"Could not get the positions of the atoms: {e}")

    return charges, sigmas, epsilons, positions


def create_gpu_memory(
    num_insertions,
    dimensions,
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

    num_insertions: int
        The number of insertions to attempt.

    dimensions: numpy.ndarray
        The box cell dimensions.

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

    dimensions_gpu: pycuda.gpuarray.GPUArray
        The box cell dimensions.

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

    num_atoms: int
        The number of atoms.

    num_blocks: int
        The number of blocks to use.

    idx_max: int
        The maximum index to use.

    idx_min: int
        The minimum index to use.
    """

    # Place the arrays on the GPU.
    dimensions_gpu = gpuarray.to_gpu(np.array(dimensions).astype(np.float32))
    charges_gpu = gpuarray.to_gpu(charges.astype(np.float32))
    charge_water_gpu = gpuarray.to_gpu(np.array(charge_water).astype(np.float32))
    sigmas_gpu = gpuarray.to_gpu(sigmas.astype(np.float32))
    sigma_water_gpu = gpuarray.to_gpu(np.array(sigma_water).astype(np.float32))
    epsilons_gpu = gpuarray.to_gpu(epsilons.astype(np.float32))
    epsilon_water_gpu = gpuarray.to_gpu(np.array(epsilon_water).astype(np.float32))
    positions_gpu = gpuarray.to_gpu(positions.flatten().astype(np.float32))

    # Store the number of atoms.
    num_atoms = len(charges)

    # Work out the number of blocks to use. We parallelise over the the largest
    # dimension, i.e. atoms or insertions.
    if num_atoms > num_insertions:
        num_blocks = num_atoms // threads_per_block + 1
        idx_max = num_atoms
        idx_min = num_insertions
    else:
        num_blocks = num_insertions // threads_per_block + 1
        idx_max = num_insertions
        idx_min = num_atoms

    # Create an array to hold the results.
    result_array = np.zeros((1, num_insertions * num_atoms)).astype(np.float32)
    energy_coul = gpuarray.to_gpu(result_array)
    energy_lj = gpuarray.to_gpu(result_array)

    return (
        dimensions_gpu,
        charges_gpu,
        charge_water_gpu,
        sigmas_gpu,
        sigma_water_gpu,
        epsilons_gpu,
        epsilon_water_gpu,
        positions_gpu,
        energy_coul,
        energy_lj,
        num_atoms,
        num_blocks,
        idx_max,
        idx_min,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCMC benchmarking")
    parser.add_argument("input", help="Input file(s)", type=str, nargs="+")
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
        "--target",
        help="Coordinates for targetting insertions, in Angstrom",
        type=float,
        nargs=3,
        required=False,
    )
    parser.add_argument(
        "--max-distance",
        help="Maximum distance from the target, in Angstrom",
        type=float,
        default=10.0,
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
        "--tolerance",
        help="The tolerance for the minimum insertion energy, in kT.",
        type=float,
        default=5,
        required=False,
    )
    parser.add_argument(
        "--adaptive",
        help="Adaptively search for the optimal insertion position.",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )

    args = parser.parse_args()

    # Set the max threads per block.
    threads_per_block = args.num_threads

    # Make sure it's a multiple of 32.
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
        dimensions = [x.value() for x in space.dimensions()]
    except Exception as e:
        raise ValuError(f"System does not contain a periodic box information!")

    # Get the number of insertions.
    num_insertions = args.num_insertions

    # Use the first water as a template.
    water = waters[0]

    # Get the positions as a numpy array.
    water_positions = sr.io.get_coords_array(water)

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

    # Create the atom search string.
    if args.target is not None:
        search = (
            f"atoms within {args.cut_off + args.max_distance} of "
            f"{args.target[0]}, {args.target[1]}, {args.target[2]}"
        )
    else:
        search = "all"

    # Get the atomic properties.
    charges, sigmas, epsilons, positions = get_atom_properties(system, search=search)

    # Create the GPU memory.
    (
        dimensions_gpu,
        charges_gpu,
        charge_water_gpu,
        sigmas_gpu,
        sigma_water_gpu,
        epsilons_gpu,
        epsilon_water_gpu,
        positions_gpu,
        energy_coul,
        energy_lj,
        num_atoms,
        num_blocks,
        idx_max,
        idx_min,
    ) = create_gpu_memory(
        num_insertions,
        dimensions,
        charges,
        charge_water,
        sigmas,
        sigma_water,
        epsilons,
        epsilon_water,
        positions,
        threads_per_block,
    )

    # Create a kernel to calculate the non-bonded energy.
    mod = SourceModule(
        """
        __global__ void energy0(
            int idx_max,
            float cutoff,
            float* dimensions,
            float *charges,
            float* charge_water,
            float* sigmas,
            float* sigma_water,
            float* epsilons,
            float* epsilon_water,
            float* positions,
            float* waters,
            float* energy_coul,
            float* energy_lj)
        {
            // Work out the atom index.
            int idx_atom = threadIdx.x + blockDim.x * blockIdx.x;

            // Make sure we're in bounds.
            if (idx_atom < idx_max)
            {
                // Store the squared cut-off distance.
                auto cutoff2 = cutoff * cutoff;

                // Work out the water index.
                int idx_water = blockIdx.y;

                // Work out the index for the result.
                int idx = (idx_water * idx_max) + idx_atom;

                // Get the atom position.
                float x1 = positions[3 * idx_atom];
                float y1 = positions[3 * idx_atom + 1];
                float z1 = positions[3 * idx_atom + 2];

                // Store the charge on the atom.
                auto c11 = charges[idx_atom] * charges[idx_atom];

                // Store the epsilon and sigma for the atom.
                float s1 = sigmas[idx_atom];
                float e1 = epsilons[idx_atom];

                // Zero the energies.
                energy_coul[idx] = 0.0;
                energy_lj[idx] = 0.0;

                // Loop over all atoms in the water molecule.
                for (int i = 0; i < 3; i++)
                {
                    // Get the water atom position.
                    float x2 = waters[9 * idx_water + 3 * i];
                    float y2 = waters[9 * idx_water + 3 * i + 1];
                    float z2 = waters[9 * idx_water + 3 * i + 2];

                    // Calculate the distance.
                    float dx = x1 - x2;
                    float dy = y1 - y2;
                    float dz = z1 - z2;

                    // Apply periodic boundary conditions.
                    if (dx >= 0.5*dimensions[0])
                    {
                        dx -= dimensions[0];
                    }
                    else if (dx < -0.5*dimensions[0])
                    {
                        dx += dimensions[0];
                    }
                    if (dy >= 0.5*dimensions[1])
                    {
                        dy -= dimensions[1];
                    }
                    else if (dy < -0.5*dimensions[1])
                    {
                        dy += dimensions[1];
                    }
                    if (dz >= 0.5*dimensions[2])
                    {
                        dz -= dimensions[2];
                    }
                    else if (dz < -0.5*dimensions[2])
                    {
                        dz += dimensions[2];
                    }

                    // Calculate the distance squared.
                    float r2 = dx * dx + dy * dy + dz * dz;

                    // The distance is within the cut-off.
                    if (r2 < cutoff2)
                    {
                        // Don't divide by zero.
                        if (r2 < 1e-6)
                        {
                            energy_coul[idx] = 1e6;
                        }
                        else
                        {
                            // Accumulate the squared Coulomb energy. We can take
                            // the square root of the total energy and rescale at
                            // the end.
                            auto c2 = charge_water[i];
                            energy_coul[idx] += (c11 * c2*c2) / r2;

                            // Accumulate the LJ energy.
                            auto s2 = sigma_water[i];
                            auto e2 = epsilon_water[i];
                            auto s = 0.5 * (s1 + s2);
                            auto e = 0.5 * (e1 * e2);
                            s2 = s * s;
                            auto sr2 = s2 / r2;
                            auto sr6 = sr2 * sr2 * sr2;
                            auto sr12 = sr6 * sr6;
                            energy_lj[idx] += 4 * e * (sr12 - sr6);
                        }
                    }
                }
            }
        }

        __global__ void energy1(
            int idx_max,
            float cutoff,
            float* dimensions,
            float *charges,
            float* charge_water,
            float* sigmas,
            float* sigma_water,
            float* epsilons,
            float* epsilon_water,
            float* positions,
            float* waters,
            float* energy_coul,
            float* energy_lj)
        {
            // Work out the water index.
            int idx_water = threadIdx.x + blockDim.x * blockIdx.x;

            // Make sure we're in bounds.
            if (idx_water < idx_max)
            {
                // Store the squared cut-off distance.
                auto cutoff2 = cutoff * cutoff;

                // Work out the atom index.
                int idx_atom = threadIdx.y + blockDim.y * blockIdx.y;

                // Work out the index for the result.
                int idx = (idx_water * gridDim.y) + idx_atom;

                // Get the atom position.
                float x1 = positions[3 * idx_atom];
                float y1 = positions[3 * idx_atom + 1];
                float z1 = positions[3 * idx_atom + 2];

                // Store the charge on the atom.
                auto c11 = charges[idx_atom] * charges[idx_atom];

                // Store the epsilon and sigma for the atom.
                float s1 = sigmas[idx_atom];
                float e1 = epsilons[idx_atom];

                // Zero the energies.
                energy_coul[idx] = 0.0;
                energy_lj[idx] = 0.0;

                // Loop over all atoms in the water molecule.
                for (int i = 0; i < 3; i++)
                {
                    // Get the water atom position.
                    float x2 = waters[9 * idx_water + 3 * i];
                    float y2 = waters[9 * idx_water + 3 * i + 1];
                    float z2 = waters[9 * idx_water + 3 * i + 2];

                    // Calculate the distance.
                    float dx = x1 - x2;
                    float dy = y1 - y2;
                    float dz = z1 - z2;

                    // Apply periodic boundary conditions.
                    if (dx >= 0.5*dimensions[0])
                    {
                        dx -= dimensions[0];
                    }
                    else if (dx < -0.5*dimensions[0])
                    {
                        dx += dimensions[0];
                    }
                    if (dy >= 0.5*dimensions[1])
                    {
                        dy -= dimensions[1];
                    }
                    else if (dy < -0.5*dimensions[1])
                    {
                        dy += dimensions[1];
                    }
                    if (dz >= 0.5*dimensions[2])
                    {
                        dz -= dimensions[2];
                    }
                    else if (dz < -0.5*dimensions[2])
                    {
                        dz += dimensions[2];
                    }

                    // Calculate the distance squared.
                    float r2 = dx * dx + dy * dy + dz * dz;

                    // The distance is within the cut-off.
                    if (r2 < cutoff2)
                    {
                        // Don't divide by zero.
                        if (r2 < 1e-6)
                        {
                            energy_coul[idx] = 1e6;
                        }
                        else
                        {
                            // Accumulate the squared Coulomb energy. We can take
                            // the square root of the total energy and rescale at
                            // the end.
                            auto c2 = charge_water[i];
                            energy_coul[idx] += (c11 * c2*c2) / r2;

                            // Accumulate the LJ energy.
                            auto s2 = sigma_water[i];
                            auto e2 = epsilon_water[i];
                            auto s = 0.5 * (s1 + s2);
                            auto e = 0.5 * (e1 * e2);
                            s2 = s * s;
                            auto sr2 = s2 / r2;
                            auto sr6 = sr2 * sr2 * sr2;
                            auto sr12 = sr6 * sr6;
                            energy_lj[idx] += 4 * e * (sr12 - sr6);
                        }
                    }
                }
            }
        }
        """
    )

    # Get the kernel.
    if num_atoms > num_insertions:
        energy_kernel = mod.get_function("energy0")
        kernel = 0
    else:
        energy_kernel = mod.get_function("energy1")
        kernel = 1

    # Set the dielectric constant.
    dielectric = 78.5

    # Energy conversion factor.
    kcal_per_mol_to_kt = 1 / (sr.units.k_boltz.to("kcal/(mol*kelvin)") * 298)

    # Loop over the batches.
    for i in range(args.num_batches):
        # Initialise the water position array.
        try:
            waters = generate_waters(
                water_positions,
                dimensions,
                num_insertions,
                target=args.target,
                distance=args.max_distance,
            )
        except Exception as e:
            raise RuntimeError(f"Could not generate water positions: {e}")

        # Copy the waters to the GPU.
        waters_gpu = gpuarray.to_gpu(waters.flatten().astype(np.float32))

        start = time.time()

        # Run the kernel.
        energy_kernel(
            np.int32(idx_max),
            np.float32(args.cut_off),
            dimensions_gpu,
            charges_gpu,
            charge_water_gpu,
            sigmas_gpu,
            sigma_water_gpu,
            epsilons_gpu,
            epsilon_water_gpu,
            positions_gpu,
            waters_gpu,
            energy_coul,
            energy_lj,
            block=(threads_per_block, 1, 1),
            grid=(num_blocks, idx_min, 1),
        )

        end = time.time()

        # Copy the results back to the CPU.
        energy_coul_cpu = energy_coul.get().reshape(num_insertions, num_atoms)
        energy_lj_cpu = energy_lj.get().reshape(num_insertions, num_atoms)

        # Calculate sum of the Couloumb energy for each water in kT.
        result_coul = (
            kcal_per_mol_to_kt
            * np.sqrt(np.sum(energy_coul_cpu, axis=1))
            / (4 * np.pi * sr.units.epsilon0.value() * dielectric)
        )

        # Calculate the sum of the LJ energy for each water in kT.
        result_lj = kcal_per_mol_to_kt * np.sum(energy_lj_cpu, axis=1)

        # Sum the energies.
        energies = result_coul + result_lj

        # Get the indices of the sorted energies.
        idxs = np.argsort(energies)

        # Get the minumum energy.
        min_energy = energies[idxs[0]]

        # Print the indices and energy for the 10 lowest energy configurations.
        print(f"Batch {i+1}")
        print("Lowest energy configurations:")
        for j in idxs[:10]:
            print(
                f"  idx {j}: Coulomb: {result_coul[j]:.3f} kT, "
                f"LJ: {result_lj[j]:.3f} kT, "
                f"Total: {energies[j]:.3f} kT"
            )

        # Print the timing for the insertion calculation.
        print(f"Time taken: {1000*(end - start):.2f} ms")

        # Store candidate insertions.
        j = 0
        candidates = []
        while energies[idxs[j]] < args.tolerance:
            candidates.append(waters[j])
            j += 1
            if j == num_insertions:
                break
        # Write the candidates to a pickle file and exit.
        if min_energy < args.tolerance:
            print(f"Found {len(candidates)} candidates. Writing to candidates.pkl")
            with open("candidates.pkl", "wb") as f:
                pickle.dump(np.array(candidates), f)
            break

        if args.adaptive and args.num_batches > 1:
            # Update the maximum distance.
            if i > 1:
                energy_difference = min_energy - previous_energy
            else:
                energy_difference = 0
                previous_energy = min_energy

            # Increase or decrease the maximum distance based on the energy
            # difference.
            if energy_difference <= 0:
                if i > 1 and args.max_distance > 0.1:
                    args.max_distance *= 0.5
                if args.max_distance < 0.1:
                    args.max_distance = 0.1

                # Update the target position to the oxgen atom of the lowest
                # energy configuration.
                args.target = waters[idxs[0]][0]

                print(f"New target position: {args.target}")
            else:
                if args.max_distance < 10.0:
                    args.max_distance *= 2.0
                if args.max_distance > 10.0:
                    args.max_distance = 10.0

            print(f"New search distance: {args.max_distance:.2f} Angstrom")

            # Update the atom search string.
            search = (
                f"atoms within {args.cut_off + args.max_distance} of "
                f"{args.target[0]}, {args.target[1]}, {args.target[2]}"
            )

            # Recalculate the the atom positions.
            charges, sigmas, epsilons, positions = get_atom_properties(
                system, search=search
            )

            # Create the GPU memory.
            (
                dimensions_gpu,
                charges_gpu,
                charge_water_gpu,
                sigmas_gpu,
                sigma_water_gpu,
                epsilons_gpu,
                epsilon_water_gpu,
                positions_gpu,
                energy_coul,
                energy_lj,
                num_atoms,
                num_blocks,
                idx_max,
                idx_min,
            ) = create_gpu_memory(
                num_insertions,
                dimensions,
                charges,
                charge_water,
                sigmas,
                sigma_water,
                epsilons,
                epsilon_water,
                positions,
                threads_per_block,
            )

            # Switch the kernel if necessary.
            if num_atoms < num_insertions and kernel == 0:
                energy_kernel = mod.get_function("energy1")
                kernel = 1

            # Store the current minimum energy.
            previous_energy = min_energy
