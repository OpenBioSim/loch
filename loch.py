import argparse
import numpy as np
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

        # Generate a random position in the cell.
        if target is not None:
            xyz = target + (2.0 * np.random.rand() - 1.0) * distance * np.array(
                [1, 1, 1]
            )
            for k in range(3):
                if xyz[k] < 0.0:
                    xyz[k] += dimensions[k]
                elif xyz[k] > dimensions[k]:
                    xyz[k] -= dimensions[k]
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
        default=1.0,
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

    # Store the search string.
    if args.target is not None:
        search = f"atoms within {args.cut_off + args.max_distance} of {args.target[0]}, {args.target[1]}, {args.target[2]}"

    # Get the charges on all the atoms.
    try:
        # Loop over the molecules and get the charges.
        if args.target is None:
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
        if args.target is None:
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
        if args.target is None:
            positions = sr.io.get_coords_array(system)
        else:
            search = f"atoms within {args.cut_off + args.max_distance} of {args.target[0]}, {args.target[1]}, {args.target[2]}"
            positions = sr.io.get_coords_array(system[search])

    except Exception as e:
        raise ValueError(f"Could not get the positions of the atoms: {e}")

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
    else:
        energy_kernel = mod.get_function("energy1")

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

        # Print the indices and energy for the 10 lowest energy configurations.
        print(f"Batch {i+1}")
        print("Lowest energy configurations:")
        for j in idxs[:10]:
            print(f"  idx {j}: {energies[j]:.3f} kT")

        # Print the timing for the insertion calculation.
        print(f"Time taken: {1000*(end - start):.2f} ms")

        # If the minimum energy is less than the tolerance, print the water
        # coordinates.
        j = 0
        if energies[idxs[0]] < args.tolerance:
            print(f"Candidate water coordinates:")
        while energies[idxs[j]] < args.tolerance:
            print(f"{waters[j]}")
            j += 1
