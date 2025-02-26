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
    waters = np.zeros((num_waters, 3))

    # Calculate the displacements of the hydrogens from the oxygen.
    dh1 = template[1] - template[0]
    dh2 = template[2] - template[0]

    for i in range(num_waters):
        # Water index.
        j = i % 3

        # Copy the template.
        water = template.copy()

        # Translate to the origin.
        water -= template[0]

        # Rotate the water randomly.
        water = uniform_random_rotation(water)

        # Generate a random position in the cell.
        if target is not None:
            xyz = target + np.random.rand(3) * distance
        else:
            xyz = np.random.rand(3) * dimensions

        # Place the oxygen (first atom) at the random position.
        water[0] = xyz

        # Shift the hydrogens by the appropriate amount.
        water[1] = xyz + dh1
        water[2] = xyz + dh2

        # Store the water in the array.
        waters[i] = water[0]

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

    args = parser.parse_args()

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

    # Get the charge on the water.
    try:
        charge_water = [charge.value() for charge in water.property("charge")]
        charge_water = np.tile(charge_water, (num_insertions, 1))
    except Exception as e:
        raise ValueError(f"Could not get the charge on the water: {e}")

    # Get the charges on all the atoms.
    try:
        # Loop over the molecules and get the charges.
        charges = []
        for mol in system:
            charges_mol = [charge.value() for charge in mol.property("charge")]
            charges.extend(charges_mol)
        # Convert to a NumPy array.
        charges = np.array(charges)
    except Exception as e:
        raise ValueError(f"Could not get the charges on the atoms: {e}")

    # Get the positions of all the atoms.
    try:
        positions = sr.io.get_coords_array(system)
    except Exception as e:
        raise ValueError(f"Could not get the positions of the atoms: {e}")

    # Place the arrays on the GPU.
    dimensions_gpu = gpuarray.to_gpu(np.array(dimensions).astype(np.float32))
    charges_gpu = gpuarray.to_gpu(charges.astype(np.float32))
    charge_water_gpu = gpuarray.to_gpu(np.array(charge_water).astype(np.float32))
    positions_gpu = gpuarray.to_gpu(positions.flatten().astype(np.float32))
    waters_gpu = gpuarray.to_gpu(waters.flatten().astype(np.float32))

    # Store the number of atoms.
    num_atoms = len(charges)

    # Set the max threads per block.
    max_threads_per_block = 1024

    # Work out the number of blocks to use.
    num_blocks = num_atoms // max_threads_per_block + 1

    # Create an array to hold the results.
    result = np.zeros((1, num_insertions * num_atoms)).astype(np.float32)
    result = gpuarray.to_gpu(result)

    # Create a kernel to calculate the Coulomb energy.
    mod = SourceModule(
        """
        __global__ void coulomb_energy(
            int num_atoms,
            float* dimensions,
            float *charges,
            float* charge_water,
            float* positions,
            float* waters,
            float* result)
        {
            // Work out the atom index.
            int idx_atom = gridDim.x * blockIdx.x + threadIdx.x;

            // Make sure we're in bounds.
            if (idx_atom < num_atoms)
            {
                // Work out the water index.
                int idx_water = blockIdx.y;

                // Work out the index for the result.
                int idx = (idx_water * num_atoms) + idx_atom;

                // Get the atom position.
                float x1 = positions[3 * idx_atom];
                float y1 = positions[3 * idx_atom + 1];
                float z1 = positions[3 * idx_atom + 2];

                // Get the water position.
                float x2 = waters[3 * idx_water];
                float y2 = waters[3 * idx_water + 1];
                float z2 = waters[3 * idx_water + 2];

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

                // Don't divide by zero.
                if (r2 < 1e-6)
                {
                    result[idx] = 1e6;
                }
                else
                {
                    // Calculate the squared energy. We can take the square root of the total
                    // energy and rescale at the end.
                    auto c1 = charges[idx_atom];
                    auto c2 = charge_water[idx_water];
                    result[idx] = (c1*c1 * c2*c2) / r2;
                }
            }
        }
        """
    )

    # Get the kernel.
    coulomb_energy = mod.get_function("coulomb_energy")

    start = time.time()

    # Run the kernel.
    coulomb_energy(
        np.int32(num_atoms),
        dimensions_gpu,
        charges_gpu,
        charge_water_gpu,
        positions_gpu,
        waters_gpu,
        result,
        block=(max_threads_per_block, 1, 1),
        grid=(num_blocks, num_insertions, 1),
    )

    end = time.time()

    # Copy the results back.
    result = result.get()

    # Get a view of the results.
    result = result.reshape(num_insertions, num_atoms)

    # Calculate the energies.
    energies = np.sum(result, axis=1) / (sr.units.epsilon0.value() * 4 * np.pi)

    # Print the indices and energy for the 10 lowest energy configurations.
    print("Lowest energy configurations:")
    for i in np.argsort(energies)[:10]:
        print(f"  idx {i}: {energies[i]:.3f} kcal/mol")

    # Print the timing for the insertion calculation.
    print(f"Time taken: {1000*(end - start):.2f} ms")
