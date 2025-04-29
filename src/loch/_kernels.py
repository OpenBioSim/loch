######################################################################
# Loch: GPU accelerated GCMC water sampling engine.
#
# Copyright: 2025
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# Loch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Loch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Loch. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""
GCMC CUDA kernels.
"""

code = """
    #include <curand_kernel.h>

    // Constants.
    const float pi = 3.14159265359f;
    const int num_points = %(NUM_POINTS)s;
    const int num_batch = %(NUM_BATCH)s;
    const int num_atoms = %(NUM_ATOMS)s;
    const int num_waters = %(NUM_WATERS)s;
    const int num_water_atoms = 3*num_points;
    const float prefactor = 332.0637090025476f;

    // Random number generator state for each water thread.
    __device__ curandState_t* states[num_batch];

    // Reaction field parameters.
    __device__ float rf_dielectric;
    __device__ float rf_kappa;
    __device__ float rf_cutoff;
    __device__ float rf_correction;

    // Triclinic box cell information.
    __device__ float cell_matrix[3][3];
    __device__ float cell_matrix_inverse[3][3];
    __device__ float M[3][3];

    // Atom properties.
    __device__ float sigma[num_atoms];
    __device__ float epsilon[num_atoms];
    __device__ float charge[num_atoms];
    __device__ float position[num_atoms * 3];
    __device__ int is_ghost[num_atoms];

    // Water properties.
    __device__ float sigma_water[num_points];
    __device__ float epsilon_water[num_points];
    __device__ float charge_water[num_points];
    __device__ int water_idx[num_waters];
    __device__ int water_state[num_waters];

    extern "C"
    {
        // Initialisation of the random number generator state for each attempt thread.
        __global__ void initialiseRNG(int* seed)
        {
            const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < num_batch)
            {
                curandState_t* s = new curandState_t;
                if (s != 0)
                {
                    curand_init(seed[tidx], tidx, 0, s);
                }

                states[tidx] = s;
            }
        }

        // Intialisation of the cell information for periodic triclinic boxes.
        __global__ void setCellMatrix(
            float* matrix,
            float* matrix_inverse,
            float* m)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    cell_matrix[i][j] = matrix[i * 3 + j];
                    cell_matrix_inverse[i][j] = matrix_inverse[i * 3 + j];
                    M[i][j] = m[i * 3 + j];
                }
            }
        }

        // Set the reaction field parameters.
        __global__ void setReactionField(float cutoff, float dielectric)
        {
            rf_dielectric = dielectric;
            rf_cutoff = cutoff;
            const auto rf_cutoff2 = cutoff * cutoff;
            const auto rf_cutoff3_inv = 1.0f / (rf_cutoff * rf_cutoff2);
            rf_kappa = rf_cutoff3_inv * (dielectric - 1.0f) / (2.0f * dielectric + 1.0f);
            rf_correction = (1.0 / rf_cutoff) + rf_kappa * rf_cutoff2;
        }

        // Set the properties of each atom.
        __global__ void setAtomProperties(
            float* charges,
            float* sigmas,
            float* epsilons,
            int* is_ghost_atom)
        {
            const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < num_atoms)
            {
                charge[tidx] = charges[tidx];
                sigma[tidx] = sigmas[tidx];
                epsilon[tidx] = epsilons[tidx];
                is_ghost[tidx] = is_ghost_atom[tidx];
            }
        }

        // Set the positions of each atom.
        __global__ void setAtomPositions(float* positions, float scale=1.0)
        {
            const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < num_atoms)
            {
                position[tidx * 3] = scale * positions[tidx * 3];
                position[tidx * 3 + 1] = scale * positions[tidx * 3 + 1];
                position[tidx * 3 + 2] = scale * positions[tidx * 3 + 2];
            }
        }

        // Set the properties of each water atom.
        __global__ void setWaterProperties(
            float* charges,
            float* sigmas,
            float* epsilons,
            int* idx,
            int* state)
        {
            for (int i = 0; i < num_points; i++)
            {
                charge_water[i] = charges[i];
                sigma_water[i] = sigmas[i];
                epsilon_water[i] = epsilons[i];
            }

            for (int i = 0; i < num_waters; i++)
            {
                water_idx[i] = idx[i];
                water_state[i] = state[i];
            }
        }

        // Update a single water.
        __global__ void updateWater(int idx, int state)
        {
            // Set the new state.
            water_state[idx] = state;

            // Get the water oxygen index in the context.
            int idx_context = water_idx[idx];

            for (int i = 0; i < num_points; i++)
            {
                // Ghost water.
                if (state == 0)
                {
                    charge[idx_context + i] = 0.0f;
                    sigma[idx_context + i] = 1.0f;
                    epsilon[idx_context + i] = 0.0f;
                    is_ghost[idx_context + i] = 1;
                }
                else
                {
                    charge[idx_context + i] = charge_water[i];
                    sigma[idx_context + i] = sigma_water[i];
                    epsilon[idx_context + i] = epsilon_water[i];
                    is_ghost[idx_context + i] = 0;
                }
            }
        }

        // Calculate the delta that needs to be subtracted from the interatomic distance
        // so that the atoms are wrapped to the same periodic box.
        __device__ void wrapDelta(float* v0, float* v1, float* delta_box)
        {
            // Work out the positions of v0 and v1 in "box" space.
            float v0_box[3];
            float v1_box[3];
            for (int i = 0; i < 3; i++)
            {
                v0_box[i] = 0.0f;
                v1_box[i] = 0.0f;

                for (int j = 0; j < 3; j++)
                {
                    v0_box[i] += cell_matrix_inverse[i][j] * v0[j];
                    v1_box[i] += cell_matrix_inverse[i][j] * v1[j];
                }
            }

            // Now work out the distance between v0 and v1 in "box" space.
            for (int i = 0; i < 3; i++)
            {
                delta_box[i] = v1_box[i] - v0_box[i];
            }

            // Extract the integer and fractional parts of the distance.
            int int_x = (int)delta_box[0];
            int int_y = (int)delta_box[1];
            int int_z = (int)delta_box[2];
            float frac_x = delta_box[0] - int_x;
            float frac_y = delta_box[1] - int_y;
            float frac_z = delta_box[2] - int_z;

            // Shift to the box.

            // x

            if (frac_x > 0.5f)
            {
                int_x += 1;
            }
            else if (frac_x < -0.5f)
            {
                int_x -= 1;
            }

            // y

            if (frac_y > 0.5f)
            {
                int_y += 1;
            }
            else if (frac_y < -0.5f)
            {
                int_y -= 1;
            }

            // z

            if (frac_z > 0.5f)
            {
                int_z += 1;
            }
            else if (frac_z < -0.5f)
            {
                int_z -= 1;
            }

            // Calculate the shifts over the box vectors.
            delta_box[0] = 0.0f;
            delta_box[1] = 0.0f;
            delta_box[2] = 0.0f;
            for (int i = 0; i < 3; i++)
            {
                delta_box[0] += cell_matrix[i][0] * int_x;
                delta_box[1] += cell_matrix[i][1] * int_y;
                delta_box[2] += cell_matrix[i][2] * int_z;
            }
        }

        // Calculate the distance between two atoms within the periodic box.
        __device__ void distance2(
            float* v0,
            float* v1,
            float& dist2)
        {
            // Work out the positions of v0 and v1 in "box" space.
            float v0_box[3];
            float v1_box[3];
            for (int i = 0; i < 3; i++)
            {
                v0_box[i] = 0.0f;
                v1_box[i] = 0.0f;

                for (int j = 0; j < 3; j++)
                {
                    v0_box[i] += cell_matrix_inverse[i][j] * v0[j];
                    v1_box[i] += cell_matrix_inverse[i][j] * v1[j];
                }
            }

            // Now work out the distance between v0 and v1 in "box" space.
            float delta_box[3];
            for (int i = 0; i < 3; i++)
            {
                delta_box[i] = v1_box[i] - v0_box[i];
            }

            // Extract the integer and fractional parts of the distance.
            int int_x = (int)delta_box[0];
            int int_y = (int)delta_box[1];
            int int_z = (int)delta_box[2];
            float frac_x = delta_box[0] - int_x;
            float frac_y = delta_box[1] - int_y;
            float frac_z = delta_box[2] - int_z;

            // Shift to the box.

            // x

            if (frac_x >= 0.5f)
            {
                frac_x -= 1.0f;
            }
            else if (frac_x <= -0.5f)
            {
                frac_x += 1.0f;
            }

            // y

            if (frac_y >= 0.5f)
            {
                frac_y -= 1.0f;
            }
            else if (frac_y <= -0.5f)
            {
                frac_y += 1.0f;
            }

            // z

            if (frac_z >= 0.5f)
            {
                frac_z -= 1.0f;
            }
            else if (frac_z <= -0.5f)
            {
                frac_z += 1.0f;
            }

            float frac_dist[3];
            frac_dist[0] = frac_x;
            frac_dist[1] = frac_y;
            frac_dist[2] = frac_z;
            for (int i = 0; i < 3; i++)
            {
                delta_box[i] = 0.0f;

                for (int j = 0; j < 3; j++)
                {
                    delta_box[i] += M[i][j] * frac_dist[j];
                }
            }
            dist2 = frac_x * delta_box[0] + frac_y * delta_box[1] + frac_z * delta_box[2];
        }

        // Perform a random rotation about a unit sphere.
        __device__ void uniform_random_rotation(float* v, float r0, float r1, float r2)
        {
            /* Adapted from:
                https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/

               Algorthm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
                https://doi.org/10.1016/B978-0-08-050755-2.50034-8
            */

            // First, generate a random rotation about the z axis.
            float x2 = 2.0f * pi * r0;
            float x3 = r1;
            float R[3][3];
            R[0][0] = R[1][1] = cosf(2.0f * pi * r2);
            R[0][1] = -sinf(2.0f * pi * r2);
            R[1][0] = sinf(2.0f * pi * r2);
            R[0][2] = R[1][2] = R[2][0] = R[2][1] = 0.0f;
            R[2][2] = 1.0f;

            // Now compute the Householder matrix H.
            float v0 = cosf(x2) * sqrtf(x3);
            float v1 = sinf(x2) * sqrtf(x3);
            float v2 = sqrtf(1.0f - x3);
            float H[3][3];
            H[0][0] = 1.0f - 2.0f * v0 * v0;
            H[0][1] = -2.0f * v0 * v1;
            H[0][2] = -2.0f * v0 * v2;
            H[1][0] = -2.0f * v0 * v1;
            H[1][1] = 1.0f - 2.0f * v1 * v1;
            H[1][2] = -2.0f * v1 * v2;
            H[2][0] = -2.0f * v0 * v2;
            H[2][1] = -2.0f * v1 * v2;
            H[2][2] = 1.0f - 2.0f * v2 * v2;

            // Now compute M = -(H @ R), i.e. rotate all points around the x axis.
            float M[3][3];
            M[0][0] = -(H[0][0] * R[0][0] + H[0][1] * R[1][0] + H[0][2] * R[2][0]);
            M[0][1] = -(H[0][0] * R[0][1] + H[0][1] * R[1][1] + H[0][2] * R[2][1]);
            M[0][2] = -(H[0][0] * R[0][2] + H[0][1] * R[1][2] + H[0][2] * R[2][2]);
            M[1][0] = -(H[1][0] * R[0][0] + H[1][1] * R[1][0] + H[1][2] * R[2][0]);
            M[1][1] = -(H[1][0] * R[0][1] + H[1][1] * R[1][1] + H[1][2] * R[2][1]);
            M[1][2] = -(H[1][0] * R[0][2] + H[1][1] * R[1][2] + H[1][2] * R[2][2]);
            M[2][0] = -(H[2][0] * R[0][0] + H[2][1] * R[1][0] + H[2][2] * R[2][0]);
            M[2][1] = -(H[2][0] * R[0][1] + H[2][1] * R[1][1] + H[2][2] * R[2][1]);
            M[2][2] = -(H[2][0] * R[0][2] + H[2][1] * R[1][2] + H[2][2] * R[2][2]);

            // Compute the mean coordinate of the water molecule.
            float mean_coord[3];
            mean_coord[0] = (v[0] + v[3] + v[6]) / 3.0f;
            mean_coord[1] = (v[1] + v[4] + v[7]) / 3.0f;
            mean_coord[2] = (v[2] + v[5] + v[8]) / 3.0f;

            // Now compute ((v - mean_coord) @ M) + mean_coord @ M.
            float x[3][3];
            x[0][0] = v[0] - mean_coord[0];
            x[0][1] = v[1] - mean_coord[1];
            x[0][2] = v[2] - mean_coord[2];
            x[1][0] = v[3] - mean_coord[0];
            x[1][1] = v[4] - mean_coord[1];
            x[1][2] = v[5] - mean_coord[2];
            x[2][0] = v[6] - mean_coord[0];
            x[2][1] = v[7] - mean_coord[1];
            x[2][2] = v[8] - mean_coord[2];

            // Compute the rotated coordinates.
            v[0] = x[0][0] * M[0][0] + x[0][1] * M[1][0] + x[0][2] * M[2][0]
                + mean_coord[0] * M[0][0] + mean_coord[1] * M[1][0] + mean_coord[2] * M[2][0];
            v[1] = x[0][0] * M[0][1] + x[0][1] * M[1][1] + x[0][2] * M[2][1]
                + mean_coord[0] * M[0][1] + mean_coord[1] * M[1][1] + mean_coord[2] * M[2][1];
            v[2] = x[0][0] * M[0][2] + x[0][1] * M[1][2] + x[0][2] * M[2][2]
                + mean_coord[0] * M[0][2] + mean_coord[1] * M[1][2] + mean_coord[2] * M[2][2];
            v[3] = x[1][0] * M[0][0] + x[1][1] * M[1][0] + x[1][2] * M[2][0]
                + mean_coord[0] * M[0][0] + mean_coord[1] * M[1][0] + mean_coord[2] * M[2][0];
            v[4] = x[1][0] * M[0][1] + x[1][1] * M[1][1] + x[1][2] * M[2][1]
                + mean_coord[0] * M[0][1] + mean_coord[1] * M[1][1] + mean_coord[2] * M[2][1];
            v[5] = x[1][0] * M[0][2] + x[1][1] * M[1][2] + x[1][2] * M[2][2]
                + mean_coord[0] * M[0][2] + mean_coord[1] * M[1][2] + mean_coord[2] * M[2][2];
            v[6] = x[2][0] * M[0][0] + x[2][1] * M[1][0] + x[2][2] * M[2][0]
                + mean_coord[0] * M[0][0] + mean_coord[1] * M[1][0] + mean_coord[2] * M[2][0];
            v[7] = x[2][0] * M[0][1] + x[2][1] * M[1][1] + x[2][2] * M[2][1]
                + mean_coord[0] * M[0][1] + mean_coord[1] * M[1][1] + mean_coord[2] * M[2][1];
            v[8] = x[2][0] * M[0][2] + x[2][1] * M[1][2] + x[2][2] * M[2][2]
                + mean_coord[0] * M[0][2] + mean_coord[1] * M[1][2] + mean_coord[2] * M[2][2];
        }

        // Generate a random position and orientation within the GCMC sphere
        // for each trial insertion.
        __global__ void generateWater(
            float* water_template,
            float* target,
            float radius,
            float* water_position,
            int is_target)
        {
            // Work out the thread index.
            const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            // Make sure we are within the number of waters.
            if (tidx < num_batch)
            {
                // Get the RNG state.
                curandState_t state = *states[tidx];

                // Translate the oxygen atom to the origin.
                float water[num_water_atoms];
                water[0] = 0.0f;
                water[1] = 0.0f;
                water[2] = 0.0f;

                // Shift the other atoms by the appropriate amount.
                for (int i = 0; i < 3; i++)
                {
                    water[i*3 + 0] = water_template[i*3 + 0] - water_template[0];
                    water[i*3 + 1] = water_template[i*3 + 1] - water_template[1];
                    water[i*3 + 2] = water_template[i*3 + 2] - water_template[2];
                }

                // Rotate the water randomly.
                uniform_random_rotation(water, curand_uniform(&state), curand_uniform(&state), curand_uniform(&state));

                // Calculate the distance between the oxygen and the hydrogens.
                float dh[num_points][3];
                for (int i = 0; i < num_points-1; i++)
                {
                    dh[i][0] = water[(i+1)*3] - water[0];
                    dh[i][1] = water[(i+1)*3 + 1] - water[1];
                    dh[i][2] = water[(i+1)*3 + 2] - water[2];
                }

                float xyz[3];

                // Choose a random position within the GCMC sphere.
                if (is_target == 1)
                {
                    // Generate a random position around the target.
                    xyz[0] = curand_normal(&state);
                    xyz[1] = curand_normal(&state);
                    xyz[2] = curand_normal(&state);

                    float norm = sqrtf(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
                    xyz[0] /= norm;
                    xyz[1] /= norm;
                    xyz[2] /= norm;
                    float r = radius * powf(curand_uniform(&state), 1.0f / 3.0f);
                    xyz[0] = target[0] + r * xyz[0];
                    xyz[1] = target[1] + r * xyz[1];
                    xyz[2] = target[2] + r * xyz[2];
                }
                // Choose a random position within the triclinic box.
                else
                {
                    float r[3];
                    r[0] = curand_uniform(&state);
                    r[1] = curand_uniform(&state);
                    r[2] = curand_uniform(&state);

                    for (int i = 0; i < 3; i++)
                    {
                        xyz[i] = 0.0f;
                        for (int j = 0; j < 3; j++)
                        {
                            xyz[i] += r[j] * cell_matrix[i][j];
                        }
                    }
                }

                // Place the oxygen (first atom) at the random position.
                water_position[tidx * num_water_atoms] = xyz[0];
                water_position[tidx * num_water_atoms + 1] = xyz[1];
                water_position[tidx * num_water_atoms + 2] = xyz[2];

                // Shift the hydrogens by the appropriate amount.
                for (int i = 0; i < num_points-1; i++)
                {
                    water_position[tidx * num_water_atoms + 3 + i*3] = xyz[0] + dh[i][0];
                    water_position[tidx * num_water_atoms + 4 + i*3] = xyz[1] + dh[i][1];
                    water_position[tidx * num_water_atoms + 5 + i*3] = xyz[2] + dh[i][2];
                }

                // Set the new state.
                *states[tidx] = state;
            }
        }

        // Compute the Lennard-Jones and reaction field Coulomb energy between
        // the water and the atoms.
        __global__ void computeEnergy(
            float* water_position,
            float* energy_coul,
            float* energy_lj,
            int* deletion_candidates,
            int* is_deletion)
        {
            // Work out the atom index.
            const int idx_atom = threadIdx.x + blockDim.x * blockIdx.x;

            // Make sure we're in bounds.
            if (idx_atom < num_atoms)
            {
                // Store the squared cut-off distance.
                const auto cutoff2 = rf_cutoff * rf_cutoff;

                // Work out the water index.
                const int idx_water = blockIdx.y;

                // Work out the index for the result.
                const int idx = (idx_water * num_atoms) + idx_atom;

                // Zero the energies.
                energy_coul[idx] = 0.0;
                energy_lj[idx] = 0.0;

                // First apply the reaction field correction for the water atoms.
                if (idx_atom == 0)
                {
                    for (int i = 0; i < num_points; i++)
                    {
                        // Self interaction.
                        const auto c1 = charge_water[i];
                        energy_coul[idx] -= 0.5f * (c1 * c1) * rf_correction;

                        // Pair interaction.
                        for (int j = i+1; j < num_points; j++)
                        {
                            const auto c2 = charge_water[j];
                            energy_coul[idx] -= (c1 * c2) * rf_correction;
                        }
                    }
                }

                // This is a deletion move, so we need to get the correct water index.
                if (is_deletion[idx_water] == 1)
                {
                    const int idx_water_context = water_idx[deletion_candidates[idx_water]];
                    const auto delta = idx_atom - idx_water_context;

                    // Don't compute self-interactions.
                    if (delta >= 0 and delta < num_points)
                    {
                        return;
                    }
                }

                // Don't interact with ghost waters.
                if (is_ghost[idx_atom] == 1)
                {
                    return;
                }

                // Get the atom position.
                float v0[3];
                v0[0] = position[3 * idx_atom];
                v0[1] = position[3 * idx_atom + 1];
                v0[2] = position[3 * idx_atom + 2];

                // Store the charge on the atom.
                auto c0 = charge[idx_atom];

                // Store the epsilon and sigma for the atom.
                float s0 = sigma[idx_atom];
                float e0 = epsilon[idx_atom];

                // Loop over all atoms in the water molecule.
                for (int i = 0; i < num_points; i++)
                {
                    // Get the water atom position.
                    float v1[3];
                    if (is_deletion[idx_water] == 1)
                    {
                        const int idx_water_context = water_idx[deletion_candidates[idx_water]];
                        v1[0] = position[3 * idx_water_context + 3 * i];
                        v1[1] = position[3 * idx_water_context + 3 * i + 1];
                        v1[2] = position[3 * idx_water_context + 3 * i + 2];
                    }
                    else
                    {
                        v1[0] = water_position[3 * num_points * idx_water + 3 * i];
                        v1[1] = water_position[3 * num_points * idx_water + 3 * i + 1];
                        v1[2] = water_position[3 * num_points * idx_water + 3 * i + 2];
                    }

                    // Calculate the squared distance between the atoms.
                    float r2;
                    distance2(v0, v1, r2);

                    // The distance is within the cut-off.
                    if (r2 < cutoff2)
                    {
                        // Don't divide by zero.
                        if (r2 < 1e-6)
                        {
                            energy_coul[idx] = 1e6;
                            energy_lj[idx] = 1e6;
                            return;
                        }
                        else
                        {
                            // Compute the LJ interaction.
                            auto s1 = sigma_water[i];
                            const auto e1 = epsilon_water[i];
                            const auto s = 0.5 * (s0 + s1);
                            const auto e = sqrtf(e0 * e1);
                            const auto s2 = s * s;
                            const auto sr2 = s2 / r2;
                            const auto sr6 = sr2 * sr2 * sr2;
                            const auto sr12 = sr6 * sr6;
                            energy_lj[idx] += 4 * e * (sr12 - sr6);

                            // Compute the distance between the atoms.
                            const auto r = sqrtf(r2);

                            // Store the charge on the water atom.
                            const auto c1 = charge_water[i];

                            // Add the reaction field pair energy.
                            energy_coul[idx] += (c0 * c1) * ((1.0f / r) + (rf_kappa * r2) - rf_correction);
                        }
                    }
                }
            }
        }

        // Calculate whether each attempt is accepted.
        __global__ void checkAcceptance(
            int N,
            float exp_B,
            float exp_minus_B,
            float beta,
            int* is_deletion,
            float* energy_coul,
            float* energy_lj,
            float* energy_change,
            float* probability,
            int* accepted,
            float tolerance)
        {
            const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < num_batch)
            {
                // Zero the energy.
                float energy = 0.0;

                // Work out the acceptance factors based on the move type.
                float sign;
                float expB;
                int N_insert;
                int N_delete;
                if (is_deletion[tidx] == 1)
                {
                    sign = -1.0f;
                    expB = exp_minus_B;
                    N_insert = 0;
                    N_delete = N;
                }
                else
                {
                    sign = 1.0f;
                    expB = exp_B;
                    N_insert = N;
                    N_delete = 1;
                }

                // Sum the energy contributions from all the atoms.
                for (int i = 0; i < num_atoms; i++)
                {
                    int idx = (tidx * num_atoms) + i;
                    energy += prefactor * energy_coul[idx] + energy_lj[idx];
                }

                // Compute the probability.
                float prob = N_delete * expB * expf(-beta * sign * energy) / (N_insert + 1);

                // Store the energy change.
                energy_change[tidx] = sign * energy;

                // Store the probability.
                probability[tidx] = prob;

                // Get the RNG state.
                curandState_t state = *states[tidx];

                // Accept or reject based on the Boltzmann weight. A tolerance
                // can be used to reject low probability states that can cause
                // instabilities and/or crashes in the MD engine.
                if (prob > tolerance and curand_uniform(&state) < prob)
                {
                    accepted[tidx] = 1;
                }
                else
                {
                    accepted[tidx] = 0;
                }

                // Set the new state.
                *states[tidx] = state;
            }
        }

        // Find candidate waters for deletion.
        __global__ void findDeletionCandidates(
            int* candidates,
            float* target,
            float radius)
        {
            const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < num_waters)
            {
                // Null the candidate.
                candidates[tidx] = 0;

                // This isn't a ghost water, so make sure it's within the GCMC sphere.
                if (water_state[tidx] != 0)
                {
                    // Get the water oxygen index.
                    int idx = water_idx[tidx];

                    // Get the oxygen atom position.
                    float v[3];
                    v[0] = position[3 * idx];
                    v[1] = position[3 * idx + 1];
                    v[2] = position[3 * idx + 2];

                    // Calculate the distance between the water and the target.
                    float r2;
                    distance2(v, target, r2);

                    // The water is within the GCMC sphere. Flag it as a candidate.
                    if (r2 < radius * radius)
                    {
                        candidates[tidx] = 1;
                    }
                }
            }
        }
    }
"""
