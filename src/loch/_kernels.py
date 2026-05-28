######################################################################
# Loch: GPU accelerated GCMC water sampling engine.
#
# Copyright: 2025-2026
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
GCMC CUDA/OpenCL kernels.
"""

code = """
    // Platform-specific definitions for CUDA and OpenCL compatibility
    #ifdef __OPENCL_VERSION__
        #define KERNEL __kernel
        #define DEVICE
        #define GLOBAL __global
        #define LOCAL __local
        #define GET_GLOBAL_ID(dim) get_global_id(dim)
        #define BLOCK_ID_Y get_group_id(1)  // OpenCL: work-group ID in dimension 1
        // Map CUDA-style function names to OpenCL names
        #define sqrtf sqrt
        #define powf pow
        #define expf exp
        #define cosf cos
        #define sinf sin
        #define rsqrtf rsqrt
        #define floorf floor
        #define fmaf fma
        // OpenCL doesn't have sincosf, so define it
        #define sincosf(x, sptr, cptr) do { *(sptr) = sinf(x); *(cptr) = cosf(x); } while(0)
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
        #define KERNEL extern "C" __global__
        #define DEVICE __device__
        #define GLOBAL
        #define LOCAL __shared__
        #define GET_GLOBAL_ID(dim) (threadIdx.x + blockIdx.x * blockDim.x)
        #define BLOCK_ID_Y blockIdx.y  // CUDA: block index in y dimension
    #endif

    // Constants.
    const float pi = 3.14159265359f;
    const float prefactor = 332.0637090025476f;

    // Maximum number of atoms per water molecule (for stack array sizing).
    #define MAX_POINTS 5
    #define MAX_WATER_POSITIONS (3 * MAX_POINTS)

    #ifndef __OPENCL_VERSION__
    extern "C"
    {
    #endif

        // Calculate the distance between two atoms within the periodic box.
        DEVICE void distance2(
            float* v0,
            float* v1,
            float* dist2,
            GLOBAL const float* cell_matrix_inverse,
            GLOBAL const float* metric_matrix)
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
                    v0_box[i] += cell_matrix_inverse[i * 3 + j] * v0[j];
                    v1_box[i] += cell_matrix_inverse[i * 3 + j] * v1[j];
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

            // Shift to the box (branchless).
            frac_x -= floorf(frac_x + 0.5f);
            frac_y -= floorf(frac_y + 0.5f);
            frac_z -= floorf(frac_z + 0.5f);

            float frac_dist[3];
            frac_dist[0] = frac_x;
            frac_dist[1] = frac_y;
            frac_dist[2] = frac_z;
            for (int i = 0; i < 3; i++)
            {
                delta_box[i] = 0.0f;

                for (int j = 0; j < 3; j++)
                {
                    delta_box[i] += metric_matrix[i * 3 + j] * frac_dist[j];
                }
            }
            *dist2 = frac_x * delta_box[0] + frac_y * delta_box[1] + frac_z * delta_box[2];
        }

        // Perform a random rotation about a unit sphere.
        DEVICE void uniform_random_rotation(float* v, int num_points, float r0, float r1, float r2)
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
            float sin_r2, cos_r2;
            sincosf(2.0f * pi * r2, &sin_r2, &cos_r2);
            R[0][0] = R[1][1] = cos_r2;
            R[0][1] = -sin_r2;
            R[1][0] = sin_r2;
            R[0][2] = R[1][2] = R[2][0] = R[2][1] = 0.0f;
            R[2][2] = 1.0f;

            // Now compute the Householder matrix H.
            float sin_x2, cos_x2;
            sincosf(x2, &sin_x2, &cos_x2);
            float sqrt_x3 = sqrtf(x3);
            float v0 = cos_x2 * sqrt_x3;
            float v1 = sin_x2 * sqrt_x3;
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
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    M[i][j] = -(H[i][0] * R[0][j] + H[i][1] * R[1][j] + H[i][2] * R[2][j]);
                }
            }

            // Compute the mean coordinate of the water molecule.
            float mean_coord[3];
            mean_coord[0] = 0.0f;
            mean_coord[1] = 0.0f;
            mean_coord[2] = 0.0f;
            for (int i = 0; i < num_points; i++)
            {
                mean_coord[0] += v[i * 3];
                mean_coord[1] += v[i * 3 + 1];
                mean_coord[2] += v[i * 3 + 2];
            }
            mean_coord[0] /= (float)num_points;
            mean_coord[1] /= (float)num_points;
            mean_coord[2] /= (float)num_points;

            // Precompute mean_coord @ M (avoids redundant calculations).
            float mean_M[3];
            for (int j = 0; j < 3; j++)
            {
                mean_M[j] = fmaf(mean_coord[0], M[0][j], fmaf(mean_coord[1], M[1][j], mean_coord[2] * M[2][j]));
            }

            // Compute ((v - mean_coord) @ M) + mean_M for each atom.
            for (int i = 0; i < num_points; i++)
            {
                float dx = v[i * 3]     - mean_coord[0];
                float dy = v[i * 3 + 1] - mean_coord[1];
                float dz = v[i * 3 + 2] - mean_coord[2];

                v[i * 3]     = fmaf(dx, M[0][0], fmaf(dy, M[1][0], fmaf(dz, M[2][0], mean_M[0])));
                v[i * 3 + 1] = fmaf(dx, M[0][1], fmaf(dy, M[1][1], fmaf(dz, M[2][1], mean_M[1])));
                v[i * 3 + 2] = fmaf(dx, M[0][2], fmaf(dy, M[1][2], fmaf(dz, M[2][2], mean_M[2])));
            }
        }

        // Update a single water.
        KERNEL void updateWater(
            int num_points,
            int idx,
            int state,
            int is_insertion,
            GLOBAL float* new_position,
            GLOBAL float* position,
            GLOBAL float* charge,
            GLOBAL float* epsilon,
            GLOBAL int* is_ghost_water,
            GLOBAL int* water_state,
            GLOBAL const int* water_idx,
            GLOBAL const float* charge_water,
            GLOBAL const float* epsilon_water)
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
                    epsilon[idx_context + i] = 0.0f;
                    is_ghost_water[idx_context + i] = 1;
                }
                else
                {
                    charge[idx_context + i] = charge_water[i];
                    epsilon[idx_context + i] = epsilon_water[i];
                    is_ghost_water[idx_context + i] = 0;
                }

                // Update the position of the water. We don't use the state to determine
                // whether an insertion is performed, since we don't need to update the
                // positions when a deletion move is rejected, which would also set the
                // state to 1.
                if (is_insertion == 1)
                {
                    position[3 * idx_context + 3 * i] = new_position[3 * i];
                    position[3 * idx_context + 3 * i + 1] = new_position[3 * i + 1];
                    position[3 * idx_context + 3 * i + 2] = new_position[3 * i + 2];
                }
            }
        }

        // Generate a random position and orientation within the GCMC sphere
        // for each trial insertion.
        KERNEL void generateWater(
            int num_points,
            int num_batch,
            GLOBAL float* water_template,
            GLOBAL float* target,
            float radius,
            GLOBAL float* water_position,
            int is_target,
            GLOBAL float* randoms_rotation,
            GLOBAL float* randoms_position_sphere,
            GLOBAL float* randoms_position_bulk,
            GLOBAL float* randoms_radius,
            GLOBAL const float* cell_matrix)
        {
            // Work out the thread index.
            const int tidx = GET_GLOBAL_ID(0);

            // Make sure we are within the number of waters.
            if (tidx < num_batch)
            {
                const int num_water_positions = 3 * num_points;

                // Translate the oxygen atom to the origin.
                float water[MAX_WATER_POSITIONS];
                water[0] = 0.0f;
                water[1] = 0.0f;
                water[2] = 0.0f;

                // Shift the other atoms by the appropriate amount.
                for (int i = 0; i < num_points; i++)
                {
                    water[i*3 + 0] = water_template[i*3 + 0] - water_template[0];
                    water[i*3 + 1] = water_template[i*3 + 1] - water_template[1];
                    water[i*3 + 2] = water_template[i*3 + 2] - water_template[2];
                }

                // Rotate the water randomly using pre-generated randoms.
                uniform_random_rotation(water, num_points,
                    randoms_rotation[tidx * 3],
                    randoms_rotation[tidx * 3 + 1],
                    randoms_rotation[tidx * 3 + 2]);

                // Calculate the distance between the oxygen and the hydrogens.
                float dh[MAX_POINTS][3];
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
                    // Generate a random position around the target using pre-generated normals.
                    xyz[0] = randoms_position_sphere[tidx * 3];
                    xyz[1] = randoms_position_sphere[tidx * 3 + 1];
                    xyz[2] = randoms_position_sphere[tidx * 3 + 2];

                    float norm = sqrtf(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
                    xyz[0] /= norm;
                    xyz[1] /= norm;
                    xyz[2] /= norm;
                    float r = radius * powf(randoms_radius[tidx], 1.0f / 3.0f);
                    xyz[0] = target[0] + r * xyz[0];
                    xyz[1] = target[1] + r * xyz[1];
                    xyz[2] = target[2] + r * xyz[2];
                }
                // Choose a random position within the triclinic box.
                else
                {
                    // Use pre-generated uniform randoms for bulk sampling.
                    float r[3];
                    r[0] = randoms_position_bulk[tidx * 3];
                    r[1] = randoms_position_bulk[tidx * 3 + 1];
                    r[2] = randoms_position_bulk[tidx * 3 + 2];

                    for (int i = 0; i < 3; i++)
                    {
                        xyz[i] = 0.0f;
                        for (int j = 0; j < 3; j++)
                        {
                            xyz[i] += r[j] * cell_matrix[i * 3 + j];
                        }
                    }
                }

                // Place the oxygen (first atom) at the random position.
                water_position[tidx * num_water_positions] = xyz[0];
                water_position[tidx * num_water_positions + 1] = xyz[1];
                water_position[tidx * num_water_positions + 2] = xyz[2];

                // Shift the hydrogens by the appropriate amount.
                for (int i = 0; i < num_points-1; i++)
                {
                    water_position[tidx * num_water_positions + 3 + i*3] = xyz[0] + dh[i][0];
                    water_position[tidx * num_water_positions + 4 + i*3] = xyz[1] + dh[i][1];
                    water_position[tidx * num_water_positions + 5 + i*3] = xyz[2] + dh[i][2];
                }
            }
        }

        // Compute the Lennard-Jones and reaction field Coulomb energy between
        // the water and the atoms.
        KERNEL void computeEnergy(
            int num_points,
            int num_batch,
            int num_atoms,
            GLOBAL float* water_position,
            GLOBAL float* energy_coul,
            GLOBAL float* energy_lj,
            GLOBAL int* deletion_candidates,
            GLOBAL int* is_deletion,
            int is_fep,
            GLOBAL const float* position,
            GLOBAL const float* charge,
            GLOBAL const float* sigma,
            GLOBAL const float* epsilon,
            GLOBAL const float* alpha,
            GLOBAL const int* is_ghost_water,
            GLOBAL const int* is_ghost_fep,
            GLOBAL const float* sigma_water,
            GLOBAL const float* epsilon_water,
            GLOBAL const float* charge_water,
            GLOBAL const int* water_idx,
            GLOBAL const float* cell_matrix_inverse,
            GLOBAL const float* metric_matrix,
            float rf_cutoff,
            float rf_kappa,
            float rf_correction,
            int softcore_form,
            float sc_shift_coulomb,
            float sc_shift_delta,
            int sc_taylor_power,
            float sc_beutler_alpha)
        {
            // Work out the atom index.
            const int idx_atom = GET_GLOBAL_ID(0);

            // Make sure we're in bounds.
            if (idx_atom < num_atoms)
            {
                const int num_water_positions = 3 * num_points;

                // Store the squared cut-off distance.
                const float cutoff2 = rf_cutoff * rf_cutoff;

                // Work out the water index.
                const int idx_water = BLOCK_ID_Y;

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
                        const float q1 = charge_water[i];
                        energy_coul[idx] -= 0.5f * (q1 * q1) * rf_correction;

                        // Pair interaction.
                        for (int j = i+1; j < num_points; j++)
                        {
                            const float q2 = charge_water[j];
                            energy_coul[idx] -= (q1 * q2) * rf_correction;
                        }
                    }
                }

                // This is a deletion move, so we need to get the correct water index.
                if (is_deletion[idx_water] == 1)
                {
                    const int idx_water_context = water_idx[deletion_candidates[idx_water]];
                    const float delta = idx_atom - idx_water_context;

                    // Don't compute self-interactions.
                    if (delta >= 0 && delta < num_points)
                    {
                        return;
                    }
                }

                // Don't interact with ghost waters.
                if (is_ghost_water[idx_atom] == 1)
                {
                    return;
                }

                // If this an alchemical system, then we need to check whether the
                // atom is a ghost atom.
                bool is_ghost_atom = false;
                if (is_fep == 1)
                {
                    if (is_ghost_fep[idx_atom] == 1)
                    {
                        is_ghost_atom = true;
                    }
                }

                // Get the atom position.
                float v0[3];
                v0[0] = position[3 * idx_atom];
                v0[1] = position[3 * idx_atom + 1];
                v0[2] = position[3 * idx_atom + 2];

                // Store the charge on the atom.
                float q0 = charge[idx_atom];

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
                        v1[0] = water_position[num_water_positions * idx_water + 3 * i];
                        v1[1] = water_position[num_water_positions * idx_water + 3 * i + 1];
                        v1[2] = water_position[num_water_positions * idx_water + 3 * i + 2];
                    }

                    // Calculate the squared distance between the atoms.
                    float r2;
                    distance2(v0, v1, &r2, cell_matrix_inverse, metric_matrix);

                    // The distance is within the cut-off.
                    if (r2 < cutoff2)
                    {
                        // Don't divide by zero.
                        if (!is_fep && r2 < 1e-6)
                        {
                            energy_coul[idx] = 1e6;
                            energy_lj[idx] = 1e6;
                            return;
                        }
                        else
                        {
                            // Regular non-bonded forces.
                            if (!is_ghost_atom)
                            {
                                // Compute the LJ interaction.
                                float s1 = sigma_water[i];
                                const float e1 = epsilon_water[i];
                                const float s = 0.5f * (s0 + s1);
                                const float e = sqrtf(e0 * e1);
                                const float s2 = s * s;
                                const float sr2 = s2 / r2;
                                const float sr6 = sr2 * sr2 * sr2;
                                energy_lj[idx] += 4.0f * e * sr6 * (sr6 - 1.0f);

                                // Compute reciprocal distance (faster than sqrtf).
                                const float r_inv = rsqrtf(r2);

                                // Store the charge on the water atom.
                                const float q1 = charge_water[i];

                                // Add the reaction field pair energy.
                                energy_coul[idx] += (q0 * q1) * (r_inv + (rf_kappa * r2) - rf_correction);
                            }

                            // Soft-core potential for ghost atoms.
                            else
                            {
                                // Store required parameters.
                                const float q1 = charge_water[i];
                                const float s1 = sigma_water[i];
                                const float e1 = epsilon_water[i];
                                const float s = 0.5f * (s0 + s1);
                                const float e = sqrtf(e0 * e1);
                                const float a = alpha[idx_atom];

                                // Clamp r2 to avoid singularities.
                                const float r2_sc = (r2 < 1e-6f) ? 1e-6f : r2;

                                // Precompute r^6 and sigma^6 using r2 directly (avoids sqrtf and powf).
                                const float r6 = r2_sc * r2_sc * r2_sc;
                                const float s2 = s * s;
                                const float s6_val = s2 * s2 * s2;

                                // Compute the LJ interaction using the chosen soft-core form.
                                float sig6;
                                float lj_prefactor = 1.0f;
                                if (softcore_form == 1)
                                {
                                    // Taylor soft-core LJ:
                                    //   sig6 = sigma^6 / (alpha^m * sigma^6 + r^6)
                                    const float alpha_m = (sc_taylor_power == 1) ? a
                                        : (sc_taylor_power == 0) ? 1.0f
                                        : powf(a, (float)sc_taylor_power);
                                    sig6 = s6_val / (alpha_m * s6_val + r6);
                                }
                                else if (softcore_form == 2)
                                {
                                    // Beutler soft-core LJ:
                                    //   sig6 = sigma^6 / (sc_beutler_alpha * sigma^6 * alpha + r^6)
                                    //   V_LJ = (1 - alpha) * 4 * epsilon * sig6 * (sig6 - 1)
                                    sig6 = s6_val / (sc_beutler_alpha * s6_val * a + r6);
                                    lj_prefactor = 1.0f - a;
                                }
                                else
                                {
                                    // Zacharias soft-core LJ:
                                    //   sig6 = sigma^6 / (sigma*delta + r^2)^3
                                    //   delta = shift_delta * alpha
                                    const float delta_lj = sc_shift_delta * a;
                                    const float denom = (s * delta_lj) + r2_sc;
                                    sig6 = s6_val / (denom * denom * denom);
                                }
                                energy_lj[idx] += lj_prefactor * 4.0f * e * sig6 * (sig6 - 1.0f);

                                // Compute the Coulomb interaction.
                                energy_coul[idx] += (q0 * q1) *
                                    ((1.0f / sqrtf((sc_shift_coulomb * sc_shift_coulomb * a)
                                    + r2_sc)) + (rf_kappa * r2) - rf_correction);

                            }
                        }
                    }
                }
            }
        }

        // Calculate whether each attempt is accepted.
        KERNEL void checkAcceptance(
            int num_batch,
            int num_atoms,
            int N,
            float exp_B,
            float exp_minus_B,
            float beta,
            GLOBAL int* is_deletion,
            GLOBAL float* energy_coul,
            GLOBAL float* energy_lj,
            GLOBAL float* energy_change,
            GLOBAL float* probability,
            GLOBAL int* accepted,
            float tolerance,
            GLOBAL float* randoms_acceptance)
        {
            const int tidx = GET_GLOBAL_ID(0);

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

                // Accept or reject based on the Boltzmann weight using pre-generated random.
                // A tolerance can be used to reject low probability states that can cause
                // instabilities and/or crashes in the MD engine.
                if (prob > tolerance && randoms_acceptance[tidx] < prob)
                {
                    accepted[tidx] = 1;
                }
                else
                {
                    accepted[tidx] = 0;
                }
            }
        }

        // Find candidate waters for deletion.
        KERNEL void findDeletionCandidates(
            int num_waters,
            GLOBAL int* candidates,
            GLOBAL float* target,
            float radius,
            GLOBAL const float* position,
            GLOBAL const int* water_idx,
            GLOBAL const int* water_state,
            GLOBAL const float* cell_matrix_inverse,
            GLOBAL const float* metric_matrix)
        {
            const int tidx = GET_GLOBAL_ID(0);

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
                    distance2(v, target, &r2, cell_matrix_inverse, metric_matrix);

                    // The water is within the GCMC sphere. Flag it as a candidate.
                    if (r2 < radius * radius)
                    {
                        candidates[tidx] = 1;
                    }
                }
            }
        }

    #ifndef __OPENCL_VERSION__
    }
    #endif
"""
