######################################################################
# Loch: GPU accelerated GMCC water sampling engine.
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

__all__ = ["GCMCSampler"]

import numpy as _np
import openmm as _openmm
import os as _os

from loguru import logger as _logger

import pycuda.driver as _cuda
import pycuda.gpuarray as _gpuarray
from pycuda.compiler import SourceModule as _SourceModule

import BioSimSpace as _BSS
import sire as _sr

from ._kernels import code as _code


class GCMCSampler:
    """
    A class to perform GCMC water sampling on the GPU via PyCUDA.
    """

    def __init__(
        self,
        system,
        reference=None,
        radius="4.0 A",
        cutoff_type="rf",
        cutoff="10.0 A",
        excess_chemical_potential="-6.09 kcal/mol",
        standard_volume="30.543 A^3",
        temperature="298 K",
        adams_shift=0.0,
        max_gcmc_waters=10,
        num_attempts=10000,
        num_threads=1024,
        bulk_sampling_probability=0.1,
        water_template=None,
        device=None,
        log_level="info",
        seed=None,
    ):
        """
        Initialise the GCMC sampler.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        reference: str
            A selection string for the reference atoms. If None, then waters
            will be randomly inserted or deleted within the simulation box.

        radius: str
            The radius of the GCMC sphere.

        cutoff_type: str
            The type of cutoff to use.

        cutoff: str
            The cutoff distance for the non-bonded interactions.

        excess_chemical_potential: str
            The excess chemical potential.

        standard_volume: str
            The standard volume of water.

        temperature: str
            The temperature of the system.

        adams_shift: float
            The Adams shift.

        max_gcmc_waters: int
            The maximum number of GCMC waters to insert.

        num_attempts: int
            The number of attempted insertions and deletions per move.

        num_threads: int
            The number of threads per block. (Must be a multiple of 32.)

        bulk_sampling_probability: float
            The probability of perforing trial insertion and deletion
            moves within the entire simulation box, rather than within
            the GCMC sphere. This option is only relevant when 'reference'
            is not None.

        water_template: sire.molecule.Molecule
            A water molecule to use as a template. This is only required when
            the system does not contain any water molecules.

        device: int
            The CUDA device index. (This is the index in the list of visible
            devices.)

        log_level: str
            The logging level.

        seed: int
            The seed for the random number generator.
        """

        # Validate the input.

        if not isinstance(system, _sr.system.System):
            raise ValueError("'system' must be of type 'sire.system.System'")
        self._system = system

        if reference is not None:
            if not isinstance(reference, str):
                raise ValueError("'reference' must be of type 'str'")
        self._reference = reference

        if not isinstance(cutoff_type, str):
            raise ValueError("'cutoff_type' must be of type 'str'")
        cutoff_type = cutoff_type.lower().replace(" ", "")
        if not cutoff_type in ["rf", "pme"]:
            raise ValueError("The cutoff type must be 'rf' or 'pme'.")
        self._cutoff_type = cutoff_type

        if self._cutoff_type == "pme":
            self._is_pme = True
        else:
            self._is_pme = False

        try:
            self._radius = self._validate_sire_unit("radius", radius, _sr.u("A"))
        except Exception as e:
            raise ValueError(f"Could not validate the 'radius': {e}")

        try:
            self._cutoff = self._validate_sire_unit("cutoff", cutoff, _sr.u("A"))
        except Exception as e:
            raise ValueError(f"Could not validate the 'cutoff': {e}")

        try:
            self._excess_chemical_potential = self._validate_sire_unit(
                "excess_chemical_potential",
                excess_chemical_potential,
                _sr.u("kcal/mol"),
            )
        except Exception as e:
            raise ValueError(f"Could not validate the 'excess_chemical_potential': {e}")

        try:
            self._standard_volume = self._validate_sire_unit(
                "standard_volume", standard_volume, _sr.u("A^3")
            )
        except Exception as e:
            raise ValueError(f"Could not validate the 'standard_volume': {e}")

        try:
            self._temperature = self._validate_sire_unit(
                "temperature", temperature, _sr.u("K")
            )
        except Exception as e:
            raise ValueError(f"Could not validate the 'temperature': {e}")

        if not isinstance(max_gcmc_waters, int):
            raise ValueError("'max_gcmc_waters' must be of type 'int'")
        self._max_gcmc_waters = max_gcmc_waters

        if not isinstance(adams_shift, (int, float)):
            raise ValueError("'adams_shift' must be of type 'int' or 'float'")
        self._adams_shift = float(adams_shift)

        if not isinstance(num_attempts, int):
            raise ValueError("'num_attempts' must be of type 'int'")
        self._num_attempts = num_attempts

        if not isinstance(num_threads, int):
            raise ValueError("'num_threads' must be of type 'int'")
        if not num_threads % 32 == 0:
            raise ValueError("'num_threads' must be a multiple of 32")
        self._num_threads = num_threads

        if not isinstance(bulk_sampling_probability, float):
            raise ValueError("'bulk_sampling_probability' must be of type 'float'")
        if not 0.0 <= bulk_sampling_probability <= 1.0:
            raise ValueError("'bulk_sampling_probability' must be between 0 and 1")
        self._bulk_sampling_probability = bulk_sampling_probability

        if not isinstance(log_level, str):
            raise ValueError("'log_level' must be of type 'str'")
        log_level = log_level.lower().replace(" ", "")
        allowed_levels = [level.lower() for level in _logger._core.levels]
        if not log_level in allowed_levels:
            raise ValueError(
                f"Invalid 'log_level': {log_level}. Choices are: {', '.join(allowed_levels)}"
            )
        self._log_level = log_level
        if self._log_level == "debug":
            self._is_debug = True
        else:
            self._is_debug = False

        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("'seed' must be of type 'int'")
        else:
            seed = _np.random.randint(_np.iinfo(_np.int32).max)
        self._seed = seed

        # Set the seed.
        _np.random.seed(self._seed)

        # Create a random number generator.
        self._rng = _np.random.default_rng(self._seed)

        import pycuda.driver as cuda
        from pycuda.tools import make_default_context

        # Set the CUDA device.
        if device is not None:
            if not isinstance(device, int):
                raise ValueError("'device' must be of type 'int'")
            _os.environ["CUDA_DEVICE"] = str(device)
        cuda.init()
        self._context = make_default_context()
        self._device = self._context.get_device()

        # Check for waters and validate the template.
        try:
            self._water_template = system["water"][0]
        except:
            if water_template is None:
                raise ValueError(
                    "The system does not contain any water molecules. "
                    "Please provide a water template."
                )
            else:
                if not isinstance(water_template, _sr.molecule.Molecule):
                    raise ValueError(
                        "'water_template' must be of type 'sire.mol.Molecule'"
                    )
            self._water_template = water_template
        self._num_points = self._water_template.num_atoms()

        # Store the positions of the template.
        self._water_template_positions = _gpuarray.to_gpu(
            _sr.io.get_coords_array(self._water_template).flatten().astype(_np.float32)
        )

        # Get the indices of the reference atoms.
        if self._reference is not None:
            self._reference_indices = self._get_reference_indices(system, reference)

        # Set the box information.
        self._space, self._cell_matrix, self._cell_matrix_inverse, self._M = (
            self._get_box_information(system)
        )

        # Prepare the system for GCMC sampling.
        try:
            self._system, self._water_indices = self._prepare_system(
                system, self._water_template, self._rng, self._max_gcmc_waters
            )
            self._num_atoms = self._system.num_atoms()
            self._num_waters = len(self._water_indices)
            self._total_waters = self._num_waters + self._max_gcmc_waters
        except Exception as e:
            raise ValueError(f"Could not prepare the system for GCMC sampling: {e}")

        # Create the kernels.
        self._kernels = {}
        mod = _SourceModule(
            _code
            % {
                "NUM_POINTS": self._num_points,
                "NUM_ATTEMPTS": self._num_attempts,
                "NUM_WATERS": self._num_waters,
                "NUM_ATOMS": self._num_atoms,
            },
            no_extern_c=True,
        )
        self._kernels["cell"] = mod.get_function("setCellMatrix")
        self._kernels["rng"] = mod.get_function("initialiseRNG")
        self._kernels["rf"] = mod.get_function("setReactionField")
        self._kernels["atom_properties"] = mod.get_function("setAtomProperties")
        self._kernels["atom_positions"] = mod.get_function("setAtomPositions")
        self._kernels["water_properties"] = mod.get_function("setWaterProperties")
        self._kernels["update_water"] = mod.get_function("updateWater")
        self._kernels["deletion"] = mod.get_function("findDeletionCandidates")
        self._kernels["water"] = mod.get_function("generateWater")
        self._kernels["energy"] = mod.get_function("computeEnergy")
        self._kernels["acceptance"] = mod.get_function("checkAcceptance")

        # Work out the number of blocks to process the atoms.
        self._atom_blocks = self._num_atoms // self._num_threads + 1

        # Work out the number of blocks to process the attempts.
        self._attempt_blocks = self._num_attempts // self._num_threads + 1

        # Work out the number of blocks to process the waters.
        self._water_blocks = self._num_waters // self._num_threads + 1

        # Initialise the GPU memory.
        self._initialise_gpu_memory()

        # Set constants.

        # Energy conversion factors.
        self._beta = 1.0 / (
            _sr.units.k_boltz.to("kcal/(mol*kelvin)") * self._temperature.value()
        )
        self._beta_openmm = 1.0 / (
            _openmm.unit.BOLTZMANN_CONSTANT_kB
            * _openmm.unit.AVOGADRO_CONSTANT_NA
            * self._temperature.value()
            * _openmm.unit.kelvin
        )

        # Work out the volume of the system and GCMC sphere.
        volume = self._space.volume().value()
        gcmc_volume = (4.0 * _np.pi * self._radius.value() ** 3) / 3.0

        # Work out the Adams value.
        B = (
            self._beta * self._excess_chemical_potential.value()
            + _np.log(gcmc_volume / self._standard_volume.value())
        ) + self._adams_shift

        # Work out the bulk Adams value.
        B_bulk = (
            self._beta * self._excess_chemical_potential.value()
            + _np.log(volume / self._standard_volume.value())
        ) + self._adams_shift

        # Store the exponentials for the Adams values.
        self._exp_B = _np.exp(B)
        self._exp_minus_B = _np.exp(-B)
        self._exp_B_bulk = _np.exp(B_bulk)
        self._exp_minus_B_bulk = _np.exp(-B_bulk)

        # Coulomb energy prefactor.
        self._prefactor = 1.0 / (4.0 * _np.pi * _sr.units.epsilon0.value())

        # Zero the number of waters in the sampling volume.
        self._N = 0

        # Zero the statistics.
        self._num_moves = 0
        self._num_accepted = 0
        self._num_accepted_attempts = 0
        self._num_insertions = 0
        self._num_deletions = 0
        self._num_insertion_candidates = 0
        self._num_deletion_candidates = 0
        self._num_pme_attempts = 0

        # Null the nonbonded force.
        self._nonbonded_force = None

        # Flag for whether the last move was a bulk sampling move.
        self._is_bulk = False

        import sys

        # Create a logger that writes to stderr.
        _logger.remove()
        _logger.add(sys.stderr, level=self._log_level.upper())

        # Log the Adams value.
        _logger.debug(f"Adams value: {B:.6f}")

        import atexit

        # Register the cleanup function.
        atexit.register(self._cleanup)

    def __str__(self):
        """
        Return a string representation of the class.
        """

        return (
            f"GCMCSampler(system={self._system}, "
            f"reference={self._reference}, "
            f"radius={self._radius}, "
            f"cutoff_type={self._cutoff_type}, "
            f"cutoff={self._cutoff}, "
            f"excess_chemical_potential={self._excess_chemical_potential}, "
            f"standard_volume={self._standard_volume}, "
            f"temperature={self._temperature}, "
            f"max_gcmc_waters={self._max_gcmc_waters}, "
            f"adams_shift={self._adams_shift}, "
            f"num_attempts={self._num_attempts}, "
            f"num_threads={self._num_threads}), "
            f"bulk_sampling_probability={self._bulk_sampling_probability}, "
            f"water_template={self._water_template}, "
            f"device={self._device}, "
            f"log_level={self._log_level}, "
            f"seed={self._seed})"
        )

    def __repr__(self):
        """
        Return a string representation of the class.
        """

        return str(self)

    def _cleanup(self):
        """
        Clean up the CUDA context.
        """
        from pycuda.tools import clear_context_caches

        self._context.pop()
        self._context = None
        clear_context_caches()

    def system(self):
        """
        Return the GCMC system.

        Returns
        -------

        system: sire.system.System
            The GCMC system.
        """
        return self._system

    def num_waters(self):
        """
        Return the number of waters in the GCMC region.

        Returns
        -------

        num_waters: int
            The number of waters.
        """

        # The last move was a bulk sampling move, so we need to recalculate
        # the number of waters in the GCMC sphere.
        if self._reference is not None and self._is_bulk:
            # Get the OpenMM state.
            state = self._context.getState(getPositions=True)

            # Get the current positions in Angstrom.
            positions = state.getPositions(asNumpy=True) / _openmm.unit.angstrom

            # Get the position of the GCMC sphere center.
            target = _gpuarray.to_gpu(
                self._get_target_position(positions).astype(_np.float32)
            )

            # Set the positions on the GPU.
            self._kernels["atom_positions"](
                _gpuarray.to_gpu(positions.astype(_np.float32).flatten()),
                _np.float32(1.0),
                block=(self._num_threads, 1, 1),
                grid=(self._atom_blocks, 1, 1),
            )

            self._kernels["deletion"](
                self._deletion_candidates,
                _gpuarray.to_gpu(target.astype(_np.float32)),
                _np.float32(self._radius.value()),
                block=(self._num_threads, 1, 1),
                grid=(self._water_blocks, 1, 1),
            )

            # Get the candidates.
            candidates = self._deletion_candidates.get().flatten()

            # Find the waters within the GCMC sphere.
            candidates = _np.where(candidates == 1)[0]

            # Set the number of waters.
            self._N = len(candidates)

            # Reset the bulk sampling flag.
            self._is_bulk = False

        return self._N

    def num_accepted_moves(self):
        """
        Return the number of accepted moves.

        Returns
        -------

        num_accepted: int
            The number of accepted moves.
        """
        return self._num_accepted

    def num_accepted_attempts(self):
        """
        Return the number accepted attempts. This is only valid when using RF.

        Returns
        -------

        num_accepted_attempts: int
            The total number of accepted attempts.
        """
        return self._num_accepted_attempts

    def move_acceptance_probability(self):
        """
        Return the acceptance probability.

        Returns
        -------

        acceptance_probability: float
            The acceptance probability.
        """
        return self._num_accepted / self._num_moves

    def attempt_acceptance_probability(self):
        """
        Return the acceptance probability per attempt. This is only valid
        when using RF.

        Returns
        -------

        acceptance_probability: float
            The acceptance probability per attempt.
        """
        return self._num_accepted_attempts / (2 * self._num_moves * self._num_attempts)

    def pme_acceptance_probability(self):
        """
        Return the PME acceptance probability.

        Returns
        -------

        acceptance_probability: float
            The PME acceptance probability.
        """
        return self._num_accepted / self._num_pme_attempts

    def num_insertions(self):
        """
        Return the number of accepted insertions.

        Returns
        -------

        num_insertions: int
            The number of accepted insertions.
        """
        return self._num_insertions

    def num_deletions(self):
        """
        Return the number of accepted deletions.

        Returns
        -------

        num_deletions: int
            The number of accepted deletions.
        """
        return self._num_deletions

    def reset(self):
        """
        Reset the sampler.
        """
        # Zero the number of accepted moves.
        self._num_accepted = 0
        self._num_insertions = 0
        self._num_deletions = 0
        self._num_moves = 0
        self._num_accepted_attempts = 0
        self._num_pme_attempts = 0

    def move(self, context):
        """
        Perform a trial move.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.

        accepted: bool
            Whether the move was accepted.

        move: str
            The type of move. (If accepted.)
        """

        # Increment the number of moves.
        self._num_moves += 1

        # Set the NonBondedForce.
        self._set_nonbonded_force(context)

        # Get the OpenMM state.
        state = context.getState(getPositions=True, getEnergy=self._is_pme)

        # Get the current positions in Angstrom.
        positions = state.getPositions(asNumpy=True) / _openmm.unit.angstrom

        # If we're using PME, then store the initial energy.
        if self._is_pme:
            initial_energy = state.getPotentialEnergy()

        self._is_bulk = True
        if self._reference is not None:
            # Sample within the GCMC sphere.
            if self._rng.random() > self._bulk_sampling_probability:
                target = _gpuarray.to_gpu(
                    self._get_target_position(positions).astype(_np.float32)
                )
                self._is_bulk = False

        # Set the positions on the GPU.
        self._kernels["atom_positions"](
            _gpuarray.to_gpu(positions.astype(_np.float32).flatten()),
            _np.float32(1.0),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, 1, 1),
        )

        # Work out the number of waters in the sampling volume.
        if not self._is_bulk:
            self._kernels["deletion"](
                self._deletion_candidates,
                _gpuarray.to_gpu(target.astype(_np.float32)),
                _np.float32(self._radius.value()),
                block=(self._num_threads, 1, 1),
                grid=(self._water_blocks, 1, 1),
            )

            # Get the candidates.
            candidates = self._deletion_candidates.get().flatten()

            # Find the waters within the GCMC sphere.
            candidates = _np.where(candidates == 1)[0]

            # Set the Adams factors.
            exp_B = self._exp_B
            exp_minus_B = self._exp_minus_B

        # Use all non-ghost waters.
        else:
            _logger.debug("Sampling within the entire simulation box")
            candidates = _np.where(self._water_state != 0)[0]
            target = None

            # Set the Adams factors.
            exp_B = self._exp_B_bulk
            exp_minus_B = self._exp_minus_B_bulk

        # Set the number of waters.
        self._N = len(candidates)

        # Log the current number of waters.
        _logger.debug(f"Number of waters in sampling volume: {self._N}")

        # Perform a batch of insertion trials.
        insertions, energy_change_insertion = self._insertion_move(target=target)

        # Perform a batch of deletion trials.
        deletions, candidates, energy_change_deletion = self._deletion_move(candidates)

        # Store the total number of accepted attempts.
        num_insertions = len(insertions)
        num_deletions = len(deletions)
        num_accepted_attempts = num_insertions + num_deletions

        # Update the total number of insertion and deletion candidates.
        self._num_insertion_candidates += num_insertions
        self._num_deletion_candidates += num_deletions

        _logger.debug(f"Number of insertion candidates: {num_insertions}")
        _logger.debug(f"Number of deletion candidates: {num_deletions}")
        _logger.debug(
            f"Total number of insertion candidates: {self._num_insertion_candidates}"
        )
        _logger.debug(
            f"Total number of deletion candidates: {self._num_deletion_candidates}"
        )

        # Reaction field.
        if not self._is_pme:
            # No moves were accepted, stay in the same state.
            if num_accepted_attempts == 0:
                return context, False, None

            # Choose a move at random.
            state = self._rng.integers(num_accepted_attempts)

            # Insertion.
            if state < num_insertions:
                # Work out the state.
                state = insertions[state]

                # Accept the move.
                context, idx = self._accept_insertion(state, context)

                # Update the acceptance statistics.
                self._num_accepted += 1
                self._num_accepted_attempts += num_accepted_attempts
                self._num_insertions += 1

                is_accepted = True
                move = "insertion"

                # Log the insertion.
                if self._is_debug:
                    self._log_insertion(state, idx)

            # Deletion.
            else:
                # Work out the deletion state.
                state = deletions[state - num_insertions]

                # Accept the move.
                context, previous_state = self._accept_deletion(
                    candidates[state], context
                )

                # Update the acceptance statistics.
                self._num_accepted += 1
                self._num_accepted_attempts += num_accepted_attempts
                self._num_deletions += 1

                is_accepted = True
                move = "deletion"

                # Log the deletion.
                if self._is_debug:
                    self._log_deletion(state, candidates, positions)

        # PME.
        else:
            # Shuffle the candidates.
            candidate_indices = _np.arange(num_accepted_attempts)
            self._rng.shuffle(candidate_indices)

            is_accepted = False
            move = None

            # Loop through the canidates until a move is accepted.
            for candidate in candidate_indices:
                # Update the number of PME attempts.
                self._num_pme_attempts += 1

                # Determine whether this is an insertion or deletion.
                if candidate < num_insertions:
                    state = insertions[candidate]

                    # Store the RF energy difference. (in kcal/mol).
                    dE_rf = (
                        energy_change_insertion[state]
                        * _openmm.unit.kilocalories_per_mole
                    )

                    # Accept the move.
                    context, idx = self._accept_insertion(state, context)

                    # Get the new energy.
                    final_energy = context.getState(getEnergy=True).getPotentialEnergy()

                    # Compute the PME acceptance correction.
                    acc_prob = _np.exp(
                        -self._beta_openmm * (final_energy - initial_energy - dE_rf)
                    )

                    # The move was rejected.
                    if acc_prob < self._rng.random():
                        # Revert the move by deleting the water.
                        context, _ = self._accept_deletion(idx, context)
                    else:
                        # Update the acceptance statistics.
                        self._num_accepted += 1
                        self._num_accepted_attempts += num_accepted_attempts
                        self._num_insertions += 1

                        # Log the insertion.
                        if self._is_debug:
                            self._log_insertion(
                                state,
                                idx,
                                pme_energy=final_energy - initial_energy,
                                pme_probability=acc_prob,
                            )

                        # Accept the move.
                        is_accepted = True
                        move = "insertion"
                        break

                else:
                    state = deletions[candidate - num_insertions]

                    # Store the RF energy difference. (in kcal/mol).
                    dE_rf = (
                        energy_change_deletion[state]
                        * _openmm.unit.kilocalories_per_mole
                    )

                    # Accept the move.
                    context, previous_state = self._accept_deletion(
                        candidates[state], context
                    )

                    # Get the new energy.
                    final_energy = context.getState(getEnergy=True).getPotentialEnergy()

                    # Compute the PME acceptance correction.
                    acc_prob = _np.exp(
                        -self._beta_openmm * (final_energy - initial_energy - dE_rf)
                    )

                    # The move was rejected.
                    if acc_prob < self._rng.random():
                        # Revert the move.
                        context = self._reject_deletion(
                            candidates[state], previous_state, context
                        )
                    else:
                        # Update the acceptance statistics.
                        self._num_accepted += 1
                        self._num_accepted_attempts += num_accepted_attempts
                        self._num_deletions += 1

                        # Log the deletion.
                        if self._is_debug:
                            self._log_deletion(
                                state,
                                candidates,
                                positions,
                                pme_energy=final_energy - initial_energy,
                                pme_probability=acc_prob,
                            )

                        # Accept the move.
                        is_accepted = True
                        move = "deletion"
                        break

        # If this was a bulk sampling move, then store the context. This allows
        # us to work out the number of waters in the GCMC sphere if the user
        # calls self.num_waters() after the move.
        if self._reference is not None and self._is_bulk:
            self._context = context

        return context, is_accepted, move

    def _insertion_move(self, target=None):
        """
        Perform a trial insertion move.

        Parameters
        ----------

        target: pycuda.gpuarray.GPUArray
            The center of the GCMC sphere.

        Returns
        -------

        accepted: np.ndarray
            The accepted candidates.

        energy_change: np.ndarray
            The energy change for each insertion. This is needed to compute
            the PME acceptance probability correction.
        """

        _logger.debug("Performing insertion move.")

        if target is None:
            target = _gpuarray.to_gpu(_np.zeros(3, dtype=_np.float32))
            is_target = _np.int32(0)
            exp_B = self._exp_B_bulk
        else:
            is_target = _np.int32(1)
            exp_B = self._exp_B

        # Generate the random water positions and orientations.
        self._kernels["water"](
            self._water_template_positions,
            target,
            _np.float32(self._radius.value()),
            self._water_positions,
            is_target,
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Perform the energy calculation.
        self._kernels["energy"](
            self._water_positions,
            self._energy_coul_insert,
            self._energy_lj_insert,
            self._deletion_candidates,
            _np.int32(0),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, self._num_attempts, 1),
        )

        # Compute the acceptance probabilities.
        self._kernels["acceptance"](
            _np.int32(self._N),
            _np.int32(1),
            _np.float32(1.0),
            _np.float32(exp_B),
            _np.float32(self._beta),
            self._energy_coul_insert,
            self._energy_lj_insert,
            self._energy_change,
            self._probability_insert,
            self._accepted,
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Get the energy change.
        if self._is_pme:
            energy_change = self._energy_change.get().flatten()
        else:
            energy_change = None

        # Get the state acceptance array.
        accepted_cpu = self._accepted.get().flatten()

        return _np.where(accepted_cpu == 1)[0], energy_change

    def _deletion_move(self, candidates):
        """
        Perform a trial deletion move.

        Parameters
        ----------

        candidates: np.ndarray
            The indices of the candidate waters.

        Returns
        -------

        accepted: np.ndarray
            The accepted candidates.

        candidates: np.ndarray
            The full set of trial candidates.

        energy_change: np.ndarray
            The energy change for each deletion. This is needed to compute
            the PME acceptance probability correction.
        """

        _logger.debug("Performing deletion move.")

        # Log the candidates.
        _logger.debug(f"Number of deletion candidates: {len(candidates)}")
        _logger.debug(f"Deletion candidates: {candidates}")

        if len(candidates) == 0:
            return [], candidates, None
        else:
            # Draw num_attempts samples from the candidates.
            candidates = self._rng.choice(candidates, size=self._num_attempts)

        # Set the Adams factor.
        if self._is_bulk:
            exp_minus_B = self._exp_minus_B_bulk
        else:
            exp_minus_B = self._exp_minus_B

        # Compute the energy of the candidate deletions.
        self._kernels["energy"](
            self._water_positions,
            self._energy_coul_delete,
            self._energy_lj_delete,
            _gpuarray.to_gpu(candidates.astype(_np.int32)),
            _np.int32(1),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, self._num_attempts, 1),
        )

        # Compute the acceptance probabilities.
        self._kernels["acceptance"](
            _np.int32(0),
            _np.int32(self._N),
            _np.float32(-1.0),
            _np.float32(exp_minus_B),
            _np.float32(self._beta),
            self._energy_coul_delete,
            self._energy_lj_delete,
            self._energy_change,
            self._probability_delete,
            self._accepted,
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Get the energy change.
        if self._is_pme:
            energy_change = self._energy_change.get().flatten()
        else:
            energy_change = None

        # Get the state acceptance array.
        accepted_cpu = self._accepted.get().flatten()

        return _np.where(accepted_cpu == 1)[0], candidates, energy_change

    @staticmethod
    def _validate_sire_unit(parameter, value, unit):
        """
        Validate a Sire unit.

        Parameters
        ----------

        parameter: str
            The name of the parameter.

        value: str
            The value to validate.

        unit: str
            The unit to validate.

        Returns
        -------

        u: sire.units.GeneralUnit
            The validated unit.
        """

        if not isinstance(value, str):
            raise ValueError(f"'{parameter}' must be of type 'str'")

        try:
            u = _sr.u(value)
        except Exception as e:
            raise ValueError(f"Could not parse '{parameter}': {e}")

        if not u.has_same_units(unit):
            raise ValueError(f"Invalid units for '{parameter}'")

        return u

    @staticmethod
    def _get_box_information(system):
        """
        Get the box information from the system.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        Returns
        -------

        cell_matrix: pycuda.gpuarray.GPUArray
            The cell matrix.

        cell_matrix_inverse: pycuda.gpuarray.GPUArray
            The inverse of the cell matrix.

        M: pycuda.gpuarray.GPUArray
            The matrix M.
        """
        # Get the box.
        try:
            space = system.property("space")
        except Exception as e:
            raise ValueError(f"System does not contain a periodic box information!")

        cell_matrix = space.box_matrix()
        cell_matrix_inverse = cell_matrix.inverse()
        M = cell_matrix.transpose() * cell_matrix

        # Convert to NumPy.
        row0 = [x.value() for x in cell_matrix.row0()]
        row1 = [x.value() for x in cell_matrix.row1()]
        row2 = [x.value() for x in cell_matrix.row2()]
        cell_matrix = _np.array([row0, row1, row2])
        row0 = [x.value() for x in cell_matrix_inverse.row0()]
        row1 = [x.value() for x in cell_matrix_inverse.row1()]
        row2 = [x.value() for x in cell_matrix_inverse.row2()]
        cell_matrix_inverse = _np.array([row0, row1, row2])
        row0 = [x.value() for x in M.row0()]
        row1 = [x.value() for x in M.row1()]
        row2 = [x.value() for x in M.row2()]
        M = _np.array([row0, row1, row2])

        # Convert to GPU memory.
        cell_matrix = _gpuarray.to_gpu(cell_matrix.flatten().astype(_np.float32))
        cell_matrix_inverse = _gpuarray.to_gpu(
            cell_matrix_inverse.flatten().astype(_np.float32)
        )
        M = _gpuarray.to_gpu(M.flatten().astype(_np.float32))

        return space, cell_matrix, cell_matrix_inverse, M

    @staticmethod
    def _get_reference_indices(system, reference):
        """
        Get the indices of the reference atoms.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        reference: str
            A selection string for the reference atoms.

        Returns
        -------

        indices: numpy.ndarray
            The indices of the reference atoms.
        """

        # Convert the system to a BioSimSpace object.
        bss_system = _BSS._SireWrappers.System(system._system)

        try:
            atoms = bss_system.search(reference).atoms()
        except Exception as e:
            raise ValueError(f"Could not get the reference atoms: {e}")

        # Get the absolute indices of the atoms.
        indices = []
        for atom in atoms:
            indices.append(bss_system.getIndex(atom))

        return _np.array(indices)

    @staticmethod
    def _prepare_system(system, water_template, rng, max_gcmc_waters):
        """
        Prepare the system for GCMC sampling.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        water_template: sire.molecule.Molecule
            The water template.

        rng: numpy.random.Generator
            The random number generator.

        max_gcmc_waters: int
            The maximum number of GCMC waters to insert.

        Returns
        -------

        system: sire.system.System
            The prepared system.

        water_indices: numpy.ndarray
            The indices of the oxygen atoms in each water molecule.
        """

        # Edit the template so that it is non-interacting.
        cursor = water_template.cursor()
        for atom in cursor.atoms():
            atom["charge"] = 0.0 * _sr.units.mod_electron
            atom["LJ"] = _sr.legacy.MM.LJParameter(
                1.0 * _sr.units.angstrom, 0.0 * _sr.units.kcal_per_mol
            )
        water_template = _BSS._SireWrappers.Molecule(cursor.commit())

        # Create a BioSimSpace system.
        bss_system = _BSS._SireWrappers.System(system._system)

        # First create the GCMC waters.
        waters = []
        for i in range(max_gcmc_waters):
            # Create a copy of the water template with a new molecule number.
            water = water_template.copy()
            # Randomly translate the water so that it is not on top of another.
            water.translate(
                (2.0 * rng.random() - 1.0) * _BSS.Units.Length.angstrom * [1, 1, 1]
            )
            waters.append(water)

        # Add the waters to the system.
        bss_system += waters

        # Search for the water oxygen atoms.
        water_indices = []
        for atom in bss_system.search("water and element O").atoms():
            water_indices.append(bss_system.getIndex(atom))

        return _sr.system.System(bss_system._sire_object), _np.array(water_indices)

    def _initialise_gpu_memory(self):
        """
        Initialise the GPU memory.
        """

        # First get the atomic properties.

        # Get the charges on all the atoms.
        try:
            charges = []
            for mol in self._system:
                charges_mol = [charge.value() for charge in mol.property("charge")]
                charges.extend(charges_mol)

            # Convert to a GPU array.
            charges = _gpuarray.to_gpu(_np.array(charges).astype(_np.float32))

        except Exception as e:
            raise ValueError(f"Could not get the charges on the atoms: {e}")

        # Try to get the sigma and epsilon for the atoms.
        try:
            sigmas = []
            epsilons = []
            for mol in self._system:
                for lj in mol.property("LJ"):
                    sigmas.append(lj.sigma().value())
                    epsilons.append(lj.epsilon().value())

            # Convert to GPU arrays.
            sigmas = _gpuarray.to_gpu(_np.array(sigmas).astype(_np.float32))
            epsilons = _gpuarray.to_gpu(_np.array(epsilons).astype(_np.float32))

        except Exception as e:
            raise ValueError(f"Could not get the LJ parameters: {e}")

        # Get the water properties.
        try:
            charge_water = []
            sigma_water = []
            epsilon_water = []
            for atom in self._water_template.atoms():
                charge_water.append(atom.charge().value())
                lj = atom.property("LJ")
                sigma_water.append(lj.sigma().value())
                epsilon_water.append(lj.epsilon().value())

            # Store the water properties.
            self._water_charge = _np.array(charge_water)
            self._water_sigma = _np.array(sigma_water)
            self._water_epsilon = _np.array(epsilon_water)

            # Convert to GPU arrays.
            charge_water = _gpuarray.to_gpu(self._water_charge.astype(_np.float32))
            sigma_water = _gpuarray.to_gpu(self._water_sigma.astype(_np.float32))
            epsilon_water = _gpuarray.to_gpu(self._water_epsilon.astype(_np.float32))

        except Exception as e:
            raise ValueError(f"Could not get the atomic properties of the water: {e}")

        # Initialise the water state: 0 = ghost, 1 = GCMC, 2 = normal.
        water_state = []
        is_ghost = []
        for i in range(self._num_waters):
            if i < self._num_waters - self._max_gcmc_waters:
                water_state.append(2)
                is_ghost.extend([0] * self._num_points)
            else:
                water_state.append(0)
                is_ghost.extend([1] * self._num_points)
        self._water_state = _np.array(water_state).astype(_np.int32)
        is_ghost = _gpuarray.to_gpu(_np.array(is_ghost).astype(_np.int32))

        # Initialise the cell.
        self._kernels["cell"](
            self._cell_matrix,
            self._cell_matrix_inverse,
            self._M,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Initialise the random number generator.
        self._kernels["rng"](
            _gpuarray.to_gpu(
                _np.random.randint(
                    _np.iinfo(_np.int32).max, size=(1, self._num_attempts)
                ).astype(_np.int32)
            ),
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Initialise the reaction field parameters.
        self._kernels["rf"](
            _np.float32(self._cutoff.value()),
            _np.float32(78.3),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Set the atomic properties.
        self._kernels["atom_properties"](
            charges,
            sigmas,
            epsilons,
            is_ghost,
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, 1, 1),
        )

        # Set the water properties.
        self._kernels["water_properties"](
            charge_water,
            sigma_water,
            epsilon_water,
            _gpuarray.to_gpu(self._water_indices.astype(_np.int32)),
            _gpuarray.to_gpu(self._water_state.astype(_np.int32)),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Initialise the memory to store the water positions.
        self._water_positions = _gpuarray.empty(
            (1, self._num_attempts * 3 * self._num_points), _np.float32
        )

        # Initialise memory to store the energy.
        self._energy_coul_insert = _gpuarray.empty(
            (1, self._num_attempts * self._num_atoms), _np.float32
        )
        self._energy_coul_delete = _gpuarray.empty(
            (1, self._num_attempts * self._num_atoms), _np.float32
        )
        self._energy_lj_insert = _gpuarray.empty(
            (1, self._num_attempts * self._num_atoms), _np.float32
        )
        self._energy_lj_delete = _gpuarray.empty(
            (1, self._num_attempts * self._num_atoms), _np.float32
        )

        # Initialise memory to store whether each attempt is accepted and
        # the probability of acceptance.
        self._accepted = _gpuarray.empty((1, self._num_attempts), _np.int32)
        self._energy_change = _gpuarray.empty((1, self._num_attempts), _np.float32)
        self._probability_insert = _gpuarray.empty((1, self._num_attempts), _np.float32)
        self._probability_delete = _gpuarray.empty((1, self._num_attempts), _np.float32)

        # Initialise memory to store the deletion candidates.
        self._deletion_candidates = _gpuarray.empty((1, self._num_waters), _np.int32)

    def _accept_insertion(self, state, context):
        """
        Accept a insertion move.

        Parameters
        ----------

        state: int
            The index of the accepted state.

        context: openmm.Context
            The OpenMM context to update.

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.

        idx: int
            The index of the water that was inserted.
        """

        if self._N >= self._total_waters:
            raise RuntimeError(
                "Cannot insert any more waters. Please increase 'max_gcmc_waters'."
            )

        # Get the new water positions.
        water_positions = self._water_positions.get().reshape(
            (self._num_attempts, 3, self._num_points)
        )[state]

        # Choose a random ghost water.
        idx = self._rng.choice(_np.where(self._water_state == 0)[0])

        # Update the water state.
        self._water_state[idx] = 1

        # Get the starting atom index.
        start_idx = self._water_indices[idx]

        # Update the water positions and NonBondedForce.
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        for i in range(self._num_points):
            positions[start_idx + i] = _openmm.unit.Quantity(
                water_positions[i], _openmm.unit.angstrom
            )
            self._nonbonded_force.setParticleParameters(
                start_idx + i,
                self._water_charge[i] * _openmm.unit.elementary_charge,
                self._water_sigma[i] * _openmm.unit.angstrom,
                self._water_epsilon[i] * _openmm.unit.kilocalorie_per_mole,
            )

        # Set the new positions.
        context.setPositions(positions)

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(idx),
            _np.int32(1),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of waters in the sampling volume.
        self._N += 1

        return context, idx

    def _accept_deletion(self, idx, context):
        """
        Accept a deletion move.

        Parameters
        ----------

        idx: int
            The index of the accepted state.

        context: openmm.Context
            The OpenMM context to update.

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.

        previous_state: int
            The previous state of the water.
        """

        # Store the curent water state.
        previous_state = self._water_state[idx]

        # Update the water state.
        self._water_state[idx] = 0

        # Get the starting atom index.
        start_idx = self._water_indices[idx]

        # Update the NonBondedForce.
        for i in range(self._num_points):
            self._nonbonded_force.setParticleParameters(start_idx + i, 0.0, 1.0, 0.0)

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(idx),
            _np.int32(0),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of waters in the sampling volume.
        self._N -= 1

        return context, previous_state

    def _reject_deletion(self, idx, state, context):
        """
        Reject a deletion move.

        Parameters
        ----------

        idx: int
            The index of the water.

        state: int
            The previous state of the water.

        context: openmm.Context
            The OpenMM context to update.

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.
        """

        # Reset the water state.
        self._water_state[idx] = state

        # Get the starting atom index.
        start_idx = self._water_indices[idx]

        # Update the NonBondedForce.
        for i in range(self._num_points):
            self._nonbonded_force.setParticleParameters(
                start_idx + i,
                self._water_charge[i] * _openmm.unit.elementary_charge,
                self._water_sigma[i] * _openmm.unit.angstrom,
                self._water_epsilon[i] * _openmm.unit.kilocalorie_per_mole,
            )

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(idx),
            _np.int32(state),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of waters in the sampling volume.
        self._N += 1

        return context

    def _set_nonbonded_force(self, context):
        """
        Find the NonBondedForce in the system.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.
        """
        # Find the NonBondedForce.
        if self._nonbonded_force is None:
            if self._nonbonded_force is None:
                for force in context.getSystem().getForces():
                    if isinstance(force, _openmm.NonbondedForce):
                        self._nonbonded_force = force
                        break

    def _get_target_position(self, positions):
        """
        Get the current center of the GCMC sphere.

        Parameters
        ----------

        positions: numpy.ndarray
            The current positions of the system.

        Returns
        -------

        target: numpy.ndarray
            The center of the GCMC sphere.
        """

        # Work out center of geometry of the reference atoms.
        com = _sr.maths.Vector(*positions[self._reference_indices[0]])
        target = com
        for index in self._reference_indices[1:]:
            delta = self._space.calcDistVector(
                target, _sr.maths.Vector(*positions[index])
            )
            com += target + _sr.maths.Vector(delta.x(), delta.y(), delta.z())
        target = _np.array([x.value() for x in com / len(self._reference_indices)])

        _logger.debug(f"GCMC sphere center: {target}")

        return target

    def _log_insertion(self, state, idx, pme_energy=None, pme_probability=None):
        """
        Log information about the accepted insertion move.

        Parameters
        ----------

        state: int
            The index of the accepted state.

        idx: int
            The index of the water that was inserted.

        pme_energy: openmm.Quantity
            The PME energy difference.

        pme_probability: float
            The PME acceptance probability.
        """
        # Get the energies.
        energy_coul = self._energy_coul_insert.get().reshape(
            (self._num_attempts, self._num_atoms)
        )
        energy_lj = self._energy_lj_insert.get().reshape(
            (self._num_attempts, self._num_atoms)
        )

        # Get the water positions.
        water_positions = self._water_positions.get().reshape(
            (self._num_attempts, self._num_points, 3)
        )

        # Get the RF acceptance probability.
        probability = self._probability_insert.get().flatten()

        # Store debugging attributes.
        self._debug = {
            "move": "insertion",
            "idx": idx,
            "energy_coul": self._prefactor * energy_coul[state].sum(),
            "energy_lj": energy_lj[state].sum(),
            "probability_rf": probability[state],
        }

        # Log the accepted candidate.
        _logger.debug(f"Accepted insertion: candidate={state}, water={idx}")

        # Log the position of the inserted oxygen atom.
        _logger.debug(f"Inserted oxygen position: {water_positions[state, 0]}")

        # Log the energies of the accepted candidate.
        _logger.debug(f"RF coulomb energy: {self._debug['energy_coul']:.6f} kcal/mol")
        _logger.debug(f"LJ energy: {self._debug['energy_lj']:.6f} kcal/mol")
        _logger.debug(
            f"Total RF energy difference: {self._debug['energy_coul'] + self._debug['energy_lj']:.6f} kcal/mol"
        )
        _logger.debug(f"RF insertion probability: {probability[state]:.6f}")

        # Add PME energy if available.
        if pme_energy is not None:
            self._debug["pme_energy"] = pme_energy.value_in_unit(
                _openmm.unit.kilocalorie_per_mole
            )
            self._debug["probability_pme"] = pme_probability

            _logger.debug(
                f"Total PME energy difference: {self._debug['pme_energy']:.6f} kcal/mol"
            )
            _logger.debug(f"PME insertion probability: {pme_probability:.6f}")

    def _log_deletion(
        self, state, candidates, positions, pme_energy=None, pme_probability=None
    ):
        """
        Log information about the accepted deletion move.

        Parameters
        ----------

        state: int
            The index of the accepted state.

        candidates: numpy.ndarray
            The indices of the candidate waters.

        positions: numpy.ndarray
            The positions of the system.

        pme_energy: openmm.Quantity
            The PME energy difference.

        pme_probability: float
            The PME acceptance probability.
        """
        # Get the coulomb and LJ energies.
        energy_coul = self._energy_coul_delete.get().reshape(
            (self._num_attempts, self._num_atoms)
        )
        energy_lj = self._energy_lj_delete.get().reshape(
            (self._num_attempts, self._num_atoms)
        )

        # Get the RF acceptance probability.
        probability = self._probability_delete.get().flatten()

        # Log the accepted candidate.
        _logger.debug(
            f"Accepted deletion: candidate={state}, water={candidates[state]}"
        )

        # Get the water index.
        water_idx = self._water_indices[candidates[state]]

        # Store debugging attributes.
        self._debug = {
            "move": "deletion",
            "idx": self._water_indices[candidates[state]],
            "energy_coul": -self._prefactor * energy_coul[state].sum(),
            "energy_lj": -energy_lj[state].sum(),
            "probability_rf": probability[state],
        }

        # Log the oxygen position.
        _logger.debug(f"Deleted oxygen position: {positions[water_idx]}")

        # Log the energies of the accepted candidate.
        _logger.debug(f"RF coulomb energy: {self._debug['energy_coul']:.6f} kcal/mol")
        _logger.debug(f"LJ energy: {self._debug['energy_lj']:.6f} kcal/mol")
        _logger.debug(
            f"Total RF energy difference: {self._debug['energy_coul'] + self._debug['energy_lj']:.6f} kcal/mol"
        )
        _logger.debug(f"RF deletion probability: {probability[state]:.6f}")

        # Add PME energy if available.
        if pme_energy is not None:
            self._debug["pme_energy"] = pme_energy.value_in_unit(
                _openmm.unit.kilocalorie_per_mole
            )
            self._debug["probability_pme"] = pme_probability

            _logger.debug(
                f"Total PME energy difference: {self._debug['pme_energy']:.6f} kcal/mol"
            )
            _logger.debug(f"PME deletion probability: {pme_probability:.6f}")
