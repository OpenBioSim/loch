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
        cutoff_type="pme",
        cutoff="10.0 A",
        excess_chemical_potential="-6.09 kcal/mol",
        standard_volume="30.543 A^3",
        temperature="298 K",
        adams_shift=0.0,
        num_ghost_waters=20,
        batch_size=1000,
        num_attempts=10000,
        num_threads=1024,
        bulk_sampling_probability=0.1,
        water_template=None,
        device=None,
        tolerance=0.0,
        lambda_schedule=None,
        lambda_value=0.0,
        rest2_scale=1.0,
        rest2_selection=None,
        coulomb_power=0.0,
        shift_coulomb="1 A",
        shift_delta="2.25 A",
        overwrite=False,
        ghost_file="ghosts.txt",
        log_file="gcmc.txt",
        log_level="error",
        seed=None,
        **kwargs,
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
            The type of cutoff to use: "pme" or "rf".

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

        num_ghost_waters: int
            The initial number of ghost waters to add to the system. These are
            used for GCMC insertion moves, so no more insertions can be made
            once they are exhausted.

        batch_size: int
            The number of random insertions and deletion trials per batch.
            This should be tuned according to the attempt acceptance
            probability i.e. aim to accept an average of 1 move per batch
            to avoid wasted computation.

        num_attempts: int
            The total number of attempts per move. In each batch, the lowest
            candidate index of the first accepted state will be used to
            determine the number of attempts, i.e. we will only accept
            the first accepted state. This must be greater than or equal
            to the batch size.

        num_threads: int
            The number of threads per block. (Must be a multiple of 32.)

        bulk_sampling_probability: float
            The probability of perforing trial insertion and deletion
            moves within the entire simulation box, rather than within
            the GCMC sphere. This option is only relevant when 'reference'
            is not None.

        water_template: sire.molecule.Molecule
            A water molecule to use as a template. This is only required when
            the system does not contain any water molecules. If provided, water
            parameters for the GCMC insertion and deletion trials will be taken
            from the template.

        device: int
            The CUDA device index. (This is the index in the list of visible
            devices.)

        tolerance: float
            The tolerance for the acceptance probability, i.e. the minimum
            probability of acceptance for a move. This can be used to exclude
            low probability candidates that can cause instabilities or crashes
            for the MD engine.

        lambda_schedule: sire.cas.LambdaSchedule
            The lambda schedule if the passed system is an alchemical system.

        lambda_value: float
            The lambda value if the passed system is an alchemical system.

        rest2_scale: float
            The scaling factor if using Replica Exchange with Solute Tempering
            (REST2) for alchemical systems. This should specify the temperature
            of the REST2 system relative to the rest of the system.

        rest2_selection: str
            A selection string for atoms to include in the REST2 region in
            addition to any perturbable molecules. For example, "molidx 0 and
            residx 0,1,2" would select atoms from the first three residues of the
            first molecule. If None, then all atoms within perturbable molecules
            will be included in the REST2 region. When atoms within a perturbable
            molecule are included in the selection, then only those atoms will be
            considered as part of the REST2 region. This allows REST2 to be applied
            to protein mutations.

        couloumb_power : float
            Power to use for the soft-core Coulomb interaction. This is used
            to soften the electrostatic interaction.

        shift_coulomb : str
            The soft-core shift-coulomb parameter. This is used to soften the
            Coulomb interaction.

        shift_delta : str
            The soft-core shift-delta parameter. This is used to soften the
            Lennard-Jones interaction.

        overwrite: bool
            Overwrite existing log files.

        dcd_file: str
            The file to write the GCMC trajectory to.

        ghost_file: str
            The file to write the ghost residue indices to.

        log_level: str
            The logging level.

        seed: int
            The seed for the random number generator.
        """

        # Validate the input.

        if not isinstance(system, _sr.system.System):
            raise ValueError("'system' must be of type 'sire.system.System'")
        self._system = system

        # Check whether this is an alchemical system.
        try:
            if len(self._system["property is_perturbable"].molecules()) > 0:
                self._is_fep = True
                self._system = _sr.morph.link_to_reference(self._system)
            else:
                self._is_fep = False
        except:
            self._is_fep = False

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

        if not isinstance(num_ghost_waters, int):
            raise ValueError("'num_ghost_waters' must be of type 'int'")
        self._num_ghost_waters = num_ghost_waters

        if not isinstance(adams_shift, (int, float)):
            raise ValueError("'adams_shift' must be of type 'int' or 'float'")
        self._adams_shift = float(adams_shift)

        if not isinstance(batch_size, int):
            raise ValueError("'batch_size' must be of type 'int'")
        if batch_size <= 0:
            raise ValueError("'batch_size' must be greater than 0")
        self._batch_size = batch_size

        if not isinstance(num_attempts, int):
            raise ValueError("'num_attempts' must be of type 'int'")
        if num_attempts <= 0:
            raise ValueError("'num_attempts' must be greater than 0")
        if num_attempts < batch_size:
            raise ValueError(
                "'num_attempts' must be greater than or equal to 'batch_size'"
            )
        self._num_attempts = num_attempts

        if not isinstance(num_threads, int):
            raise ValueError("'num_threads' must be of type 'int'")
        if not num_threads % 32 == 0:
            raise ValueError("'num_threads' must be a multiple of 32")
        self._num_threads = num_threads

        self.set_bulk_sampling_probability(bulk_sampling_probability)

        if not isinstance(overwrite, bool):
            raise ValueError("'overwrite' must be of type 'bool'")
        self._overwrite = overwrite

        if ghost_file is not None:
            if not isinstance(ghost_file, str):
                raise ValueError("'ghost_file' must be of type 'str'")
            self._ghost_file = ghost_file
            if not isinstance(ghost_file, str):
                raise ValueError("'ghost_file' must be of type 'str'")
            self._ghost_file = ghost_file

            if _os.path.exists(self._ghost_file):
                if not self._overwrite:
                    raise ValueError(
                        "'ghost_file' already exists. Use 'overwrite=True' to overwrite it."
                    )
                else:
                    with open(self._ghost_file, "w") as f:
                        f.write("")
        else:
            self._ghost_file = None

        if log_file is not None:
            if not isinstance(log_file, str):
                raise ValueError("'log_file' must be of type 'str'")
            self._log_file = log_file
            if _os.path.exists(self._log_file):
                if not self._overwrite:
                    raise ValueError(
                        "'log_file' already exists. Use 'overwrite=True' to overwrite it."
                    )
                else:
                    with open(self._log_file, "w") as f:
                        f.write("")
        else:
            self._log_file = None

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

        from pycuda.tools import make_default_context

        # Set the CUDA device.
        _cuda.init()
        if device is not None:
            if not isinstance(device, int):
                raise ValueError("'device' must be of type 'int'")
            if device < 0 or device >= _cuda.Device.count():
                raise ValueError(
                    f"'device' must be between 0 and {cuda.Device.count() - 1}"
                )
            self._pycuda_context = _cuda.Device(device).make_context()
        else:
            self._pycuda_context = make_default_context()
        self._device = self._pycuda_context.get_device()

        # Set the tolerance.
        try:
            self._tolerance = float(tolerance)
        except Exception as e:
            raise ValueError(f"Could not convert 'tolerance' to float: {e}")

        # Check for alchemical properties.
        if lambda_schedule is not None:
            if not isinstance(lambda_schedule, _sr.cas.LambdaSchedule):
                raise ValueError(
                    "'lambda_schedule' must be of type 'sire.cas.LambdaSchedule'"
                )
            self._lambda_schedule = lambda_schedule
        else:
            if self._is_fep:
                raise ValueError(
                    "'lambda_schedule' must be provided for alchemical systems"
                )
            self._lambda_schedule = None

        try:
            lambda_value = float(lambda_value)
        except:
            raise ValueError("'lambda_value' must be of type 'float'")
        if not 0.0 <= lambda_value <= 1.0:
            raise ValueError("'lambda_value' must be between 0 and 1")
        self._lambda_value = float(lambda_value)

        try:
            rest2_scale = float(rest2_scale)
        except:
            raise ValueError("'rest2_scale' must be of type 'float'")
        if rest2_scale < 1.0:
            raise ValueError("'rest2_scale' must be greater than or equal to 1.0")
        self._rest2_scale = rest2_scale

        if rest2_selection is not None:
            if not isinstance(rest2_selection, str):
                raise ValueError("'rest2_selection' must be of type 'str'")

            from sire.mol import selection_to_atoms

            try:
                atoms = selection_to_atoms(self._system, rest2_selection)
            except:
                msg = "Invalid 'rest2_selection' value."
                _logger.error(msg)
                raise ValueError(msg)

            # Make sure the user hasn't selected all atoms.
            if len(atoms) == self._system.num_atoms():
                raise ValueError(
                    "'rest2_selection' cannot contain all atoms in the system."
                )
        self._rest2_selection = rest2_selection

        try:
            coulomb_power = float(coulomb_power)
        except:
            raise ValueError("'coulomb_power' must be of type 'float'")
        self._coulomb_power = float(coulomb_power)

        try:
            self._shift_coulomb = self._validate_sire_unit(
                "shift_coulomb", shift_coulomb, _sr.u("A")
            )
        except Exception as e:
            raise ValueError(f"Could not validate the 'shift_coulomb': {e}")

        try:
            self._shift_delta = self._validate_sire_unit(
                "shift_delta", shift_delta, _sr.u("A")
            )
        except Exception as e:
            raise ValueError(f"Could not validate the 'shift_delta': {e}")

        # Check for waters and validate the template.
        try:
            self._water_template = system["water and not property is_perturbable"][0]
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

        # Get the indices of the reference atoms.
        if self._reference is not None:
            self._reference_indices = self._get_reference_indices(system, reference)

        # Prepare the system for GCMC sampling.
        try:
            self._system, self._water_indices, self._water_residues = (
                self._prepare_system(
                    system, self._water_template, self._rng, self._num_ghost_waters
                )
            )
            self._num_atoms = self._system.num_atoms()
            self._num_waters = len(self._water_indices)
        except Exception as e:
            raise ValueError(f"Could not prepare the system for GCMC sampling: {e}")

        # Create the kernels.
        self._kernels = {}
        mod = _SourceModule(
            _code
            % {
                "NUM_POINTS": self._num_points,
                "NUM_BATCH": self._batch_size,
                "NUM_WATERS": self._num_waters,
                "NUM_ATOMS": self._num_atoms,
            },
            no_extern_c=True,
        )
        self._kernels["cell"] = mod.get_function("setCellMatrix")
        self._kernels["rng"] = mod.get_function("initialiseRNG")
        self._kernels["rf"] = mod.get_function("setReactionField")
        self._kernels["softcore"] = mod.get_function("setSoftCore")
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
        self._batch_blocks = self._batch_size // self._num_threads + 1

        # Work out the number of blocks to process the waters.
        self._water_blocks = self._num_waters // self._num_threads + 1

        # Initialise the GPU memory.
        self._initialise_gpu_memory()

        # Set the box information.
        self.set_box(self._system)

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

        # Null the nonbonded forces.
        self._nonbonded_force = None
        self._custom_nonbonded_force = None

        # Flag for whether the last move was a bulk sampling move.
        self._is_bulk = False

        import sys

        # Create a logger that writes to stderr and the log file.
        # The 'no_logger' keyword argument can be used to disable logging if
        # the sampler is being driven by an external package, e.g. SOMD2.
        if not "no_logger" in kwargs:
            _logger.remove()
            _logger.add(sys.stderr, level=self._log_level.upper())
            if self._log_file is not None:
                _logger.add(
                    self._log_file, level=self._log_level.upper(), filter="loch"
                )

        # Log the Adams value.
        _logger.debug(f"Adams value: {B:.6f}")

        import atexit

        # Register the cleanup function.
        atexit.register(self._cleanup)

        # Check for testing mode.
        if "test" in kwargs:
            if kwargs["test"] == True:
                _logger.debug("Testing mode enabled")
                self._is_test = True
            else:
                raise ValueError("'test' must be of type 'bool'")
        else:
            self._is_test = False

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
            f"num_ghost_waters={self._num_ghost_waters}, "
            f"adams_shift={self._adams_shift}, "
            f"batch_size={self._batch_size}, "
            f"num_attempts={self._num_attempts}, "
            f"num_threads={self._num_threads}), "
            f"bulk_sampling_probability={self._bulk_sampling_probability}, "
            f"water_template={self._water_template}, "
            f"device={self._device}, "
            f"tolerance={self._tolerance}, "
            f"lambda_schedule={self._lambda_schedule}, "
            f"lambda_value={self._lambda_value}, "
            f"overwrite={self._overwrite}, "
            f"ghost_file={self._ghost_file}, "
            f"log_file={self._log_file}, "
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
        Detach the PyCUDA context.
        """
        try:
            self.pop()
        except:
            pass
        self._pycuda_context.detach()
        self._pycuda_context = None

    def push(self):
        """
        Push the PyCUDA context on top of the stack.
        """
        self._pycuda_context.push()

    def pop(self):
        """
        Pop the PyCUDA context from the stack.
        """
        self._pycuda_context.pop()

    def system(self):
        """
        Return the GCMC system.

        Returns
        -------

        system: sire.system.System
            The GCMC system.
        """
        return self._system

    def set_box(self, system):
        """
        Set the box information.

        Parameters
        ----------

        system: sire.system.System, openmm.Context
            The molecular system, or OpenMM context.
        """

        # Get the space property from the system.
        if isinstance(system, _sr.system.System):
            try:
                self._space = system.property("space")
            except:
                raise ValueError("'system' must contain a 'space' property")
        # Create a Sire TriclinicBox from the OpenMM box vectors.
        elif isinstance(system, _openmm.Context):
            box = system.getState().getPeriodicBoxVectors()
            v0 = [10 * box[0].x, 10 * box[0].y, 10 * box[0].z]
            v1 = [10 * box[1].x, 10 * box[1].y, 10 * box[1].z]
            v2 = [10 * box[2].x, 10 * box[2].y, 10 * box[2].z]
            self._space = _sr.vol.TriclinicBox(
                _sr.maths.Vector(*v0), _sr.maths.Vector(*v1), _sr.maths.Vector(*v2)
            )
        else:
            raise ValueError(
                "'system' must be of type 'sire.system.System' or 'openmm.Context'"
            )

        # Get the box information.
        self._cell_matrix, self._cell_matrix_inverse, self._M = (
            self._get_box_information(self._space)
        )

        # Update the cell matrix information on the GPU.
        self._kernels["cell"](
            self._cell_matrix,
            self._cell_matrix_inverse,
            self._M,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

    def set_bulk_sampling_probability(self, probability):
        """
        Set the bulk sampling probability.

        Parameters
        ----------

        probability: float
            The bulk sampling probability. This should be between 0 and 1.
        """
        try:
            probability = float(probability)
        except Exception as e:
            raise ValueError(
                f"Could not convert 'bulk_sampling_probability' to float: {e}"
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError("'bulk_sampling_probability' must be between 0 and 1")

        self._bulk_sampling_probability = probability

    def delete_waters(self, context):
        """
        Delete any waters within the GCMC sphere. (Convert to ghosts.)

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.
        """

        # Set the NonBondedForce(s).
        self._set_nonbonded_forces(context)

        # Get the OpenMM state.
        state = context.getState(getPositions=True)

        # Get the current positions in Angstrom.
        positions = state.getPositions(asNumpy=True) / _openmm.unit.angstrom

        # Get the position of the GCMC sphere centre.
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

        # Find the non-ghost waters within the GCMC region.
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

        _logger.info(f"Deleting {len(candidates)} waters from the GCMC sphere")

        # Loop over the candidates and delete them.
        for idx in candidates:
            self._accept_deletion(idx, context)

        # Set the number of waters in the GCMC sphere to zero.
        self._N = 0

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
            if not self._openmm_context:
                msg = "OpenMM context is not set!"
                _logger.error(msg)
                raise RuntimeError(msg)

            # Get the OpenMM state.
            state = self._openmm_context.getState(getPositions=True)

            # Get the current positions in Angstrom.
            positions = state.getPositions(asNumpy=True) / _openmm.unit.angstrom

            # Get the position of the GCMC sphere centre.
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

            # Find the non-ghost waters within the GCMC region.
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
        Return the number accepted attempts. (Note that, when using PME, this
        is the number of accepted attempts for the approximate RF potential.)

        Returns
        -------

        num_accepted_attempts: int
            The total number of accepted attempts.
        """
        return self._num_accepted_attempts

    def water_state(self):
        """
        Return the current water state array: 0 = ghost water, 1 = real water.
        """
        return self._water_state.copy()

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
        Return the acceptance probability per attempt. (Note that, when using
        PME, this is acceptance probability for the approximate RF potential.)

        Returns
        -------

        acceptance_probability: float
            The acceptance probability per attempt.
        """
        return self._num_accepted_attempts / (self._num_moves * self._num_attempts)

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
        self._num_accepted_insertions = 0
        self._num_accepted_deletions = 0

        # Clear the forces.
        self._nonbonded_force = None
        self._custom_nonbonded_force = None

        # Clear the OpenMM context.
        self._openmm_context = None

    def ghost_residues(self):
        """
        Return the current indices of the ghost water residues in the OpenMM
        context.

        Returns
        -------

        ghost_residues: np.ndarray
            The indices of the ghost water residues.
        """

        # First get the indices of the ghost waters.
        ghost_waters = _np.where(self._water_state == 0)[0]

        # Now extract and return the residue indices.
        return self._water_residues[ghost_waters]

    def write_ghost_residues(self):
        """
        Write the current indices of the ghost water residues to a file.
        """

        if self._ghost_file is None:
            raise ValueError("'ghost_file' is set to None!")

        # Get the ghost residues.
        ghost_residues = self.ghost_residues()

        # Append a comma-separated list of ghost residue indices to the file.
        with open(self._ghost_file, "a") as f:
            f.write(f"{', '.join([str(x) for x in ghost_residues])}\n")

    def move(self, context):
        """
        Perform num_attempts trial moves.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.

        Returns
        -------

        moves: [int]
            A list of the accepted moves. (0 = insertion, 1 = deletion)
        """
        # Increment the number of moves.
        self._num_moves += 1

        # Set the NonBondedForce(s).
        self._set_nonbonded_forces(context)

        # Zero the number of attempts and batch index.
        num_attempts = 0
        num_batches = 1

        # Initialise the acceptance flags.
        is_accepted = False

        # Create the moves list.
        moves = []

        # Decide if this is a bulk sampling move.
        self._is_bulk = True
        if self._reference is not None:
            if self._rng.random() > self._bulk_sampling_probability:
                self._is_bulk = False

        # Loop until we have the required number of attempts.
        while num_attempts < self._num_attempts:
            _logger.debug(f"Processing batch number {num_batches}")
            _logger.debug(f"Completed {num_attempts} of {self._num_attempts} attempts")
            _logger.debug(f"Number of accepted moves: {self._num_accepted}")
            _logger.debug(f"Number of accepted insertions: {self._num_insertions}")
            _logger.debug(f"Number of accepted deletions: {self._num_deletions}")

            # Prepare the GPU state for the next batch.
            if num_batches == 1 or is_accepted:
                # We only need to get the positions and initial energy for the first
                # batch. These will be updated dynamically as moves are accepted.
                if num_batches == 1:
                    # Get the OpenMM state.
                    state = context.getState(getPositions=True, getEnergy=self._is_pme)

                    # Get the current positions in OpenMM format and in Angstrom.
                    positions_openmm = state.getPositions(asNumpy=True)
                    positions_angstrom = positions_openmm / _openmm.unit.angstrom

                    # If we're using PME, then compute the initial energy.
                    if self._is_pme:
                        initial_energy = state.getPotentialEnergy()
                    else:
                        initial_energy = None

                    # Sample within the GCMC sphere.
                    if self._reference is not None and not self._is_bulk:
                        target = self._get_target_position(positions_angstrom).astype(
                            _np.float32
                        )

                    # Set the positions on the GPU.
                    self._kernels["atom_positions"](
                        _gpuarray.to_gpu(
                            positions_angstrom.astype(_np.float32).flatten()
                        ),
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
                    deletion_candidates = self._deletion_candidates.get().flatten()

                    # Find the waters within the GCMC sphere.
                    deletion_candidates = _np.where(deletion_candidates == 1)[0]

                # Use all non-ghost waters.
                else:
                    _logger.debug("Sampling within the entire simulation box")
                    deletion_candidates = _np.where(self._water_state != 0)[0]
                    target = None

                # Get the current ghost waters.
                ghost_waters = _np.where(self._water_state == 0)[0]

                # If there are no ghost waters, then we can't perform any insertions.
                if len(ghost_waters) == 0:
                    msg = f"Cannot insert any more waters. Please increase 'num_ghost_waters'."
                    _logger.error(msg)
                    raise RuntimeError(msg)

                # Choose a random ghost water.
                idx_water = self._rng.choice(ghost_waters)

                # Get the template positions for the water insertion.
                start_idx = self._water_indices[idx_water]
                template_positions = _gpuarray.to_gpu(
                    positions_angstrom[start_idx : start_idx + self._num_points]
                    .astype(_np.float32)
                    .flatten()
                )

                # Set the number of waters.
                self._N = len(deletion_candidates)

            # Reset the batch acceptance flag.
            is_accepted = False

            # Reset the move type.
            move = None

            # Log the current number of waters.
            _logger.debug(f"Number of waters in sampling volume: {self._N}")
            _logger.debug(f"Water indices: {deletion_candidates}")

            # Draw batch_size samples from the deletion candidates.
            if len(deletion_candidates) > 0:
                candidates = self._rng.choice(
                    deletion_candidates, size=self._batch_size
                )
                candidates_gpu = _gpuarray.to_gpu(candidates.astype(_np.int32))

                # Generate the array of moves types. (0 = insertion, 1 = deletion)
                is_deletion = self._rng.choice(2, size=self._batch_size)
                is_deletion_gpu = _gpuarray.to_gpu(is_deletion.astype(_np.int32))
            # If there are no deletion candidates, then we can only perform
            # insertion moves.
            else:
                candidates = _np.zeros(self._batch_size, dtype=_np.int32)
                candidates_gpu = _gpuarray.to_gpu(candidates.astype(_np.int32))
                is_deletion = _np.zeros(self._batch_size, dtype=_np.int32)
                is_deletion_gpu = _gpuarray.to_gpu(is_deletion.astype(_np.int32))

            _logger.debug("Preparing insertion candidates")

            if target is None:
                target_gpu = _gpuarray.to_gpu(_np.zeros(3, dtype=_np.float32))
                is_target = _np.int32(0)
                exp_B = self._exp_B_bulk
                exp_minus_B = self._exp_minus_B_bulk
            else:
                target_gpu = _gpuarray.to_gpu(target.astype(_np.float32))
                is_target = _np.int32(1)
                exp_B = self._exp_B
                exp_minus_B = self._exp_minus_B

            # Generate the random water positions and orientations.
            self._kernels["water"](
                template_positions,
                target_gpu,
                _np.float32(self._radius.value()),
                self._water_positions,
                is_target,
                block=(self._num_threads, 1, 1),
                grid=(self._batch_blocks, 1, 1),
            )

            # Perform the energy calculation.
            self._kernels["energy"](
                self._water_positions,
                self._energy_coul,
                self._energy_lj,
                candidates_gpu,
                is_deletion_gpu,
                _np.int32(self._is_fep),
                block=(self._num_threads, 1, 1),
                grid=(self._atom_blocks, self._batch_size, 1),
            )

            # Check the acceptance for each trial state.
            self._kernels["acceptance"](
                _np.int32(self._N),
                _np.float32(exp_B),
                _np.float32(exp_minus_B),
                _np.float32(self._beta),
                is_deletion_gpu,
                self._energy_coul,
                self._energy_lj,
                self._energy_change,
                self._probability,
                self._accepted,
                _np.float32(self._tolerance),
                block=(self._num_threads, 1, 1),
                grid=(self._batch_blocks, 1, 1),
            )

            # Get the acceptance array.
            accepted = _np.where(self._accepted.get().flatten() == 1)[0]

            # Store the number of accepted attempts.
            num_accepted_attempts = len(accepted)
            self._num_accepted_attempts += num_accepted_attempts

            _logger.debug(f"Number of accepted attempts: {num_accepted_attempts}")
            _logger.debug(
                f"Total number of accepted attempts: {self._num_accepted_attempts}"
            )

            # No moves were accepted for this batch. Just increment the
            # number of attempts.
            if num_accepted_attempts == 0:
                num_attempts += self._batch_size
                num_batches += 1
                continue

            # For PME we consider each accepted trial in turn, checking to
            # see whether it is accepted via the Gelb correction.
            if self._is_pme:
                max_accepted = num_accepted_attempts
            # For RF we just use the first accepted trial.
            else:
                max_accepted = 1

            # Loop over the accepted trials.
            for i in range(max_accepted):
                # Get the index of the accepted trial.
                idx = accepted[i]

                # Update the number of attempts.
                num_attempts += idx + 1

                # We've exceeded the number of attempts so reject the move.
                if num_attempts > self._num_attempts:
                    move = None
                    is_accepted = False
                    break

                # Insertion move.
                if is_deletion[idx] == 0:
                    # Accept the move.
                    self._accept_insertion(
                        idx, idx_water, positions_openmm, positions_angstrom, context
                    )

                    # Update the acceptance statistics.
                    self._num_accepted += 1
                    self._num_insertions += 1

                    # Initalise the acceptance variables.
                    is_accepted = True
                    move = 0

                    # Set null values for the PME energy and probability.
                    pme_energy = None
                    pme_probability = None

                    # Apply the PME correction.
                    if self._is_pme:
                        # Get the energy change in kcal/mol.
                        dE_RF = (
                            self._energy_change.get().flatten()[idx]
                            * _openmm.unit.kilocalories_per_mole
                        )

                        # Get the new energy.
                        final_energy = context.getState(
                            getEnergy=True
                        ).getPotentialEnergy()

                        # Compute the PME acceptance correction.
                        acc_prob = _np.exp(
                            -self._beta_openmm * (final_energy - initial_energy - dE_RF)
                        )

                        # Store the PME energy change and acceptance probability.
                        pme_energy = final_energy - initial_energy
                        pme_probability = acc_prob

                        # The move was rejected.
                        if acc_prob < self._rng.random():
                            # Revert the move.
                            _ = self._accept_deletion(idx_water, context)

                            # Update the acceptance statistics.
                            self._num_accepted -= 1
                            self._num_insertions -= 1

                            # Revert the number of attempts.
                            num_attempts -= idx + 1

                            is_accepted = False
                            move = None

                            # Log that the insertion was rejected.
                            if self._is_debug:
                                dE_RF = dE_RF.value_in_unit(
                                    _openmm.unit.kilocalories_per_mole
                                )
                                dE_PME = (final_energy - initial_energy).value_in_unit(
                                    _openmm.unit.kilocalories_per_mole
                                )

                                _logger.debug(
                                    f"Rejected PME insertion: dE RF={dE_RF:.3f} kcal/mol, "
                                    f"dE PME={dE_PME:.3f} kcal/mol, acc prob={acc_prob:.3f}"
                                )

                    # Log the insertion and break.
                    if is_accepted:
                        # Log the accepted candidate.
                        _logger.debug(
                            f"Accepted insertion: candidate={idx}, water={idx}"
                        )

                        if self._is_debug:
                            self._log_insertion(
                                idx,
                                idx_water,
                                pme_energy=pme_energy,
                                pme_probability=pme_probability,
                            )
                        break

                # Deletion move.
                else:
                    # Accept the move.
                    self._accept_deletion(candidates[idx], context)

                    # Update the acceptance statistics.
                    self._num_accepted += 1
                    self._num_deletions += 1

                    # Initalise the acceptance variables.
                    is_accepted = True
                    move = 1

                    # Set null values for the PME energy and probability.
                    pme_energy = None
                    pme_probability = None

                    # Apply the PME correction.
                    if self._is_pme:
                        # Get the energy change in kcal/mol.
                        dE_RF = (
                            self._energy_change.get().flatten()[idx]
                            * _openmm.unit.kilocalories_per_mole
                        )

                        # Get the new energy.
                        final_energy = context.getState(
                            getEnergy=True
                        ).getPotentialEnergy()

                        # Compute the PME acceptance correction.
                        acc_prob = _np.exp(
                            -self._beta_openmm * (final_energy - initial_energy - dE_RF)
                        )

                        # Store the PME energy change and acceptance probability.
                        pme_energy = final_energy - initial_energy
                        pme_probability = acc_prob

                        # The move was rejected.
                        if acc_prob < self._rng.random():
                            # Revert the move.
                            self._reject_deletion(candidates[idx], context)

                            # Update the acceptance statistics.
                            self._num_accepted -= 1
                            self._num_deletions -= 1

                            # Revert the number of attempts.
                            num_attempts -= idx + 1

                            is_accepted = False
                            move = None

                            # Log that the deletion was rejected.
                            if self._is_debug:
                                dE_RF = dE_RF.value_in_unit(
                                    _openmm.unit.kilocalories_per_mole
                                )
                                dE_PME = (final_energy - initial_energy).value_in_unit(
                                    _openmm.unit.kilocalories_per_mole
                                )

                                _logger.debug(
                                    f"Rejected PME deletion: dE RF={dE_RF:.3f} kcal/mol, "
                                    f"dE PME={dE_PME:.3f} kcal/mol, acc prob={acc_prob:.3f}"
                                )

                    # Log the deletion and break.
                    if is_accepted:
                        # Log the accepted candidate.
                        _logger.debug(
                            f"Accepted deletion: candidate={idx}, water={candidates[idx]}"
                        )

                        if self._is_debug:
                            self._log_deletion(
                                idx,
                                candidates,
                                positions_angstrom,
                                pme_energy=pme_energy,
                                pme_probability=pme_probability,
                            )
                        break

            # Update the move acceptance flag and append the move.
            if is_accepted:
                moves.append(move)

                # Update the initial energy.
                if self._is_pme:
                    initial_energy = final_energy

                # Return immediately if we're in test mode.
                if self._is_test:
                    return moves
            # If no moves were accepted at the PME level, then update the
            # number of attempts by the batch size.
            else:
                if self._is_pme:
                    num_attempts += self._batch_size

            # Increment the number of batches.
            num_batches += 1

        # If this was a bulk sampling move, then store the context. This allows
        # us to work out the number of waters in the GCMC sphere if the user
        # calls self.num_waters() after the move.
        if self._reference is not None and self._is_bulk:
            self._openmm_context = context

        return moves

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
    def _get_box_information(space):
        """
        Get the box information from the system.

        Parameters
        ----------

        space: sire.vol.PeriodicBox, sire.vol.TriclinicBox
            The simulation box.

        Returns
        -------

        cell_matrix: pycuda.gpuarray.GPUArray
            The cell matrix.

        cell_matrix_inverse: pycuda.gpuarray.GPUArray
            The inverse of the cell matrix.

        M: pycuda.gpuarray.GPUArray
            The matrix M.
        """

        # Validate input.
        if not isinstance(space, _sr.vol.Cartesian):
            raise ValueError(
                "'space' must be of type 'sire.vol.PeriodicBox' or 'sire.vol.TriclinicBox'"
            )

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

        return cell_matrix, cell_matrix_inverse, M

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

        # Get the atoms in the selection.
        try:
            atoms = system[reference].atoms()
        except Exception as e:
            raise ValueError(f"Could not get the reference atoms: {e}")

        # Get the indices of the atoms.
        indices = _np.array(system.atoms().find(atoms))

        return indices

    @staticmethod
    def _prepare_system(system, water_template, rng, num_ghost_waters):
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

        num_ghost_waters: int
            The maximum number of GCMC waters to insert.

        Returns
        -------

        system: sire.system.System
            The prepared system.

        water_indices: numpy.ndarray
            The indices of the oxygen atoms in each water molecule.

        water_residues: numpy.ndarray
            The indices of the water residues.
        """

        # Get the space property from the system.
        try:
            space = system.property("space")
        except:
            raise ValueError("'system' must contain a 'space' property")

        # Get the box matrix and diagonal.
        box_matrix = space.box_matrix()
        box = _np.array([box_matrix.xx(), box_matrix.yy(), box_matrix.zz()])

        # Edit the template so that it is non-interacting.
        cursor = water_template.cursor()
        for atom in cursor.atoms():
            atom["charge"] = 0.0 * _sr.units.mod_electron
            atom["LJ"] = _sr.legacy.MM.LJParameter(
                atom["LJ"].sigma(), 0.0 * _sr.units.kcal_per_mol
            )
        water_template = _BSS._SireWrappers.Molecule(cursor.commit())

        # Create a BioSimSpace system.
        bss_system = _BSS._SireWrappers.System(system._system)

        # Get the initial positions of the atoms.
        positions = _sr.io.get_coords_array(water_template._sire_object)

        # Create the GCMC waters.
        waters = []
        for i in range(num_ghost_waters):
            # Create a copy of the water template with a new molecule number.
            water = water_template.copy()

            # Make the water editable.
            cursor = water._sire_object.cursor()

            # Work out the new position for the oxygen atom.
            oxygen = rng.random(3) * box

            # Loop over the atoms and update the positions.
            for j, atom in enumerate(cursor.atoms()):
                new_position = positions[j] + oxygen - positions[0]
                atom["coordinates"] = _sr.maths.Vector(*new_position)

            # Commit the changes to the water.
            water._sire_object = cursor.commit()

            # Append the water to the list.
            waters.append(water)

        # Add the waters to the system.
        bss_system += waters

        # Convert back to a Sire system.
        system = _sr.system.System(bss_system._sire_object)

        # Search for non-perturbable water oxygen atoms and their residues.
        selection = system["(water and not property is_perturbable) and element O"]

        # Get the atoms and residues from the selection.
        water_atoms = selection.atoms()
        water_residues = selection.residues()

        # Get the indices of the water oxygen atoms and their residues.
        water_atoms = _np.array(system.atoms().find(water_atoms))
        water_residues = _np.array(system.residues().find(water_residues))

        return system, water_atoms, water_residues

    def _initialise_gpu_memory(self):
        """
        Initialise the GPU memory.
        """

        # First get the atomic properties.

        # If this is a regular system, then we can just get the properties directly.
        if not self._is_fep:
            # Get the charges on all the atoms.
            try:
                charges = _np.zeros(self._num_atoms, dtype=_np.float32)
                i = 0
                for mol in self._system:
                    for q in mol.property("charge"):
                        charges[i] = q.value()
                        i += 1

                # Convert to a GPU array.
                charges = _gpuarray.to_gpu(charges.astype(_np.float32))

            except Exception as e:
                raise ValueError(f"Could not get the charges on the atoms: {e}")

            # Try to get the sigma and epsilon for the atoms.
            try:
                sigmas = _np.zeros(self._num_atoms, dtype=_np.float32)
                epsilons = _np.zeros(self._num_atoms, dtype=_np.float32)
                i = 0
                for mol in self._system:
                    for lj in mol.property("LJ"):
                        sigmas[i] = lj.sigma().value()
                        epsilons[i] = lj.epsilon().value()
                        i += 1

                # Convert to GPU arrays.
                sigmas = _gpuarray.to_gpu(sigmas.astype(_np.float32))
                epsilons = _gpuarray.to_gpu(epsilons.astype(_np.float32))

            except Exception as e:
                raise ValueError(f"Could not get the LJ parameters: {e}")

            # Set the alphas to zero.
            alphas = _gpuarray.to_gpu(_np.zeros(self._num_atoms, dtype=_np.float32))

            # Set the is_ghost_fep array to zero.
            is_ghost_fep = _gpuarray.to_gpu(_np.zeros(self._num_atoms, dtype=_np.int32))

        # Otherwise, we need to create an OpenMM context using the specified lambda
        # schedule and value, then extract the required properties from the forces
        # within the context. (The system just contains the end-state properties.)
        else:
            # Create a dynamics object.
            d = self._system.dynamics(
                cutoff_type=self._cutoff,
                cutoff=self._cutoff,
                lambda_value=self._lambda_value,
                schedule=self._lambda_schedule,
                pressure=None,
                timestep="2fs",
                constraint="h_bonds",
                perturbable_constraint="h_bonds_not_heavy_perturbed",
                rest2_scale=self._rest2_scale,
                rest2_selection=self._rest2_selection,
                platform="cpu",
            )

            # Flags for the required force.
            has_gng = False

            # Find the required forces.
            for force in d.context().getSystem().getForces():
                if force.getName() == "GhostNonGhostNonbondedForce":
                    gng_force = force
                    has_gng = True
                    break

            # Make sure the force was found.
            if not has_gng:
                raise ValueError(
                    "Could not find the GhostNonGhostNonbondedForce in the system"
                )

            # Get the parameters for the GhostNonGhostNonbondedForce.
            charges = _np.zeros(self._num_atoms, dtype=_np.float32)
            sigmas = _np.zeros(self._num_atoms, dtype=_np.float32)
            epsilons = _np.zeros(self._num_atoms, dtype=_np.float32)
            alphas = _np.zeros(self._num_atoms, dtype=_np.float32)
            for i in range(gng_force.getNumParticles()):
                # Custom force parameters are returned as floats.
                q, half_sigma, two_sqrt_epsilon, alpha, _ = (
                    gng_force.getParticleParameters(i)
                )
                # Charge in |e|, sigma in nm, epsilon in kJ/mol.
                charges[i] = q
                # Rescale and convert units.
                sigmas[i] = _sr.u(f"{2.0 * half_sigma} nm").to("angstrom")
                epsilons[i] = _sr.u(f"{(0.5 * two_sqrt_epsilon)**2} kJ/mol").to(
                    "kcal/mol"
                )
                # Store the softening parameter.
                alphas[i] = alpha

            # Convert to GPU arrays.
            charges = _gpuarray.to_gpu(charges.astype(_np.float32))
            sigmas = _gpuarray.to_gpu(sigmas.astype(_np.float32))
            epsilons = _gpuarray.to_gpu(epsilons.astype(_np.float32))
            alphas = _gpuarray.to_gpu(alphas.astype(_np.float32))

            # Create the ghost atom array.
            is_ghost_fep = _np.zeros(self._num_atoms, dtype=_np.int32)

            # Get the atoms in the system.
            atoms = self._system.atoms()

            # Loop over all perturbable molecules.
            for mol in self._system["property is_perturbable"].molecules():
                # Loop over all atoms in the molecule.
                for atom in mol.atoms():
                    # Get the end-state charge.
                    charge0 = atom.property("charge0").value()
                    charge1 = atom.property("charge1").value()

                    # The charge at the reference state is zero.
                    if _np.isclose(charge0, 0.0):
                        # Get the end-state LJ parameters.
                        lj = atom.property("LJ0")

                        # This is a null LJ parameter.
                        if _np.isclose(lj.epsilon().value(), 0.0):
                            idx = atoms.find(atom)
                            is_ghost_fep[idx] = 1

                    # The charge at the perturbed state is zero.
                    elif _np.isclose(charge1, 0.0):
                        # Get the end-state LJ parameters.
                        lj = atom.property("LJ1")

                        # This is a null LJ parameter.
                        if _np.isclose(lj.epsilon().value(), 0.0):
                            idx = atoms.find(atom)
                            is_ghost_fep[idx] = 1

            # Convert to GPU array.
            is_ghost_fep = _gpuarray.to_gpu(is_ghost_fep.astype(_np.int32))

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

            # Convert sigma and epsilon for use in custom forces.
            # These are half-sigma in nanometers and 2 * sqrt(epsilon) in kJ/mol.
            self._water_sigma_custom = 0.05 * self._water_sigma
            self._water_epsilon_custom = 2.0 * _np.sqrt(4.184 * self._water_epsilon)

            # Convert to GPU arrays.
            charge_water = _gpuarray.to_gpu(self._water_charge.astype(_np.float32))
            sigma_water = _gpuarray.to_gpu(self._water_sigma.astype(_np.float32))
            epsilon_water = _gpuarray.to_gpu(self._water_epsilon.astype(_np.float32))

        except Exception as e:
            raise ValueError(f"Could not get the atomic properties of the water: {e}")

        # Initialise the water state: 0 = ghost, 1 = real.
        water_state = []
        is_ghost_water = _np.zeros(self._num_atoms, dtype=_np.int32)
        for i in range(self._num_waters):
            if i < self._num_waters - self._num_ghost_waters:
                water_state.append(1)
            else:
                water_state.append(0)
                for j in range(self._num_points):
                    is_ghost_water[self._water_indices[i] + j] = 1
        self._water_state = _np.array(water_state).astype(_np.int32)

        # Initialise the random number generator.
        self._kernels["rng"](
            _gpuarray.to_gpu(
                _np.random.randint(
                    _np.iinfo(_np.int32).max, size=(1, self._batch_size)
                ).astype(_np.int32)
            ),
            block=(self._num_threads, 1, 1),
            grid=(self._batch_blocks, 1, 1),
        )

        # Initialise the reaction field parameters.
        self._kernels["rf"](
            _np.float32(self._cutoff.value()),
            _np.float32(78.3),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Initialise the soft-core parameters.
        if self._is_fep:
            self._kernels["softcore"](
                _np.float32(self._coulomb_power),
                _np.float32(self._shift_coulomb.value()),
                _np.float32(self._shift_delta.value()),
                block=(1, 1, 1),
                grid=(1, 1, 1),
            )

        # Set the atomic properties.
        self._kernels["atom_properties"](
            charges,
            sigmas,
            epsilons,
            alphas,
            _gpuarray.to_gpu(is_ghost_water.astype(_np.int32)),
            is_ghost_fep,
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
            (1, self._batch_size * 3 * self._num_points), _np.float32
        )

        # Initialise memory to store the energy.
        self._energy_coul = _gpuarray.empty(
            (1, self._batch_size * self._num_atoms), _np.float32
        )
        self._energy_lj = _gpuarray.empty(
            (1, self._batch_size * self._num_atoms), _np.float32
        )

        # Initialise memory to store whether each attempt is accepted and
        # the probability of acceptance.
        self._accepted = _gpuarray.empty((1, self._batch_size), _np.int32)
        self._energy_change = _gpuarray.empty((1, self._batch_size), _np.float32)
        self._probability = _gpuarray.empty((1, self._batch_size), _np.float32)

        # Initialise memory to store the deletion candidates.
        self._deletion_candidates = _gpuarray.empty((1, self._num_waters), _np.int32)

    def _accept_insertion(
        self, idx, idx_water, positions_openmm, positions_angstrom, context
    ):
        """
        Accept a insertion move.

        Parameters
        ----------

        idx: int
            The index of the accepted state.

        idx_water: int
            The index of the ghost water to use for the insertion.

        positions_openmm: numpy.ndarray
            The positions of the atoms in the system in OpenMM units.

        positions_angstrom: numpy.ndarray
            The positions of the atoms in the system in Angstroms.

        context: openmm.Context
            The OpenMM context to update.
        """

        # Get the new water positions.
        water_positions = self._water_positions.get().reshape(
            (self._batch_size, 3, self._num_points)
        )[idx]

        # Update the water state.
        self._water_state[idx_water] = 1

        # Get the starting atom index.
        start_idx = self._water_indices[idx_water]

        # Update the water positions and NonBondedForce.
        for i in range(self._num_points):
            # Update the water positions.
            positions_openmm[start_idx + i] = _openmm.unit.Quantity(
                water_positions[i], _openmm.unit.angstrom
            )
            positions_angstrom[start_idx + i] = water_positions[i]
            # Update the NonBondedForce parameters.
            self._nonbonded_force.setParticleParameters(
                start_idx + i,
                self._water_charge[i] * _openmm.unit.elementary_charge,
                self._water_sigma[i] * _openmm.unit.angstrom,
                self._water_epsilon[i] * _openmm.unit.kilocalories_per_mole,
            )
            # Update the custom NonBondedForce parameters.
            if self._is_fep:
                self._custom_nonbonded_force.setParticleParameters(
                    start_idx + i,
                    (
                        self._water_charge[i],
                        self._water_sigma_custom[i],
                        self._water_epsilon_custom[i],
                        0.0,
                        0.0,
                    ),
                )

        # Set the new positions.
        context.setPositions(positions_openmm)

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the CustomNonbondedForce parameters in the context.
        if self._is_fep:
            self._custom_nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(idx_water),
            _np.int32(1),
            _np.int32(1),
            _gpuarray.to_gpu(water_positions.flatten().astype(_np.float32)),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of waters in the sampling volume.
        self._N += 1

    def _accept_deletion(self, idx, context):
        """
        Accept a deletion move.

        Parameters
        ----------

        idx: int
            The index of the deleted water.

        context: openmm.Context
            The OpenMM context to update.
        """

        # Update the water state.
        self._water_state[idx] = 0

        # Get the starting atom index.
        start_idx = self._water_indices[idx]

        for i in range(self._num_points):
            # Update the NonBondedForce parameters.
            self._nonbonded_force.setParticleParameters(
                start_idx + i, 0.0, self._water_sigma[i] * _openmm.unit.angstrom, 0.0
            )
            # Update the CustomNonBondedForce parameters.
            if self._is_fep:
                self._custom_nonbonded_force.setParticleParameters(
                    start_idx + i,
                    (
                        0.0,
                        self._water_sigma_custom[i],
                        0.0,
                        0.0,
                        0.0,
                    ),
                )

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the CustomNonbondedForce parameters in the context.
        if self._is_fep:
            self._custom_nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(idx),
            _np.int32(0),
            _np.int32(0),
            _gpuarray.to_gpu(
                _np.zeros((self._num_points, 3), dtype=_np.float32).flatten()
            ),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of waters in the sampling volume.
        self._N -= 1

    def _reject_deletion(self, idx, context):
        """
        Reject a deletion move.

        Parameters
        ----------

        idx: int
            The index of the water.

        context: openmm.Context
            The OpenMM context to update.
        """

        # Reset the water state.
        self._water_state[idx] = 1

        # Get the starting atom index.
        start_idx = self._water_indices[idx]

        for i in range(self._num_points):
            # Update the NonBondedForce parameters.
            self._nonbonded_force.setParticleParameters(
                start_idx + i,
                self._water_charge[i] * _openmm.unit.elementary_charge,
                self._water_sigma[i] * _openmm.unit.angstrom,
                self._water_epsilon[i] * _openmm.unit.kilocalories_per_mole,
            )
            # Update the CustomNonBondedForce parameters.
            if self._is_fep:
                self._custom_nonbonded_force.setParticleParameters(
                    start_idx + i,
                    (
                        self._water_charge[i],
                        self._water_sigma_custom[i],
                        self._water_epsilon_custom[i],
                        0.0,
                        0.0,
                    ),
                )

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the CustomNonbondedForce parameters in the context.
        if self._is_fep:
            self._custom_nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(idx),
            _np.int32(1),
            _np.int32(0),
            _gpuarray.to_gpu(
                _np.zeros((self._num_points, 3), dtype=_np.float32).flatten()
            ),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of waters in the sampling volume.
        self._N += 1

    def _set_water_state(self, context, indices=None, states=None, force=False):
        """
        Update the state for a list of waters. This can be used by external
        packages when swapping OpenMM state between different replicas when
        GCMC sampling.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to update.

        indices: np.array
            The indices of the waters to update. If None, then all waters
            are updated. Default: None.

        states: np.array
            The new states of the water. If None, then the states are set
            to their current state. This is useful if the context has been
            recreated externally, e.g. following a crash, so the water state
            will have been lost. Default: None.

        force: bool
            If True, then update the state even if it is unchanged.
            Default: False.
        """

        if indices is None:
            # Update all waters.
            indices = _np.arange(self._num_waters, dtype=_np.int32)

        if states is None:
            # Update all waters to their current state.
            states = self._water_state[indices]
            # Assume the context has been recreated, so we need to get the
            # new forces.
            self._nonbonded_force = None
            self._custom_nonbonded_force = None
            # Update even if the state is unchanged.
            force = True

        # Set the NonBondedForce(s).
        self._set_nonbonded_forces(context)

        # Loop over the indices and states.
        for idx, state in zip(indices, states):

            # Skip if the state is unchanged.
            if not force and self._water_state[idx] == state:
                continue

            _logger.debug(f"Updating water {idx} to state {state}")

            # Get the water starting index.
            start_idx = self._water_indices[idx]

            # Ghost water.
            if state == 0:
                for i in range(self._num_points):
                    # Update the NonbondedForce parameters.
                    self._nonbonded_force.setParticleParameters(
                        start_idx + i,
                        0.0,
                        self._water_sigma[i] * _openmm.unit.angstrom,
                        0.0,
                    )
                    # Update the CustomNonbondedForce parameters.
                    if self._is_fep:
                        self._custom_nonbonded_force.setParticleParameters(
                            start_idx + i,
                            (
                                0.0,
                                self._water_sigma_custom[i],
                                0.0,
                                0.0,
                                0.0,
                            ),
                        )

                # Update the state of the water on the GPU.
                self._kernels["update_water"](
                    _np.int32(idx),
                    _np.int32(0),
                    _np.int32(0),
                    _gpuarray.to_gpu(
                        _np.zeros((self._num_points, 3), dtype=_np.float32).flatten()
                    ),
                    block=(1, 1, 1),
                    grid=(1, 1, 1),
                )

                # Set the new water state.
                self._water_state[idx] = 0

            # Real water.
            else:
                for i in range(self._num_points):
                    # Update the NonbondedForce parameters.
                    self._nonbonded_force.setParticleParameters(
                        start_idx + i,
                        self._water_charge[i] * _openmm.unit.elementary_charge,
                        self._water_sigma[i] * _openmm.unit.angstrom,
                        self._water_epsilon[i] * _openmm.unit.kilocalories_per_mole,
                    )
                    # Update the CustomNonBondedForce parameters.
                    if self._is_fep:
                        self._custom_nonbonded_force.setParticleParameters(
                            start_idx + i,
                            (
                                self._water_charge[i],
                                self._water_sigma_custom[i],
                                self._water_epsilon_custom[i],
                                0.0,
                                0.0,
                            ),
                        )

                # Update the state of the water on the GPU.
                self._kernels["update_water"](
                    _np.int32(idx),
                    _np.int32(1),
                    _np.int32(0),
                    _gpuarray.to_gpu(
                        _np.zeros((self._num_points, 3), dtype=_np.float32).flatten()
                    ),
                    block=(1, 1, 1),
                    grid=(1, 1, 1),
                )

                # Set the new water state.
                self._water_state[idx] = 1

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the CustomNonbondedForce parameters in the context.
        if self._is_fep:
            self._custom_nonbonded_force.updateParametersInContext(context)

    def _set_nonbonded_forces(self, context):
        """
        Find the required nonbonded force(s) in the system.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.
        """
        if self._nonbonded_force is None or (
            self._is_fep and self._custom_nonbonded_force is None
        ):
            for force in context.getSystem().getForces():
                if isinstance(force, _openmm.NonbondedForce):
                    self._nonbonded_force = force
                elif self._is_fep and force.getName() == "GhostNonGhostNonbondedForce":
                    self._custom_nonbonded_force = force
                elif "Barostat" in force.getName():
                    msg = (
                        f"GCMC must be used at constant volume: "
                        f"'{force.getName()}' is not supported."
                    )
                    _logger.error(msg)
                    raise TypeError(msg)

        if self._nonbonded_force is None:
            msg = "Could not find a NonbondedForce in the system"
            _logger.error(msg)
            raise ValueError(msg)

        if self._is_fep and self._custom_nonbonded_force is None:
            msg = "Could not find a CustomNonbondedForce in the system"
            _logger.error(msg)
            raise ValueError(msg)

    def _get_target_position(self, positions):
        """
        Get the current centre of the GCMC sphere.

        Parameters
        ----------

        positions: numpy.ndarray
            The current positions of the system.

        Returns
        -------

        target: numpy.ndarray
            The centre of the GCMC sphere.
        """

        # Work out the centre of geometry of the reference atoms.
        centre = _sr.maths.Vector(*positions[self._reference_indices[0]])
        target = centre
        for index in self._reference_indices[1:]:
            delta = self._space.calc_dist_vector(
                target, _sr.maths.Vector(*positions[index])
            )
            centre += target + _sr.maths.Vector(delta.x(), delta.y(), delta.z())
        target = _np.array([x.value() for x in centre / len(self._reference_indices)])

        _logger.debug(f"GCMC sphere centre: {target}")

        return target

    def _log_insertion(self, idx, idx_water, pme_energy=None, pme_probability=None):
        """
        Log information about the accepted insertion move.

        Parameters
        ----------

        idx: int
            The index of the accepted trial move.

        idx_water: int
            The index of the water that was inserted.

        pme_energy: openmm.Quantity
            The PME energy difference.

        pme_probability: float
            The PME acceptance probability.
        """
        # Get the energies.
        energy_coul = self._energy_coul.get().reshape(
            (self._batch_size, self._num_atoms)
        )
        energy_lj = self._energy_lj.get().reshape((self._batch_size, self._num_atoms))

        # Get the water positions.
        water_positions = self._water_positions.get().reshape(
            (self._batch_size, self._num_points, 3)
        )

        # Get the RF acceptance probability.
        probability = self._probability.get().flatten()

        # Store debugging attributes.
        self._debug = {
            "move": "insertion",
            "idx": idx_water,
            "energy_coul": self._prefactor * energy_coul[idx].sum(),
            "energy_lj": energy_lj[idx].sum(),
            "probability_rf": probability[idx],
        }

        # Log the position of the inserted oxygen atom.
        _logger.debug(f"Inserted oxygen position: {water_positions[idx, 0]}")

        # Log the energies of the accepted candidate.
        _logger.debug(f"RF coulomb energy: {self._debug['energy_coul']:.6f} kcal/mol")
        _logger.debug(f"LJ energy: {self._debug['energy_lj']:.6f} kcal/mol")
        _logger.debug(
            f"Total RF energy difference: {self._debug['energy_coul'] + self._debug['energy_lj']:.6f} kcal/mol"
        )
        _logger.debug(f"RF insertion probability: {probability[idx]:.6f}")

        # Add PME energy if available.
        if pme_energy is not None:
            self._debug["pme_energy"] = pme_energy.value_in_unit(
                _openmm.unit.kilocalories_per_mole
            )
            self._debug["probability_pme"] = pme_probability

            _logger.debug(
                f"Total PME energy difference: {self._debug['pme_energy']:.6f} kcal/mol"
            )
            _logger.debug(f"PME insertion probability: {pme_probability:.6f}")

    def _log_deletion(
        self, idx, candidates, positions, pme_energy=None, pme_probability=None
    ):
        """
        Log information about the accepted deletion move.

        Parameters
        ----------

        idx: int
            The index of the accepted deletion.

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
        energy_coul = self._energy_coul.get().reshape(
            (self._batch_size, self._num_atoms)
        )
        energy_lj = self._energy_lj.get().reshape((self._batch_size, self._num_atoms))

        # Get the RF acceptance probability.
        probability = self._probability.get().flatten()

        # Get the water index.
        idx_water = self._water_indices[candidates[idx]]

        # Store debugging attributes.
        self._debug = {
            "move": "deletion",
            "idx": self._water_indices[candidates[idx]],
            "energy_coul": -self._prefactor * energy_coul[idx].sum(),
            "energy_lj": -energy_lj[idx].sum(),
            "probability_rf": probability[idx],
        }

        # Log the oxygen position.
        _logger.debug(f"Deleted oxygen position: {positions[idx_water]}")

        # Log the energies of the accepted candidate.
        _logger.debug(f"RF coulomb energy: {self._debug['energy_coul']:.6f} kcal/mol")
        _logger.debug(f"LJ energy: {self._debug['energy_lj']:.6f} kcal/mol")
        _logger.debug(
            f"Total RF energy difference: {self._debug['energy_coul'] + self._debug['energy_lj']:.6f} kcal/mol"
        )
        _logger.debug(f"RF deletion probability: {probability[idx]:.6f}")

        # Add PME energy if available.
        if pme_energy is not None:
            self._debug["pme_energy"] = pme_energy.value_in_unit(
                _openmm.unit.kilocalories_per_mole
            )
            self._debug["probability_pme"] = pme_probability

            _logger.debug(
                f"Total PME energy difference: {self._debug['pme_energy']:.6f} kcal/mol"
            )
            _logger.debug(f"PME deletion probability: {pme_probability:.6f}")

    def _flag_ghost_waters(self, system):
        """
        Flag the ghost waters in the system.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        Returns

        system: sire.system.System
            The system with the ghost waters flagged.
        """

        if not isinstance(system, _sr.system.System):
            raise ValueError("'system' must be a Sire system")

        # First get the indices of the ghost waters.
        ghost_waters = _np.where(self._water_state == 0)[0]

        # Now extract the oxygen indices.
        ghost_oxygens = self._water_indices[ghost_waters]

        # Loop over the ghost waters and set the is_ghost property.
        for i in ghost_oxygens:
            cursor = system[system.atoms()[int(i)].molecule()].cursor()
            cursor["is_ghost_water"] = True
            system.update(cursor.commit())

        # Return the system.
        return system
