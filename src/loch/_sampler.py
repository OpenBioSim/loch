import numpy as _np
import openmm as _openmm

import pycuda.gpuarray as _gpuarray
import pycuda.autoinit as _autoinit
from pycuda.compiler import SourceModule as _SourceModule

import BioSimSpace as _BSS
import sire as _sr

from ._kernels import code as _code


class GCMCSampler:
    """
    A class to perform GCMC sampling using the GPU.
    """

    def __init__(
        self,
        system,
        reference,
        radius="4 A",
        cutoff_type="rf",
        cut_off="10.0 A",
        excess_chemical_potential="-6.09 kcal/mol",
        standard_volume="30.345 A^3",
        temperature="298 K",
        adams_shift=0.0,
        max_gcmc_waters=10,
        num_attempts=10000,
        num_threads=1024,
        water_template=None,
        seed=None,
    ):
        """
        Initialise the GCMC sampler.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        reference: str
            A selection string for the reference atoms.

        radius: str
            The radius of the GCMC sphere.

        cutoff_type: str
            The type of cutoff to use.

        cut_off: str
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
            The number of attempts to make.

        num_threads: int
            The number of threads per block. (Must be a multiple of 32.)

        water_template: sire.molecule.Molecule
            A water molecule to use as a template. This is only required when
            the system does not contain any water molecules.

        seed: int
            The seed for the random number generator.
        """

        # Validate the input.

        if not isinstance(system, _sr.system.System):
            raise ValueError("The system must be a Sire system.")
        self._system = system

        if not isinstance(reference, str):
            raise ValueError("The reference must be a string.")
        self._reference = reference

        cutoff_type = cutoff_type.lower().replace(" ", "")
        if not cutoff_type in ["rf", "pme"]:
            raise ValueError("The cutoff type must be 'rf' or 'pme'.")
        self._cutoff_type = cutoff_type

        try:
            self._radius = self._validate_sire_unit("radius", radius, _sr.u("A"))
        except Exception as e:
            raise ValueError(f"Could not validate the 'radius': {e}")

        try:
            self._cut_off = self._validate_sire_unit("cut_off", cut_off, _sr.u("A"))
        except Exception as e:
            raise ValueError(f"Could not validate the 'cut_off': {e}")

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
            raise ValueError("The maximum number of GCMC waters must be of type 'int'.")
        self._max_gcmc_waters = max_gcmc_waters

        if not isinstance(adams_shift, (int, float)):
            raise ValueError("The Adams shift must be a of type 'int' or 'float'")
        self._adams_shift = float(adams_shift)

        if not isinstance(max_gcmc_waters, int):
            raise ValueError("The maximum number of GCMC waters must be of type 'int'.")
        self._max_gcmc_waters = max_gcmc_waters

        if not isinstance(num_attempts, int):
            raise ValueError("The number of attempts must be of type 'int'.")
        self._num_attempts = num_attempts

        if not isinstance(num_threads, int):
            raise ValueError("The number of threads must be of type 'int'.")
        if not num_threads % 32 == 0:
            raise ValueError("The number of threads must be a multiple of 32.")
        self._num_threads = num_threads

        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("The seed must be of type 'int'.")
        else:
            seed = _np.random.randint(_np.iinfo(_np.int32).max)
        self._seed = seed

        # Set the seed.
        _np.random.seed(self._seed)

        # Create a random number generator.
        self._rng = _np.random.default_rng(self._seed)

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
                    raise ValueError("The water template must be a Sire molecule.")
            self._water_template = water_template
        self._num_points = self._water_template.num_atoms()

        # Store the positions of the template.
        self._water_template_positions = _gpuarray.to_gpu(
            _sr.io.get_coords_array(self._water_template).flatten().astype(_np.float32)
        )

        # Get the indices of the reference atoms.
        self._reference_indices = self._get_reference_indices(system, reference)

        # Set the box information.
        self._space, self._cell_matrix, self._cell_matrix_inverse, self._M = (
            self._get_box_information(system)
        )

        # Prepare the system for GCMC sampling.
        try:
            self._system, self._water_indices = self._prepare_system(
                system, self._water_template, self._max_gcmc_waters
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
        self._kernels["probability"] = mod.get_function("computeAcceptanceProbability")

        # Work out the number of blocks to process the atoms.
        self._atom_blocks = self._num_atoms // self._num_threads + 1

        # Work out the number of blocks to process the attempts.
        self._attempt_blocks = self._num_attempts // self._num_threads + 1

        # Work out the number of blocks to process the waters.
        self._water_blocks = self._num_waters // self._num_threads + 1

        # Initialise the GPU memory.
        self._initialise_gpu_memory()

        # Create memory to store the trial states.
        self._states = _np.arange(self._num_attempts + 1).astype(_np.int32)

        # Set constants.

        # Energy conversion factor.
        self._beta = 1 / (
            _sr.units.k_boltz.to("kcal/(mol*kelvin)") * self._temperature.value()
        )

        # Work out the volume of the GCMC sphere.
        volume = (4.0 * _np.pi * self._radius.value() ** 3) / 3.0

        # Work out the Adams value.
        B = (
            self._beta * self._excess_chemical_potential.value()
            + _np.log(volume / self._standard_volume.value())
        ) + self._adams_shift

        # Store the exponentials for the Adams value.
        self._exp_B = _np.exp(B)
        self._exp_minus_B = _np.exp(-B)

        # Zero the number of waters in the GCMC region.
        self._N = 0

        # Null the nonbonded force.
        self._nonbonded_force = None

    def __str__(self):
        """
        Return a string representation of the class.
        """

        return (
            f"GCMCSampler(system={self._system}, "
            f"reference={self._reference}, "
            f"radius={self._radius}, "
            f"cutoff_type={self._cutoff_type}, "
            f"cut_off={self._cut_off}, "
            f"excess_chemical_potential={self._excess_chemical_potential}, "
            f"standard_volume={self._standard_volume}, "
            f"temperature={self._temperature}, "
            f"max_gcmc_waters={self._max_gcmc_waters}, "
            f"adams_shift={self._adams_shift}, "
            f"num_attempts={self._num_attempts}, "
            f"num_threads={self._num_threads}), "
            f"water_template={self._water_template}, "
            f"seed={self._seed})"
        )

    def __repr__(self):
        """
        Return a string representation of the class.
        """

        return str(self)

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
    def _prepare_system(system, water_template, max_gcmc_waters):
        """
        Prepare the system for GCMC sampling.

        Parameters
        ----------

        system: sire.system.System
            The molecular system.

        water_template: sire.molecule.Molecule
            The water template.

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
                1e-9 * _sr.units.angstrom, 1e-9 * _sr.units.kcal_per_mol
            )
        water_template = _BSS._SireWrappers.Molecule(cursor.commit())

        # Create a BioSimSpace system.
        bss_system = _BSS._SireWrappers.System(system._system)

        # First create the GCMC waters.
        waters = []
        for i in range(max_gcmc_waters):
            waters.append(water_template.copy())

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
        for i in range(self._num_waters):
            if i < self._num_waters - self._max_gcmc_waters:
                water_state.append(2)
            else:
                water_state.append(0)
        self._water_state = _np.array(water_state).astype(_np.int32)

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
            _np.float32(self._cut_off.value()),
            _np.float32(78.3),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Set the atomic properties.
        self._kernels["atom_properties"](
            charges,
            sigmas,
            epsilons,
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
        self._energy_coul = _gpuarray.empty(
            (1, self._num_attempts * self._num_atoms), _np.float32
        )
        self._energy_lj = _gpuarray.empty(
            (1, self._num_attempts * self._num_atoms), _np.float32
        )

        # Initialise memory to store the acceptance probabilities.
        self._probability = _gpuarray.empty((1, self._num_attempts + 1), _np.float32)

        # Initialise memory to store the deletion candidates.
        self._deletion_candidates = _gpuarray.empty((1, self._num_waters), _np.int32)

    @staticmethod
    def _evaluate_candidate(system, candidate_position, cutoff, context=None):
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

    @staticmethod
    def _choose_state(rng, states, probability, num_insertions, threshold=1e-6):
        """
        Choose a trial move according to the probabilities.

        Parameters
        ----------

        rng: numpy.random.Generator
            The random number generator.

        states: numpy.ndarray
            The states to choose from.

        probability: numpy.ndarray
            The probabilities of each move.

        num_insertions: int
            The number of insertions.

        Returns
        -------

        state: int
            The state to move to.
        """

        # Zero the probability of staying in the same state.
        probability[-1] = 0.0

        # Compute the total probability.
        total_probability = _np.sum(probability)

        print("Total probability:", total_probability)

        # Update the probability of staying in the same state.
        if total_probability < 1.0:
            probability[-1] = 1.0 - total_probability

        # Remove entries with low probability.
        mask = probability > threshold
        states = states[mask]
        probability = probability[mask]

        # Choose a state according to its probability.
        return rng.choice(states, p=probability / _np.sum(probability))

    def get_system(self):
        """
        Return the GCMC system.

        Returns
        -------

        system: sire.system.System
            The GCMC system.
        """
        return self._system

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

        move: str
            The type of move.

        accepted: bool
            Whether the move was accepted.
        """

        # Choose a move.
        if _np.random.randint(2) == 1:
            return self.insertion_move(context)
        else:
            return self.deletion_move(context)

    def insertion_move(self, context):
        """
        Perform a trial insertion move.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.

        move: str
            The type of move.

        accepted: bool
            Whether the move was accepted.
        """

        print("\nPerforming insertion move.")

        # Set the NonBondedForce.
        self._set_nonbonded_force(context)

        # Get the current positions in Angstrom.
        positions = (
            context.getState(getPositions=True).getPositions(asNumpy=True)
            / _openmm.unit.angstrom
        )

        # Get the target position.
        target = self._get_target_position(positions)

        # Set the positions on the GPU.
        self._kernels["atom_positions"](
            _gpuarray.to_gpu(positions.astype(_np.float32).flatten()),
            _np.float32(1.0),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, 1, 1),
        )

        # Generate the random water positions and orientations.
        self._kernels["water"](
            self._water_template_positions,
            _gpuarray.to_gpu(target.astype(_np.float32)),
            _np.float32(self._radius.value()),
            self._water_positions,
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Perform the energy calculation.
        self._kernels["energy"](
            self._water_positions,
            self._energy_coul,
            self._energy_lj,
            _np.int32(1),
            _np.int32(0),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, self._num_attempts, 1),
        )

        # Compute the acceptance probabilities.
        self._kernels["probability"](
            _np.int32(self._N),
            _np.float32(1.0),
            _np.float32(self._exp_B),
            _np.float32(self._beta),
            self._energy_coul,
            self._energy_lj,
            self._probability,
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Get the probabilities and choose a new state.
        probability_cpu = self._probability.get().flatten()
        state = self._choose_state(
            self._rng, self._states, probability_cpu, self._num_attempts
        )

        # A candidate insertion was accepted.
        if state != self._num_attempts:
            # If we're using reaction field, then we're good to go.
            if self._cutoff_type == "rf":
                context = self._accept_insertion(state, context)
            else:
                # Otherwise we need to evaluate the energy of the candidate state
                # at the PME level.
                context = self._evaluate_candidate(context, state, 0)

        # Get the energies.
        energy_couls = self._energy_coul.get().reshape(
            (self._num_attempts, self._num_atoms)
        )
        energy_ljs = self._energy_lj.get().reshape(
            (self._num_attempts, self._num_atoms)
        )

        if state < self._num_attempts:
            print("Coulomb energy:", energy_couls[state].sum())
            print("LJ energy:", energy_ljs[state].sum())

        return context, "insertion", state != self._num_attempts

    def deletion_move(self, context):
        """
        Perform a trial deletion move.

        Parameters
        ----------

        context: openmm.Context
            The OpenMM context to use.

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.

        move: str
            The type of move.

        accepted: bool
            Whether the move was accepted.
        """

        print("\nPerforming deletion move.")

        # Set the NonBondedForce.
        self._set_nonbonded_force(context)

        # Get the current positions in Angstrom.
        positions = (
            context.getState(getPositions=True).getPositions(asNumpy=True)
            / _openmm.unit.angstrom
        )

        # Get the target position.
        target = self._get_target_position(positions)

        # Set the positions on the GPU.
        self._kernels["atom_positions"](
            _gpuarray.to_gpu(positions.astype(_np.float32).flatten()),
            _np.float32(1.0),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, 1, 1),
        )

        # First work out the candidate waters for deletion.
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
        candidates = _np.argwhere(candidates == 1).flatten()

        # Print the candidates.
        print("Number of deletion candidates:", len(candidates))
        print("Deletion candidates:", candidates)

        if len(candidates) == 0:
            return context, "deletion", False
        else:
            # Draw num_attempts samples from the candidates.
            candidates = self._rng.choice(candidates, size=self._num_attempts)

        # Compute the energy of the candidate deletions.
        self._kernels["energy"](
            self._water_positions,
            self._energy_coul,
            self._energy_lj,
            _gpuarray.to_gpu(candidates.astype(_np.int32)),
            _np.int32(1),
            block=(self._num_threads, 1, 1),
            grid=(self._atom_blocks, self._num_attempts, 1),
        )

        # Get the coulomb and LJ energies.
        energy_coul = self._energy_coul.get().reshape(
            (self._num_attempts, self._num_atoms)
        )
        energy_lj = self._energy_lj.get().reshape((self._num_attempts, self._num_atoms))

        print("Coulomb energy:", energy_coul[1].sum())
        print("LJ energy:", energy_lj[1].sum())

        # Compute the acceptance probabilities.
        self._kernels["probability"](
            _np.int32(0),
            _np.float32(-1.0),
            _np.float32(self._exp_minus_B),
            _np.float32(self._beta),
            self._energy_coul,
            self._energy_lj,
            self._probability,
            block=(self._num_threads, 1, 1),
            grid=(self._attempt_blocks, 1, 1),
        )

        # Get the probabilities and choose a new state.
        probability_cpu = self._probability.get().flatten()
        state = self._choose_state(
            self._rng, self._states, probability_cpu, self._num_attempts
        )

        # A candidate deletion was accepted.
        if state != self._num_attempts:
            # If we're using reaction field, then we're good to go.
            if self._cutoff_type == "rf":
                context = self._accept_deletion(candidates[state], context)
            else:
                # Otherwise we need to evaluate the energy of the candidate state
                # at the PME level.
                context = self._evaluate_candidate(context, candidates[state], 1)

        return context, "deletion", state != self._num_attempts

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
        """

        if self._N == self._max_gcmc_waters:
            raise RuntimeError(
                "Cannot insert any more waters. Please increase 'max_gcmc_waters'."
            )

        # Get the new water positions.
        water_positions = self._water_positions.get().reshape(
            (self._num_attempts, 3, self._num_points)
        )[state]

        # Choose the first ghost water.
        idx = _np.unravel_index(
            (self._water_state == 0).argmax(), self._water_state.shape
        )[0]

        print(
            f"Accepting insertion of water {state} idx {idx}, {self._water_indices[idx]}"
        )

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

        # Update the number of GCMC waters.
        self._N += 1

        return context

    def _accept_deletion(self, state, context):
        """
        Accept a deletion move.

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
        """

        print(f"Accepting deletion of water {state} idx {self._water_indices[state]}")

        # Update the water state.
        self._water_state[state] = 0

        # Get the starting atom index.
        start_idx = self._water_indices[state]

        # Update the water positions and NonBondedForce.
        for i in range(self._num_points):
            self._nonbonded_force.setParticleParameters(start_idx + i, 0.0, 1e-9, 1e-9)

        # Update the NonbondedForce parameters in the context.
        self._nonbonded_force.updateParametersInContext(context)

        # Update the state of the water on the GPU.
        self._kernels["update_water"](
            _np.int32(state),
            _np.int32(0),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )

        # Update the number of GCMC waters.
        self._N -= 1

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

        return target

    def _evaluate_candidate(self, state, context, move):
        """
        Evaluate the energy of a candidate move at the PME level.

        Parameters
        ----------

        state: int
            The index of the candidate state.

        context: openmm.Context
            The OpenMM context to update.

        move: int
            The type of move to evaluate. (0 = insertion, 1 = deletion)

        Returns
        -------

        context: openmm.Context
            The updated OpenMM context.
        """

        return context
