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
Utility functions for calibrating the GCMC potential.
"""

__all__ = ["excess_chemical_potential", "standard_volume"]


def excess_chemical_potential(
    topology_file,
    coordinate_file,
    temperature="298 K",
    pressure="1 bar",
    cutoff="10 A",
    runtime="5 ns",
    num_lambda=24,
    replica_exchange=False,
    work_dir=None,
):
    """
    Calculate the excess chemical potential of a water molecule at the given
    temperature and pressure via alchemical decoupling.

    Parameters
    ----------

    topology_file : str
        Path to the topology file.

    coordinate_file : str
        Path to the coordinate file.

    temperature : str, optional
        Temperature of the system (default is "298 K").

    pressure : str, optional
        Pressure of the system (default is "1 bar").

    cutoff : str, optional
        Non-bonded interaction cutoff distance (default is "10 A").

    runtime : str, optional
        Simulation runtime (default is "5 ns").

    num_lambda : int, optional
        Number of lambda windows to use in the calculation (default is 24).

    replica_exchange : bool, optional
        Whether to use replica exchange during the calculation (default is False).

    work_dir : str, optional
        Working directory for the decoupling simulation (default is None,
        which uses a temporary directory).

    Returns
    -------

    float
        Excess chemical potential in kcal/mol.
    """

    import os
    import sire as sr

    from BioSimSpace.FreeEnergy import Relative

    if not isinstance(topology_file, str):
        raise TypeError("'topology_file' must be a of type 'str'.")
    if not os.path.isfile(topology_file):
        raise FileNotFoundError(f"'topology_file' not found: {topology_file}")

    if not isinstance(coordinate_file, str):
        raise TypeError("'coordinate_file' must be a of type 'str'.")
    if not os.path.isfile(coordinate_file):
        raise FileNotFoundError(f"'coordinate_file' not found: {coordinate_file}")

    if not isinstance(temperature, str):
        raise TypeError("'temperature' must be a of type 'str'.")
    try:
        u = sr.u(temperature)
    except Exception as e:
        raise ValueError(f"Unable to parse 'temperature': {e}")
    if not u.has_same_units(sr.units.kelvin):
        raise ValueError("'temperature' has incorrect units.")

    try:
        u = sr.u(pressure)
    except Exception as e:
        raise ValueError(f"Unable to parse 'pressure': {e}")
    if not u.has_same_units(sr.units.bar):
        raise ValueError("'pressure' has incorrect units.")

    try:
        u = sr.u(cutoff)
    except Exception as e:
        raise ValueError(f"Unable to parse 'cutoff': {e}")
    if not u.has_same_units(sr.units.angstrom):
        raise ValueError("'cutoff' has incorrect units.")

    try:
        runtime = sr.u(runtime)
    except Exception as e:
        raise ValueError(f"Unable to parse 'runtime': {e}")
    if not runtime.has_same_units(sr.units.nanosecond):
        raise ValueError("'runtime' has incorrect units.")

    if not isinstance(num_lambda, int):
        raise TypeError("'num_lambda' must be a of type 'int'.")

    if not isinstance(replica_exchange, bool):
        raise TypeError("'replica_exchange' must be a of type 'bool'.")

    if work_dir is not None:
        if not isinstance(work_dir, str):
            raise TypeError("'work_dir' must be a of type 'str'.")
    else:
        import tempfile

        tmp_dir = tempfile.TemporaryDirectory()
        work_dir = tmp_dir.name

    # Import the required runner.
    if replica_exchange:
        from somd2.runner import RepexRunner as Runner
    else:
        from somd2.runner import Runner

    from somd2.config import Config

    # Set up the configuration.
    try:
        config = Config(
            temperature=temperature,
            pressure=pressure,
            cutoff=cutoff,
            replica_exchange=replica_exchange,
            num_lambda=num_lambda,
            runtime=runtime,
            timestep="2 fs",
            h_mass_factor=1,
            shift_delta="2.25 A",
            output_directory=work_dir,
        )
    except Exception as e:
        raise ValueError(f"Unable to create SOMD2 configuration: {e}")

    # Try to read the topology and coordinate files.
    try:
        mols = sr.load(topology_file, coordinate_file)
    except Exception as e:
        raise ValueError(f"Unable to read topology/coordinate files: {e}")

    # Make sure there are only water molecules.
    waters = mols["water"]
    if not len(waters) == len(mols):
        raise ValueError(
            "The provided topology/coordinate files must contain only water molecules."
        )

    # Decouple a single water molecule.
    mol = waters[0]
    mol = sr.morph.decouple(mol, as_new_molecule=False)
    mols.update(mol)
    mols = sr.morph.link_to_reference(mols)

    # Get the lambda schedule from the molecule.
    l = mol.property("schedule")

    # Avoid scaling kappa during decouple stage.
    l.set_equation(stage="decouple", lever="kappa", force="ghost/ghost", equation=0)
    l.set_equation(stage="decouple", lever="kappa", force="ghost-14", equation=0)

    # Add new discharging stage.
    l.set_equation(stage="decouple", lever="charge", equation=l.final())
    l.prepend_stage("decharge", l.initial())
    l.set_equation(
        stage="decharge",
        lever="charge",
        equation=l.lam() * l.final() + l.initial() * (1 - l.lam()),
    )
    l.set_equation(stage="decharge", force="ghost/ghost", equation=l.initial())
    l.set_equation(stage="decharge", force="ghost-14", equation=l.initial())
    l.set_equation(
        stage="decharge", lever="kappa", force="ghost/ghost", equation=-l.lam() + 1
    )
    l.set_equation(
        stage="decharge", lever="kappa", force="ghost-14", equation=-l.lam() + 1
    )

    # Update the schedule in the configuration.
    config.lambda_schedule = l

    # Set up the runner.
    try:
        runner = Runner(mols, config)
    except Exception as e:
        raise ValueError(f"Unable to set up the decoupling simulation: {e}")

    # Run the decoupling simulation.
    try:
        runner.run()
    except Exception as e:
        raise RuntimeError(f"Decoupling simulation failed: {e}")

    # Analyse the results.
    try:
        pmf, overlap = Relative.analyse(work_dir)
    except Exception as e:
        raise RuntimeError(f"Unable to analyse the decoupling results: {e}")

    # Return the excess chemical potential.
    return -pmf[-1][1].value()


def standard_volume(
    topology_file,
    coordinate_file,
    temperature="298 K",
    pressure="1 bar",
    cutoff="10 A",
    num_samples=5000,
    sample_interval="1 ps",
):
    """
    Calculate the standard volume of water at the given temperature and pressure.

    Parameters
    ----------

    topology_file : str
        Path to the topology file.

    coordinate_file : str
        Path to the coordinate file.

    temperature : str, optional
        Temperature of the system (default is "298 K").

    pressure : str, optional
        Pressure of the system (default is "1 bar").

    cutoff : str, optional
        Non-bonded interaction cutoff distance (default is "10 A").

    num_samples : int, optional
        Number of volume samples to collect (default is 5000).

    sample_interval : str, optional
        Interval at which to sample the volume (default is "1 ps").

    Returns
    -------

    float
        Standard volume in A^3/molecule.
    """
    import os
    import sire as sr

    from openmm.unit import angstrom

    if not isinstance(topology_file, str):
        raise TypeError("'topology_file' must be a of type 'str'.")
    if not os.path.isfile(topology_file):
        raise FileNotFoundError(f"'topology_file' not found: {topology_file}")

    if not isinstance(coordinate_file, str):
        raise TypeError("'coordinate_file' must be a of type 'str'.")
    if not os.path.isfile(coordinate_file):
        raise FileNotFoundError(f"'coordinate_file' not found: {coordinate_file}")

    if not isinstance(temperature, str):
        raise TypeError("'temperature' must be a of type 'str'.")
    try:
        u = sr.u(temperature)
    except Exception as e:
        raise ValueError(f"Unable to parse 'temperature': {e}")
    if not u.has_same_units(sr.units.kelvin):
        raise ValueError("'temperature' has incorrect units.")

    try:
        u = sr.u(pressure)
    except Exception as e:
        raise ValueError(f"Unable to parse 'pressure': {e}")
    if not u.has_same_units(sr.units.bar):
        raise ValueError("'pressure' has incorrect units.")

    try:
        u = sr.u(cutoff)
    except Exception as e:
        raise ValueError(f"Unable to parse 'cutoff': {e}")
    if not u.has_same_units(sr.units.angstrom):
        raise ValueError("'cutoff' has incorrect units.")

    if not isinstance(num_samples, int):
        raise TypeError("'num_samples' must be a of type 'int'.")
    if num_samples <= 1:
        raise ValueError("'num_samples' must be greater than 1.")

    if not isinstance(sample_interval, str):
        raise TypeError("'sample_interval' must be a of type 'str'.")
    try:
        u = sr.u(sample_interval)
    except Exception as e:
        raise ValueError(f"Unable to parse 'sample_interval': {e}")
    if not u.has_same_units(sr.units.picosecond):
        raise ValueError("'sample_interval' has incorrect units.")

    # Disable the dynamics progress bar.
    sr.base.ProgressBar.set_silent()

    # Try to read the topology and coordinate files.
    try:
        mols = sr.load(topology_file, coordinate_file)
    except Exception as e:
        raise ValueError(f"Unable to read topology/coordinate files: {e}")

    # Make sure there are only water molecules.
    waters = mols["water"]
    if not len(waters) == len(mols):
        raise ValueError(
            "The provided topology/coordinate files must contain only water molecules."
        )

    # Store the number of water molecules.
    num_waters = len(waters)

    # Set up the NPT simulation.
    try:
        d = mols.dynamics(temperature=temperature, pressure=pressure, timestep="2 fs")
    except Exception as e:
        raise ValueError(f"Unable to set up NPT dynamics: {e}")

    # Run the NPT simulation and collect volume samples.
    volumes = []
    for _ in range(num_samples):
        d.run(sample_interval, energy_frequency=0, frame_frequency=0, save_frequency=0)

        # Get the current OpenMM volume in A^3.
        volume = (
            d.context().getState().getPeriodicBoxVolume().value_in_unit(angstrom**3)
        )

        # Store the volume.
        volumes.append(volume)

    # Calculate the average volume per water molecule in A^3/molecule.
    avg_volume = sum(volumes) / (len(volumes) * num_waters)

    return avg_volume
