import argparse
import openmm

from time import time

from loch import GCMCSampler

import sire as sr

parser = argparse.ArgumentParser("GCMC sampler demo")
parser.add_argument(
    "--cutoff-type",
    help="The non-bonded cutoff type",
    type=str,
    default="pme",
    choices=["rf", "pme"],
    required=False,
)
parser.add_argument(
    "--cutoff",
    help="The non-bonded cutoff",
    type=str,
    default="10 A",
    required=False,
)
parser.add_argument(
    "--temperature",
    help="The simulation temperature",
    type=str,
    default="298 K",
    required=False,
)
parser.add_argument(
    "--cycle-time",
    help="The duration of the dynamics cycle",
    type=str,
    default="1 ps",
    required=False,
)
parser.add_argument(
    "--batch-size",
    help="The number of GCMC trials per batch",
    type=int,
    default=1000,
    required=False,
)
parser.add_argument(
    "--num-attempts",
    help="The number of GCMC attempts per move",
    type=int,
    default=10000,
    required=False,
)
parser.add_argument(
    "--num-cycles",
    help="The number of dynamics cycles",
    type=int,
    default=100000,
    required=False,
)
parser.add_argument(
    "--log-level",
    help="The logging level",
    type=str,
    default="error",
    choices=["info", "debug", "error"],
    required=False,
)
args = parser.parse_args()

# Load the water box.
mols = sr.load_test_files("water_box.prm7", "water_box.rst7")

# Store the box information.
space = mols.property("space")

# Store the volume.
volume = space.volume()

# Store Avagadro's number.
NA = openmm.unit.AVOGADRO_CONSTANT_NA._value

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    reference=None,
    batch_size=args.batch_size,
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    temperature=args.temperature,
    num_ghost_waters=100,
    log_level=args.log_level,
)

# Create a dynamics object using the modified GCMC system.
# This contains a number of ghost waters that can be used
# for insertion moves.
d = sampler.system().dynamics(
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    temperature=args.temperature,
    integrator="langevin_middle",
    pressure=None,
    constraint="h_bonds",
    timestep="2 fs",
)
d.randomise_velocities()

# Bind the GCMC sampler to the dynamics object. This ensures that the GCMC
# water state can be reset following a crash.
sampler.bind_dynamics(d)

# Store the mass of a water molecule.
mass = mols[0].mass()

num_waters = mols.num_molecules()

# Run dynamics cycles with a GCMC move after each.
total = 0
for i in range(args.num_cycles):
    # Run 1ps of dynamics.
    d.run(args.cycle_time, save_frequency=0, energy_frequency=0, frame_frequency=0)

    # Perform a GCMC move.
    start = time()
    moves = sampler.move(d.context())
    end = time()
    if i > 0:
        total += end - start

    print(
        f"Cycle {i}, N = {sampler.num_waters()}, "
        f"insertions = {sampler.num_insertions()}, "
        f"deletions = {sampler.num_deletions()}"
    )
    print(
        f"Current potential energy: {d.current_potential_energy().value():.3f} kcal/mol"
    )

    # Work out the current density in g/mL.
    total_mass = sampler.num_waters() * mass
    density = ((total_mass * sr.units.mole) / (volume * NA)).to("g/centimeter^3")
    print(f"volume: {volume.value():.5f} A^3, density: {density:.5f} g/mL")

print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Move acceptance probability: {sampler.move_acceptance_probability():.4f}")
print(f"Attempt acceptance probability: {sampler.attempt_acceptance_probability():.4f}")
print(f"Average time: {1000*total / (args.num_cycles - 1):.3f} ms")
