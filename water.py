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
    default="rf",
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
    "--num-attempts",
    help="The number of GCMC insertion attempts",
    type=int,
    default=10000,
    required=False,
)
parser.add_argument(
    "--num-cycles",
    help="The number of dynamics cycles",
    type=int,
    default=100,
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
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    excess_chemical_potential="-6.16 kcal/mol",
    standard_volume="30.543 A^3",
    temperature=args.temperature,
    max_gcmc_waters=100,
    log_level="info",
)

# Create a dynamics object using the modified GCMC system.
# This contains a number of ghost waters that can be used
# for insertion moves.
d = sampler.system().dynamics(
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    temperature=args.temperature,
    pressure=None,
    constraint="h_bonds",
    timestep="2 fs",
)

# Get the context.
context = d.context()

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
    context, accepted, move = sampler.move(d.context())
    end = time()
    if i > 0:
        total += end - start

    # If the move was accepted, update the dynamics object.
    if accepted:
        d._d._omm_mols = context

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
    print(f"volume {volume.value():.5f} A^3, density: {density:.5f} g/mL")

print(f"Accepted: {sampler.num_accepted()}")
print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Average time: {1000*total / (args.num_cycles - 1):.3f} ms")
