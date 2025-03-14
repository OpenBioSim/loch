import argparse

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
    "--radius",
    help="The radius of the GCMC sphere",
    type=str,
    default="4 A",
    required=False,
)
parser.add_argument(
    "--ligand",
    help="The ligand index",
    type=int,
    default=None,
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

# Load the scytalone dehydratase system
if args.ligand is None:
    mols = sr.load(f"examples/scytalone-dehydratase/outputs/*7")
else:
    mols = sr.load(f"examples/scytalone-dehydratase/outputs/lig{args.ligand}/*7")

# Create the PDB suffix.
suffix = f"_lig{args.ligand}" if args.ligand is not None else ""

# Store the reference selection.
reference = "(residx 22 or residx 42) and (atomname OH)"

# Save the initial configuration.
sr.save(
    mols[f"mols within {args.radius} of {reference}"],
    f"initial{suffix}.pdb",
)

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    reference,
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    radius=args.radius,
    temperature=args.temperature,
    log_level="debug",
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

# Run dynamics cycles with a GCMC move after each.
total = 0
for i in range(args.num_cycles):
    print(f"Cycle {i}")

    # Run 1ps of dynamics.
    d.run(args.cycle_time, save_frequency=0, energy_frequency=0, frame_frequency=0)

    # Perform a GCMC move.
    start = time()
    context, move, accepted = sampler.move(d.context())
    end = time()
    if i > 0:
        total += end - start

    # If the move was accepted, update the dynamics object.
    if accepted:
        d._d._omm_mols = context

print(f"Accepted: {sampler.num_accepted()}")
print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Average time: {1000*total / (args.num_cycles - 1):.3f} ms")

# Save the final configuration.
mols = d.commit()
sr.save(
    mols[f"mols within {sampler._radius.value()} of {reference}"],
    f"final_{args.cutoff_type}{suffix}.pdb",
)
