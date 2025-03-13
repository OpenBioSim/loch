import argparse

from loch import GCMCSampler

import sire as sr
from sire.base import ProgressBar as _ProgressBar

_ProgressBar.set_silent()
del _ProgressBar

parser = argparse.ArgumentParser("GCMC sampler demo")
parser.add_argument(
    "--cutoff-type",
    type=str,
    default="rf",
    help="The non-bonded cutoff type",
    choices=["rf", "pme"],
    required=False,
)
parser.add_argument(
    "--cutoff",
    type=str,
    default="10 A",
    help="The non-bonded cutoff",
    required=False,
)
parser.add_argument(
    "--num-attempts",
    type=int,
    default=10000,
    help="The number of GCMC insertion attempts",
    required=False,
)
args = parser.parse_args()

# Load the scytalone dehydratase system
mols = sr.load("examples/scytalone-dehydratase/outputs/*7")

# Store the reference selection.
reference = "(residx 22 or residx 42) and (atomname OH)"

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    reference,
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    log_level="debug",
)

# Create a dynamics object using the modified GCMC system.
# This contains a number of ghost waters that can be used
# for insertion moves.
d = sampler.system().dynamics(
    cutoff_type=args.cutoff_type, cutoff=args.cutoff, pressure=None
)

# Run 100 dynamics cycles with a GCMC move after each cycle.
for i in range(100):
    print(f"Cycle {i}")

    # Run 1ps of dynamics.
    d.run("1ps", save_frequency=0)

    # Perform a GCMC move.
    context, move, accepted = sampler.move(d.context())

    # If the move was accepted, update the dynamics object.
    if accepted:
        d._d._omm_mols = context

print(f"Accepted: {sampler.num_accepted()}")
print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")

# Save the final configuration.
mols = d.commit()
sr.save(mols[f"mols within 10 of {reference}"], f"final_{args.cutoff_type}.pdb")
