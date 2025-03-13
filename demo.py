from loch import GCMCSampler

import sire as sr
from sire.base import ProgressBar as _ProgressBar

_ProgressBar.set_silent()
del _ProgressBar

# Load the scytalone dehydratase system
mols = sr.load("examples/scytalone-dehydratase/outputs/*7")

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    "(residx 22 or residx 42) and (atomname OH)",
    num_attempts=10000,
    cutoff_type="rf",
    log_level="debug",
)

# Create a dynamics object using the modified GCMC system.
# This contains a number of ghost waters that can be used
# for insertion moves.
d = sampler.system().dynamics(cutoff_type="rf", pressure=None)

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
