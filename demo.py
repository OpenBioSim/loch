from loch import GCMCSampler

import sire as sr

mols = sr.load("examples/scytalone-dehydratase/outputs/*7")

sampler = GCMCSampler(mols, "(residx 22 or residx 42) and (atomname OH)", num_attempts=10000)

d = sampler.get_system().dynamics(cutoff_type="rf", pressure=None)
context = d.context()

num_accepted = 0
for i in range(100):
    context, _, accepted = sampler.move(context)
    if accepted:
        num_accepted += 1

print(num_accepted)
