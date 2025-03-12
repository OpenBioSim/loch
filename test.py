from loch import GCMCSampler

import sire as sr

mols = sr.load("examples/scytalone-dehydratase/outputs/*7")

sampler = GCMCSampler(mols, "(residx 22 or residx 42) and (atomname OH)", num_attempts=10000)

d = sampler.get_system().dynamics(cutoff_type="rf", pressure=None)

for i in range(100):
    sampler.move(d.context())
