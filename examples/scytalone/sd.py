import argparse
import math

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
    default=1,
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
    default=100,
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

# Load the scytalone dehydratase system.
mols = sr.load_test_files(f"sd{args.ligand}.prm7", f"sd{args.ligand}.rst7")

# Store the reference selection.
reference = "(residx 22 or residx 42) and (atomname OH)"

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    reference=reference,
    batch_size=args.batch_size,
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    radius=args.radius,
    temperature=args.temperature,
    max_gcmc_waters=100,
    bulk_sampling_probability=0,
    log_level=args.log_level,
    overwrite=True,
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

# Delete any existing waters from the GCMC region.
sampler.delete_waters(d.context())

# Perform initial GCMC equilibration on the initial structure.
print("Equilibrating the system with GCMC moves...")
for i in range(100):
    sampler.move(d.context())

# Run dynamics cycles with a GCMC move after each.
print("Runing dynamics with GCMC moves...")
total = 0
for i in range(args.num_cycles):
    # Run a dynamics block.
    d.run(args.cycle_time, save_frequency=0, energy_frequency=0, frame_frequency=0)

    # Perform a GCMC move.
    moves = sampler.move(d.context())

    # Report.
    N = sampler.num_waters()
    total += N
    print(
        f"Cycle {i+1}, N = {N}, "
        f"insertions = {sampler.num_insertions()}, "
        f"deletions = {sampler.num_deletions()}"
    )
    print(
        f"Current potential energy: {d.current_potential_energy().value():.3f} kcal/mol"
    )

print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Average number of waters: {total / args.num_cycles:.2f}")
print(f"Move acceptance probability: {sampler.move_acceptance_probability():.4f}")
print(f"Attempt acceptance probability: {sampler.attempt_acceptance_probability():.4f}")

# Save the final configuration.
mols = d.commit()

# Store the periodic space.
space = mols.space()

# Save the final positions for SD and the ligand.
sr.save(mols["molidx 1"], f"sd_{args.ligand}.pdb")
sr.save(mols["molidx 0"], f"ligand_{args.ligand}.pdb")

# Get the reference coordinates.
centre = mols[reference].coordinates()

# Find the water oxygens within the GCMC sphere.
nums = []
radius = sampler._radius.value()
for i, atom in enumerate(mols.atoms()):
    # This is a water oxygen.
    if atom.element() == sr.mol.Element("O") and atom.residue().name().value() == "WAT":
        # Get the charge for this atom from the OpenMM nonbonded force.
        charge, _, _ = sampler._nonbonded_force.getParticleParameters(i)
        # This is a physical water.
        if not math.isclose(charge._value, 0.0):
            dist = space.calc_dist(centre, atom.coordinates())
            # The oxygen is within the GCMC sphere.
            if dist < radius:
                nums.append(str(atom.molecule().number().value()))

# Save the waters.
if len(nums) > 0:
    sr.save(mols[f"molnum {','.join(nums)}"], f"water_{args.ligand}.pdb")
