import argparse
import grand

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
    "--log-level",
    help="The logging level",
    type=str,
    default="info",
    choices=["info", "debug"],
    required=False,
)

args = parser.parse_args()

# Load the system.
mols = sr.load_test_files("bpti.prm7", "bpti.rst7")

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    reference="(resnum 10 and atomname CA) or (resnum 43 and atomname CA)",
    batch_size=args.batch_size,
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    radius="4.2 A",
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
d.minimise()

# Get the context.
context = d.context()

# Equilibrate the system.

# 1) Perform 100 GCMC moves.
print("Equilibrating the system with GCMC moves...")
for i in range(10000):
    moves = sampler.move(context)

# 2) Run 1ps of dynamics, performing GCMC moves every 10fs.
print("Running 1ps of dynamics with GCMC moves...")
for i in range(100):
    # Run 10fs of dynamics.
    d.run("10 fs", save_frequency=0, energy_frequency=0, frame_frequency=0)

    # Perform a GCMC move.
    moves = sampler.move(d.context())

# 3) Run 500ps of regular NPT dynamics.
print("Running 500ps of NPT dynamics...")

# Get a new Sire system from the dynamics object.
mols = d.commit()

# Create a NPT dynamics object.
d_npt = mols.dynamics(
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    temperature=args.temperature,
    integrator="langevin_middle",
    pressure="1 bar",
    constraint="h_bonds",
    timestep="2 fs",
)
d.minimise()

# Run the dynamics.
d_npt.run("500 ps", save_frequency=0, energy_frequency=0, frame_frequency=0)

# Get the updated Sire system from the dynamics object.
mols = d_npt.commit()

# Copy the state between the two contexts and re-minimise.
d._d._omm_mols.setState(d_npt._d._omm_mols.getState())
d.minimise()

# Update the box information in the GCMC sampler.
sampler.set_box(mols)

# Store the frame frequency.
frame_frequency = 50

# 4) Run 10ns dynamics with GCMC moves every 1ps.
print("Running 10ns of dynamics with GCMC moves...")
for i in range(1000):
    # Run 1ps of dynamics.
    d.run("1ps", energy_frequency="50ps", frame_frequency="50ps")

    # Perform a GCMC move.
    moves = sampler.move(d.context())

    # If we hit the frame frequency, then save the current ghost residue indices.
    if i > 0 and i % frame_frequency == 0:
        sampler.write_ghost_residues()

    print(
        f"Cycle {i}, N = {sampler.num_waters()}, "
        f"insertions = {sampler.num_insertions()}, "
        f"deletions = {sampler.num_deletions()}"
    )
    print(
        f"Current potential energy: {d.current_potential_energy().value():.3f} kcal/mol"
    )

# Save the trajectory.
mols = d.commit()
sr.save(mols, "bpti_final.prm7")
sr.save(mols.trajectory(), "bpti.dcd")

# Define reference atoms for the GCMC sphere (grand format).
ref_atoms = [
    {"name": "CA", "resname": "TYR", "resid": "10", "chain": 0},
    {"name": "CA", "resname": "ASN", "resid": "43", "chain": 0},
]

# Remove ghost waters from GCMC region.
trj = grand.utils.shift_ghost_waters(
    ghost_file="ghost_file.txt", topology="bpti_final.prm7", trajectory="bpti.dcd"
)

# Centre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname="TYR", name="CA", resid=9)

# Align the trajectory to the protein.
grand.utils.align_traj(t=trj, output="bpti_aligned.dcd")

# Write out a PDB trajectory of the GCMC sphere
grand.utils.write_sphere_traj(
    radius=4.2,
    ref_atoms=ref_atoms,
    topology="bpti_final.prm7",
    trajectory="bpti_aligned.dcd",
    output="bpti_gcmc_sphere.pdb",
    initial_frame=True,
)

# Cluster water sites.
grand.utils.cluster_waters(
    topology="bpti_final.prm7",
    trajectory="bpti_aligned.dcd",
    sphere_radius=4.2,
    ref_atoms=ref_atoms,
    cutoff=2.4,
    output="bpti_clusters.pdb",
)

# Read and write the clustered waters with Sire to recover the element records.
mols = sr.load("bpti_clusters.pdb")
sr.save(mols, "bpti_clusters.pdb")

print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Move acceptance probability: {sampler.move_acceptance_probability():.4f}")
print(f"Attempt acceptance probability: {sampler.attempt_acceptance_probability():.4f}")
