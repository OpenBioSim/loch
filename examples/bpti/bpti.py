import argparse
import grand
import MDAnalysis as mda

from MDAnalysis.analysis import align

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
    default="error",
    choices=["info", "debug", "error"],
    required=False,
)

args = parser.parse_args()

# Load the system.
mols = sr.load_test_files("bpti.prm7", "bpti.rst7")

# Store the reference for the center of geometry of the GCMC sphere.
reference = "(resnum 10 and atomname CA) or (resnum 43 and atomname CA)"

# Store the GCMC radius.
radius = 4.2

# Create a GCMC sampler.
sampler = GCMCSampler(
    mols,
    reference=reference,
    batch_size=args.batch_size,
    num_attempts=args.num_attempts,
    cutoff_type=args.cutoff_type,
    cutoff=args.cutoff,
    radius=f"{radius} A",
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

# Store the frame frequency.
frame_frequency = 50

# Delete any existing waters from the GCMC region.
sampler.delete_waters(d.context())

# Perform initial GCMC equilibration on the initial structure.
print("Equilibrating the system with GCMC moves...")
for i in range(100):
    sampler.move(d.context())

# Run 10ns dynamics with GCMC moves every 1ps.
print("Running 10ns of dynamics with GCMC moves...")
for i in range(10000):
    # Run 1ps of dynamics.
    d.run("1ps", energy_frequency="50ps", frame_frequency="50ps")

    # Perform a GCMC move.
    moves = sampler.move(d.context())

    # If we hit the frame frequency, then save the current ghost residue indices.
    if i > 0 and (i + 1) % frame_frequency == 0:
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
sr.save(mols, "bpti_final.pdb")
sr.save(mols, "bpti_final.prm7")
sr.save(mols.trajectory(), "bpti.dcd")

# Define reference atoms for the GCMC sphere (grand format).
ref_atoms = [
    {"name": "CA", "resname": "TYR", "resid": "10", "chain": 0},
    {"name": "CA", "resname": "ASN", "resid": "43", "chain": 0},
]

# Remove ghost waters from GCMC region.
trj = grand.utils.shift_ghost_waters(
    ghost_file="ghosts.txt", topology="bpti_final.pdb", trajectory="bpti.dcd"
)

# Centre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname="TYR", name="CA", resid=10)

# Align the trajectory to the protein.
grand.utils.align_traj(t=trj, output="bpti_aligned.dcd")

# Write out a PDB trajectory of the GCMC sphere
grand.utils.write_sphere_traj(
    radius=4.2,
    ref_atoms=ref_atoms,
    topology="bpti_final.pdb",
    trajectory="bpti_aligned.dcd",
    output="bpti_gcmc_sphere.pdb",
    initial_frame=True,
)

# Cluster water sites.
grand.utils.cluster_waters(
    topology="bpti_final.pdb",
    trajectory="bpti_aligned.dcd",
    sphere_radius=4.2,
    ref_atoms=ref_atoms,
    cutoff=2.4,
    output="bpti_clusters.pdb",
)

# Read and write the clustered waters with Sire to recover the element records.
mols = sr.load("bpti_clusters.pdb")
sr.save(mols, "bpti_clusters.pdb")

# Average the position of the protein over the aligned trajectory.
mols = sr.load("bpti_final.pdb", "bpti_aligned.dcd")
mol_num = mols[reference].molecules()[0].number()
for frame in mols.trajectory():
    # Get the molecule containing the reference atoms.
    mol = frame[mol_num]

    # Get the coordinates array.
    coords = sr.io.get_coords_array(mol)

    # Udpate the average position of the first molecule.
    try:
        average_position += coords
    except:
        average_position = coords

# Write the average positions to file.
average_position /= mols.num_frames()
mol = mols[mol_num]
cursor = mol.cursor()
for i, atom in enumerate(cursor.atoms()):
    coords = sr.maths.Vector(*average_position[i])
    atom["coordinates"] = coords
mol = cursor.commit()
sr.save(mol, "bpti_reference.pdb")

# Align the crystal structure to the average position.
u0 = mda.Universe("bpti_reference.pdb")
u1 = mda.Universe("5pti.pdb")
rmsds = align.alignto(u1, u0, select="name CA", match_atoms=False)
u1.atoms.write("5pti_aligned.pdb")

# Load the aligned structure.
mols = sr.load("5pti_aligned.pdb")
space = mols.space()

# Get the center of geometry of the reference atoms.
center = mols[reference].coordinates()

# Find the water oxygens close to the reference position.
atom_nums = []
for atom in mols.atoms():
    if atom.element() == sr.mol.Element("O") and atom.residue().name().value() == "DOD":
        dist = space.calc_dist(center, atom.coordinates())
        if dist < radius:
            atom_nums.append(str(atom.number().value()))

# Save the positions of the crystal water oxygens.
sr.save(mols[f"atomnum {','.join(atom_nums)}"], "bpti_crystal_waters.pdb")

print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Move acceptance probability: {sampler.move_acceptance_probability():.4f}")
print(f"Attempt acceptance probability: {sampler.attempt_acceptance_probability():.4f}")
