import argparse
import grand

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

# Store the ligand index.
lig = args.ligand

# Load the scytalone dehydratase system.
mols = sr.load_test_files(f"sd{lig}.prm7", f"sd{lig}.rst7")

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
    ghost_file=f"ghosts_{lig}.txt",
    log_file=f"gcmc_{lig}.txt",
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

# Store the frame frequency.
frame_frequency = 5

# Store the radius.
radius = sampler._radius.value()

# Perform initial GCMC equilibration on the initial structure.
print("Equilibrating the system with GCMC moves...")
for i in range(100):
    sampler.move(d.context())

# Run 1ns dynamics with GCMC moves every 1ps.
print("Running 1ns of dynamics with GCMC moves...")
for i in range(1000):
    # Run 1ps of dynamics.
    d.run("1ps", energy_frequency="5ps", frame_frequency="5ps")

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
sr.save(mols, f"sd_{lig}_final.pdb")
sr.save(mols, f"sd_{lig}_final.prm7")
sr.save(mols.trajectory(), f"sd_{lig}.dcd")

# Define reference atoms for the GCMC sphere (grand format).
ref_atoms = [
    {"name": "OH", "resname": "TYR", "resid": "24"},
    {"name": "OH", "resname": "TYR", "resid": "44"},
]

# Remove ghost waters from GCMC region.
trj = grand.utils.shift_ghost_waters(
    ghost_file=f"ghosts_{lig}.txt",
    topology=f"sd_{lig}_final.pdb",
    trajectory=f"sd_{lig}.dcd",
)

# Centre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname="TYR", name="CA", resid=10)

# Align the trajectory to the protein.
grand.utils.align_traj(t=trj, output=f"sd_{lig}_aligned.dcd")

# Write out a PDB trajectory of the GCMC sphere
grand.utils.write_sphere_traj(
    radius=radius,
    ref_atoms=ref_atoms,
    topology=f"sd_{lig}_final.pdb",
    trajectory=f"sd_{lig}_aligned.dcd",
    output=f"sd_{lig}_gcmc_sphere.pdb",
    initial_frame=True,
)

# Cluster water sites.
grand.utils.cluster_waters(
    topology=f"sd_{lig}_final.pdb",
    trajectory=f"sd_{lig}_aligned.dcd",
    sphere_radius=radius,
    ref_atoms=ref_atoms,
    cutoff=2.4,
    output=f"clusters_{lig}.pdb",
)

# Read and write the clustered waters with Sire to recover the element records.
mols = sr.load(f"clusters_{lig}.pdb")
sr.save(mols, f"clusters_{lig}.pdb")

# Average the position of the protein and ligand over the aligned trajectory.
mols = sr.load(f"sd_{lig}_final.pdb", f"sd_{lig}_aligned.dcd")
protein_num = mols[reference].molecules()[0].number()
ligand_num = mols["resname MOL"].molecules()[0].number()
for frame in mols.trajectory():
    # Get the molecule containing the reference atoms.
    protein = frame[protein_num]
    ligand = frame[ligand_num]

    # Get the coordinates array.
    protein_coords = sr.io.get_coords_array(protein)
    ligand_coords = sr.io.get_coords_array(ligand)

    # Udpate the average position of the first molecule.
    try:
        average_protein_position += protein_coords
        average_ligand_position += ligand_coords
    except:
        average_protein_position = protein_coords
        average_ligand_position = ligand_coords

# Write the average protein position to file.
average_protein_position /= mols.num_frames()
mol = mols[protein_num]
cursor = mol.cursor()
for i, atom in enumerate(cursor.atoms()):
    coords = sr.maths.Vector(*average_protein_position[i])
    atom["coordinates"] = coords
protein = cursor.commit()
sr.save(protein, f"sd_{lig}_reference.pdb")

# Write the average ligand position to file.
average_ligand_position /= mols.num_frames()
mol = mols[ligand_num]
cursor = mol.cursor()
for i, atom in enumerate(cursor.atoms()):
    coords = sr.maths.Vector(*average_ligand_position[i])
    atom["coordinates"] = coords
ligand = cursor.commit()
sr.save(ligand, f"ligand_{lig}_reference.pdb")

print(f"Insertions: {sampler.num_insertions()}")
print(f"Deletions: {sampler.num_deletions()}")
print(f"Move acceptance probability: {sampler.move_acceptance_probability():.4f}")
print(f"Attempt acceptance probability: {sampler.attempt_acceptance_probability():.4f}")
