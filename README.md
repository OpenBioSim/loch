# Loch

Prototype CUDA accelerated GCMC water sampling code.

## Installation

```
conda create -f environment.yml
```

## Examples

### Bulk water

Run 10ns of bulk water sampling, with GCMC moves every 1ps:

```
cd examples/water
python water.py
```

### Bovine pancreatic trypsin inhibitor (BPTI)

Sample water positions in the binding site of BPTI:

```
cd examples/bpti
python bpti.py
```

When complete, the script will write a PDB file, `bpti_clusters.pdb`, containing
the water clusters that were sampled within the GCMC sphere radius over the course
of the trajectory. Aligned positions of the crystal waters are also written to the
`bpti_crystal_waters.pdb` file. The script will also write a reference structure for
BPTI, `bpti_reference.pdb`. The clusters can be visualised using `VMD` or `PyMOL`,
e.g. for `VMD`:

```
vmd -m bpti_reference.pdb bpti_crystal_waters.pdb bpti_clusters.pdb -startup vmd.tcl
```

To view the GCMC sphere over the course of the trajectory, use the `PyMOL`
visualisation script from [grand](https://github.com/essex-lab/grand/blob/v1.0.0/grand/scripts/gcmc_pymol.py):

```
python gcmc_pymol.py --topology bpti_final.pdb --trajectory bpti_aligned.dcd --sphere bpti_gcmc_sphere.pdb
```

(You will likely need to reset the view in `PyMOL` once loaded.)

### Scytalone dehydratase (SD)

Perform 100 ps of GCMC sampling of water in the binding site of scytalone dehydratase (SD):

```
cd examples/sd
python sd.py --ligand 1
````

The ligand index specifes which ligand will used in the complex. The script
will output a PDB file for the final SD and ligand strucures, e.g.
`sd_1.pdb` and `ligand_1.pdb`, along with a PDB file for any waters within
the GCMC sampling region, e.g. `water_1.pdb`. (Here the suffix `_1` indicates
the ligand index that was specified.) To visualise with `VMD`:

```
vmd -m sd_1.pdb ligand_1.pdb water_1.pdb -startup vmd.tcl
```

While running, the script will output the current number of waters in the GCMC
region after each cycle, with the average occupancy reported at the end.
