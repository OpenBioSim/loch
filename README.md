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
