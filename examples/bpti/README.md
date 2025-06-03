# Bovine pancreatic trypsin inhibitor (BPTI)

Sample water positions in the binding site of BPTI:

```
cd examples/bpti
python bpti.py
```

When complete, the script will write a PDB file, `bpti_clusters.pdb`, containing
the water clusters that were sampled within the GCMC sphere radius over the course
of the trajectory. Aligned positions of the crystal waters are also written to the
`bpti_crystal_waters.pdb` file. The script will also write a reference structure for
BPTI, `bpti_reference.pdb`. The clusters can be visualised using
[VMD](https://www.ks.uiuc.edu/Research/vmd/) or [PyMOL](https://pymol.org/),
e.g. for `VMD`:

```
vmd -m bpti_reference.pdb bpti_crystal_waters.pdb bpti_clusters.pdb -startup vmd.tcl
```

which should look something like the following:

![BPTI water clusters](clusters.png)

(Blue atoms show the positions of the oxygen atoms for the crystal waters. The
other atoms indicate the water oxygen that was closest to the centre of each
water cluster sampled during the simulation. Darker red indicates higher
occupancy.)

To view the GCMC sphere over the course of the trajectory, use the `PyMOL`
visualisation script from [grand](https://github.com/essex-lab/grand/blob/v1.0.0/grand/scripts/gcmc_pymol.py):

```
python gcmc_pymol.py --topology bpti_final.pdb --trajectory bpti_aligned.dcd --sphere bpti_gcmc_sphere.pdb
```
