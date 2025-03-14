# Loch

Prototype GCMC water sampling code.

## Installation

```bash
conda create -f environment.yml
```

## Usage

Run 100 1ps dynamics blocks for scytalone dehydratase, performing 10000
GCMC trial moves (insertions or deletions) after each block:

```bash
python demo.py --cutoff-type rf
```

When complete, the script will write a PDB file containing all molecules
within the GCMC sphere radius, e.g. `final_rf.pdb`. For comparison, the initial
configuration is written to `initial.pdb`.
