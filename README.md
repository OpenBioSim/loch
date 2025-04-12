# Loch

Prototype GCMC water sampling code.

## Installation

```
conda create -f environment.yml
```

## Examples

### Scytalone dehydratase

Run 100 1ps dynamics blocks for scytalone dehydratase, performing GCMC trial
moves after each block:

```
cd examples/scytalone
python sd.py --cutoff-type pme
```

When complete, the script will write a PDB file containing all molecules
within the GCMC sphere radius, e.g. `final_rf.pdb`. For comparison, the initial
configuration is written to `initial.pdb`.

### Bulk water

Run 10ns of bulk water sampling, with GCMC moves every 1ps:

```
cd examples/water
python water.py --cutoff-type pme
```
