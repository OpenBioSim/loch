# Loch

Prototype GCMC water sampling code.

## Installation

```bash
conda create -f environment.yml
```

## Usage

Attempt 10000 random insertions within the periodic box of the system:

```bash
python loch.py examples/scytalone-dehydratase/outputs/*7 \
    --num-insertions 10000
```

To run 10 batches of the above command in serial:

```bash
python loch.py examples/scytalone-dehydratase/outputs/*7 \
    --num-insertions 10000 \
    --num-batches 10
```

Attempt 10000 random insertions targeted at the hydration site:

```bash
python loch.py examples/scytalone-dehydratase/outputs/*7 \
    --num-insertions 10000 \
    --reference "(residx 22 or residx 42) and (atomname OH)" \ 
    --max-distance d
```

Here `reference` is a selection string used to specify the atoms whose center
of geometry will be used as the target site for insertions and `radius` is the
maximum distance from the target site in Angstroms.

The code parallelises work over blocks of GPU threads in batches. Parallelism
is performed across the *largest* dimension, i.e. the number of insertions if
this is larger than the number of atoms to calculate energies against, e.g.
when targeting a small region of the system, or the number of atoms if this
is larger.
