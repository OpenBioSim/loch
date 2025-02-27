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
    --target x y z \
    --max-distance d
```

Here `x`, `y`, and `z` are the coordinates of the target site in Angstrom, and `d`
is the maximum distance from the target site, i.e. insertions will only be
attempted within a cube of side length `2d` centered at the target site.

The code parallelises work over blocks of GPU threads in batches. Parallelism
is performed across the *largest* dimension, i.e. the number of insertions if
this is larger than the number of atoms to calculate energies against, e.g.
when targeting a small region of the system, or the number of atoms if this
is larger.
