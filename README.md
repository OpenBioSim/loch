# Loch

[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

CUDA accelerated GCMC water sampling code. Built on top of [Sire](https://github.com/OpenBioSim/sire)
, [BioSimSpace](https://github.com/OpenBioSim/biosimspace) and [OpenMM](https://github.com/openmm/openmm).

## Installation

First, create a conda environment with the required dependencies:

```
conda create -f environment.yml
conda activate loch
```

Next, clone the repository and install the package:

```
git clone https://github.com/openbiosim/loch
cd loch
pip install -e .
```

## How does it work?

Loch is based on [grand](https://github.com/essex-lab/grand). Instead of
computing the energy change for a trial insertion/deletion with OpenMM, the
calculation is performed at the reaction field (RF) level using a custom CUDA
kernel, which gives _exact_ agreement with OpenMM. Particle mesh Ewald (PME)
is handled using the method for sampling from an approximate potential (in this
case the RF potential) described [here](https://doi.org/10.1063/1.1563597).
Parallelisation of the insertion and deletion trials is achieved using the
strategy described in [this](https://doi.org/10.1021/acs.jctc.0c00660) paper.

## Usage

1) Load the molecular system of interest, e.g.:

```python
import sire as sr

mols = sr.load_test_files("bpti.prm7", "bpti.rst7")
```

2) Create a `GCMCSampler`:

```python
from loch import GCMCSampler

sampler = GCMCSampler(
    mols,
    reference = "(resnum 10 and atomname CA) or (resnum 43 and atomname CA)",
    num_attempts=10000,
    batch_size=1000,
    cutoff_type="pme",
    cutoff="10 A",
    radius="4 A",
    temperature="298 K",
    max_gcmc_waters=50,
    bulk_sampling_probability=0.1,
    log_level="info",
)
```

Here the `reference` is a Sire selection string for the atoms that define
the centre of geometry of the GCMC sphere. Each GCMC move consists of
a total of `num_attempts` random insertion and deletion attempts, with
`batch_size` number of attempts being performed in parallel. The
`bulk_sampling_probability` controls the probability performing a bulk
sampling move, i.e. performing attempts within the entire simulation box,
rather than just within the GCMC sphere.

3) Get the GCMC system:

In order to perform a simulation we need to get back the GCMC system, which
contains an additional `max_gcmc_waters` number of ghost water molecules
that are used for insertion moves.

```python
gcmc_system = sampler.get_system()
```

4) Create an OpenMM context:

We can directly use the Sire dynamics interface to create an OpenMM context
for us, e.g.:

```python
d = gcmc_system.dynamics(
    integrator="langevin_middle",
    temperature="298 K",
    pressure=None,
    cutoff_type="pme",
    cutoff="10 A",
    constraint="h_bonds"
    timestep="2 fs",
)
```

> [!Note]
> While we have used Sire to create the OpenMM context, you can also write
> the GCMC system to file and create the OpenMM context manually.

> [!Note]
> GCMC sampling must be performed in the NVT ensemble, hence the pressure
> is set to `None` in the above example. However, bulk sampling moves can
> be used as an effective barostat.

5) Run dynamics with GCMC sampling:

```python
# Set the cycle frequency for saving ghost residue indices.
frame_frequency = 50

# Run 1ns of dynamics and perform GCMC moves every 1ps.
for i in range(1000):
    # Run 1ps of dynamics.
    d.run("1ps", energy_frequency="50ps", frame_frequency="50ps")

    # Perform a GCMC move.
    moves = sampler.move(d.context())

    # If we hit the frame frequency, then save the current ghost residue indices.
    if i > 0 and (i + 1) % frame_frequency == 0:
        sampler.write_ghost_residues()

    # Print the current status.
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
sr.save(mols.trajectory(), "gcmc_traj.dcd")
```

> [!Note]
> `Loch` is designed to be compatible with `grand`, so you can make use of
> `grand` to perform post-simulation analysis, such as trajectory processing
> and water cluster analysis.

## Examples

A full set of examples can be found in the [examples](examples) directory.

## GCMC Free Energy Perturbation

Free Energy Perturbation (FEP) with GCMC using `Loch` is supported via the
[SOMD2](https://github.com/OpenBioSim/somd2) package.

## Notes

* Make sure that `nvcc` is in your `PATH`.

* A future version supporting AMD GPUs via PyOpenCL is planned.
