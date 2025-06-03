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

## Examples

A full set of examples can be found in the [examples](examples) directory.

## GCMC Free Energy Perturbation

Free Energy Perturbation (FEP) with GCMC using `Loch` is supported via the
[SOMD2](https://github.com/OpenBioSim/somd2) package.

## Notes

* Make sure that `nvcc` is in your `$PATH`.

* A future version supporting AMD GPUs via PyOpenCL is planned.
