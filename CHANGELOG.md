Changelog
=========

[2026.1.0](https://github.com/openbiosim/loch/compare/2025.2.0...2026.1.0) - ********
-------------------------------------------------------------------------------------

* Add support for getting and restoring sampling statistics.
* Handle XED force field virtual sites.
* Add support for long-range Lennard-Jones dispersion correction.
* Add support for Beutler soft-core Lennard-Jones form.

[2025.2.0](https://github.com/openbiosim/loch/compare/2025.1.0...2025.2.0) - Feb 2026
-------------------------------------------------------------------------------------

* Fixed handling of four- and five-point water models [#2](https://github.com/OpenBioSim/loch/pull/2).
* Add support for the OpenCL platform and optimise GPU kernels [#6](https://github.com/OpenBioSim/loch/pull/6).
* Clamp exponent to avoid exponential overflow when calculation the acceptance
  probability for the PME correction [#9](https://github.com/OpenBioSim/loch/pull/9).
* Reduce memory footprint by using a shared primary CUDA context on each GPU device [#10](https://github.com/OpenBioSim/loch/pull/10).

[2025.1.0](https://github.com/OpenBioSim/loch/releases/tag/2025.1.0) - Nov 2025
-------------------------------------------------------------------------------

* Initial public release.
