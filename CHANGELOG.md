Changelog
=========

[2026.2.0](https://github.com/openbiosim/loch/compare/2026.1.0...2026.2.0) - ********
--------------------------------------------------------------------------------------

* Please add an item to this CHANGELOG for any new features or bug fixes when creating a PR.
* Restrict PME energy calculation to required force groups [#29](https://github.com/OpenBioSim/loch/pull/29).
* Add `set_lambda` and `_precompute_lambdas` so that a sampler can be re-used across lambda values without rebuilding an OpenMM context.

[2026.1.0](https://github.com/openbiosim/loch/compare/2025.2.0...2026.1.0) - Jun 2026
-------------------------------------------------------------------------------------

* Add support for getting and restoring sampling statistics.
* Handle XED force field virtual sites [#17](https://github.com/OpenBioSim/loch/pull/17).
* Add support for long-range Lennard-Jones dispersion correction [#18](https://github.com/OpenBioSim/loch/pull/18).
* Add support for Beutler soft-core Lennard-Jones form [#18](https://github.com/OpenBioSim/loch/pull/18).
* Fixed type check for ``water_template`` [#19](https://github.com/OpenBioSim/loch/pull/19).
* Add support for simulations in the osmotic ensemble [#22](https://github.com/OpenBioSim/loch/pull/22).
* Fixed non-uniform bulk insertion positions caused by use of normal rather than uniform random numbers [#24](https://github.com/OpenBioSim/loch/pull/24).
* Add methods to update the system with the current water state and return system without ghost waters [#26](https://github.com/OpenBioSim/loch/pull/26).

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
