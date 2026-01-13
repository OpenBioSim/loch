######################################################################
# Loch: GPU accelerated GCMC water sampling engine.
#
# Copyright: 2025-2026
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# Loch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Loch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Loch. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""
Random number generation utilities for GPU platforms.
"""

from typing import Dict as _Dict, Optional as _Optional

import numpy as _np


class RNGManager:
    """
    Host-side random number generator for both CUDA and OpenCL backends.

    This class generates random numbers on the host (CPU) that are then
    transferred to the GPU for use in kernels. This approach eliminates
    the need for device-side RNG libraries (like CURAND) and simplifies
    cross-platform support.
    """

    def __init__(self, batch_size: int, seed: _Optional[int] = None) -> None:
        """
        Initialize the RNG manager.

        Parameters
        ----------
        batch_size : int
            Number of parallel trials in a batch.

        seed : int, optional
            Random seed for reproducibility. If None, a random seed is used.
        """
        self.batch_size = batch_size
        self.rng = _np.random.default_rng(seed)

    def generate_insertion_randoms(self) -> _Dict[str, _np.ndarray]:
        """
        Generate random numbers for the generateWater kernel.

        This method generates all random numbers needed for water insertion:
        - 3 uniform randoms per trial for rotation (Householder method)
        - 3 normal randoms per trial for position direction
        - 1 uniform random per trial for radial distance

        Returns
        -------
        dict
            Dictionary with keys:
            - 'rotation': (batch_size, 3) array of uniform [0,1) randoms
            - 'position': (batch_size, 3) array of normal randoms
            - 'radius': (batch_size,) array of uniform [0,1) randoms
        """
        return {
            "rotation": self.rng.uniform(0, 1, size=(self.batch_size, 3)).astype(
                _np.float32
            ),
            "position": self.rng.normal(0, 1, size=(self.batch_size, 3)).astype(
                _np.float32
            ),
            "radius": self.rng.uniform(0, 1, size=self.batch_size).astype(_np.float32),
        }

    def generate_acceptance_randoms(self) -> _np.ndarray:
        """
        Generate random numbers for the checkAcceptance kernel.

        Returns
        -------
        numpy.ndarray
            Array of shape (batch_size,) with uniform [0,1) random numbers
            for Metropolis acceptance checks.
        """
        return self.rng.uniform(0, 1, size=self.batch_size).astype(_np.float32)
