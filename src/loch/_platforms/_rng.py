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

from dataclasses import dataclass as _dataclass
from queue import Queue as _Queue
from threading import Thread as _Thread
from typing import Optional as _Optional

import numpy as _np


@_dataclass
class BatchRandoms:
    """
    Container for all random numbers needed for a single batch.

    Attributes
    ----------

    rotation : numpy.ndarray
        Shape (batch_size, 3) array of uniform [0,1) randoms for water rotation.

    position : numpy.ndarray
        Shape (batch_size, 3) array of normal randoms for position direction.

    radius : numpy.ndarray
        Shape (batch_size,) array of uniform [0,1) randoms for radial distance.

    acceptance : numpy.ndarray
        Shape (batch_size,) array of uniform [0,1) randoms for Metropolis acceptance.
    """

    rotation: _np.ndarray
    position: _np.ndarray
    radius: _np.ndarray
    acceptance: _np.ndarray


class RNGManager:
    """
    Host-side random number generator for both CUDA and OpenCL backends.

    This class generates random numbers on the host (CPU) that are then
    transferred to the GPU for use in kernels. This approach eliminates
    the need for device-side RNG libraries (like CURAND) and simplifies
    cross-platform support.

    Random number batches are pre-computed in a background thread while GPU
    kernels are running, so that they are ready when the next batch is requested.
    """

    _MAX_QUEUE_SIZE = 10

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
        self._batch_size = batch_size
        self._rng = _np.random.default_rng(seed)
        self._queue: _Queue[BatchRandoms] = _Queue(maxsize=self._MAX_QUEUE_SIZE)
        self._shutdown = False

        # Start the background thread that fills the queue.
        self._thread = _Thread(target=self._fill_queue, daemon=True)
        self._thread.start()

    def _generate_batch(self) -> BatchRandoms:
        """Generate a single batch of random numbers."""
        return BatchRandoms(
            rotation=self._rng.uniform(0, 1, size=(self._batch_size, 3)).astype(
                _np.float32
            ),
            position=self._rng.normal(0, 1, size=(self._batch_size, 3)).astype(
                _np.float32
            ),
            radius=self._rng.uniform(0, 1, size=self._batch_size).astype(_np.float32),
            acceptance=self._rng.uniform(0, 1, size=self._batch_size).astype(
                _np.float32
            ),
        )

    def _fill_queue(self) -> None:
        """Background thread that continuously fills the queue with batches."""
        while not self._shutdown:
            batch = self._generate_batch()
            self._queue.put(batch)  # Blocks if queue is full.

    def get_batch_randoms(self) -> BatchRandoms:
        """
        Get all random numbers needed for a single batch.

        Returns
        -------

        BatchRandoms
            Container with rotation, position, radius, and acceptance arrays.
        """
        return self._queue.get()

    def shutdown(self) -> None:
        """Stop the background thread. Call when done using the RNG manager."""
        self._shutdown = True
        # Drain one item to unblock the thread if it's waiting on put().
        try:
            self._queue.get_nowait()
        except Exception:
            pass
        self._thread.join(timeout=1.0)
