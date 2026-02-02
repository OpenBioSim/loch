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
Abstract base class for GPU platform backends.
"""

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from typing import Any as _Any, Callable as _Callable, Dict as _Dict

import numpy as _np


class PlatformBackend(_ABC):
    """
    Abstract base class for GPU platform backends.

    This class defines the interface that all GPU platform implementations
    (CUDA, OpenCL, etc.) must implement. It abstracts platform-specific
    details like kernel compilation, memory management, and context handling.
    """

    @_abstractmethod
    def __init__(
        self,
        device,
        num_points,
        num_batch,
        num_waters,
        num_atoms,
        num_threads,
        nvcc=None,
        compiler_optimisations=True,
    ):
        """
        Initialize the platform backend.

        Parameters
        ----------
        device : int
            The GPU device index to use.

        num_points : int
            Number of atoms per water molecule (typically 3).

        num_batch : int
            Number of parallel GCMC trials per batch.

        num_waters : int
            Number of ghost water molecules.

        num_atoms : int
            Total number of atoms in the system.

        num_threads : int
            Number of threads per block (CUDA) or work-group size (OpenCL).

        nvcc : str, optional
            Path to NVCC compiler (CUDA only). If None, uses default.

        compiler_optimisations : bool, optional
            Enable compiler optimisations for faster math operations.
            Default: True (matches OpenMM defaults).
        """
        pass

    @_abstractmethod
    def compile_kernels(self) -> _Dict[str, _Callable]:
        """
        Compile GPU kernels and return callable functions.

        Returns
        -------
        dict
            Dictionary mapping kernel names to callable kernel functions.
            Expected keys: 'update_water', 'deletion', 'water', 'energy',
            'acceptance'.
        """
        pass

    @_abstractmethod
    def to_gpu(self, array: _np.ndarray) -> _Any:
        """
        Transfer a NumPy array to GPU memory.

        Parameters
        ----------
        array : numpy.ndarray
            Array to transfer to GPU.

        Returns
        -------
        GPU buffer
            Platform-specific GPU buffer object.
        """
        pass

    @_abstractmethod
    def empty(self, shape, dtype) -> _Any:
        """
        Allocate an empty GPU buffer.

        Parameters
        ----------
        shape : tuple
            Shape of the array to allocate.

        dtype : numpy.dtype
            Data type of the array.

        Returns
        -------
        GPU buffer
            Platform-specific GPU buffer object.
        """
        pass

    @_abstractmethod
    def from_gpu(self, buffer: _Any) -> _np.ndarray:
        """
        Transfer data from GPU memory to host NumPy array.

        Parameters
        ----------
        buffer : GPU buffer
            Platform-specific GPU buffer to transfer from.

        Returns
        -------
        numpy.ndarray
            Array containing the data from GPU.
        """
        pass

    @_abstractmethod
    def cleanup(self):
        """
        Clean up GPU resources and release context.

        This method should release all GPU memory and context references.
        It is called during shutdown to ensure proper resource cleanup.
        """
        pass

    @property
    @_abstractmethod
    def platform_name(self) -> str:
        """
        Get the name of the platform backend.

        Returns
        -------
        str
            Platform name ('cuda' or 'opencl').
        """
        pass

    @property
    def compiler_log(self) -> str:
        """
        Get the kernel compiler log including any warnings.

        Returns
        -------
        str
            Compiler log output, or empty string if no warnings/messages.
        """
        return getattr(self, "_compiler_log", "")
