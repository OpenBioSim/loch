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
CUDA platform backend implementation.
"""

import atexit as _atexit
import io as _io
import sys as _sys
from typing import Any as _Any, Callable as _Callable, Dict as _Dict

import numpy as _np
import pycuda.driver as _cuda
import pycuda.gpuarray as _gpuarray
from pycuda.compiler import SourceModule as _SourceModule

from .._kernels import code as _kernel_code
from ._base import PlatformBackend as _PlatformBackend


class CUDAPlatform(_PlatformBackend):
    """
    CUDA platform backend using PyCUDA.

    This backend wraps PyCUDA functionality to provide GPU-accelerated
    GCMC sampling on NVIDIA GPUs.
    """

    def __init__(
        self,
        device,
        num_points,
        num_batch,
        num_waters,
        num_atoms,
        num_threads,
        nvcc=None,
    ):
        """
        Initialize the CUDA platform backend.

        Parameters
        ----------
        device : int
            The CUDA device index to use.

        num_points : int
            Number of atoms per water molecule (typically 3).

        num_batch : int
            Number of parallel GCMC trials per batch.

        num_waters : int
            Number of ghost water molecules.

        num_atoms : int
            Total number of atoms in the system.

        num_threads : int
            Number of threads per block.

        nvcc : str, optional
            Path to NVCC compiler. If None, uses default from PATH.
        """
        from pycuda.tools import make_default_context

        # Initialize CUDA driver
        _cuda.init()

        # Validate and set device
        if device is not None:
            if not isinstance(device, int):
                raise ValueError("'device' must be of type 'int'")
            if device < 0 or device >= _cuda.Device.count():
                raise ValueError(
                    f"'device' must be between 0 and {_cuda.Device.count() - 1}"
                )
            self._pycuda_context = _cuda.Device(device).make_context()
        else:
            self._pycuda_context = make_default_context()

        self._device = self._pycuda_context.get_device()

        # Store parameters
        self._num_points = num_points
        self._num_batch = num_batch
        self._num_waters = num_waters
        self._num_atoms = num_atoms
        self._num_threads = num_threads
        self._nvcc = nvcc

        # Register cleanup
        _atexit.register(self._cleanup_wrapper)

    def compile_kernels(self) -> _Dict[str, _Callable]:
        """
        Compile CUDA kernels and return callable functions.

        Returns
        -------
        dict
            Dictionary mapping kernel names to callable kernel functions.
        """
        # Compile kernel module with template substitution.
        # Suppress stderr but capture it for error reporting.
        stderr_capture = _io.StringIO()
        old_stderr = _sys.stderr
        try:
            _sys.stderr = stderr_capture
            mod = _SourceModule(
                _kernel_code
                % {
                    "NUM_POINTS": self._num_points,
                    "NUM_BATCH": self._num_batch,
                    "NUM_WATERS": self._num_waters,
                    "NUM_ATOMS": self._num_atoms,
                },
                no_extern_c=True,
                nvcc=self._nvcc,
            )
        except Exception as e:
            stderr_output = stderr_capture.getvalue().strip()
            error_msg = f"CUDA kernel compilation failed: {e}"
            if stderr_output:
                error_msg += f"\n{stderr_output}"
            raise RuntimeError(error_msg)
        finally:
            _sys.stderr = old_stderr

        # Store any compiler warnings.
        self._compiler_log = stderr_capture.getvalue().strip()

        # Extract kernel functions
        kernels = {
            "cell": mod.get_function("setCellMatrix"),
            "rf": mod.get_function("setReactionField"),
            "softcore": mod.get_function("setSoftCore"),
            "atom_properties": mod.get_function("setAtomProperties"),
            "atom_positions": mod.get_function("setAtomPositions"),
            "water_properties": mod.get_function("setWaterProperties"),
            "update_water": mod.get_function("updateWater"),
            "deletion": mod.get_function("findDeletionCandidates"),
            "water": mod.get_function("generateWater"),
            "energy": mod.get_function("computeEnergy"),
            "acceptance": mod.get_function("checkAcceptance"),
        }

        return kernels

    def to_gpu(self, array: _np.ndarray) -> _Any:
        """
        Transfer a NumPy array to GPU memory.

        Parameters
        ----------
        array : numpy.ndarray
            Array to transfer to GPU.

        Returns
        -------
        pycuda.gpuarray.GPUArray
            GPU array containing the data.
        """
        return _gpuarray.to_gpu(array)

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
        pycuda.gpuarray.GPUArray
            Allocated GPU array.
        """
        return _gpuarray.empty(shape, dtype)

    def from_gpu(self, buffer: _Any) -> _np.ndarray:
        """
        Transfer data from GPU memory to host NumPy array.

        Parameters
        ----------
        buffer : pycuda.gpuarray.GPUArray
            GPU array to transfer from.

        Returns
        -------
        numpy.ndarray
            Array containing the data from GPU.
        """
        return buffer.get()

    def push_context(self):
        """
        Push the CUDA context onto the context stack.
        """
        self._pycuda_context.push()

    def pop_context(self):
        """
        Pop the CUDA context from the context stack.
        """
        self._pycuda_context.pop()

    def cleanup(self):
        """
        Clean up CUDA resources and detach context.
        """
        try:
            self.pop_context()
        except:
            pass
        self._pycuda_context.detach()
        self._pycuda_context = None

    def _cleanup_wrapper(self):
        """
        Wrapper for cleanup to handle atexit registration.
        """
        try:
            if self._pycuda_context is not None:
                self.cleanup()
        except:
            pass

    @property
    def platform_name(self) -> str:
        """
        Get the name of the platform backend.

        Returns
        -------
        str
            Platform name ('cuda').
        """
        return "cuda"
