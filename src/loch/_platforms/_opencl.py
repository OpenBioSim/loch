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
OpenCL platform backend implementation.
"""

import io as _io
import sys as _sys
import warnings as _warnings
from typing import Any as _Any, Callable as _Callable, Dict as _Dict

import numpy as _np
import pyopencl as _cl
import pyopencl.array as _cl_array

from .._kernels import code as _kernel_code
from ._base import PlatformBackend as _PlatformBackend

# Module-level kernel compilation cache. Keyed on
# (device_index, compiler_optimisations). Stores compiled program binaries.
_kernel_cache = {}


class OpenCLPlatform(_PlatformBackend):
    """
    OpenCL platform backend using PyOpenCL.

    This backend wraps PyOpenCL functionality to provide GPU-accelerated
    GCMC sampling on various GPU vendors (Intel, AMD, NVIDIA).
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
        compiler_optimisations=True,
    ):
        """
        Initialize the OpenCL platform backend.

        Parameters
        ----------

        device : int
            The OpenCL device index to use.

        num_points : int
            Number of atoms per water molecule (typically 3).

        num_batch : int
            Number of parallel GCMC trials per batch.

        num_waters : int
            Number of ghost water molecules.

        num_atoms : int
            Total number of atoms in the system.

        num_threads : int
            Work-group size (threads per work-group).

        nvcc : str, optional
            Ignored for OpenCL (included for API compatibility).

        compiler_optimisations : bool, optional
            Enable compiler optimisations for faster math operations.
            When True, passes -cl-mad-enable and -cl-no-signed-zeros to the compiler.
            Default: True (matches OpenMM defaults).
        """
        # Get platforms and devices
        platforms = _cl.get_platforms()
        devices = []
        for platform in platforms:
            devices.extend(platform.get_devices(device_type=_cl.device_type.GPU))

        if not devices:
            raise RuntimeError("No OpenCL GPU devices found")

        # Validate device index
        if device is not None:
            if not isinstance(device, int):
                raise ValueError("'device' must be of type 'int'")
            if device < 0 or device >= len(devices):
                raise ValueError(f"'device' must be between 0 and {len(devices) - 1}")
            self._device_index = device
        else:
            self._device_index = 0
        self._device = devices[self._device_index]

        # Create context and command queue
        self._context = _cl.Context([self._device])
        self._queue = _cl.CommandQueue(self._context)

        # Store parameters
        self._num_points = num_points
        self._num_batch = num_batch
        self._num_waters = num_waters
        self._num_atoms = num_atoms
        self._num_threads = num_threads
        self._compiler_optimisations = compiler_optimisations

    def compile_kernels(self) -> _Dict[str, _Callable]:
        """
        Compile OpenCL kernels and return callable functions.

        Uses a module-level cache so that only the first sampler on a given
        device pays the compilation cost.

        Returns
        -------

        dict
            Dictionary mapping kernel names to callable kernel functions.
        """
        cache_key = (self._device_index, self._compiler_optimisations)

        # Build compiler options
        build_options = []
        if self._compiler_optimisations:
            build_options.extend(["-cl-mad-enable", "-cl-no-signed-zeros"])

        if cache_key in _kernel_cache:
            cached_binary = _kernel_cache[cache_key]

            # Create program from cached binary.
            stderr_capture = _io.StringIO()
            old_stderr = _sys.stderr
            try:
                _sys.stderr = stderr_capture
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    program = _cl.Program(
                        self._context, [self._device], [cached_binary]
                    )
                    program.build(options=build_options)
            except _cl.RuntimeError as e:
                stderr_output = stderr_capture.getvalue().strip()
                error_msg = f"OpenCL kernel build from cached binary failed: {e}"
                if stderr_output:
                    error_msg += f"\n{stderr_output}"
                raise RuntimeError(error_msg)
            finally:
                _sys.stderr = old_stderr

            self._compiler_log = ""
            self._cache_hit = True
        else:
            # Compile program from source, suppressing stderr and warnings.
            stderr_capture = _io.StringIO()
            old_stderr = _sys.stderr
            try:
                _sys.stderr = stderr_capture
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    program = _cl.Program(self._context, _kernel_code).build(
                        options=build_options
                    )
            except _cl.RuntimeError as e:
                stderr_output = stderr_capture.getvalue().strip()
                error_msg = f"OpenCL kernel compilation failed: {e}"
                if stderr_output:
                    error_msg += f"\n{stderr_output}"
                raise RuntimeError(error_msg)
            finally:
                _sys.stderr = old_stderr

            # Capture the compiler log (including any warnings).
            self._compiler_log = program.get_build_info(
                self._device, _cl.program_build_info.LOG
            ).strip()

            self._cache_hit = False

            # Cache the compiled binary.
            _kernel_cache[cache_key] = program.get_info(_cl.program_info.BINARIES)[0]

        # Create kernel wrappers that match PyCUDA calling convention.
        # OpenCL kernels need (queue, global_size, local_size, *args)
        # We wrap them to match CUDA's (args..., block, grid) signature.
        def make_kernel_wrapper(kernel):
            def wrapper(*args, **kwargs):
                block = kwargs.get("block", (self._num_threads, 1, 1))
                grid = kwargs.get("grid", (1, 1, 1))

                global_size = tuple(b * g for b, g in zip(block, grid))
                local_size = block

                processed_args = []
                for arg in args:
                    if isinstance(arg, _cl_array.Array):
                        processed_args.append(arg.data)
                    else:
                        processed_args.append(arg)

                kernel(self._queue, global_size, local_size, *processed_args)
                self._queue.finish()

            return wrapper

        kernels = {
            "update_water": make_kernel_wrapper(program.updateWater),
            "deletion": make_kernel_wrapper(program.findDeletionCandidates),
            "water": make_kernel_wrapper(program.generateWater),
            "energy": make_kernel_wrapper(program.computeEnergy),
            "acceptance": make_kernel_wrapper(program.checkAcceptance),
        }

        return kernels

    @staticmethod
    def clear_cache():
        """Clear the kernel compilation cache."""
        _kernel_cache.clear()

    def to_gpu(self, array: _np.ndarray) -> _Any:
        """
        Transfer a NumPy array to GPU memory.

        Parameters
        ----------

        array : numpy.ndarray
            Array to transfer to GPU.

        Returns
        -------

        pyopencl.array.Array
            GPU array containing the data.
        """
        return _cl_array.to_device(self._queue, array)

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

        pyopencl.array.Array
            Allocated GPU array.
        """
        return _cl_array.empty(self._queue, shape, dtype)

    def from_gpu(self, buffer: _Any) -> _np.ndarray:
        """
        Transfer data from GPU memory to host NumPy array.

        Parameters
        ----------

        buffer : pyopencl.array.Array
            GPU array to transfer from.

        Returns
        -------

        numpy.ndarray
            Array containing the data from GPU.
        """
        return buffer.get()

    def cleanup(self):
        """
        Clean up OpenCL resources.
        """
        self._queue = None
        self._context = None

    @property
    def platform_name(self) -> str:
        """
        Get the name of the platform backend.

        Returns
        -------

        str
            Platform name ('opencl').
        """
        return "opencl"
