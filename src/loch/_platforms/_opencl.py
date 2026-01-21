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
            self._device = devices[device]
        else:
            self._device = devices[0]

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

        Returns
        -------
        dict
            Dictionary mapping kernel names to callable kernel functions.
        """
        # Substitute template parameters
        kernel_source = _kernel_code % {
            "NUM_POINTS": self._num_points,
            "NUM_BATCH": self._num_batch,
            "NUM_WATERS": self._num_waters,
            "NUM_ATOMS": self._num_atoms,
        }

        # Build compiler options
        build_options = []
        if self._compiler_optimisations:
            build_options.extend(["-cl-mad-enable", "-cl-no-signed-zeros"])

        # Compile program, suppressing stderr and warnings but capturing for errors.
        stderr_capture = _io.StringIO()
        old_stderr = _sys.stderr
        try:
            _sys.stderr = stderr_capture
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                program = _cl.Program(self._context, kernel_source).build(
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

        # Create kernel wrappers that match PyCUDA calling convention
        # OpenCL kernels need (queue, global_size, local_size, *args)
        # We'll wrap them to match CUDA's (args..., block, grid) signature
        def make_kernel_wrapper(kernel):
            def wrapper(*args, **kwargs):
                # Extract block and grid from kwargs
                block = kwargs.get("block", (self._num_threads, 1, 1))
                grid = kwargs.get("grid", (1, 1, 1))

                # Calculate global work size
                global_size = tuple(b * g for b, g in zip(block, grid))
                local_size = block

                # Convert PyOpenCL array objects to their .data buffers
                processed_args = []
                for arg in args:
                    if isinstance(arg, _cl_array.Array):
                        processed_args.append(arg.data)
                    else:
                        processed_args.append(arg)

                # Execute kernel
                kernel(self._queue, global_size, local_size, *processed_args)
                self._queue.finish()

            return wrapper

        # Extract and wrap kernel functions
        kernels = {
            "cell": make_kernel_wrapper(program.setCellMatrix),
            "rf": make_kernel_wrapper(program.setReactionField),
            "softcore": make_kernel_wrapper(program.setSoftCore),
            "atom_properties": make_kernel_wrapper(program.setAtomProperties),
            "atom_positions": make_kernel_wrapper(program.setAtomPositions),
            "water_properties": make_kernel_wrapper(program.setWaterProperties),
            "update_water": make_kernel_wrapper(program.updateWater),
            "deletion": make_kernel_wrapper(program.findDeletionCandidates),
            "water": make_kernel_wrapper(program.generateWater),
            "energy": make_kernel_wrapper(program.computeEnergy),
            "acceptance": make_kernel_wrapper(program.checkAcceptance),
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

    def push_context(self):
        """
        Push context (no-op for OpenCL).

        OpenCL doesn't use context stacking like CUDA, so this method
        does nothing. It's provided for API compatibility.
        """
        pass

    def pop_context(self):
        """
        Pop context (no-op for OpenCL).

        OpenCL doesn't use context stacking like CUDA, so this method
        does nothing. It's provided for API compatibility.
        """
        pass

    def cleanup(self):
        """
        Clean up OpenCL resources.
        """
        # OpenCL resources are automatically released when objects are deleted
        # No explicit cleanup needed, but we'll clear references
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
