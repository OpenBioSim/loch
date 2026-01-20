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
GPU platform abstraction for GCMC sampling.

This module provides a factory interface for creating GPU platform backends
that abstract the differences between CUDA and OpenCL.
"""

from typing import Optional as _Optional

from ._base import PlatformBackend as _PlatformBackend


def detect_platform() -> str:
    """
    Auto-detect available GPU platform.

    Tries CUDA first (preferred), then falls back to OpenCL if CUDA is not available.

    Returns
    -------
    str
        Platform name ('cuda' or 'opencl').

    Raises
    ------
    RuntimeError
        If no CUDA or OpenCL GPU devices are found.
    """
    # Try CUDA first
    try:
        import pycuda.driver as cuda

        cuda.init()
        if cuda.Device.count() > 0:
            return "cuda"
    except Exception:
        pass

    # Fall back to OpenCL
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                return "opencl"
    except Exception:
        pass

    # No GPU found
    raise RuntimeError(
        "No CUDA or OpenCL GPU devices found. "
        "Please ensure PyCUDA or PyOpenCL is installed and a GPU is available."
    )


def create_backend(
    platform: str,
    device: _Optional[int],
    num_points: int,
    num_batch: int,
    num_waters: int,
    num_atoms: int,
    num_threads: int,
    nvcc: _Optional[str] = None,
    compiler_optimisations: bool = True,
) -> _PlatformBackend:
    """
    Create a GPU platform backend instance.

    Parameters
    ----------
    platform : str
        Platform to use ('auto', 'cuda', or 'opencl').
        If 'auto', CUDA will be tried first, falling back to OpenCL.

    device : int
        GPU device index.

    num_points : int
        Number of atoms per water molecule (typically 3).

    num_batch : int
        Batch size for parallel trials.

    num_waters : int
        Number of ghost water molecules.

    num_atoms : int
        Total number of atoms in the system.

    num_threads : int
        Number of threads per block/work-group.

    nvcc : str, optional
        Path to NVCC compiler (CUDA only).

    compiler_optimisations : bool, optional
        Enable compiler optimisations for faster math operations.
        Default: True (matches OpenMM defaults).

    Returns
    -------
    PlatformBackend
        Initialized platform backend (CUDAPlatform or OpenCLPlatform).
    """
    # Detect platform if auto
    if platform == "auto":
        platform = detect_platform()

    # Create appropriate backend
    if platform == "cuda":
        from ._cuda import CUDAPlatform as _CUDAPlatform

        return _CUDAPlatform(
            device,
            num_points,
            num_batch,
            num_waters,
            num_atoms,
            num_threads,
            nvcc,
            compiler_optimisations,
        )
    elif platform == "opencl":
        from ._opencl import OpenCLPlatform as _OpenCLPlatform

        return _OpenCLPlatform(
            device,
            num_points,
            num_batch,
            num_waters,
            num_atoms,
            num_threads,
            nvcc,
            compiler_optimisations,
        )
    else:
        raise ValueError(
            f"Unknown platform '{platform}'. Must be 'auto', 'cuda', or 'opencl'."
        )
