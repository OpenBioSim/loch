import os
import shutil

import pytest


def _get_nvcc():
    """Get nvcc path from environment or PATH."""
    return os.environ.get("PYCUDA_NVCC") or shutil.which("nvcc")


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
class TestOpenCLCompiler:
    """Tests for OpenCL compiler log functionality."""

    def test_compiler_log_returns_string(self):
        """Test that compiler_log returns a string after successful compilation."""
        from loch._platforms._opencl import OpenCLPlatform

        backend = OpenCLPlatform(
            device=0,
            num_points=3,
            num_batch=10,
            num_waters=5,
            num_atoms=100,
            num_threads=32,
        )
        backend.compile_kernels()

        log = backend.compiler_log
        assert isinstance(log, str)

    def test_compiler_log_empty_before_compilation(self):
        """Test that compiler_log is empty before compile_kernels is called."""
        from loch._platforms._opencl import OpenCLPlatform

        backend = OpenCLPlatform(
            device=0,
            num_points=3,
            num_batch=10,
            num_waters=5,
            num_atoms=100,
            num_threads=32,
        )

        # Before compilation, should return empty string
        log = backend.compiler_log
        assert log == ""


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
class TestCUDACompiler:
    """Tests for CUDA compiler log functionality."""

    def test_compiler_log_returns_string(self):
        """Test that compiler_log returns a string after successful compilation."""
        from loch._platforms._cuda import CUDAPlatform

        backend = CUDAPlatform(
            device=0,
            num_points=3,
            num_batch=10,
            num_waters=5,
            num_atoms=100,
            num_threads=32,
            nvcc=_get_nvcc(),
        )
        backend.compile_kernels()

        log = backend.compiler_log
        assert isinstance(log, str)

        backend.cleanup()

    def test_compiler_log_empty_before_compilation(self):
        """Test that compiler_log is empty before compile_kernels is called."""
        from loch._platforms._cuda import CUDAPlatform

        backend = CUDAPlatform(
            device=0,
            num_points=3,
            num_batch=10,
            num_waters=5,
            num_atoms=100,
            num_threads=32,
            nvcc=_get_nvcc(),
        )

        # Before compilation, should return empty string
        log = backend.compiler_log
        assert log == ""

        backend.cleanup()

    def test_compilation_error_raises_exception(self):
        """Test that compilation errors raise RuntimeError with message."""
        import loch._platforms._cuda as cuda_module
        from loch._platforms._cuda import CUDAPlatform

        backend = CUDAPlatform(
            device=0,
            num_points=3,
            num_batch=10,
            num_waters=5,
            num_atoms=100,
            num_threads=32,
            nvcc=_get_nvcc(),
        )

        # Clear the kernel cache so the patched code is actually compiled.
        CUDAPlatform.clear_cache()

        # Patch kernel code directly in the cuda module (not the kernels module,
        # since it's already imported as _kernel_code at module load time).
        original_code = cuda_module._kernel_code
        # Use code with syntax error - will fail on CUDA compiler.
        cuda_module._kernel_code = (
            'extern "C" { __global__ void test( { syntax error here } }'
        )

        try:
            with pytest.raises(RuntimeError) as exc_info:
                backend.compile_kernels()

            assert "CUDA kernel compilation failed" in str(exc_info.value)
        finally:
            cuda_module._kernel_code = original_code
            backend.cleanup()
