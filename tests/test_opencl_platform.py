from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if PyOpenCL is not installed.
pytest.importorskip("pyopencl")

_CPU = 2
_GPU = 4


def _mock_platform(devices):
    """A mock OpenCL platform with (name, type) devices, filtered by type."""
    platform = MagicMock()
    platform.get_devices.side_effect = lambda device_type: [
        name for name, kind in devices if kind & device_type
    ]
    return platform


def _make_backend(mock_cl, device, platform_index, platforms=None):
    """Instantiate OpenCLPlatform with a mocked PyOpenCL module."""
    from loch._platforms._opencl import OpenCLPlatform

    if platforms is None:
        platforms = [
            _mock_platform([("a0", _GPU), ("a1", _GPU)]),
            _mock_platform([("b_cpu", _CPU), ("b0", _GPU)]),
        ]

    mock_cl.device_type.CPU = _CPU
    mock_cl.device_type.GPU = _GPU
    mock_cl.get_platforms.return_value = platforms

    return OpenCLPlatform(
        device=device,
        num_points=3,
        num_batch=10,
        num_waters=5,
        num_atoms=100,
        num_threads=32,
        platform_index=platform_index,
    )


class TestOpenCLPlatformIndex:
    """Tests for OpenCLPlatform device selection by platform index."""

    def test_all_platforms_when_none(self):
        """GPU devices from all platforms are indexed when platform_index is None."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            backend = _make_backend(mock_cl, device=2, platform_index=None)
            assert backend._device == "b0"

    def test_selected_platform(self):
        """Devices are indexed within the chosen platform."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            backend = _make_backend(mock_cl, device=1, platform_index=0)
            assert backend._device == "a1"

    def test_cpu_devices_indexed_as_openmm(self):
        """CPU devices count towards the index, matching OpenMM's DeviceIndex."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            backend = _make_backend(mock_cl, device=1, platform_index=1)
            assert backend._device == "b0"

    def test_device_out_of_range_for_platform(self):
        """A device index beyond the chosen platform's devices raises."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            with pytest.raises(ValueError, match="'device'"):
                _make_backend(mock_cl, device=2, platform_index=1)

    def test_platform_index_out_of_range(self):
        """An invalid platform index raises."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            with pytest.raises(ValueError, match="'platform_index'"):
                _make_backend(mock_cl, device=0, platform_index=2)

    def test_platform_without_devices(self):
        """A platform with no devices raises a clear error."""
        platforms = [_mock_platform([])]
        with patch("loch._platforms._opencl._cl") as mock_cl:
            with pytest.raises(RuntimeError, match="platform 0"):
                _make_backend(mock_cl, device=0, platform_index=0, platforms=platforms)

    def test_sampler_forwards_platform_index(self):
        """create_backend passes opencl_platform_index to OpenCLPlatform."""
        from loch._platforms import create_backend

        with patch("loch._platforms._opencl.OpenCLPlatform") as mock_backend:
            create_backend(
                platform="opencl",
                device=0,
                num_points=3,
                num_batch=10,
                num_waters=5,
                num_atoms=100,
                num_threads=32,
                opencl_platform_index=1,
            )
            assert mock_backend.call_args.args[-1] == 1
