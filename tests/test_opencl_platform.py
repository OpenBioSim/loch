from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if PyOpenCL is not installed.
pytest.importorskip("pyopencl")


def _mock_platforms():
    """Two mock OpenCL platforms with two and one GPU devices."""
    platforms = []
    for devices in (["a0", "a1"], ["b0"]):
        platform = MagicMock()
        platform.get_devices.return_value = devices
        platforms.append(platform)
    return platforms


def _make_backend(mock_cl, device, platform_index):
    """Instantiate OpenCLPlatform with a mocked PyOpenCL module."""
    from loch._platforms._opencl import OpenCLPlatform

    mock_cl.get_platforms.return_value = _mock_platforms()

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
        """Devices from all platforms are indexed when platform_index is None."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            backend = _make_backend(mock_cl, device=2, platform_index=None)
            assert backend._device == "b0"

    def test_selected_platform(self):
        """Devices are indexed within the chosen platform."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            backend = _make_backend(mock_cl, device=0, platform_index=1)
            assert backend._device == "b0"

    def test_device_out_of_range_for_platform(self):
        """A device index beyond the chosen platform's devices raises."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            with pytest.raises(ValueError, match="'device'"):
                _make_backend(mock_cl, device=1, platform_index=1)

    def test_platform_index_out_of_range(self):
        """An invalid platform index raises."""
        with patch("loch._platforms._opencl._cl") as mock_cl:
            with pytest.raises(ValueError, match="'platform_index'"):
                _make_backend(mock_cl, device=0, platform_index=2)
