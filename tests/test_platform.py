from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if PyCUDA is not installed.
pytest.importorskip("pycuda")


def _make_backend(mock_driver):
    """Instantiate CUDAPlatform with a mocked PyCUDA driver."""
    from loch._platforms._cuda import CUDAPlatform

    mock_driver.Device.count.return_value = 1
    mock_context = MagicMock()
    mock_driver.Device.return_value.retain_primary_context.return_value = mock_context

    backend = CUDAPlatform(
        device=0,
        num_points=3,
        num_batch=10,
        num_waters=5,
        num_atoms=100,
        num_threads=32,
    )
    return backend, mock_context


class TestCUDAPushCount:
    """Tests for CUDAPlatform context push-count tracking."""

    def test_initial_push_count(self):
        """Push count starts at 1 after __init__ (one push for the lifetime context)."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, mock_context = _make_backend(mock_driver)
            assert backend._push_count == 1
            mock_context.push.assert_called_once()

    def test_push_increments_count(self):
        """push_context() increments _push_count."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, _ = _make_backend(mock_driver)
            backend.push_context()
            assert backend._push_count == 2
            backend.push_context()
            assert backend._push_count == 3

    def test_pop_decrements_count(self):
        """pop_context() decrements _push_count."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, _ = _make_backend(mock_driver)
            backend.push_context()
            backend.pop_context()
            assert backend._push_count == 1

    def test_cleanup_pops_once_normally(self):
        """cleanup() pops exactly once when no extra pushes are outstanding."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, mock_context = _make_backend(mock_driver)
            backend.cleanup()
            assert mock_context.pop.call_count == 1
            assert backend._push_count == 0
            assert backend._pycuda_context is None

    def test_cleanup_pops_all_outstanding(self):
        """cleanup() pops all outstanding pushes, simulating a crash mid-move."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, mock_context = _make_backend(mock_driver)
            # Simulate two push_context() calls that were never popped (e.g.
            # two GCMC moves crashed before their paired pop_context()).
            backend.push_context()
            backend.push_context()
            assert backend._push_count == 3
            backend.cleanup()
            assert mock_context.pop.call_count == 3
            assert backend._push_count == 0
            assert backend._pycuda_context is None

    def test_cleanup_handles_pop_exception(self):
        """cleanup() continues safely if pop() raises (e.g. stack already empty)."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, mock_context = _make_backend(mock_driver)
            mock_context.pop.side_effect = Exception("context stack is empty")
            backend.cleanup()
            assert backend._pycuda_context is None

    def test_cleanup_idempotent(self):
        """Calling cleanup() a second time is a no-op."""
        with patch("loch._platforms._cuda._cuda") as mock_driver:
            backend, mock_context = _make_backend(mock_driver)
            backend.cleanup()
            pop_count_after_first = mock_context.pop.call_count
            backend.cleanup()
            assert mock_context.pop.call_count == pop_count_after_first
