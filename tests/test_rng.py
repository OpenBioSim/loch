import numpy as np
import time

from loch._platforms._rng import RNGManager, BatchRandoms


class TestBatchRandoms:
    """Tests for the BatchRandoms dataclass."""

    def test_dataclass_fields(self):
        """Test that BatchRandoms has the expected fields."""
        batch = BatchRandoms(
            rotation=np.zeros((10, 3)),
            position=np.zeros((10, 3)),
            radius=np.zeros(10),
            acceptance=np.zeros(10),
        )
        assert hasattr(batch, "rotation")
        assert hasattr(batch, "position")
        assert hasattr(batch, "radius")
        assert hasattr(batch, "acceptance")


class TestRNGManager:
    """Tests for the RNGManager class."""

    def test_batch_shapes(self):
        """Test that generated batches have correct shapes."""
        batch_size = 64
        rng = RNGManager(batch_size=batch_size, seed=42)

        batch = rng.get_batch_randoms()

        assert batch.rotation.shape == (batch_size, 3)
        assert batch.position.shape == (batch_size, 3)
        assert batch.radius.shape == (batch_size,)
        assert batch.acceptance.shape == (batch_size,)

        rng.shutdown()

    def test_batch_dtypes(self):
        """Test that generated batches have float32 dtype."""
        rng = RNGManager(batch_size=32, seed=42)

        batch = rng.get_batch_randoms()

        assert batch.rotation.dtype == np.float32
        assert batch.position.dtype == np.float32
        assert batch.radius.dtype == np.float32
        assert batch.acceptance.dtype == np.float32

        rng.shutdown()

    def test_uniform_range(self):
        """Test that uniform randoms are in [0, 1) range."""
        rng = RNGManager(batch_size=1000, seed=42)

        # Get several batches to have enough samples.
        for _ in range(5):
            batch = rng.get_batch_randoms()

            assert np.all(batch.rotation >= 0) and np.all(batch.rotation < 1)
            assert np.all(batch.radius >= 0) and np.all(batch.radius < 1)
            assert np.all(batch.acceptance >= 0) and np.all(batch.acceptance < 1)

        rng.shutdown()

    def test_normal_distribution(self):
        """Test that position randoms follow a normal distribution."""
        rng = RNGManager(batch_size=1000, seed=42)

        # Collect samples from multiple batches.
        samples = []
        for _ in range(10):
            batch = rng.get_batch_randoms()
            samples.append(batch.position.flatten())

        all_samples = np.concatenate(samples)

        # Check mean and std are approximately 0 and 1.
        assert np.abs(np.mean(all_samples)) < 0.1
        assert np.abs(np.std(all_samples) - 1.0) < 0.1

        rng.shutdown()

    def test_reproducibility_with_seed(self):
        """Test that the same seed produces the same sequence."""
        rng1 = RNGManager(batch_size=32, seed=12345)
        rng2 = RNGManager(batch_size=32, seed=12345)

        batch1 = rng1.get_batch_randoms()
        batch2 = rng2.get_batch_randoms()

        np.testing.assert_array_equal(batch1.rotation, batch2.rotation)
        np.testing.assert_array_equal(batch1.position, batch2.position)
        np.testing.assert_array_equal(batch1.radius, batch2.radius)
        np.testing.assert_array_equal(batch1.acceptance, batch2.acceptance)

        rng1.shutdown()
        rng2.shutdown()

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different sequences."""
        rng1 = RNGManager(batch_size=32, seed=111)
        rng2 = RNGManager(batch_size=32, seed=222)

        batch1 = rng1.get_batch_randoms()
        batch2 = rng2.get_batch_randoms()

        # At least one array should differ.
        assert not np.array_equal(batch1.rotation, batch2.rotation)

        rng1.shutdown()
        rng2.shutdown()

    def test_multiple_batches_are_different(self):
        """Test that consecutive batches are different."""
        rng = RNGManager(batch_size=32, seed=42)

        batch1 = rng.get_batch_randoms()
        batch2 = rng.get_batch_randoms()

        assert not np.array_equal(batch1.rotation, batch2.rotation)
        assert not np.array_equal(batch1.acceptance, batch2.acceptance)

        rng.shutdown()

    def test_queue_prefilling(self):
        """Test that the background thread pre-fills the queue."""
        rng = RNGManager(batch_size=32, seed=42)

        # Give the background thread time to fill the queue.
        time.sleep(0.2)

        # Getting multiple batches should be fast since they're pre-computed.
        start = time.perf_counter()
        for _ in range(5):
            rng.get_batch_randoms()
        elapsed = time.perf_counter() - start

        # Should be very fast since batches are pre-computed.
        assert elapsed < 0.1

        rng.shutdown()

    def test_shutdown(self):
        """Test that shutdown stops the background thread."""
        rng = RNGManager(batch_size=32, seed=42)

        # Verify thread is running.
        assert rng._thread.is_alive()

        rng.shutdown()

        # Thread should have stopped.
        assert not rng._thread.is_alive()
