import pytest

from loch import GCMCSampler


def make_sampler(lambda_value=0.0, lambda_values=None, is_fep=True):
    """
    Create a sampler with only the attributes the statistics use, so that the
    bookkeeping can be tested without a system or a GPU.
    """
    sampler = object.__new__(GCMCSampler)
    sampler._lambda_value = lambda_value
    sampler._rest2_scale = 1.0
    sampler._is_fep = is_fep
    sampler._lambda_values = lambda_values
    sampler._stats = {}
    sampler._zero_stats()
    return sampler


def do_moves(sampler, num_moves):
    """Pretend that a number of moves were performed and all were accepted."""
    sampler._num_moves += num_moves
    sampler._num_accepted += num_moves


def switch(sampler, lambda_value):
    """Switch lambda, as set_lambda() does."""
    sampler._switch_stats(lambda_value)
    sampler._lambda_value = lambda_value


class TestStatsKey:
    """Tests for the key used to store statistics."""

    @pytest.mark.parametrize(
        "lambda_value, expected",
        [
            (0.0, "0.00000"),
            (1, "1.00000"),
            (0.33333, "0.33333"),
            (1.0 / 3.0, "0.33333"),
        ],
    )
    def test_key_format(self, lambda_value, expected):
        """Keys are formatted to five decimal places, as SOMD2 does."""
        assert GCMCSampler.stats_key(lambda_value) == expected

    def test_key_is_stable_across_representations(self):
        """Values that agree to five decimal places share a key."""
        assert GCMCSampler.stats_key(0.1 + 0.2) == GCMCSampler.stats_key(0.3)


class TestPerLambdaStats:
    """Tests for statistics accumulated per lambda value."""

    def test_isolated_between_lambdas(self):
        """Moves at one lambda must not be counted at another."""
        sampler = make_sampler(lambda_values=[0.0, 0.5])

        do_moves(sampler, 3)
        switch(sampler, 0.5)

        # The new lambda starts from zero.
        assert sampler._num_moves == 0

        do_moves(sampler, 7)
        switch(sampler, 0.0)

        # Returning restores the original count, not the total.
        assert sampler._num_moves == 3

        stats = sampler.get_stats()
        assert stats["0.00000"]["num_moves"] == 3
        assert stats["0.50000"]["num_moves"] == 7

    def test_accumulates_across_visits(self):
        """Revisiting a lambda continues from where it left off."""
        sampler = make_sampler(lambda_values=[0.0, 0.5])

        do_moves(sampler, 3)
        switch(sampler, 0.5)
        do_moves(sampler, 7)
        switch(sampler, 0.0)
        do_moves(sampler, 2)

        stats = sampler.get_stats()
        assert stats["0.00000"]["num_moves"] == 5
        assert stats["0.50000"]["num_moves"] == 7

    def test_current_lambda_is_reported(self):
        """The lambda in use is included alongside the archived ones."""
        sampler = make_sampler(lambda_values=[0.0, 0.5])
        do_moves(sampler, 4)

        assert sampler.get_stats() == {
            "0.00000": {
                "num_moves": 4,
                "num_accepted": 4,
                "num_insertions": 0,
                "num_deletions": 0,
                "num_accepted_attempts": 0,
            }
        }

    def test_non_alchemical_has_a_single_key(self):
        """A non-alchemical system reports the same shape, with one key."""
        sampler = make_sampler(is_fep=False)
        do_moves(sampler, 6)

        stats = sampler.get_stats()
        assert list(stats) == ["0.00000"]
        assert stats["0.00000"]["num_moves"] == 6

    def test_reset_clears_every_lambda(self):
        """reset() zeroes the current lambda and discards the others."""
        sampler = make_sampler(lambda_values=[0.0, 0.5])
        do_moves(sampler, 3)
        switch(sampler, 0.5)
        do_moves(sampler, 7)

        sampler.reset()

        assert sampler.get_stats() == {
            "0.50000": {
                "num_moves": 0,
                "num_accepted": 0,
                "num_insertions": 0,
                "num_deletions": 0,
                "num_accepted_attempts": 0,
            }
        }


class TestRestoreStats:
    """Tests for restoring statistics, e.g. from a checkpoint."""

    def test_round_trip(self):
        """Statistics survive a save and restore."""
        sampler = make_sampler(lambda_values=[0.0, 0.5])
        do_moves(sampler, 3)
        switch(sampler, 0.5)
        do_moves(sampler, 7)
        stats = sampler.get_stats()

        restored = make_sampler(lambda_value=0.5, lambda_values=[0.0, 0.5])
        restored.restore_stats(stats)

        assert restored.get_stats() == stats
        assert restored._num_moves == 7

    def test_unvisited_lambdas_are_ignored(self):
        """
        A sampler keeps only its own lambda values.

        Each sampler can be handed the statistics for a whole simulation. If it
        kept the others, it would report stale values for lambdas it never
        samples, which could overwrite the live ones when merged.
        """
        stats = {
            "0.00000": {
                "num_moves": 5,
                "num_accepted": 5,
                "num_insertions": 0,
                "num_deletions": 0,
                "num_accepted_attempts": 0,
            },
            "1.00000": {
                "num_moves": 9,
                "num_accepted": 9,
                "num_insertions": 0,
                "num_deletions": 0,
                "num_accepted_attempts": 0,
            },
        }

        sampler = make_sampler(lambda_values=[0.0])
        sampler.restore_stats(stats)

        assert list(sampler.get_stats()) == ["0.00000"]

    def test_merge_order_cannot_clobber(self):
        """Merging several samplers is safe regardless of order."""
        first = make_sampler(lambda_value=0.0, lambda_values=[0.0])
        second = make_sampler(lambda_value=1.0, lambda_values=[1.0])

        do_moves(first, 5)
        do_moves(second, 9)

        merged = {}
        merged.update(first.get_stats())
        merged.update(second.get_stats())

        # Restart both from the merged statistics, then advance one of them.
        first = make_sampler(lambda_value=0.0, lambda_values=[0.0])
        second = make_sampler(lambda_value=1.0, lambda_values=[1.0])
        first.restore_stats(merged)
        second.restore_stats(merged)
        do_moves(first, 100)

        for order in ([first, second], [second, first]):
            remerged = {}
            for sampler in order:
                remerged.update(sampler.get_stats())
            assert remerged["0.00000"]["num_moves"] == 105
            assert remerged["1.00000"]["num_moves"] == 9

    def test_missing_lambda_is_zeroed(self):
        """A lambda absent from the statistics starts from zero."""
        sampler = make_sampler(lambda_value=0.5, lambda_values=[0.0, 0.5])
        sampler.restore_stats(
            {
                "0.00000": {
                    "num_moves": 5,
                    "num_accepted": 5,
                    "num_insertions": 0,
                    "num_deletions": 0,
                    "num_accepted_attempts": 0,
                }
            }
        )

        assert sampler._num_moves == 0
        assert sampler.get_stats()["0.00000"]["num_moves"] == 5
