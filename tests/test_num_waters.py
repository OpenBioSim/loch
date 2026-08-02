import pytest

from loch import GCMCSampler


def make_sampler(reference="resname LIG", N=0, N_region=None, openmm_context=None):
    """
    Create a sampler with only the attributes num_waters() uses, so that the
    counting logic can be tested without a system or a GPU.
    """
    sampler = object.__new__(GCMCSampler)
    sampler._reference = reference
    sampler._N = N
    sampler._N_region = N_region
    sampler._openmm_context = openmm_context
    sampler._is_bulk = False
    return sampler


def test_num_waters_without_a_region():
    """
    Without a GCMC region every move samples the whole box, so the count that
    move() maintains is already the answer. Counting a region would need a
    reference to take a sphere centre from, which does not exist in this case.
    """
    sampler = make_sampler(reference=None, N=7)

    assert sampler.num_waters() == 7

    # Passing a context must not send it down the recount path either, which
    # would dereference the reference indices that were never set.
    assert sampler.num_waters(context=object()) == 7


def test_num_waters_is_seeded_before_the_first_move():
    """
    self._N is set as each move runs, so without a region it would report zero
    until the first one had. A cycle can complete without any GCMC moves when
    'gcmc_frequency' is coarser than the checkpoint interval, so seeding it
    during setup is what stops a checkpoint logging zero waters.
    """
    import numpy as np

    from loch import GCMCSampler

    sampler = object.__new__(GCMCSampler)

    # Four waters, of which the last two are ghosts.
    sampler._water_state = np.array([1, 1, 0, 0], dtype=np.int32)
    sampler._invalidate_water_caches()

    # The seeding that _initialise_gpu_memory() performs.
    sampler._N = len(sampler._non_ghost_waters_cache)

    sampler._reference = None
    assert sampler.num_waters() == 2


def test_num_waters_reports_the_stored_region_count():
    """With a region and nothing to count from, the stored count is returned."""
    sampler = make_sampler(N=99, N_region=4)

    assert sampler.num_waters() == 4


def test_num_waters_refuses_a_whole_box_count():
    """
    A bulk move leaves self._N counting the whole box, so it cannot answer for
    the region. With no context to recount from, that must raise rather than
    report the box count as though it were the region count.
    """
    sampler = make_sampler(N=99, N_region=None)

    with pytest.raises(RuntimeError, match="OpenMM context is not set"):
        sampler.num_waters()
