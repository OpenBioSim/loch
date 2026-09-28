import os

import pytest

from loch import GCMCSampler


def make_sampler(mols, ghost_file, restart=False, overwrite=False):
    return GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        ghost_file=str(ghost_file),
        log_file=None,
        restart=restart,
        overwrite=overwrite,
        test=True,
        platform="cuda",
        seed=42,
    )


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize(
    "restart, overwrite, expected",
    [
        (True, False, "1, 2, 3\n"),
        (True, True, "1, 2, 3\n"),
        (False, True, ""),
    ],
)
def test_existing_ghost_file(water_box, tmp_path, restart, overwrite, expected):
    """
    An existing ghost file is kept on restart, so that it stays aligned with
    the trajectory, and is only cleared when overwriting a fresh run.
    """
    mols, _ = water_box

    ghost_file = tmp_path / "ghosts.txt"
    ghost_file.write_text("1, 2, 3\n")

    make_sampler(mols, ghost_file, restart=restart, overwrite=overwrite)

    assert ghost_file.read_text() == expected


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
def test_existing_ghost_file_raises(water_box, tmp_path):
    """
    An existing ghost file can't be silently overwritten by a fresh run.
    """
    mols, _ = water_box

    ghost_file = tmp_path / "ghosts.txt"
    ghost_file.write_text("1, 2, 3\n")

    with pytest.raises(ValueError, match="ghost_file"):
        make_sampler(mols, ghost_file)

    assert ghost_file.read_text() == "1, 2, 3\n"
