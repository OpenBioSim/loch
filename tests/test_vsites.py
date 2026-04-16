import os

import pytest
import sire as sr

from loch import GCMCSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_vsites_to_mol(mol, n_vs):
    """
    Return a copy of mol with n_vs zero-charge virtual sites all parented to
    atom 0.

    Using atom 0 for all parent references works for any molecule regardless
    of its size. Virtual sites have zero charge so they do not affect energies,
    making it possible to compare sampler results with and without virtual sites.
    """
    n_atoms = mol.num_atoms()

    # LocalCoordinatesSite requires three parent atoms; clamp to valid range so
    # this helper works even for single-atom molecules (e.g. ions).
    p1 = min(1, n_atoms - 1)
    p2 = min(2, n_atoms - 1)

    vsite_dict = {
        str(k): {
            "vs_indices": [0, p1, p2],
            "vs_ows": [1, 0, 0],
            "vs_xs": [1, -1, 0],
            "vs_ys": [0, 1, -1],
            # Offset each vsite slightly so their positions are distinct.
            "vs_local": [(k + 1) * 0.03, 0, 0],
        }
        for k in range(n_vs)
    }

    # All vsites are children of atom 0.
    parents = {str(i): [] for i in range(n_atoms)}
    parents["0"] = list(range(n_vs))

    cursor = mol.cursor()
    cursor.set("n_virtual_sites", n_vs)
    cursor.set("vs_charges", [0.0] * n_vs)
    cursor.set("virtual_sites", vsite_dict)
    cursor.set("parents", parents)
    return cursor.commit()


# ---------------------------------------------------------------------------
# Unit tests for _get_vsite_offsets (no GPU required)
# ---------------------------------------------------------------------------


def test_get_vsite_offsets_no_vsites():
    """
    _get_vsite_offsets on a system with no virtual sites returns all-zero
    offsets and empty per-molecule charge lists.
    """
    mols = sr.load_test_files("bpti.prm7", "bpti.rst7")

    total_vsites, offsets, mol_vsite_charges = GCMCSampler._get_vsite_offsets(mols)

    assert total_vsites == 0
    assert offsets.shape == (mols.num_atoms(),)
    assert (offsets == 0).all()
    assert mol_vsite_charges == {}


def test_get_vsite_offsets_with_vsites():
    """
    Adding N virtual sites to molecule 0 gives a zero offset for every atom
    inside that molecule and an offset of N for all atoms in subsequent
    molecules.
    """
    mols = sr.load_test_files("bpti.prm7", "bpti.rst7")

    n_vs = 2
    n_atoms_mol0 = mols[0].num_atoms()

    mols_with_vs = mols.clone()
    mols_with_vs.update(_add_vsites_to_mol(mols_with_vs[0], n_vs))

    total_vsites, offsets, mol_vsite_charges = GCMCSampler._get_vsite_offsets(
        mols_with_vs
    )

    assert total_vsites == n_vs

    # Atoms in molecule 0 precede no vsite-bearing molecule, so offset is 0.
    assert (offsets[:n_atoms_mol0] == 0).all()

    # All atoms in molecules 1..N follow molecule 0 and are shifted by n_vs.
    assert (offsets[n_atoms_mol0:] == n_vs).all()

    # Only molecule 0 appears in the dict, with n_vs zero charges.
    assert len(mol_vsite_charges) == 1
    assert list(mol_vsite_charges.values())[0] == [0.0] * n_vs


# ---------------------------------------------------------------------------
# Integration tests (GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_vsite_index_offsets(bpti, platform):
    """
    GCMCSampler correctly offsets _water_indices and _reference_indices when
    virtual sites are present on a preceding molecule.

    In BPTI the protein is molecule 0 and all water molecules follow it.
    Adding N vsites to the protein shifts every water's OpenMM particle index
    by N. The reference atoms also live in the protein, so their OpenMM
    indices are not shifted (no vsites precede molecule 0).
    """
    mols, reference = bpti

    common_kwargs = dict(
        reference=reference,
        cutoff_type="rf",
        cutoff="10 A",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
        seed=42,
    )

    # Baseline: no virtual sites.
    baseline = GCMCSampler(mols, **common_kwargs)

    # Add 2 zero-charge virtual sites to molecule 0 (the protein). All
    # water molecules follow the protein, so their OpenMM indices shift by 2.
    n_vs = 2
    mols_with_vs = mols.clone()
    mols_with_vs.update(_add_vsites_to_mol(mols_with_vs[0], n_vs))

    sampler = GCMCSampler(mols_with_vs, **common_kwargs)

    # Total vsite count must match what we added.
    assert sampler._total_vsites == n_vs

    # _num_atoms must include the virtual site particles.
    assert sampler._num_atoms == baseline._num_atoms + n_vs

    # Every water oxygen index is shifted by n_vs (waters follow the protein).
    assert (sampler._water_indices == baseline._water_indices + n_vs).all()

    # Reference atoms are inside molecule 0 (the protein), which has no
    # preceding vsites, so their OpenMM indices are unchanged.
    assert (sampler._reference_indices == baseline._reference_indices).all()
