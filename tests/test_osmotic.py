import os

import openmm
import pytest

from loch import GCMCSampler


@pytest.mark.skipif(
    "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Requires CUDA enabled GPU.",
)
@pytest.mark.parametrize("platform", ["cuda", "opencl"])
def test_osmotic_ensemble(water_box, platform):
    """
    When pressure is set, move() updates box parameters and Adams values
    from the current OpenMM context state after each call.

    Two moves are performed with the box scaled between them. After the
    second move, _v_nm3, _exp_B_bulk, and _exp_minus_B_bulk must all
    reflect the new volume.
    """
    mols, _ = water_box

    sampler = GCMCSampler(
        mols,
        cutoff_type="rf",
        cutoff="10 A",
        pressure="1 atm",
        ghost_file=None,
        log_file=None,
        test=True,
        platform=platform,
        seed=42,
    )

    # NPT dynamics so the context carries a barostat (bypasses the muVT guard).
    d = sampler.system().dynamics(
        cutoff_type="rf",
        cutoff="10 A",
        temperature="298 K",
        pressure="1 atm",
        constraint="h_bonds",
        timestep="2 fs",
        platform=platform,
    )
    context = d.context()

    # First move: record initial box-dependent values.
    sampler.move(context)
    v_nm3_before = sampler._v_nm3
    exp_B_bulk_before = sampler._exp_B_bulk
    exp_minus_B_bulk_before = sampler._exp_minus_B_bulk

    # Scale the box uniformly to simulate a barostat volume move.
    scale = 1.1
    box = context.getState().getPeriodicBoxVectors(asNumpy=True)
    box_nm = box.value_in_unit(openmm.unit.nanometer)
    context.setPeriodicBoxVectors(
        openmm.Vec3(*box_nm[0] * scale) * openmm.unit.nanometer,
        openmm.Vec3(*box_nm[1] * scale) * openmm.unit.nanometer,
        openmm.Vec3(*box_nm[2] * scale) * openmm.unit.nanometer,
    )

    # Second move: box parameters must now reflect the scaled volume.
    sampler.move(context)

    assert sampler._v_nm3 == pytest.approx(v_nm3_before * scale**3, rel=1e-5)
    assert sampler._exp_B_bulk != exp_B_bulk_before
    assert sampler._exp_minus_B_bulk != exp_minus_B_bulk_before
