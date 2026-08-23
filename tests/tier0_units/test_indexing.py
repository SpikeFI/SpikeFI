"""Tier 0 — index arithmetic: FaultSite.unroll's 3-/4-tuple and full_5D
forms against a known-valued tensor, and Fault.unroll's canonical,
insertion-order-independent site ordering.
"""


import pytest
import torch

from spikefi.fault import Fault, FaultSite
from spikefi.models import DeadNeuron, DeadSynapse


@pytest.mark.neuron
def test_unroll_neuron_site_returns_position_verbatim() -> None:
    """A 3-tuple (neuron) site's unroll() is its own position, unchanged."""
    site = FaultSite('SF1', (1, 0, 0))
    assert site.unroll() == (1, 0, 0)


@pytest.mark.synapse
def test_unroll_synapse_site_returns_position_verbatim() -> None:
    """A 4-tuple (synapse) site's unroll() is its own position, unchanged."""
    site = FaultSite('SF1', (2, 1, 0, 0))
    assert site.unroll() == (2, 1, 0, 0)


@pytest.mark.neuron
def test_unroll_full_5d_neuron_site_selects_batch_and_time() -> None:
    """A neuron site's full_5D form pads to (:, c, h, w, :), selecting the
    whole batch and time axes at that fixed neuron -- checked on the
    returned tuple itself, since torch's implicit trailing-slice padding
    would mask a missing final slice(None) if only indexing were checked."""
    site = FaultSite('SF1', (1, 0, 0))
    assert site.unroll(full_5D=True) == (slice(None), 1, 0, 0, slice(None))

    tensor = torch.arange(2 * 3 * 1 * 1 * 4).reshape(2, 3, 1, 1, 4).float()
    indexed = tensor[site.unroll(full_5D=True)]
    assert torch.equal(indexed, tensor[:, 1, 0, 0, :])


@pytest.mark.synapse
def test_unroll_full_5d_synapse_site_selects_time() -> None:
    """A synapse site's full_5D form only appends the time axis: its
    4-tuple already addresses (out, C, H, W) exactly -- checked on the
    returned tuple itself, since torch's implicit trailing-slice padding
    would mask a missing final slice(None) if only indexing were checked."""
    site = FaultSite('SF1', (2, 1, 0, 0))
    assert site.unroll(full_5D=True) == (2, 1, 0, 0, slice(None))

    tensor = torch.arange(4 * 3 * 1 * 1 * 1).reshape(4, 3, 1, 1, 1).float()
    indexed = tensor[site.unroll(full_5D=True)]
    assert torch.equal(indexed, tensor[2, 1, 0, 0, :])


@pytest.mark.neuron
@pytest.mark.synapse
def test_unroll_site_less() -> None:
    """A site-less FaultSite unrolls to an empty index (position itself),
    and its full_5D form selects only the batch axis."""
    site = FaultSite()
    assert site.unroll() == ()
    assert site.unroll(full_5D=True) == (slice(None),)


@pytest.mark.neuron
def test_fault_unroll_is_insertion_order_independent() -> None:
    """unroll() returns sites in ascending-position order regardless of the
    order they were inserted in -- inserted in an order that Python's own
    set iteration would *not* happen to already produce ascending order in,
    since unroll() must sort explicitly rather than trust set internals."""
    insertion_order = [(2, 0, 0), (0, 0, 0), (1, 0, 0), (3, 0, 0)]
    sites = [FaultSite('SF1', p) for p in insertion_order]
    fault = Fault(DeadNeuron(), sites)

    c_idx, h_idx, w_idx = fault.unroll()
    assert torch.equal(c_idx, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(h_idx, torch.tensor([0, 0, 0, 0]))
    assert torch.equal(w_idx, torch.tensor([0, 0, 0, 0]))


@pytest.mark.synapse
def test_fault_unroll_groups_synapse_sites_by_ascending_position() -> None:
    """unroll() groups per-dimension indices, one tensor per position
    coordinate, in ascending site-position order -- inserted in an order
    that Python's own set iteration would *not* happen to already produce
    ascending order in, since unroll() must sort explicitly rather than
    trust set internals."""
    insertion_order = [(1, 0, 0, 0), (0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]
    sites = [FaultSite('SF1', p) for p in insertion_order]
    fault = Fault(DeadSynapse(), sites)

    out_idx, c_idx, h_idx, w_idx = fault.unroll()
    assert torch.equal(out_idx, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(c_idx, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(h_idx, torch.tensor([0, 0, 0, 0]))
    assert torch.equal(w_idx, torch.tensor([0, 0, 0, 0]))
