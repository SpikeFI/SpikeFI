"""Tier 1 — reference-forward fidelity: campaign.golden's optimized forward
must reproduce the network's own, ordinary forward exactly, since every
other Tier 1 oracle trusts campaign.golden(x) as ground truth.
"""


from collections.abc import Callable
from copy import deepcopy

import pytest
import torch
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi

from nets import NetSpec


@pytest.mark.neuron
@pytest.mark.synapse
def test_golden_forward_matches_the_network_own_forward(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """campaign.golden(x) reproduces type(net).forward(net, x) -- the
    network's own, un-wrapped forward -- bit-identically. If this doesn't
    hold, no other Tier 1 assertion means anything."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)

    own_forward = type(dense_net.net).forward(dense_net.net, x)
    golden_forward = cmpn.golden(x)

    assert torch.equal(own_forward, golden_forward)


@pytest.mark.neuron
@pytest.mark.synapse
def test_golden_slayer_neuron_matches_net_slayer_neuron(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """The campaign's own slayer.neuron dict (deepcopied at construction)
    still matches the original net's slayer.neuron: if these diverged, the
    golden/faulty forward passes would use different neuron dynamics than
    the net they're supposed to model."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)

    assert cmpn.slayer.neuron == dense_net.net.slayer.neuron


@pytest.mark.neuron
@pytest.mark.synapse
def test_deepcopied_net_forward_rebinds_to_the_copy(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """A deepcopy of campaign.golden's forward is re-bound to the new
    object, not left pointing at the original -- Campaign.reset() relies on
    this every round, deepcopying self.golden into self.faulty."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)

    copy_net = deepcopy(cmpn.golden)

    assert copy_net.forward.__self__ is copy_net
    assert torch.equal(copy_net(x), cmpn.golden(x))
