"""Tier 1 — null-fault identity: a fault whose model reduces to the
identity transform at its injected site is a provable no-op, so its
output must match golden bit-for-bit.
"""


from collections.abc import Callable

import pytest
import torch
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import PerturbedSynapse, StuckSynapse, ThresholdFaultNeuron

from nets import NetSpec
from helpers import run_round


@pytest.mark.parametric
def test_threshold_fault_at_1x_is_a_no_op(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """ThresholdFaultNeuron(1.0) multiplies theta by 1.0: the perturbed
    threshold must equal the original exactly -- checked directly on the
    dummy layer's own parameter, since a wrong theta can still fail to flip
    any single random input's spike decision, masking the bug from an
    output-only comparison. The faulty output must also match golden."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_out = cmpn.golden(x)

    fault = sff.Fault(ThresholdFaultNeuron(1.0), sff.FaultSite('SF1', (0, 0, 0)))
    cmpn.inject(fault, round_idx=0)

    faulty_out = run_round(cmpn, 0, x)

    installed = cmpn.rounds[0].grouped[('SF1', sff.FaultTarget.PARAMETER)][0]
    assert installed.model.flayer.neuron['theta'] == cmpn.slayer.neuron['theta']
    assert torch.equal(faulty_out, golden_out)


@pytest.mark.synapse
def test_perturbed_synapse_at_1x_is_a_no_op(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """PerturbedSynapse(1.0) multiplies the weight by 1.0: a provable
    no-op, so the faulty output must match golden bit-for-bit."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_out = cmpn.golden(x)

    fault = sff.Fault(PerturbedSynapse(1.0), sff.FaultSite('SF1', (0, 0, 0, 0)))
    cmpn.inject(fault, round_idx=0)

    faulty_out = run_round(cmpn, 0, x)
    assert torch.equal(faulty_out, golden_out)


@pytest.mark.synapse
def test_stuck_synapse_at_original_value_is_a_no_op(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """StuckSynapse(w_original) sets the weight to its own current value:
    a provable no-op, so the faulty output must match golden bit-for-bit.
    Also captures the weight actually used *during* the forward pass, since
    a small enough write-time error could be absorbed by the neuron's
    spiking threshold and never show up in the output at all."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_out = cmpn.golden(x)

    site_index = (0, 0, 0, 0)
    w_original = cmpn.golden.SF1.weight[site_index].item()
    fault = sff.Fault(StuckSynapse(w_original), sff.FaultSite('SF1', site_index))
    cmpn.inject(fault, round_idx=0)

    cmpn._pre_run(sfi.CampaignOptimization.O0)
    used_weight = {}
    handle = cmpn.faulty.SF1.register_forward_pre_hook(
        lambda _, __: used_weight.__setitem__('w', cmpn.faulty.SF1.weight[site_index].item())
    )
    cmpn.r_idx_ref.r = 0
    faulty_out = cmpn.faulty(x)
    handle.remove()

    assert used_weight['w'] == w_original
    assert torch.equal(faulty_out, golden_out)
