"""Tier 2 — propagation: a fault's local effect reaching the following
layer's input and, in forced (all-dead / all-weights-dead) constructions
only, the network's final output. Masking is real and never asserted as a
law outside those forced constructions.
"""


from collections.abc import Callable

import pytest
import torch
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, DeadSynapse

from nets import NetSpec
from helpers import assert_active, run_round


def _capture_following_input(
        cmpn: sfi.Campaign,
        following_name: str,
        x: Tensor,
        round_idx: int = 0
) -> Tensor:
    """Runs round `round_idx` and returns the exact tensor the following
    layer received as input, captured via a pre-hook registered after the
    fault pre-hook so it observes the already-perturbed value."""
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    captured = {}
    handle = getattr(cmpn.faulty, following_name).register_forward_pre_hook(
        lambda _, inputs: captured.__setitem__('in', inputs[0])
    )
    cmpn.r_idx_ref.r = round_idx
    cmpn.faulty(x)
    handle.remove()
    return captured['in']


@pytest.mark.neuron
def test_dead_neuron_changes_only_the_faulted_site_in_next_layer_input(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """A neuron fault's effect appears in the following layer's input
    tensor at exactly the faulted site, leaving every other site of that
    tensor bit-identical to golden's."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)

    channels = cmpn.layers_info.shapes_neu['SF1'][0]
    site = next(c for c in range(channels) if activity['SF1'][:, c, 0, 0, :].sum() > 0)
    index = (slice(None), site, 0, 0, slice(None))
    assert_active(activity['SF1'], index)

    fault = sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    next_input = _capture_following_input(cmpn, 'SF2', x)

    assert not torch.equal(next_input[index], activity['SF1'][index])

    elsewhere = torch.ones_like(next_input, dtype=torch.bool)
    elsewhere[index] = False
    assert torch.equal(next_input[elsewhere], activity['SF1'][elsewhere])


@pytest.mark.neuron
def test_all_dead_layer_forces_next_layers_output_to_zero(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """Forced construction: DeadNeuron on every SF1 site zeroes SF2's whole
    input outright, and since slayer's dense layers carry no bias, an
    all-zero input can only ever produce an all-zero PSP that never crosses
    threshold and SF2's output is exactly zero."""
    # SF2's own threshold is rarely crossed by this tiny net's
    # default-initialized weights, so it is amplified to get a
    # real, non-degenerate golden output to force to zero.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)
    assert activity['SF2'].sum() > 0, (
        "Golden SF2 output is already all zero; the fault would be vacuous."
    )

    channels = cmpn.layers_info.shapes_neu['SF1'][0]
    sites = [sff.FaultSite('SF1', (c, 0, 0)) for c in range(channels)]
    cmpn.inject(sff.Fault(DeadNeuron(), sites), round_idx=0)

    output = run_round(cmpn, 0, x)
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.neuron
def test_dead_conv_layer_propagates_through_pool_to_next_layers_output(
        conv_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """Structural case: SP1 is non-injectable, so the neuron perturb
    pre-hook for SC1 is registered on SP1 rather than on SF2 directly.
    Forced construction: DeadNeuron on every SC1 site zeroes SP1's whole
    input, and since pooling and dense are both bias-free the zero survives
    the pool and forces SF2's output to be exactly zero."""
    cmpn = make_campaign(conv_net.net, conv_net.shape_in, slayer)
    assert cmpn.layers_info.get_following('SC1') == 'SP1'
    assert not cmpn.layers_info.is_injectable('SP1')

    x = fixed_input(conv_net.shape_in)
    activity = golden_activity(cmpn, x)
    assert activity['SF2'].sum() > 0, (
        "Golden SF2 output is already all zero; the fault would be vacuous."
    )

    channels, height, width = cmpn.layers_info.shapes_neu['SC1']
    sites = [
        sff.FaultSite('SC1', (c, h, w))
        for c in range(channels) for h in range(height) for w in range(width)
    ]
    cmpn.inject(sff.Fault(DeadNeuron(), sites), round_idx=0)

    output = run_round(cmpn, 0, x)
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.synapse
def test_dead_weight_layer_propagates_through_intervening_layer_to_final_output(
        three_layer_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """Forced construction, one layer earlier than a last-layer weight kill:
    zeroing every incoming weight of SF2 forces SF2's own output to zero
    directly and that zero must then propagate through SF3 to reach the
    network's final output, which is exactly zero as a result."""
    # SF2/SF3's own thresholds are rarely crossed by this tiny net's
    # default-initialized weights, so they are amplified to get a real,
    # non-degenerate golden output to force to zero.
    with torch.no_grad():
        three_layer_net.net.SF2.weight.mul_(20)
        three_layer_net.net.SF3.weight.mul_(20)
    cmpn = make_campaign(three_layer_net.net, three_layer_net.shape_in, slayer)
    x = fixed_input(three_layer_net.shape_in)
    activity = golden_activity(cmpn, x)
    assert activity['SF3'].sum() > 0, (
        "Golden final output is already all zero; the fault would be vacuous."
    )

    out_channels, in_channels = cmpn.golden.SF2.weight.shape[:2]
    sites = [
        sff.FaultSite('SF2', (o, c, 0, 0))
        for o in range(out_channels) for c in range(in_channels)
    ]
    cmpn.inject(sff.Fault(DeadSynapse(), sites), round_idx=0)

    output = run_round(cmpn, 0, x)
    assert torch.equal(output, torch.zeros_like(output))
