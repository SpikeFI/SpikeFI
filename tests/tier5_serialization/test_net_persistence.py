"""Tier 5 -- net persistence (Campaign.save_net / Campaign.load_net):
the single save_net contract across training, post-training, and golden
rounds, its independence from a prior run(), and backward compatibility
with plain state_dict files.
"""


from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
import spikefi.hooks as sfh
from spikefi.models import DeadNeuron, PerturbedSynapse, SaturatedNeuron, StuckSynapse, ThresholdFaultNeuron

from nets import NetSpec
from helpers import assert_active, assert_differs, assert_not_saturated, run_round


@pytest.mark.serialization
@pytest.mark.training
@pytest.mark.neuron
@pytest.mark.parametric
def test_training_campaign_reload_matches_live_faulty_output_bit_identically(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """After run_train(), save_net(r)/load_net() reproduces the live
    faulties[r] net's output bit-for-bit, including a neuron hard fault and a
    parametric fault, neither of which a bare state_dict alone can carry."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)
    x = fixed_input(dense_net.shape_in)

    # Two distinct, genuinely firing sites on SF1: one for each fault type,
    # so neither fault would be a no-op on this net's own weights.
    golden_sf1 = golden_activity(cmpn, x)['SF1']
    dead_site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), dead_site, 0, 0, slice(None)))
    param_site = next(
        c for c in range(4)
        if c != dead_site and golden_sf1[:, c, 0, 0, :].sum() > 0
    )
    assert_active(golden_sf1, (slice(None), param_site, 0, 0, slice(None)))

    cmpn.inject([
        sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (dead_site, 0, 0))),
        sff.Fault(ThresholdFaultNeuron(0.05), sff.FaultSite('SF1', (param_site, 0, 0))),
    ], round_idx=0)

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )
    expected_output = faulties[0](x).clone()
    assert expected_output.any(), (
        'Live faulty output is all-zero; bit-identical equality below would be vacuous.'
    )

    fpath = cmpn.save_net(0)
    device = next(dense_net.net.parameters()).device
    loaded = sfi.Campaign.load_net(fpath, dense_net.net, device)

    # Weights alone (no envelope hooks) must NOT reproduce the faulty
    # output, or this test would pass even if load_net() dropped the round
    # entirely and only restored the state_dict.
    bare = deepcopy(dense_net.net).to(device)
    bare.load_state_dict(
        torch.load(fpath, map_location=device, weights_only=False)['state_dict']
    )
    bare.eval()
    assert not torch.equal(bare(x), expected_output), (
        'Weights alone reproduce the faulty output; this test would not '
        'actually exercise the neuron/parametric hooks load_net() reattaches.'
    )

    assert torch.equal(loaded(x), expected_output)


@pytest.mark.serialization
@pytest.mark.neuron
@pytest.mark.synapse
def test_post_training_campaign_reload_matches_round_output_with_campaign_deleted(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """For a post-training round carrying both a synapse and a neuron
    fault, save_net(r)/load_net() reproduces the round's output exactly
    once the originating Campaign is gone -- checked both as the combined
    output and as its two separable halves: the synapse fault baked into
    the state_dict at exactly its own index, and the neuron fault's direct
    pre-hook reattached on the layer following its own."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_output = cmpn.golden(x).clone()

    # SaturatedNeuron forces its site to fire at every time bin -- unlike
    # DeadNeuron, strong enough to push this net's naturally quiet second
    # layer into actually firing, so the round's own output is non-trivial.
    golden_sf1 = golden_activity(cmpn, x)['SF1']
    n_time_bins = x.shape[-1]
    neuron_site = next(
        c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() < n_time_bins
    )
    assert_not_saturated(golden_sf1, (slice(None), neuron_site, 0, 0, slice(None)), n_time_bins)

    synapse_site = (0, 0, 0, 0)
    target = 0.37
    assert_differs(dense_net.net.SF1.weight[synapse_site].item(), target)

    cmpn.inject([
        sff.Fault(SaturatedNeuron(), sff.FaultSite('SF1', (neuron_site, 0, 0))),
        sff.Fault(StuckSynapse(target), sff.FaultSite('SF1', synapse_site)),
    ], round_idx=0)

    expected_output = run_round(cmpn, 0, x).clone()
    assert expected_output.any(), (
        "The round's output is all-zero; the reload equality below would be vacuous."
    )
    assert not torch.equal(expected_output, golden_output), (
        'The injected round produced golden output; the reload-matches-faulty '
        'check below would be indistinguishable from a reload of golden.'
    )

    fpath = cmpn.save_net(0)
    device = next(dense_net.net.parameters()).device
    payload = torch.load(fpath, map_location=device, weights_only=False)

    # The synapse fault must be the only weight difference from golden,
    # baked in at exactly its own site.
    loaded_state = payload['state_dict']
    golden_state = dense_net.net.state_dict()
    for key, tensor in loaded_state.items():
        diff_mask = tensor != golden_state[key]
        if key == 'SF1.weight':
            expected_mask = torch.zeros_like(diff_mask)
            expected_mask[synapse_site] = True
            assert torch.equal(diff_mask, expected_mask), (
                f'SF1.weight differs from golden outside the synapse fault site {synapse_site}.'
            )
        else:
            assert not diff_mask.any(), f'{key} unexpectedly differs from golden.'

    del cmpn

    loaded = sfi.Campaign.load_net(fpath, dense_net.net, device)

    # The neuron fault's direct pre-hook must be reattached on SF2, the
    # layer following SF1 (the faulty layer).
    pre_hooks = loaded.SF2._forward_pre_hooks.values()
    assert any(isinstance(h, sfh.DirectNeuronPerturbPreHook) for h in pre_hooks), (
        'No DirectNeuronPerturbPreHook found on the layer following the neuron fault.'
    )

    assert torch.equal(loaded(x), expected_output)


@pytest.mark.serialization
def test_empty_round_save_net_writes_bare_state_dict(
        conv_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """A round with no faults injected makes save_net() write a bare
    state_dict with no fault envelope, and load_net() reproduces golden's
    own output exactly."""
    cmpn = make_campaign(conv_net.net, conv_net.shape_in, slayer)
    x = fixed_input(conv_net.shape_in)
    golden_output = cmpn.golden(x).clone()
    assert golden_output.any(), 'Golden output is all-zero; equality below would be vacuous.'

    fpath = cmpn.save_net(0)
    device = next(conv_net.net.parameters()).device
    payload = torch.load(fpath, map_location=device, weights_only=False)
    assert not (isinstance(payload, dict) and 'state_dict' in payload), (
        'save_net() wrote a fault envelope for an empty round instead of a bare state_dict.'
    )

    loaded = sfi.Campaign.load_net(fpath, conv_net.net, device)
    assert torch.equal(loaded(x), golden_output)


@pytest.mark.serialization
@pytest.mark.synapse
def test_post_training_save_net_state_dict_is_independent_of_a_prior_run(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        artifact_name: str
) -> None:
    """A post-training round's perturbed weight is derived from golden's
    own weights every time save_net() is called, not from the fault
    model's cached `.perturbed` value that a prior run() would have left
    behind -- so save_net() immediately after inject() and again after
    run() must write the identical state_dict."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)

    site = (0, 0, 0, 0)
    rho = 1.7
    w_original = dense_net.net.SF1.weight[site].item()
    cmpn.inject(sff.Fault(PerturbedSynapse(rho), sff.FaultSite('SF1', site)), round_idx=0)

    fpath_before_run = cmpn.save_net(0, fname=f'{artifact_name}_before_run')
    run_round(cmpn, 0, x)
    fpath_after_run = cmpn.save_net(0, fname=f'{artifact_name}_after_run')

    device = next(dense_net.net.parameters()).device
    state_before = torch.load(fpath_before_run, map_location=device, weights_only=False)['state_dict']
    state_after = torch.load(fpath_after_run, map_location=device, weights_only=False)['state_dict']

    expected = w_original * rho
    assert state_before['SF1.weight'][site].item() == pytest.approx(expected), (
        'The pre-run save did not derive the perturbed weight from golden.'
    )
    assert torch.equal(state_before['SF1.weight'], state_after['SF1.weight'])


@pytest.mark.serialization
def test_load_net_accepts_a_plain_state_dict_file(
        conv_net: NetSpec,
        fixed_input: Callable[..., Tensor],
        tmp_path: Path
) -> None:
    """A file written the old way -- a bare torch.save(net.state_dict(),
    path) with no envelope at all -- still loads through load_net(): the
    weights come back, but with no hooks attached."""
    device = next(conv_net.net.parameters()).device
    x = fixed_input(conv_net.shape_in)
    reference_output = conv_net.net(x).clone()
    assert reference_output.any(), 'Reference output is all-zero; equality below would be vacuous.'

    fpath = str(tmp_path / 'plain_state_dict.pt')
    torch.save(conv_net.net.state_dict(), fpath)

    loaded = sfi.Campaign.load_net(fpath, conv_net.net, device)

    assert torch.equal(loaded(x), reference_output)
    assert loaded.SC1._forward_pre_hooks == {}
    assert loaded.SF2._forward_pre_hooks == {}
