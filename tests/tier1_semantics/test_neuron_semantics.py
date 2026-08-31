"""Tier 1 — neuron (OUTPUT-target) fault semantics: exact known-answer
values at the following layer's input, and SaturatedNeuron's exact
end-to-end answer through the tail on an output-layer site.
"""


import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, SaturatedNeuron, StuckNeuron

from helpers import assert_active, assert_differs


def _capture_following_input(cmpn: sfi.Campaign, following_name: str, x, round_idx: int = 0):
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
def test_dead_neuron_sets_next_layer_input_to_exactly_zero(
        dense_net, slayer, make_campaign, fixed_input, golden_activity
) -> None:
    """DeadNeuron is set_value(_, 0.): the following layer's input at the
    fault site is exactly 0.0 for every batch sample and time bin."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)

    site = next(c for c in range(4) if activity['SF1'][:, c, 0, 0, :].sum() > 0)
    assert_active(activity['SF1'], (slice(None), site, 0, 0, slice(None)))

    fault = sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0)))
    cmpn.inject(fault, round_idx=0)

    next_input = _capture_following_input(cmpn, 'SF2', x)
    assert torch.equal(next_input[:, site, 0, 0, :], torch.zeros_like(next_input[:, site, 0, 0, :]))


@pytest.mark.neuron
def test_saturated_neuron_sets_next_layer_input_to_exactly_one(
        dense_net, slayer, make_campaign, fixed_input, golden_activity
) -> None:
    """SaturatedNeuron is set_value(_, 1.): the following layer's input at
    the fault site is exactly 1.0 for every batch sample and time bin."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)

    site = next(
        c for c in range(4)
        if (activity['SF1'][:, c, 0, 0, :].sum(dim=1) < 16).any()
    )
    assert (activity['SF1'][:, site, 0, 0, :].sum(dim=1) < 16).any(), (
        f'Site {site} is already saturated for every sample; '
        'SaturatedNeuron would be a no-op here.'
    )

    fault = sff.Fault(SaturatedNeuron(), sff.FaultSite('SF1', (site, 0, 0)))
    cmpn.inject(fault, round_idx=0)

    next_input = _capture_following_input(cmpn, 'SF2', x)
    assert torch.equal(next_input[:, site, 0, 0, :], torch.ones_like(next_input[:, site, 0, 0, :]))


@pytest.mark.neuron
def test_stuck_neuron_sets_next_layer_input_to_exactly_x(
        dense_net, slayer, make_campaign, fixed_input, golden_activity
) -> None:
    """StuckNeuron(x) is set_value(_, x): the following layer's input at
    the fault site is exactly x for every batch sample and time bin."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)

    site = (0, 0, 0)
    stuck_value = 0.5
    assert_differs(activity['SF1'][0, 0, 0, 0, 0].item(), stuck_value)

    fault = sff.Fault(StuckNeuron(stuck_value), sff.FaultSite('SF1', site))
    cmpn.inject(fault, round_idx=0)

    next_input = _capture_following_input(cmpn, 'SF2', x)
    expected = torch.full_like(next_input[:, 0, 0, 0, :], stuck_value)
    assert torch.equal(next_input[:, 0, 0, 0, :], expected)


@pytest.mark.neuron
def test_saturated_neuron_end_to_end_forces_prediction_and_accuracy(
        dense_net, slayer, make_campaign
) -> None:
    """SaturatedNeuron on an output-layer neuron o propagates untouched
    through the synthetic tail: output[:,o,0,0,:] is 1.0 at every time bin,
    so pred is always o, and testing.maxAccuracy equals the fraction of
    samples whose true label is o."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    assert cmpn.layers_info.is_output('SF2')

    device = next(dense_net.net.parameters()).device
    n_samples = 8
    generator = torch.Generator(device=device).manual_seed(7)
    x = (torch.rand(n_samples, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    # Deliberately asymmetric class counts (5/2/1, not an even 3-way split):
    # an even split could make an argmax-vs-argmin-style prediction bug
    # produce the *same* accuracy by coincidence, masking it.
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2], device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    o = 0
    fault = sff.Fault(SaturatedNeuron(), sff.FaultSite('SF2', (o, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')
    # Campaign.run()'s own stats.update() resets correctSamples/numSamples for
    # the next epoch, so maxAccuracy must be read right after run() -- a
    # second _pre_run() below (to inspect the raw output) rebuilds
    # self.performance from scratch and would wipe it.
    max_accuracy = cmpn.performance[0].testing.maxAccuracy

    cmpn._pre_run(sfi.CampaignOptimization.O0)
    cmpn.r_idx_ref.r = 0
    output = cmpn.faulty(x)

    assert torch.equal(output[:, o, 0, 0, :], torch.ones_like(output[:, o, 0, 0, :]))
    pred = output.sum(dim=(2, 3, 4)).argmax(dim=1)
    assert torch.all(pred == o)
    assert max_accuracy == (y == o).float().mean().item()


@pytest.mark.neuron
def test_dead_neuron_end_to_end_reduces_accuracy_by_the_dead_class_rate(
        dense_net, slayer, make_campaign
) -> None:
    """DeadNeuron on an output-layer neuron o forces that channel to 0
    forever: every sample the golden net correctly predicted as o is now
    misclassified, while every other sample's prediction is untouched.
    testing.maxAccuracy therefore drops by exactly the fraction of
    samples that were correctly classified as o beforehand."""
    # SF2's own threshold is rarely crossed by this tiny net's
    # default-initialized weights (see test_synapse_semantics.py), so it is
    # amplified to get a real, non-degenerate prediction per sample.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    assert cmpn.layers_info.is_output('SF2')

    device = next(dense_net.net.parameters()).device
    n_samples = 16
    generator = torch.Generator(device=device).manual_seed(7)
    x = (torch.rand(n_samples, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    golden_pred = cmpn.golden(x).sum(dim=(2, 3, 4)).argmax(dim=1)
    # Labelled to match golden's own prediction, so golden is 100% accurate
    # and every later drop is attributable to the fault alone.
    y = golden_pred.clone()

    golden_channel_sums = cmpn.golden(x).sum(dim=(2, 3, 4))
    o = next(
        c for c in range(3)
        # A class predicted only by an all-channels-silent tie (argmax
        # defaults to index 0) would make DeadNeuron a no-op for those
        # samples: it was already forced to 0 by having nothing to say.
        if 0 < (golden_pred == c).sum() < n_samples
        and (golden_channel_sums[golden_pred == c, c] > 0).all()
    )
    correctly_classified_as_o = int((golden_pred == o).sum())
    assert correctly_classified_as_o > 0, f'No sample is predicted {o} in golden; fault would be vacuous.'
    expected_drop = correctly_classified_as_o / n_samples

    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    fault = sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (o, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')
    max_accuracy = cmpn.performance[0].testing.maxAccuracy

    cmpn._pre_run(sfi.CampaignOptimization.O0)
    cmpn.r_idx_ref.r = 0
    output = cmpn.faulty(x)

    assert torch.equal(output[:, o, 0, 0, :], torch.zeros_like(output[:, o, 0, 0, :]))
    pred = output.sum(dim=(2, 3, 4)).argmax(dim=1)

    was_o = golden_pred == o
    assert torch.all(pred[was_o] != o)
    assert torch.equal(pred[~was_o], golden_pred[~was_o])
    assert max_accuracy == pytest.approx(1.0 - expected_drop)
