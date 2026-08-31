"""Tier 1 — parametric (PARAMETER-target) fault semantics: threshold
monotonicity, the perturbed dummy layer's isolation from the campaign's own
slayer, and post-round stash hygiene.
"""


from collections.abc import Callable

import pytest
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import ThresholdFaultNeuron

from nets import NetSpec
from helpers import assert_active


def _spike_count_at_site(
        cmpn: sfi.Campaign,
        x: Tensor,
        site: tuple[int, int, int]
) -> Tensor:
    """Runs round 0 and returns the spike count (summed over batch and
    time) the following layer received at `site`, i.e. after any neuron
    fault at that site has already been applied."""
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    captured = {}
    handle = cmpn.faulty.SF2.register_forward_pre_hook(
        lambda _, inputs: captured.__setitem__('in', inputs[0])
    )
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)
    handle.remove()
    return captured['in'][:, site[0], site[1], site[2], :].sum()


@pytest.mark.parametric
def test_threshold_increase_does_not_increase_spike_count(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """ThresholdFaultNeuron(rho > 1) raises theta -> a higher bar to fire,
    so the site's spike count cannot increase relative to golden."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)

    site = next(c for c in range(4) if activity['SF1'][:, c, 0, 0, :].sum() > 0)
    assert_active(activity['SF1'], (slice(None), site, 0, 0, slice(None)))
    golden_count = activity['SF1'][:, site, 0, 0, :].sum()

    fault = sff.Fault(ThresholdFaultNeuron(1.5), sff.FaultSite('SF1', (site, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    faulty_count = _spike_count_at_site(cmpn, x, (site, 0, 0))

    assert faulty_count != golden_count, (
        "The fault had no effect on this site's spike count; the "
        'non-increasing check below would be vacuously true regardless of '
        'whether theta actually changed.'
    )
    assert faulty_count <= golden_count


@pytest.mark.parametric
def test_threshold_decrease_does_not_decrease_spike_count(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """ThresholdFaultNeuron(rho < 1) lowers theta -> an easier bar to fire,
    so the site's spike count cannot decrease relative to golden."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)

    site = next(c for c in range(4) if activity['SF1'][:, c, 0, 0, :].sum() > 0)
    assert_active(activity['SF1'], (slice(None), site, 0, 0, slice(None)))
    golden_count = activity['SF1'][:, site, 0, 0, :].sum()

    fault = sff.Fault(ThresholdFaultNeuron(0.5), sff.FaultSite('SF1', (site, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    faulty_count = _spike_count_at_site(cmpn, x, (site, 0, 0))

    assert faulty_count != golden_count, (
        "The fault had no effect on this site's spike count; the "
        'non-decreasing check below would be vacuously true regardless of '
        'whether theta actually changed.'
    )
    assert faulty_count >= golden_count


@pytest.mark.parametric
def test_parametric_fault_isolated_from_campaign_slayer(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """param_perturb() builds its dummy layer from a shallow copy of
    slayer.neuron: the campaign's own slayer.neuron is unchanged after a
    round runs, while the fault's dummy flayer holds the perturbed value."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    original_theta = cmpn.slayer.neuron['theta']

    rho = 1.5
    fault = sff.Fault(ThresholdFaultNeuron(rho), sff.FaultSite('SF1', (0, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)

    installed = cmpn.rounds[0].grouped[('SF1', sff.FaultTarget.PARAMETER)][0]
    assert cmpn.slayer.neuron['theta'] == original_theta
    assert installed.model.flayer.neuron['theta'] == original_theta * rho


@pytest.mark.parametric
def test_parametric_fault_stash_is_none_after_a_round(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """After running a round, a PARAMETER fault's cached perturbed value is
    None again -- unlike a WEIGHT fault's, unstore() consumes it during the
    same forward pass that produces it."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)

    fault = sff.Fault(ThresholdFaultNeuron(1.5), sff.FaultSite('SF1', (0, 0, 0)))
    cmpn.inject(fault, round_idx=0)
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)

    installed = cmpn.rounds[0].grouped[('SF1', sff.FaultTarget.PARAMETER)][0]
    assert installed.model.perturbed is None
