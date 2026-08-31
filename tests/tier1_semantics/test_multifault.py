"""Tier 1 — multi-fault rounds: locality (every site not targeted by a
fault stays bit-identical to golden, even with several simultaneous
faults), and coverage across the whole batch, every time bin, and
independently of what else is in the batch.
"""


from collections.abc import Callable

import pytest
import torch
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, DeadSynapse, ThresholdFaultNeuron

from nets import NetSpec
from helpers import assert_active


def _capture_following_input(
        cmpn: sfi.Campaign,
        following_name: str,
        x: Tensor,
        round_idx: int = 0
) -> Tensor:
    """Runs `round_idx` and returns the exact tensor the following layer
    received as input, captured after the fault pre-hook has already run."""
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    captured = {}
    handle = getattr(cmpn.faulty, following_name).register_forward_pre_hook(
        lambda _, inputs: captured.__setitem__('in', inputs[0].clone())
    )
    cmpn.r_idx_ref.r = round_idx
    cmpn.faulty(x)
    handle.remove()
    return captured['in']


@pytest.mark.neuron
@pytest.mark.synapse
def test_locality_with_simultaneous_faults_on_different_layers(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """A neuron fault on SF1 and a synapse fault on SF2, injected in the
    same round, each affect only their own site: every other position of
    SF1's own output stays bit-identical to golden, proving one fault's
    hook does not leak into a site it was never given."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    activity = golden_activity(cmpn, x)
    golden_sf1 = activity['SF1']

    site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), site, 0, 0, slice(None)))

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0))), round_idx=0)
    cmpn.inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF2', (0, 0, 0, 0))), round_idx=0)

    faulty_sf2_input = _capture_following_input(cmpn, 'SF2', x)

    mask = torch.ones_like(golden_sf1, dtype=torch.bool)
    mask[:, site, 0, 0, :] = False
    assert torch.equal(faulty_sf2_input[mask], golden_sf1[mask])
    assert torch.equal(
        faulty_sf2_input[:, site, 0, 0, :], torch.zeros_like(faulty_sf2_input[:, site, 0, 0, :])
    )


@pytest.mark.neuron
def test_fault_applies_to_every_batch_sample_and_time_bin(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """A DeadNeuron fault zeroes its site for every sample in the batch and
    every time bin, not just the specific (sample, bin) pairs that happened
    to be firing beforehand."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in, batch=4)
    activity = golden_activity(cmpn, x)
    golden_sf1 = activity['SF1']

    site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), site, 0, 0, slice(None)))

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0))), round_idx=0)
    faulty_sf2_input = _capture_following_input(cmpn, 'SF2', x)

    at_site = faulty_sf2_input[:, site, 0, 0, :]
    assert at_site.shape == (4, 16)
    assert torch.equal(at_site, torch.zeros_like(at_site))


@pytest.mark.parametric
def test_fault_result_identical_regardless_of_other_batch_members(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """The same sample's faulty result is identical whether it runs alone
    (batch of 1) or alongside three other samples (batch of 4): the fault
    hooks do not mix information across the batch dimension. Uses a
    parametric fault, not an OUTPUT fault: DeadNeuron/StuckNeuron/etc.
    overwrite with a fixed constant regardless of input, so a bug that
    leaked data across the batch would stay invisible to them -- a
    parametric fault's dummy layer genuinely recomputes from each sample's
    own input current, and so is actually sensitive to such a leak."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x4 = fixed_input(dense_net.shape_in, batch=4)
    x1 = x4[0:1]

    # A low enough rho, on a site chosen for strong activity, that the
    # dummy layer's threshold is actually crossed -- unlike most
    # (rho, site) combinations on this tiny net's default-initialized
    # weights (see test_synapse_semantics.py).
    cmpn.inject(
        sff.Fault(ThresholdFaultNeuron(0.05), sff.FaultSite('SF1', (2, 0, 0))),
        round_idx=0
    )

    out_batch4 = _capture_following_input(cmpn, 'SF2', x4)
    out_batch1 = _capture_following_input(cmpn, 'SF2', x1)

    assert torch.equal(out_batch1[0], out_batch4[0])
