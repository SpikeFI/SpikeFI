"""Tier 4 — training mode (run_train): persistent vs. one-shot synapse
fault semantics, neuron faults surviving training, the fault's
participation in the backward pass, per-round isolation, and the
self-containment of a returned net (no live Campaign required).
"""


from collections.abc import Callable
from copy import deepcopy
import gc
import weakref

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
import spikefi.hooks as sfh
from spikefi.models import DeadNeuron, mul_value, PerturbedSynapse, StuckSynapse

from nets import NetSpec
from helpers import assert_active


def _capture_sf2_input(net: nn.Module, x: Tensor) -> Tensor:
    """Returns exactly what SF2 receives as input, captured via its own
    forward pre-hook: closer to a neuron fault (evaluated on this same
    pre-hook) than the network's final output, and immune to a trained net's
    output having gone fully silent downstream -- a real failure mode after
    just a few epochs on this tiny random dataset, which would otherwise
    make any two results compare equal regardless of the fault."""
    captured = {}
    handle = net.SF2.register_forward_pre_hook(
        lambda _, inputs: captured.__setitem__('in', inputs[0].clone())
    )
    net(x)
    handle.remove()
    return captured['in']


@pytest.mark.synapse
@pytest.mark.training
def test_persistent_weight_fault_is_exact_after_training(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A persistent synapse fault's optimizer step-post-hook
    re-clamps the weight after every optimizer step, so no matter how
    training moves it in between, the final weight lands exactly on the
    fault's target value."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)

    site = (0, 0, 0, 0)
    target = 0.42
    cmpn.inject(sff.Fault(StuckSynapse(target), sff.FaultSite('SF1', site)), round_idx=0)

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        3, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    assert faulties[0].SF1.weight[site].item() == pytest.approx(target)


@pytest.mark.synapse
@pytest.mark.training
def test_non_persistent_weight_fault_moves_away_after_training(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A non-persistent fault (PerturbedSynapse) is applied once, as the
    net's initial faulty state, and nothing re-clamps it afterwards: plain
    gradient descent is free to move the weight away during training.
    Checked in two parts, not just the end state -- 'the final weight
    isn't at the target value' is equally true of a fault that was never
    applied to begin with, so the initial application is confirmed
    directly before training even starts."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)

    site = (0, 0, 0, 0)
    rho = 1.5
    w_original = dense_net.net.SF1.weight[site].item()
    initial_value = w_original * rho
    cmpn.inject(sff.Fault(PerturbedSynapse(rho), sff.FaultSite('SF1', site)), round_idx=0)

    pre_training_faulties = cmpn._pre_run_train()
    assert pre_training_faulties[0].SF1.weight[site].item() == pytest.approx(initial_value)

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        3, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    assert faulties[0].SF1.weight[site].item() != pytest.approx(initial_value)


@pytest.mark.synapse
@pytest.mark.training
def test_persistent_mul_value_fault_compounds_across_reapplications(
        dense_net: NetSpec
) -> None:
    """DirectSynapsePersistentOptimizerHook re-applies model.perturb() to
    whatever the weight currently is, every time it fires -- for a
    mul_value fault that means each call multiplies by rho again, so N
    calls compound to w * rho**N. Exercised directly against the hook,
    isolated from an optimizer's own gradient updates, which would
    otherwise perturb the value between calls too."""
    site = (0, 0, 0, 0)
    w0 = dense_net.net.SF1.weight[site].clone()
    rho = 1.1
    model = sff.FaultModel(sff.FaultTarget.WEIGHT, mul_value, rho, persistent=True)
    round = sff.FaultRound([sff.Fault(model, sff.FaultSite('SF1', site))])
    hook = sfh.DirectSynapsePersistentOptimizerHook(dense_net.net, round)

    n_calls = 5
    for _ in range(n_calls):
        hook(None, (), {})

    assert dense_net.net.SF1.weight[site].item() == pytest.approx(
        (w0 * rho ** n_calls).item(), rel=1e-5
    )


@pytest.mark.neuron
@pytest.mark.training
def test_neuron_fault_still_active_after_training_completes(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A neuron fault's direct pre-hook is attached to the trained net
    itself (not routed through the campaign), so the fault site is still
    forced to its fault value on every forward pass after training
    completes, exactly as during training."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)
    x, _ = next(iter(test_loader))

    # A site that never fires would make DeadNeuron a no-op regardless of
    # whether the fault hook actually works, so a genuinely active one is
    # searched for rather than assumed.
    golden_sf1 = cmpn.golden(x, 0, 0)
    site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), site, 0, 0, slice(None)))
    site = (site, 0, 0)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', site)), round_idx=0)

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        2, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    captured = {}
    handle = faulties[0].SF2.register_forward_pre_hook(
        lambda _, inputs: captured.__setitem__('in', inputs[0])
    )
    faulties[0](x)
    handle.remove()

    at_site = captured['in'][:, site[0], site[1], site[2], :]
    assert torch.equal(at_site, torch.zeros_like(at_site))


@pytest.mark.synapse
@pytest.mark.training
def test_fault_participates_in_the_backward_pass(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """The faulty weight must actually feed the loss the gradient is taken
    from, not merely be displayed while gradients are computed from the
    original value: one optimizer step on the campaign's faulty net must
    match a hand-built net that starts training from the same perturbed
    weight bit-for-bit, and must differ from the same recipe starting at
    golden's own weight."""
    device = next(dense_net.net.parameters()).device
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in, batch_size=8)
    x, y = next(iter(train_loader))

    site = (0, 0, 0, 0)
    rho = 1.5
    w_original = dense_net.net.SF1.weight[site].clone()
    w_perturbed = w_original * rho
    cmpn.inject(sff.Fault(PerturbedSynapse(rho), sff.FaultSite('SF1', site)), round_idx=0)

    spike_loss = snn.loss(net_params).to(device)
    faulties = cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.SGD(params, lr=0.1), progress_mode='silent'
    )

    def _one_sgd_step(net: nn.Module) -> nn.Module:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        output = net(x)
        target = torch.zeros_like(output[..., :1]).scatter_(1, y.view(-1, 1, 1, 1, 1), 1.0)
        loss = spike_loss.numSpikes(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return net

    hand_net = deepcopy(dense_net.net)
    with torch.no_grad():
        hand_net.SF1.weight[site] = w_perturbed
    _one_sgd_step(hand_net)

    golden_start_net = _one_sgd_step(deepcopy(dense_net.net))

    assert torch.allclose(faulties[0].SF1.weight, hand_net.SF1.weight, atol=1e-6)
    assert torch.allclose(faulties[0].SF2.weight, hand_net.SF2.weight, atol=1e-6)
    assert not torch.allclose(faulties[0].SF1.weight, golden_start_net.SF1.weight, atol=1e-6)


@pytest.mark.training
def test_per_round_training_isolation(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """Each round trains its own deepcopy of the net: a persistent fault
    present only in round 1 must show up only in that round's returned net,
    leaving round 0's own weight at that same site free to train normally."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)

    site = (0, 0, 0, 0)
    target = 0.5
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(StuckSynapse(target), sff.FaultSite('SF2', site)))

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        2, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    assert faulties[1].SF2.weight[site].item() == pytest.approx(target)
    assert faulties[0].SF2.weight[site].item() != pytest.approx(target)


@pytest.mark.training
def test_returned_net_output_is_invariant_to_round_index_mutation(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A trained round's hooks hold their own Fault objects directly
    (DirectFaultHook), not resolved through the campaign's shared
    RoundIndex, so mutating campaign.r_idx_ref.r must not change what a
    returned net computes. run_train() itself leaves r_idx_ref.r sitting on
    the last round trained (1 here), so the value is forced back to 0
    first -- comparing against whatever it already happened to be would
    let a real dependency on r_idx_ref slip through unnoticed."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)
    x, _ = next(iter(test_loader))

    # A site that never fires would make DeadNeuron a no-op regardless of
    # whether the fault hook actually works, so a genuinely active one is
    # searched for rather than assumed -- without this, round 0's net would
    # look identical whether or not its own fault (or round 1's, wrongly
    # dispatched to it) were actually being applied.
    golden_sf1 = cmpn.golden(x, 0, 0)
    site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), site, 0, 0, slice(None)))

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(StuckSynapse(0.5), sff.FaultSite('SF2', (0, 0, 0, 0))))

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    cmpn.r_idx_ref.r = 0
    before = _capture_sf2_input(faulties[0], x)
    cmpn.r_idx_ref.r = 1
    after = _capture_sf2_input(faulties[0], x)

    assert torch.equal(before, after)


@pytest.mark.training
def test_returned_net_output_survives_campaign_deletion(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A returned net's fault hooks hold their Fault objects via an
    ordinary reference, not a lookup back through the campaign, so the
    campaign itself must actually become collectible once dropped -- 'the
    output is unchanged after del campaign' on its own proves nothing:
    Python's refcounting GC keeps any object a surviving hook still
    references alive regardless (a hook holding a live reference to
    r_idx_ref or rounds, as a dispatching-style hook would, keeps those
    specific objects around, but never the Campaign itself, since neither
    is a back-reference to it), so a dispatching-style hook would pass that
    comparison too. Checked instead with a weakref to the campaign object:
    it must actually die."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)
    x, _ = next(iter(test_loader))

    # A site that never fires would make DeadNeuron a no-op regardless of
    # whether the fault hook actually works, so a genuinely active one is
    # searched for rather than assumed.
    golden_sf1 = cmpn.golden(x, 0, 0)
    site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), site, 0, 0, slice(None)))

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0))), round_idx=0)

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    before = _capture_sf2_input(faulties[0], x)
    campaign_ref = weakref.ref(cmpn)
    del cmpn
    gc.collect()

    assert campaign_ref() is None, 'The Campaign object is still referenced from somewhere.'
    after = _capture_sf2_input(faulties[0], x)
    assert torch.equal(before, after)


@pytest.mark.training
def test_dispatching_hooks_stay_empty_and_direct_hooks_are_keyed_by_round_and_layer(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """run_train() uses _perturb_net_train exclusively: it must never
    populate campaign.dispatching_hooks (the post-training-only dispatch
    path), and every entry it does add to direct_hooks must be keyed by
    its own (round, layer|None, hook type) -- checked with a neuron fault
    on SF1 in *both* rounds (not just round 0), or a bug that hardcoded
    every key's round index to 0 would go unnoticed: round 0's own key is
    already 0, so only round 1's entry can tell the two apart."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject([
        sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))),
        sff.Fault(StuckSynapse(0.5), sff.FaultSite('SF2', (0, 0, 0, 0)))
    ])

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )

    assert cmpn.dispatching_hooks == {}
    assert (0, 'SF1', sfh.DirectNeuronPerturbPreHook) in cmpn.direct_hooks
    assert (0, None, sfh.DirectSynapsePersistentOptimizerHook) in cmpn.direct_hooks
    assert (1, 'SF1', sfh.DirectNeuronPerturbPreHook) in cmpn.direct_hooks
    assert (1, None, sfh.DirectSynapsePersistentOptimizerHook) in cmpn.direct_hooks
    assert (
        cmpn.direct_hooks[(0, 'SF1', sfh.DirectNeuronPerturbPreHook)]
        is not cmpn.direct_hooks[(1, 'SF1', sfh.DirectNeuronPerturbPreHook)]
    )


@pytest.mark.synapse
@pytest.mark.training
@pytest.mark.serialization
def test_persistent_synapse_fault_survives_reload_with_no_campaign(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A persistent synapse fault is carried by the weight matrix itself,
    so it needs no hook to be delivered: save_net()/load_net() reproduces
    the trained faulty net's output with the originating campaign deleted."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)
    x, _ = next(iter(test_loader))

    site = (0, 0, 0, 0)
    cmpn.inject(sff.Fault(StuckSynapse(0.42), sff.FaultSite('SF1', site)), round_idx=0)

    spike_loss = snn.loss(net_params).to(next(dense_net.net.parameters()).device)
    faulties = cmpn.run_train(
        2, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )
    expected_output = faulties[0](x).clone()

    fpath = cmpn.save_net(0)
    device = next(dense_net.net.parameters()).device
    del cmpn

    loaded = sfi.Campaign.load_net(fpath, dense_net.net, device)

    assert loaded.SF1.weight[site].item() == pytest.approx(0.42)
    assert torch.equal(loaded(x), expected_output)


@pytest.mark.neuron
@pytest.mark.training
def test_neuron_fault_participates_in_the_backward_pass(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A neuron fault is delivered by overwriting the next layer's incoming
    spikes in place, and that overwrite sits inside the autograd graph
    rather than under no_grad as the synapse hooks do, so it has to be
    differentiated through. The dead site's activation is then a constant:
    no gradient can reach the weights feeding it, and its whole row of SF1
    must come out of training bit-identical, while the rows behind the
    surviving neurons train normally. A hook that detached, or masked
    outside the graph, would leave the dead row learning as though its
    neuron were alive -- which no forward-only check can see."""
    device = next(dense_net.net.parameters()).device
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    train_loader, test_loader = tiny_loaders(dense_net.shape_in, batch_size=8)
    x, y = next(iter(train_loader))

    # A site that never fires would make DeadNeuron a no-op regardless of
    # whether the fault hook actually works, so a genuinely active one is
    # searched for rather than assumed.
    golden_sf1 = cmpn.golden(x, 0, 0)
    site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    assert_active(golden_sf1, (slice(None), site, 0, 0, slice(None)))

    w_before = dense_net.net.SF1.weight.detach().clone()
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0))), round_idx=0)

    spike_loss = snn.loss(net_params).to(device)
    faulties = cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.SGD(params, lr=0.1), progress_mode='silent'
    )

    moved = [not torch.equal(faulties[0].SF1.weight[c], w_before[c]) for c in range(4)]
    assert moved == [c != site for c in range(4)], (
        f'Only the dead neuron\'s row should be frozen, but rows moved: {moved} (dead: {site}).'
    )

    # The same run reproduced without spikefi, by masking the identical
    # channel on SF2's own pre-hook, must land on the same weights.
    hand_net = deepcopy(dense_net.net)

    # Returns None, so the tuple is modified in place rather than replaced,
    # which is how the framework's own neuron pre-hook delivers the fault.
    def _mask_site(_: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        inputs[0][:, site, 0, 0, :] = 0.0

    handle = hand_net.SF2.register_forward_pre_hook(_mask_site)
    optimizer = torch.optim.SGD(hand_net.parameters(), lr=0.1)
    output = hand_net(x)
    one_hot = torch.zeros_like(output[..., :1]).scatter_(1, y.view(-1, 1, 1, 1, 1), 1.0)
    optimizer.zero_grad()
    spike_loss.numSpikes(output, one_hot).backward()
    optimizer.step()
    handle.remove()

    assert torch.allclose(faulties[0].SF1.weight, hand_net.SF1.weight, atol=1e-6)
    assert torch.allclose(faulties[0].SF2.weight, hand_net.SF2.weight, atol=1e-6)
