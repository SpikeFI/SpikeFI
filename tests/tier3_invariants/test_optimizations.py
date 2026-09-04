"""Tier 3 — do the optimizations actually optimize: late start must skip
real work (not merely produce the right answer), and the early-stop mask's
exact threshold, its growth with es_tol, its full-tolerance limit, its
irrelevance below O2, and the partial-batch code path all match the
documented semantics.
"""


from collections.abc import Callable

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, SaturatedNeuron

from nets import NetSpec
from helpers import capture_invocation_widths, capture_run_outputs, count_invocations_during_run


def _partial_early_stop_setup(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> tuple[Callable[[], sfi.Campaign], Tensor, DataLoader, Tensor]:
    """SF2's threshold is rarely crossed by this tiny net's
    default-initialized weights, which leaves golden silent and turns every
    comparison against it into a comparison against zeros, so it is
    amplified first -- once per test, which is why the campaign comes back
    as a factory rather than as an instance. On this seed the two rounds
    then diverge from golden on a different part of each batch of four --
    round 0 on none of the first batch, two of the second, two of the third
    and one of the last -- putting both of early stop's branches inside a
    single batch."""
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)

    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(9)
    x = (torch.rand(16, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(16, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    golden_final = make_campaign(dense_net.net, dense_net.shape_in, slayer).golden(x)

    def _new_campaign() -> sfi.Campaign:
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.rounds = [
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0)))]),
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (2, 0, 0)))]),
        ]
        return cmpn

    return _new_campaign, x, loader, golden_final


@pytest.mark.optimization
def test_late_start_skips_invoking_layers_before_the_late_start_index(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A round whose only faults are plain OUTPUT faults on SF1 advances
    late_start_idx to SF2. Under O2/O4 this must show up as SF1 genuinely
    never being called during run()."""
    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(5)
    x = (torch.rand(16, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(16, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    n_batches = len(loader)

    def _rounds() -> list[sff.FaultRound]:
        return [
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (2, 0, 0)))]),
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (3, 0, 0)))]),
        ]

    for opt in (sfi.CampaignOptimization.O0, sfi.CampaignOptimization.O1):
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.rounds = _rounds()
        counts = count_invocations_during_run(
            cmpn, ['SF1', 'SF2'], loader, es_tol=0, opt=opt, progress_mode='silent'
        )
        assert counts == {'SF1': n_batches * 2, 'SF2': n_batches * 2}, f'{opt}: {counts}'

    for opt in (sfi.CampaignOptimization.O2, sfi.CampaignOptimization.O4):
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.rounds = _rounds()
        counts = count_invocations_during_run(
            cmpn, ['SF1', 'SF2'], loader, es_tol=0, opt=opt, progress_mode='silent'
        )
        assert counts == {'SF1': 0, 'SF2': n_batches * 2}, f'{opt}: {counts}'


@pytest.mark.optimization
def test_early_stop_mask_matches_the_diff_threshold_exactly_at_every_es_tol(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A sample must early-stop exactly when its divergence from golden at
    the early-stop layer is at most es_tol. Two saturated-neuron rounds on
    this seed give per-sample divergences spanning 0..3 and 1..2, so sweeping es_tol
    over 0..3 walks a four-step staircase rather than flipping the whole
    batch at once, and every step is checked against the threshold applied
    here independently of the framework's own comparison."""
    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(10)
    x = (torch.rand(16, *dense_net.shape_in, 16, device=device, generator=generator) < 0.7).float()
    y = torch.zeros(16, dtype=torch.long, device=device)
    # One sample per batch, so counting the continued-forward calls reads the
    # mask off per sample: a whole batch would only reveal whether at least
    # one of its samples kept going.
    loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)

    def _rounds() -> list[sff.FaultRound]:
        return [
            sff.FaultRound([sff.Fault(SaturatedNeuron(), sff.FaultSite('SF1', (c, 0, 0)))])
            for c in (1, 2)
        ]

    # The left-hand side of the framework's 'divergence <= es_tol' contract,
    # taken from the same golden and faulty layer ranges the optimization
    # runs over. Only the comparison against es_tol is re-derived below --
    # that is the part under test.
    probe = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    probe.rounds = _rounds()
    probe._pre_run(sfi.CampaignOptimization.O4)
    golden_spikes = [x]
    for layer_idx in range(len(probe.layers_info)):
        golden_spikes.append(probe.golden(golden_spikes[layer_idx], layer_idx, layer_idx))

    divergences: list[Tensor] = []
    for r_idx in range(len(probe.rounds)):
        probe.r_idx_ref.r = r_idx
        oround = probe.orounds[r_idx]
        assert oround.early_stop_en, f'Round {r_idx} has early stop disabled; nothing to measure.'
        ls_idx, es_idx = oround.late_start_idx, oround.early_stop_idx
        next_out = probe.faulty(golden_spikes[ls_idx], ls_idx, es_idx + 1)
        divergences.append(
            torch.sum(next_out.ne(golden_spikes[es_idx + 2]), dim=(1, 2, 3, 4))
        )

    n_evaluations = len(probe.rounds) * len(x)
    stopped: list[int] = []
    for es_tol in (0, 1, 2, 3):
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.rounds = _rounds()
        counts = count_invocations_during_run(
            cmpn, ['tail'], loader, es_tol=es_tol,
            opt=sfi.CampaignOptimization.O4, progress_mode='silent'
        )
        expected = sum(int((d <= es_tol).sum()) for d in divergences)
        stopped.append(n_evaluations - counts['tail'])
        assert stopped[-1] == expected, (
            f'es_tol={es_tol}: {stopped[-1]} samples early-stopped, expected {expected}.'
        )

    # Non-vacuity: the sweep has to be an actual staircase, or the equality
    # above would hold just as well for a mask that ignores es_tol.
    assert stopped[0] > 0, 'Nothing stopped at es_tol=0; the exact-threshold case is untested.'
    assert stopped[-1] == n_evaluations
    assert len(set(stopped)) == len(stopped), f'es_tol did not move the mask at every step: {stopped}'


@pytest.mark.optimization
def test_es_tol_at_least_sample_size_reproduces_golden_output_exactly(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Once es_tol is at least as large as one sample's own element count,
    every sample's diff-from-golden count is trivially within tolerance, so
    early stop fires for the whole batch and the round's raw output is
    golden's output, exactly."""
    new_campaign, x, loader, golden_final = _partial_early_stop_setup(
        dense_net, slayer, make_campaign
    )
    numel_per_sample = golden_final[0].numel()

    outputs = capture_run_outputs(
        new_campaign(), loader, es_tol=numel_per_sample,
        opt=sfi.CampaignOptimization.O4, progress_mode='silent'
    )

    assert golden_final.any(), 'Golden is silent; equality with it would not check any content.'
    assert torch.equal(outputs[0], golden_final)


@pytest.mark.optimization
def test_es_tol_has_no_effect_below_O2(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """run() only forwards es_tol to the evaluate method at O2 and above; at
    O0/O1 it is never read at all, so an arbitrarily large es_tol must not
    change a single result there."""
    new_campaign, x, loader, golden_final = _partial_early_stop_setup(
        dense_net, slayer, make_campaign
    )

    outputs = {}
    for es_tol in (0, 10_000):
        outputs[es_tol] = capture_run_outputs(
            new_campaign(), loader, es_tol=es_tol,
            opt=sfi.CampaignOptimization.O0, progress_mode='silent'
        )

    assert torch.equal(outputs[0][0], outputs[10_000][0])
    # Non-vacuity: confirm this scenario does diverge from golden at O0, so
    # an es_tol that silently leaked through and forced early stop would
    # actually be caught by the comparison above.
    assert not torch.equal(outputs[0][0], golden_final)


@pytest.mark.optimization
def test_es_tol_is_inert_when_the_round_faults_the_last_two_layers(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Early stop needs two fault-free trailing layers to have anywhere to
    reconverge before the output, so a round faulting the output layer
    itself must disable it outright. es_tol then has nothing to act on, and
    the round must report its own faulty output however large the tolerance
    gets. Were the guard off by a layer, a wide tolerance would compare at
    the network's own output and substitute golden's there -- reporting the
    fault as harmless, which is exactly what the guard exists to prevent."""
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)

    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(9)
    x = (torch.rand(16, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(16, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    golden_final = make_campaign(dense_net.net, dense_net.shape_in, slayer).golden(x)

    # Killing a channel that never fires would leave the output identical to
    # golden, making the comparison below unable to see a wrongly substituted
    # one, so the faulted channels are searched for rather than assumed.
    active = [c for c in range(golden_final.shape[1]) if golden_final[:, c].sum() > 0]
    assert len(active) >= 2, f'Need two active output channels, found {active}.'

    def _new_campaign() -> sfi.Campaign:
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.rounds = [
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (c, 0, 0)))])
            for c in active[:2]
        ]
        return cmpn

    metadata = _new_campaign()
    metadata._pre_run(sfi.CampaignOptimization.O4)
    assert not metadata.orounds[0].early_stop_en
    assert metadata.orounds[0].early_stop_name is None

    outputs = {}
    for es_tol in (0, 10_000):
        outputs[es_tol] = capture_run_outputs(
            _new_campaign(), loader, es_tol=es_tol,
            opt=sfi.CampaignOptimization.O4, progress_mode='silent'
        )

    assert not torch.equal(outputs[0][0], golden_final), (
        'The fault leaves the output unchanged; substituting golden would not be visible.'
    )
    assert torch.equal(outputs[0][0], outputs[10_000][0])


@pytest.mark.optimization
def test_partial_batch_early_stop_continues_only_the_samples_that_diverge(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Batches here hold both reconverging and diverging samples at
    es_tol=0, so _evaluate_optimized has to take its early-stopped
    assignment and its torch.any(~early_stop) continued-forward branch
    within one batch. Each continued call must then carry only the diverging
    samples: an output-only check could not tell a narrowed call apart from
    a run that early-stopped nothing and recomputed the whole batch, since a
    sample that reconverges yields golden's result either way. Their results
    must also come back to their own slots, which is what the boolean-mask
    round trip in and out of the sub-batch can get wrong, and the stopped
    samples must carry golden's spikes rather than the zeros the output
    buffer starts as."""
    new_campaign, x, loader, golden_final = _partial_early_stop_setup(
        dense_net, slayer, make_campaign
    )

    # 'tail' is the first layer a continued forward reaches, so its
    # invocations are exactly the continued calls of the whole run, and their
    # widths are how many samples each one kept. Sorted, because the two
    # rounds interleave with the batches in run order.
    widths = capture_invocation_widths(
        new_campaign(), ['tail'], loader, es_tol=0,
        opt=sfi.CampaignOptimization.O4, progress_mode='silent'
    )
    assert sorted(widths['tail']) == [1, 2, 2, 3, 3, 3, 4], (
        f'Continued forwards did not carry the diverging samples alone: {widths["tail"]}.'
    )

    outputs = capture_run_outputs(
        new_campaign(), loader, es_tol=0,
        opt=sfi.CampaignOptimization.O4, progress_mode='silent'
    )
    last_batch = range(12, 16)
    matches_golden = [torch.equal(outputs[0][i], golden_final[i]) for i in last_batch]

    assert golden_final[last_batch].flatten(1).any(dim=1).all(), (
        'Golden is silent for some of the last batch; matching it would not check any content.'
    )
    assert matches_golden == [True, True, False, True], (
        f'Early stop and the continued forward disagree on which slot is whose: {matches_golden}.'
    )


@pytest.mark.optimization
def test_early_stop_skips_the_continued_forward_when_it_actually_reconverges(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Early stop producing the right answer is not enough on its own to
    prove it skips work: a broken threshold comparison that never actually
    triggers would still reach the correct output by falling through to a
    full recompute every time (see test_es_tol_increase_is_monotone... --
    an output-only oracle cannot tell the two apart). Checked directly by
    counting calls to 'tail', the layer the continued-forward call reaches
    in this two-layer net: es_tol=0 needs one call (for the one sample that
    does not reconverge), while an es_tol covering every sample's full
    element count needs none at all."""
    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(5)
    x = (torch.rand(16, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(16, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    def _rounds() -> list[sff.FaultRound]:
        return [
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (2, 0, 0)))]),
            sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (3, 0, 0)))]),
        ]

    cmpn0 = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    numel_per_sample = cmpn0.golden(x)[0].numel()

    cmpn_partial = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn_partial.rounds = _rounds()
    counts_partial = count_invocations_during_run(
        cmpn_partial, ['tail'], loader, es_tol=0, opt=sfi.CampaignOptimization.O4, progress_mode='silent'
    )

    cmpn_full = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn_full.rounds = _rounds()
    counts_full = count_invocations_during_run(
        cmpn_full, ['tail'], loader, es_tol=numel_per_sample,
        opt=sfi.CampaignOptimization.O4, progress_mode='silent'
    )

    assert counts_partial == {'tail': 1}
    assert counts_full == {'tail': 0}
