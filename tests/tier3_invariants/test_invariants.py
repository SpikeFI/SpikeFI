"""Tier 3 — metamorphic invariants: agreement across optimization levels,
late-start index correctness, duplicate-fault-application algebra, round
isolation across injection order, golden/faulty weight invariants, rgroups
metadata, and eject() semantics.
"""


from collections.abc import Callable

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import (
    bfl_value, DeadNeuron, DeadSynapse, mul_value, qua_value, ThresholdFaultNeuron
)
from spikefi.utils.quantization import qargs_from_range

from nets import NetSpec
from helpers import capture_run_outputs


def _find_nonvacuous_sites(
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        dense_net: NetSpec,
        slayer: spikeLayer,
        model_factory: Callable[[], sff.FaultModel],
        layer: str,
        site_candidates: list[tuple],
        x: Tensor,
        golden_out: Tensor,
        count: int
) -> list[tuple]:
    """Scans site_candidates for `count` distinct sites whose single-fault
    output actually differs from golden, so the round-isolation oracle below
    is never accidentally vacuous."""
    found = []
    for site in site_candidates:
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.inject(sff.Fault(model_factory(), sff.FaultSite(layer, site)), round_idx=0)
        cmpn._pre_run(sfi.CampaignOptimization.O0)
        cmpn.r_idx_ref.r = 0
        out = cmpn.faulty(x)
        if not torch.equal(out, golden_out):
            found.append(site)
            if len(found) == count:
                break
    assert len(found) == count, f'Could not find {count} non-vacuous sites on {layer}.'
    return found


@pytest.mark.optimization
def test_cross_optimization_agreement_at_zero_tolerance(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """O0..O4 must agree exactly on every round's per-round statistics at
    es_tol=0: accuracy (correctSamples/numSamples), average loss (a raw,
    per-spike-sensitive signal unlike accuracy alone) and N_critical."""
    # A fault confined to SF1 alone (unlike the three SF2 rounds below) is
    # the only kind that leaves both trailing layers fault-free, so it is
    # the only one for which early stop is ever enabled.
    # SF2's own threshold is rarely crossed by this tiny net's
    # default-initialized weights, so it is amplified for a genuinely
    # varied, non-degenerate golden prediction across the batch.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn0 = make_campaign(dense_net.net, dense_net.shape_in, slayer)

    device = next(dense_net.net.parameters()).device
    n_samples = 16
    generator = torch.Generator(device=device).manual_seed(42)
    x = (torch.rand(n_samples, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    golden_pred = cmpn0.golden(x).sum(dim=(2, 3, 4)).argmax(dim=1)
    # Labelled to match golden's own prediction, so any accuracy drop below
    # is attributable to the faults alone.
    y = golden_pred.clone()
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    results = {}
    for opt in (
        sfi.CampaignOptimization.O0, sfi.CampaignOptimization.O1,
        sfi.CampaignOptimization.O2, sfi.CampaignOptimization.O4
    ):
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (0, 0, 0))), round_idx=0)
        cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (1, 0, 0))))
        cmpn.then_inject(
            sff.Fault(DeadSynapse(), [sff.FaultSite('SF2', (2, c, 0, 0)) for c in range(4)])
        )
        cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (2, 0, 0))))
        n_critical = cmpn.run(
            loader, es_tol=0, opt=opt, compute_critical=True, progress_mode='silent'
        )
        results[opt] = (
            [round(p.testing.accuracyLog[0], 6) for p in cmpn.performance],
            [round(p.testing.lossLog[0], 6) for p in cmpn.performance],
            n_critical.tolist()
        )

    baseline = results[sfi.CampaignOptimization.O0]
    assert len(set(baseline[0])) > 1, 'The three rounds produced the same accuracy; not a real check.'
    assert any(n > 0 for n in baseline[2]), 'No round was ever misclassified; N_critical is vacuously 0.'
    for opt, result in results.items():
        assert result == baseline, f'{opt} disagreed with O0: {result} != {baseline}'


@pytest.mark.optimization
def test_late_start_index_not_advanced_for_parametric_fault_on_late_start_layer(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A round carrying an OUTPUT and a PARAMETER fault on the same
    (late-start) layer must not advance late_start_idx past it: a
    parametric fault is evaluated on its own layer's forward hook, so
    skipping ahead would leave no stashed value to consume."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    round = sff.FaultRound([
        sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))),
        sff.Fault(ThresholdFaultNeuron(1.5), sff.FaultSite('SF1', (1, 0, 0)))
    ])
    oround = round.optimized(cmpn.layers_info, late_start_en=True, early_stop_en=True)

    assert oround.late_start_name == 'SF1'
    assert oround.late_start_idx == cmpn.layers_info.index('SF1')


@pytest.mark.optimization
def test_late_start_index_advanced_for_neuronal_only_non_parametric_round(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A round whose only fault is a plain OUTPUT fault has no dummy layer
    to stash from, so late start can safely skip straight to the layer
    where the neuron perturb pre-hook actually lives: the one following
    the faulty layer."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    round = sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0)))])
    oround = round.optimized(cmpn.layers_info, late_start_en=True, early_stop_en=True)

    assert oround.late_start_name == 'SF2'
    assert oround.late_start_idx == cmpn.layers_info.index('SF2')


@pytest.mark.optimization
def test_early_stop_layer_is_the_last_faulty_layer_not_the_first(
        three_layer_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Early stop compares against golden at the layer after the round's
    *last* faulty one, since that is the first point at which every fault
    has been applied. Taking the first faulty layer instead would let a
    sample that reconverged there be handed golden's output while the
    round's later faults were never applied at all -- reporting them as
    harmless. Needs three layers to show: on a two-layer net the first and
    last faulty layer of an early-stoppable round are always the same. The
    round carries three keys rather than two, so stopping partway through
    them is a distinguishable answer as well."""
    cmpn = make_campaign(three_layer_net.net, three_layer_net.shape_in, slayer)
    round = sff.FaultRound([
        sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))),
        sff.Fault(ThresholdFaultNeuron(1.5), sff.FaultSite('SF1', (1, 0, 0))),
        sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (1, 0, 0)))
    ])
    oround = round.optimized(cmpn.layers_info, late_start_en=True, early_stop_en=True)

    assert oround.early_stop_en
    assert oround.early_stop_name == 'SF2'
    assert oround.early_stop_idx == cmpn.layers_info.index('SF2')
    # The first faulty layer is a different layer here, so naming it instead
    # would be a distinguishable answer rather than the same one twice.
    assert oround.early_stop_name != 'SF1'


@pytest.mark.optimization
def test_mixed_output_and_parametric_round_agrees_across_optimization(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """The scenario the previous two tests guard the index for: a round
    mixing an OUTPUT and a PARAMETER fault on the same late-start layer
    must run cleanly under full optimization and agree with the
    unoptimized baseline, not just report a plausible-looking late_start_idx."""
    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(11)
    x = (torch.rand(8, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(8, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    def _mixed_round() -> sff.FaultRound:
        return sff.FaultRound([
            sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))),
            sff.Fault(ThresholdFaultNeuron(1.5), sff.FaultSite('SF1', (1, 0, 0)))
        ])

    outputs = {}
    for opt in (sfi.CampaignOptimization.O0, sfi.CampaignOptimization.O4):
        cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
        cmpn.rounds = [_mixed_round(), sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (2, 0, 0)))])]
        outputs[opt] = capture_run_outputs(cmpn, loader, es_tol=0, opt=opt, progress_mode='silent')

    assert torch.equal(outputs[sfi.CampaignOptimization.O0][0], outputs[sfi.CampaignOptimization.O4][0])


@pytest.mark.synapse
def test_duplicate_application_of_a_set_value_fault_is_idempotent(
        dense_net: NetSpec
) -> None:
    """A set_value-based fault model (e.g. DeadSynapse) always overwrites
    with the same constant regardless of the current value, so applying it
    twice in a row is identical to applying it once."""
    model = DeadSynapse()
    w = torch.tensor(0.37)

    once = model.perturb(w)
    twice = model.perturb(once)

    assert twice == once == 0.


@pytest.mark.synapse
def test_duplicate_application_of_a_mul_value_fault_composes(
        dense_net: NetSpec
) -> None:
    """Unlike set_value, mul_value's effect depends on the current value, so
    two applications compose multiplicatively rather than collapsing to one:
    perturb(perturb(w)) == w * rho**2, not w * rho."""
    rho = 1.5
    model = sff.FaultModel(sff.FaultTarget.WEIGHT, mul_value, rho)
    w = torch.tensor(0.4)

    once = model.perturb(w)
    twice = model.perturb(once)

    assert twice == pytest.approx(w.item() * rho ** 2)
    assert twice != pytest.approx(once.item())


@pytest.mark.synapse
def test_duplicate_application_of_a_bfl_value_fault_is_a_quantizing_involution_not_identity(
        dense_net: NetSpec
) -> None:
    """bfl_value flips a bit in a weight's quantized integer representation
    and dequantizes it, so applying it twice flips the same bit back --
    but through a float, so the result is the *quantized* original value,
    not the original float itself."""
    dtype = torch.qint8
    scale, zero_point = qargs_from_range(-2.0, 2.0, dtype)
    w = torch.tensor(0.37)

    # A bit that is currently 0 flips to 1 on the first application and
    # back to 0 on the second: an OR-instead-of-XOR bug would set it once
    # and then leave it stuck at 1, so starting from an unset bit is what
    # makes the round trip discriminating.
    original_int_repr = torch.quantize_per_tensor(w, scale, zero_point, dtype).int_repr()
    unsigned_repr = original_int_repr.item() & 0xFF
    bit = next(b for b in range(8) if not (unsigned_repr >> b) & 1)

    once = bfl_value(w, bit, scale, zero_point, dtype)
    twice = bfl_value(once, bit, scale, zero_point, dtype)
    quantized_original = qua_value(w, scale, zero_point, dtype)

    assert torch.equal(twice, quantized_original)
    assert not torch.equal(twice, w)
    assert abs(w.item() - quantized_original.item()) <= scale / 2


@pytest.mark.neuron
def test_duplicate_site_and_model_injection_merges_into_one_fault(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Injecting the same (site, model) pair twice, in two separate inject()
    calls, must not create a duplicate site: FaultRound keys faults by
    (layer, model) and each Fault's sites are a set, so the second
    injection is absorbed into the first."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    site = sff.FaultSite('SF1', (0, 0, 0))

    cmpn.inject(sff.Fault(DeadNeuron(), site), round_idx=0)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)

    faults = cmpn.rounds[0].get_faults()
    assert len(faults) == 1
    assert len(faults[0].sites) == 1


@pytest.mark.neuron
@pytest.mark.synapse
@pytest.mark.parametric
@pytest.mark.parametrize('target, model_factory, site_candidates', [
    pytest.param(
        'OUTPUT', DeadNeuron,
        [(o, 0, 0) for o in range(3)], id='OUTPUT'
    ),
    pytest.param(
        'WEIGHT', DeadSynapse,
        [(o, c, 0, 0) for o in range(3) for c in range(4)], id='WEIGHT'
    ),
    pytest.param(
        'PARAMETER', lambda: ThresholdFaultNeuron(1.5),
        [(o, 0, 0) for o in range(3)], id='PARAMETER'
    ),
])
def test_round_isolation_is_independent_of_injection_order(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        target: str,
        model_factory: Callable[[], sff.FaultModel],
        site_candidates: list[tuple]
) -> None:
    """Each round's own result must not depend on what other round ran
    before it in the same campaign. Checked by running two single-fault rounds
    in both orders and comparing each round's own output across the swap."""
    # SF2's own threshold is rarely crossed by this tiny net's
    # default-initialized weights, so it is amplified and the fault is
    # placed on SF2 itself (rather than SF1, one layer upstream) to get a
    # real, non-vacuous effect on the network's own output.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn0 = make_campaign(dense_net.net, dense_net.shape_in, slayer)

    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(42)
    x = (torch.rand(4, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(4, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    golden_out = cmpn0.golden(x)

    site_a, site_b = _find_nonvacuous_sites(
        make_campaign, dense_net, slayer, model_factory, 'SF2', site_candidates, x, golden_out, count=2
    )

    def _round(site: tuple) -> sff.FaultRound:
        return sff.FaultRound([sff.Fault(model_factory(), sff.FaultSite('SF2', site))])

    cmpn_ab = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn_ab.rounds = [_round(site_a), _round(site_b)]
    out_ab = capture_run_outputs(cmpn_ab, loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    cmpn_ba = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn_ba.rounds = [_round(site_b), _round(site_a)]
    out_ba = capture_run_outputs(cmpn_ba, loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    assert torch.equal(out_ab[0], out_ba[1]), f'{target} round A depends on injection order.'
    assert torch.equal(out_ab[1], out_ba[0]), f'{target} round B depends on injection order.'


@pytest.mark.synapse
def test_golden_state_dict_unchanged_after_run(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """golden is the campaign's own reference network and must never be
    mutated by fault injection or by run(), regardless of fault target."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(3)
    x = (torch.rand(4, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(4, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    before = {k: v.clone() for k, v in cmpn.golden.state_dict().items()}

    cmpn.inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF1', (0, 0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))))
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    # golden and faulty must be genuinely separate objects, not just
    # equal-by-value after restore hooks put weights back: a shared
    # reference would still pass the before/after comparison below.
    assert cmpn.golden is not cmpn.faulty
    after = cmpn.golden.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"golden's '{k}' changed after run()."


@pytest.mark.synapse
def test_faulty_weights_restored_to_golden_after_run(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A WEIGHT fault's perturbed value is only ever active during its own
    round's forward pass: once run() completes, DispatchingSynapseRestoreHook
    must have put every weight back, so faulty's weights end up
    bit-identical to golden's."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    device = next(dense_net.net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(3)
    x = (torch.rand(4, *dense_net.shape_in, 16, device=device, generator=generator) < 0.3).float()
    y = torch.zeros(4, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    cmpn.inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF1', (0, 0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF2', (0, 0, 0, 0))))
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    assert torch.equal(cmpn.faulty.SF1.weight, cmpn.golden.SF1.weight)
    assert torch.equal(cmpn.faulty.SF2.weight, cmpn.golden.SF2.weight)


@pytest.mark.optimization
def test_rgroups_are_grouped_and_sorted_by_late_start_layer(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """rgroups buckets round indices by their oround's late_start_name and
    orders the buckets by that layer's position in the network, not by
    insertion order: round 0 here lands in the later 'SF2' bucket (its
    neuronal-only fault advances late start past SF1) while rounds 1 and 2,
    inserted after it, land in the earlier 'SF1' bucket."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.rounds = [
        sff.FaultRound([sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0)))]),
        sff.FaultRound(),  # empty (golden) round
        sff.FaultRound([sff.Fault(DeadSynapse(), sff.FaultSite('SF1', (0, 0, 0, 0)))]),
    ]
    cmpn._pre_run(sfi.CampaignOptimization.FO)

    assert cmpn.rgroups == {'SF1': [1, 2], 'SF2': [0]}
    # dict equality ignores key order, so the content check above would
    # still pass even if rgroups kept plain insertion order ('SF2' first,
    # from round 0) instead of sorting by layer index -- checked explicitly.
    assert list(cmpn.rgroups.keys()) == ['SF1', 'SF2']


@pytest.mark.neuron
def test_eject_by_fault_drops_only_the_rounds_it_empties(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Ejecting a fault present in several rounds removes it from all of
    them, but only drops the rounds that become empty as a result -- a
    round that still holds another fault survives, shrunk."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    shared = sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0)))
    other = sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0)))

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject([
        sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))),
        sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0)))
    ])
    assert len(cmpn.rounds) == 2

    cmpn.eject(faults=[shared])

    assert len(cmpn.rounds) == 1
    remaining_sites = {s.position for f in cmpn.rounds[0].get_faults() for s in f.sites}
    assert remaining_sites == {other.get_sites()[0].position}


@pytest.mark.neuron
def test_eject_by_round_index_pops_the_round_unconditionally(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """eject(round_idx=r) with no faults given removes round r outright,
    regardless of whether it still holds faults."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))))
    assert len(cmpn.rounds) == 2

    cmpn.eject(round_idx=0)

    assert len(cmpn.rounds) == 1
    remaining_sites = {s.position for f in cmpn.rounds[0].get_faults() for s in f.sites}
    assert remaining_sites == {(1, 0, 0)}


@pytest.mark.neuron
def test_eject_to_nothing_leaves_exactly_one_empty_round(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A campaign is never left with zero rounds: ejecting everything drops
    every round but one empty (golden) round is appended back."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))))

    cmpn.eject()

    assert len(cmpn.rounds) == 1
    assert len(cmpn.rounds[0]) == 0
