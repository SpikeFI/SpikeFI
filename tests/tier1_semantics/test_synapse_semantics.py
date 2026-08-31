"""Tier 1 — synapse (WEIGHT-target) fault semantics: differential agreement
with a hand-built mutant for every weight model, the dead-column/dead-neuron
cross-path oracle, DeadSynapse's exact end-to-end answer, BitflippedSynapse's
weight-value oracle, and post-round stash hygiene.
"""


from collections.abc import Callable
from copy import deepcopy

import pytest
import torch
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, DeadSynapse, PerturbedSynapse, StuckSynapse
from spikefi.utils.quantization import qargs_from_range

from nets import NetSpec
from helpers import hand_mutate_weight, run_round


SITE = (0, 0, 0, 0)


def _find_nonvacuous_sf1_site(
        dense_net: NetSpec,
        x: Tensor,
        golden_sf1: Tensor,
        build_model: Callable[[float], sff.FaultModel]
) -> tuple[tuple[int, int, int, int], sff.FaultModel]:
    """SF2's own threshold is high enough relative to this tiny net's
    default-initialized weights that its output rarely fires at all, no
    matter what SF1 does -- so this searches for a site whose effect is
    already visible one layer down, at SF1's own output, rather than
    risking every candidate site being masked by SF2 never firing."""
    out_channels, in_channels = dense_net.net.SF1.weight.shape[:2]
    for o in range(out_channels):
        for c in range(in_channels):
            site = (o, c, 0, 0)
            w = dense_net.net.SF1.weight[site].item()
            model = build_model(w)
            hand_net = deepcopy(dense_net.net)
            hand_mutate_weight(hand_net.SF1, site, model.perturb(w))
            hand_sf1 = hand_net.slayer.spike(hand_net.slayer.psp(hand_net.SF1(x)))
            if not torch.equal(hand_sf1, golden_sf1):
                return site, model
    raise AssertionError('No SF1 site has a visible effect for this model.')


@pytest.mark.synapse
@pytest.mark.parametrize('build_model', [
    pytest.param(lambda w: DeadSynapse(), id='DeadSynapse'),
    pytest.param(lambda w: StuckSynapse(w + 0.3), id='StuckSynapse'),
    pytest.param(lambda w: PerturbedSynapse(1.5), id='PerturbedSynapse'),
])
def test_weight_fault_matches_hand_mutant(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        build_model: Callable[[float], sff.FaultModel]
) -> None:
    """A WEIGHT fault's output is bit-identical to directly overwriting the
    same weight site by hand, bypassing SpikeFI's machinery entirely.
    Compared at SF1's own output, the layer the fault is actually on, since
    that is the layer this guarantee is about."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_sf1 = cmpn.golden(x, 0, 0)

    site, model = _find_nonvacuous_sf1_site(dense_net, x, golden_sf1, build_model)
    w_original = dense_net.net.SF1.weight[site].item()

    fault = sff.Fault(model, sff.FaultSite('SF1', site))
    cmpn.inject(fault, round_idx=0)
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    cmpn.r_idx_ref.r = 0
    faulty_sf1 = cmpn.faulty(x, 0, 0)

    hand_net = deepcopy(dense_net.net)
    hand_mutate_weight(hand_net.SF1, site, model.perturb(w_original))
    hand_sf1 = hand_net.slayer.spike(hand_net.slayer.psp(hand_net.SF1(x)))

    assert torch.equal(faulty_sf1, hand_sf1)


@pytest.mark.synapse
@pytest.mark.neuron
def test_dead_weight_column_matches_dead_neuron(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """Zeroing every SF2 weight fed by one SF1 output neuron (a 'dead
    column') must reproduce a DeadNeuron fault on that SF1 neuron
    bit-for-bit: two separate hook paths (WEIGHT pre/post hooks on SF2 vs.
    the OUTPUT pre-hook on SF1's following layer) computing the same thing,
    with no reference model needed. Unlike SF1 (fed directly by the raw
    input, which carries no FaultSite-addressable neuron of its own), SF2
    is fed by SF1's own output, so this only works one layer downstream."""
    # SF2's own threshold is rarely crossed by this tiny net's
    # default-initialized weights, so a column that reaches only silent
    # output neurons would make the comparison below vacuously true;
    # amplifying SF2 makes it responsive enough to search for one that isn't.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn_col = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_out = cmpn_col.golden(x)

    out_channels, in_channels = cmpn_col.golden.SF2.weight.shape[:2]
    column_out = None
    for c in range(in_channels):
        dead_column_sites = [sff.FaultSite('SF2', (o, c, 0, 0)) for o in range(out_channels)]
        cmpn_col.inject(sff.Fault(DeadSynapse(), dead_column_sites), round_idx=0)
        candidate = run_round(cmpn_col, 0, x)
        if not torch.equal(candidate, golden_out):
            column_out = candidate
            break
        cmpn_col.eject(round_idx=0)
    assert column_out is not None, 'No dead column has a visible effect on the output.'

    cmpn_neu = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn_neu.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (c, 0, 0))), round_idx=0)
    neuron_out = run_round(cmpn_neu, 0, x)

    assert torch.equal(column_out, neuron_out)


@pytest.mark.synapse
def test_dead_synapse_zeroes_the_output_row(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """DeadSynapse on every weight feeding output neuron o zeroes that
    neuron's entire row of the network's final output exactly -- on a row
    that actually fired beforehand, or zeroing it would prove nothing."""
    # See test_dead_weight_column_matches_dead_neuron: SF2 rarely crosses
    # threshold at default init, so it is amplified here too.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_out = cmpn.golden(x)

    out_channels, in_channels = cmpn.golden.SF2.weight.shape[:2]
    o = next(oc for oc in range(out_channels) if golden_out[:, oc].sum() > 0)
    row_sites = [sff.FaultSite('SF2', (o, c, 0, 0)) for c in range(in_channels)]
    cmpn.inject(sff.Fault(DeadSynapse(), row_sites), round_idx=0)
    faulty_out = run_round(cmpn, 0, x)

    assert faulty_out[:, o].sum() == 0


@pytest.mark.synapse
def test_bitflipped_synapse_uses_the_independently_computed_bfl_value(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """The weight actually used during forward for a BitflippedSynapse
    fault has exactly the named bit flipped in its quantized integer
    representation, checked via torch's own quantize_per_tensor."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)

    dtype = torch.qint8
    scale, zero_point = qargs_from_range(-2.0, 2.0, dtype)
    w_original = cmpn.golden.SF1.weight[SITE].detach().clone()
    original_int_repr = torch.quantize_per_tensor(w_original, scale, zero_point, dtype).int_repr()

    # A bit that is currently 0 flips to 1 under an OR-instead-of-XOR bug
    # exactly as it would under a correct flip, so the chosen bit must
    # actually be set beforehand for the check below to be discriminating.
    unsigned_repr = original_int_repr.item() & 0xFF
    bit = next(b for b in range(8) if (unsigned_repr >> b) & 1)

    fault = sff.Fault(
        sfi.fm.BitflippedSynapse(bit, scale, zero_point, dtype), sff.FaultSite('SF1', SITE)
    )
    cmpn.inject(fault, round_idx=0)
    cmpn._pre_run(sfi.CampaignOptimization.O0)

    used_weight = {}
    handle = cmpn.faulty.SF1.register_forward_pre_hook(
        lambda _, __: used_weight.__setitem__('w', cmpn.faulty.SF1.weight[SITE].detach().clone())
    )
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)
    handle.remove()

    used_int_repr = torch.quantize_per_tensor(used_weight['w'], scale, zero_point, dtype).int_repr()
    # XOR in unsigned byte space: int_repr is signed, so a flipped sign bit
    # (bit 7) would otherwise compare a negative Python int against 1 << 7.
    xor_bits = (original_int_repr.item() ^ used_int_repr.item()) & 0xFF
    assert xor_bits == (1 << bit)


@pytest.mark.synapse
def test_weight_fault_stash_is_not_none_after_a_round(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """After running a round, a WEIGHT fault's cached perturbed value is
    still populated (unlike a PARAMETER fault's, which unstore() clears),
    since the restore hook only reads it back, never clears it."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)

    cmpn.inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF1', SITE)), round_idx=0)
    run_round(cmpn, 0, x)

    installed = cmpn.rounds[0].grouped[('SF1', sff.FaultTarget.WEIGHT)][0]
    assert installed.model.perturbed is not None
