"""Tier 5 — serialization: save()/load() round-trip fidelity, export()'s
CampaignData snapshot, the grad-free cached fault state both rely on being
pickle-safe, and restore()'s exact (rounds-only) carrying contract.
"""


from collections.abc import Callable, Generator
from contextlib import contextmanager

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
import spikefi.utils.io as sfio
from spikefi.models import DeadNeuron, DeadSynapse, SaturatedSynapse

from nets import NetSpec
from helpers import assert_active, assert_differs, capture_run_outputs


@contextmanager
def _spy_saved_path() -> Generator[dict[str, str]]:
    """Campaign.save()/CampaignData.save() both return None and the suite
    forbids reconstructing or globbing a fixed filename, so this wraps
    spikefi.utils.io.make_res_filepath to record the actual path it
    returns, restoring the original function afterwards."""
    captured: dict[str, str] = {}
    original = sfio.make_res_filepath

    def _wrapped(fname: str, rename: bool = False) -> str:
        path = original(fname, rename)
        captured['path'] = path
        return path

    sfio.make_res_filepath = _wrapped
    try:
        yield captured
    finally:
        sfio.make_res_filepath = original


@pytest.mark.serialization
@pytest.mark.neuron
@pytest.mark.synapse
def test_save_load_round_trip_reproduces_identical_per_round_results(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """save() -> Campaign.load() -> re-run reproduces the exact same
    correctSamples, lossSum and raw output tensor for every round as the
    live campaign that was saved. Two rounds (one OUTPUT, one WEIGHT
    fault) so the multi-round (optimized) run path is exercised on both
    sides of the round trip."""
    # SF2's own threshold is rarely crossed by this tiny net's
    # default-initialized weights, so it is amplified for a genuinely
    # non-degenerate (not all-zero) output to compare across the round trip.
    with torch.no_grad():
        dense_net.net.SF2.weight.mul_(20)
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0], shuffle=False)
    activity = golden_activity(cmpn, x)

    neuron_site = next(c for c in range(4) if activity['SF1'][:, c, 0, 0, :].sum() > 0)
    assert_active(activity['SF1'], (slice(None), neuron_site, 0, 0, slice(None)))
    weight_site = (0, 0, 0, 0)
    assert_differs(dense_net.net.SF2.weight[weight_site].item(), 0.)

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (neuron_site, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF2', weight_site)))

    spike_loss = snn.loss(net_params).to(x.device)
    live_outputs = capture_run_outputs(
        cmpn, loader, spike_loss=spike_loss,
        opt=sfi.CampaignOptimization.FO, progress_mode='silent'
    )
    live_correct = [p.testing.correctSamples for p in cmpn.performance]
    live_loss = [p.testing.lossSum for p in cmpn.performance]

    # Guard: the two rounds are actually distinguishable from each other, so
    # a save/load bug that silently collapsed or aliased rounds would not
    # slip past the identical-per-round comparison below.
    assert not torch.equal(live_outputs[0], live_outputs[1]), (
        'The two rounds produced the same output; not a real check.'
    )

    with _spy_saved_path() as saved:
        cmpn.save()
    path = saved['path']

    restored = sfi.Campaign.load(path)
    restored_outputs = capture_run_outputs(
        restored, loader, spike_loss=spike_loss,
        opt=sfi.CampaignOptimization.FO, progress_mode='silent'
    )
    restored_correct = [p.testing.correctSamples for p in restored.performance]
    restored_loss = [p.testing.lossSum for p in restored.performance]

    assert restored_correct == live_correct
    assert restored_loss == live_loss
    for live_out, restored_out in zip(live_outputs, restored_outputs):
        assert torch.equal(live_out, restored_out)


@pytest.mark.serialization
@pytest.mark.synapse
def test_export_with_synapse_faults_matches_live_campaign_state(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """export() succeeds with a synapse fault, and the returned
    CampaignData's rounds, orounds, rgroups, performance and duration all
    match the live campaign's own state after run() -- a snapshot of it,
    not a fresh reconstruction."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0], shuffle=False)

    weight_site = (0, 0, 0, 0)
    assert_differs(dense_net.net.SF1.weight[weight_site].item(), 0.)

    cmpn.inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF1', weight_site)), round_idx=0)
    spike_loss = snn.loss(net_params).to(x.device)
    cmpn.run(loader, spike_loss=spike_loss, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    # Guard: the live campaign's own runtime state is non-trivially
    # populated (not the default single empty round with nothing in
    # orounds/rgroups/performance), so the comparison below checks real
    # content rather than two empty shells.
    assert len(cmpn.rounds[0].get_faults()) == 1
    assert cmpn.orounds and cmpn.rgroups and cmpn.performance

    data = cmpn.export()

    assert data.rounds == cmpn.rounds
    assert len(data.orounds) == len(cmpn.orounds)
    for data_oround, live_oround in zip(data.orounds, cmpn.orounds):
        assert data_oround.late_start_name == live_oround.late_start_name
        assert data_oround.early_stop_name == live_oround.early_stop_name
    assert data.rgroups == cmpn.rgroups
    assert len(data.performance) == len(cmpn.performance)
    for data_perf, live_perf in zip(data.performance, cmpn.performance):
        assert data_perf.testing.correctSamples == live_perf.testing.correctSamples
        assert data_perf.testing.lossSum == live_perf.testing.lossSum
    assert data.duration == cmpn.duration


@pytest.mark.serialization
@pytest.mark.neuron
def test_restore_carries_only_rounds_and_reproduces_live_output(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """CampaignData.restore() rebuilds a fresh Campaign from golden/
    layers_info/slayer/name and hands it `rounds` -- but does not carry
    `orounds`, `rgroups`, `performance` or `duration` across, those being
    run()-time artifacts of the live campaign rather than reconstructible
    state. The restored campaign is nonetheless runnable: run again, its
    own rounds reproduce the exact same per-round output as the live one."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0], shuffle=False)
    activity = golden_activity(cmpn, x)

    site = next(c for c in range(4) if activity['SF1'][:, c, 0, 0, :].sum() > 0)
    assert_active(activity['SF1'], (slice(None), site, 0, 0, slice(None)))

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (site, 0, 0))), round_idx=0)
    live_outputs = capture_run_outputs(
        cmpn, loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent'
    )

    # Guard: the live campaign's runtime state is actually populated, so
    # what follows checks a real absence in the restored campaign, not two
    # objects that were both empty to begin with.
    assert cmpn.orounds and cmpn.rgroups and cmpn.performance

    restored = cmpn.export().restore()

    assert restored.rounds == cmpn.rounds
    assert restored.orounds == []
    assert restored.rgroups == {}
    assert restored.performance == []
    assert restored.duration == 0.

    restored_outputs = capture_run_outputs(
        restored, loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent'
    )
    for live_out, restored_out in zip(live_outputs, restored_outputs):
        assert torch.equal(live_out, restored_out)


@pytest.mark.serialization
def test_restore_rebuilds_the_campaign_on_its_original_device(
        dense_net: NetSpec,
        slayer: spikeLayer,
        artifact_name: str
) -> None:
    """A campaign built on an explicitly named device comes back on that
    same device: restore() reconstructs a Campaign, whose own default would
    otherwise silently pick whatever device happens to be available."""
    explicit = torch.device('cuda', torch.cuda.current_device())
    # Guard: the default a fresh Campaign would fall back to is a different
    # device object, so the assertion below cannot pass by coincidence.
    assert explicit != torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cmpn = sfi.Campaign(
        dense_net.net, dense_net.shape_in, slayer,
        name=artifact_name, device=explicit
    )
    assert cmpn.device == explicit

    restored = cmpn.export().restore()

    assert restored.device == explicit
    assert next(restored.golden.parameters()).device == explicit


@pytest.mark.serialization
def test_restore_falls_back_when_the_captured_device_is_unavailable(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """Data read on a different machine than the one that produced it may
    name a device that no longer exists here (a CPU-only machine reading
    GPU-captured data, or one with fewer GPUs) -- restore() must fall back
    to Campaign's own default rather than propagate a CUDA error trying to
    honor a device that isn't there. An out-of-range CUDA index stands in
    for 'unavailable', since this device really is absent regardless of
    what hardware the test itself happens to run on."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    data = cmpn.export()

    # Guard: this index is out of range on any real machine, so the
    # fallback below is exercised for a genuine reason, not a coincidence.
    assert data.device.type != 'cuda' or torch.cuda.device_count() < 99
    data.device = torch.device('cuda', 99)

    restored = data.restore()

    assert restored.device != data.device
    # A bare torch.device('cuda') and an explicitly-indexed torch.device
    # ('cuda', 0) refer to the same physical placement but do not compare
    # equal, so only .type is checked here.
    assert next(restored.golden.parameters()).device.type == restored.device.type


@pytest.mark.serialization
@pytest.mark.synapse
def test_restore_places_fault_args_on_the_resolved_device(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """A fault's own Tensor-valued args are normalized to CPU for export,
    since they are permanent configuration rather than scratch state a run()
    would recompute. restore() must place them back on whatever device the
    restored campaign actually resolves to or a later perturb() mixing a CPU
    arg with a non-CPU weight risks failing (masked here only by a 0-dim
    Tensor's device-match exemption, which is why the check below is on
    .device directly rather than on perturb() actually succeeding)."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)

    weight = cmpn.golden.SF1.weight.detach()
    Q1 = torch.quantile(weight, 0.25)
    Q3 = torch.quantile(weight, 0.75)
    cmpn.inject(sff.Fault(SaturatedSynapse(Q1, Q3), sff.FaultSite('SF1', (0, 0, 0, 0))), round_idx=0)

    data = cmpn.export()
    exported_fault = data.rounds[0].get_faults()[0]
    # Guard: args really were normalized to CPU by export, so restore()
    # placing them back on a real device below is an actual, checkable
    # change rather than a no-op.
    assert all(arg.device.type == 'cpu' for arg in exported_fault.model.args)

    restored = data.restore()
    restored_fault = restored.rounds[0].get_faults()[0]

    assert all(arg.device.type == restored.device.type for arg in restored_fault.model.args)
