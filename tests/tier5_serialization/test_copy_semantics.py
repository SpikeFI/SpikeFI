"""Tier 5 — copy semantics of CampaignData's export(): which fields stay
identity-shared with the live campaign (Fault objects) versus which are
genuinely deep-copied and safe to mutate independently.
"""


from collections.abc import Callable
from copy import deepcopy

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, SaturatedSynapse, ThresholdFaultNeuron

from nets import NetSpec


@pytest.mark.serialization
@pytest.mark.neuron
def test_exported_rounds_and_orounds_share_fault_objects_but_not_with_live(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """CampaignData.rounds[r] and CampaignData.orounds[r] still share the
    same Fault objects with each other (`is` identity), exactly as the live
    campaign's own rounds[r]/orounds[r] do: two separate deepcopy calls
    would have split the pair and left the optimized view reading stale
    faults. The exported Fault is a copy of that shared pair, though, not
    the same object the live campaign holds."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0], shuffle=False)

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    # Baseline: the live campaign itself must share the Fault object between
    # its round and its oround, or the assertions below would be meaningless.
    live_fault = cmpn.rounds[0].get_faults()[0]
    live_ofault = cmpn.orounds[0].get_faults()[0]
    assert live_fault is live_ofault, (
        "Baseline invariant broken: the live campaign's own round and "
        'oround no longer share a Fault object.'
    )

    data = cmpn.export()

    exported_fault = data.rounds[0].get_faults()[0]
    exported_ofault = data.orounds[0].get_faults()[0]
    assert exported_fault is exported_ofault

    # The exported pair is its own copy, not the live campaign's objects.
    assert exported_fault is not live_fault


@pytest.mark.serialization
@pytest.mark.synapse
def test_exported_campaign_data_lives_entirely_on_cpu(
        dense_net: NetSpec,
        slayer: spikeLayer,
        device: torch.device,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """CampaignData must be loadable on a machine other than the one that
    produced it, so nothing in it may still be bound to a GPU-only device:
    golden is moved to CPU with its forward put back to the net class's
    own, slayer's own buffers are moved to CPU, and a fault's own fixed
    configuration when it happens to be Tensor-valued is moved to CPU too.
    Cached variables are cleared rather than moved, since run()/run_train()
    recompute them unconditionally before ever reading them back. The live
    campaign's own copies must be untouched either way."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0], shuffle=False)

    # Q1/Q3 derived from the live (GPU-resident) weight, so args is
    # genuinely Tensor-valued and on the same device as everything else,
    # not a plain float that torch.as_tensor would place on CPU regardless.
    weight = dense_net.net.SF1.weight.detach()
    weight_fault = sff.Fault(
        SaturatedSynapse(torch.quantile(weight, 0.25), torch.quantile(weight, 0.75)),
        sff.FaultSite('SF1', (0, 0, 0, 0))
    )

    golden_sf1 = golden_activity(cmpn, x)['SF1']
    param_site = next(c for c in range(4) if golden_sf1[:, c, 0, 0, :].sum() > 0)
    param_fault = sff.Fault(ThresholdFaultNeuron(2.0), sff.FaultSite('SF1', (param_site, 0, 0)))

    cmpn.inject([weight_fault, param_fault], round_idx=0)
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    live_weight_fault = next(
        f for f in cmpn.rounds[0].get_faults() if not f.model.is_parametric()
    )
    live_param_fault = next(
        f for f in cmpn.rounds[0].get_faults() if f.model.is_parametric()
    )

    # Guard: the live campaign's own state really is on a non-CPU-only-by-
    # coincidence device, keeps the forward_opt wrapper (a different
    # function than the net class's own forward), and really is
    # populated, so the checks below are actual, checkable changes rather
    # than trivial (already-CPU, already-None) coincidences.
    assert cmpn.golden.forward.__func__ is not type(cmpn.golden).forward
    assert next(cmpn.golden.parameters()).device.type == device.type
    assert cmpn.slayer.srmKernel.device.type == device.type
    assert all(arg.device.type == device.type for arg in live_weight_fault.model.args)
    assert live_weight_fault.model.original is not None
    assert live_weight_fault.model.perturbed is not None
    assert live_param_fault.model.flayer is not None
    assert live_param_fault.model.param_original is not None
    assert live_param_fault.model.param_perturbed is not None

    data = cmpn.export()
    exported_weight_fault = next(
        f for f in data.rounds[0].get_faults() if not f.model.is_parametric()
    )
    exported_param_fault = next(
        f for f in data.rounds[0].get_faults() if f.model.is_parametric()
    )

    assert data.golden.forward.__func__ is type(data.golden).forward
    assert next(data.golden.parameters()).device.type == 'cpu'
    assert data.slayer.srmKernel.device.type == 'cpu'
    assert all(arg.device.type == 'cpu' for arg in exported_weight_fault.model.args)
    assert exported_weight_fault.model.original is None
    assert exported_weight_fault.model.perturbed is None
    assert exported_param_fault.model.flayer is None
    assert exported_param_fault.model.param_original is None
    assert exported_param_fault.model.param_perturbed is None

    # The live campaign's own state is unaffected either way.
    assert cmpn.golden.forward.__func__ is not type(cmpn.golden).forward
    assert next(cmpn.golden.parameters()).device.type == device.type
    assert cmpn.slayer.srmKernel.device.type == device.type
    assert all(arg.device.type == device.type for arg in live_weight_fault.model.args)
    assert live_weight_fault.model.original is not None
    assert live_weight_fault.model.perturbed is not None
    assert live_param_fault.model.flayer is not None
    assert live_param_fault.model.param_original is not None
    assert live_param_fault.model.param_perturbed is not None


@pytest.mark.serialization
@pytest.mark.neuron
def test_mutating_exported_golden_weights_does_not_cross_with_live_campaign(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]
) -> None:
    """CampaignData.golden is a deep copy: mutating its weights after
    export() must not affect the live campaign's golden, and mutating the
    live campaign's golden afterwards must not affect the already-exported
    copy either."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    data = cmpn.export()

    live_before = cmpn.golden.SF1.weight.clone()
    data_before = data.golden.SF1.weight.clone()
    # Guard: the two start out equal (same net, same seed), so a divergence
    # detected below is caused by the mutation, not by a pre-existing
    # difference.
    assert torch.equal(live_before, data_before.to(cmpn.device))

    # Distinct, asymmetric deltas: if the two objects were secretly shared,
    # both mutations below would land on the same tensor and the final
    # values would betray it, however the two deltas happened to combine.
    data_delta, live_delta = 1.0, -3.0

    with torch.no_grad():
        data.golden.SF1.weight.add_(data_delta)
    assert torch.equal(cmpn.golden.SF1.weight, live_before), (
        "Mutating exported data's golden weights leaked into the live campaign."
    )

    with torch.no_grad():
        cmpn.golden.SF1.weight.add_(live_delta)
    assert torch.equal(cmpn.golden.SF1.weight, live_before + live_delta), (
        "Mutating the live campaign's golden weights leaked into exported data."
    )
    assert torch.equal(data.golden.SF1.weight, data_before.to(data.golden.SF1.weight.device) + data_delta), (
        "Mutating the live campaign's golden weights leaked into exported data."
    )


@pytest.mark.serialization
@pytest.mark.neuron
def test_mutating_exported_rgroups_and_performance_does_not_cross_with_live_campaign(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """CampaignData.rgroups and .performance are deep copies of the live
    campaign's own: mutating the exported copy must not affect the live
    campaign's, and mutating the live campaign's afterwards must not affect
    the already-exported copy either."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=x.shape[0], shuffle=False)

    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.run(loader, opt=sfi.CampaignOptimization.O0, progress_mode='silent')

    # Guard: both fields are actually populated on the live campaign before
    # export, so the divergences checked below are real, not two empty
    # containers that trivially stay equal (or unequal) forever.
    assert cmpn.rgroups
    assert cmpn.performance[0].testing.correctSamples is not None

    data = cmpn.export()
    live_rgroups_before = deepcopy(cmpn.rgroups)
    live_correct_before = cmpn.performance[0].testing.correctSamples

    data.rgroups[next(iter(data.rgroups))].append(999)
    assert cmpn.rgroups == live_rgroups_before, (
        'Mutating exported rgroups leaked into the live campaign.'
    )

    data.performance[0].testing.correctSamples += 1000
    assert cmpn.performance[0].testing.correctSamples == live_correct_before, (
        'Mutating exported performance leaked into the live campaign.'
    )

    data_rgroups_before = deepcopy(data.rgroups)
    data_correct_before = data.performance[0].testing.correctSamples

    cmpn.rgroups[next(iter(cmpn.rgroups))].append(-999)
    assert data.rgroups == data_rgroups_before, (
        "Mutating the live campaign's rgroups leaked into exported data."
    )

    cmpn.performance[0].testing.correctSamples += 1000
    assert data.performance[0].testing.correctSamples == data_correct_before, (
        "Mutating the live campaign's performance leaked into exported data."
    )
