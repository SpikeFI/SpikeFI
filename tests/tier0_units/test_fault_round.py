"""Tier 0 — FaultRound bookkeeping: insertion merging, extraction,
grouped/fault_map consistency, and what optimized() carries across.
"""


import pytest

from spikefi.fault import Fault, FaultRound, FaultSite, FaultTarget
from spikefi.models import DeadNeuron, DeadSynapse, StuckNeuron
from spikefi.utils.layer import LayersInfo


@pytest.mark.neuron
def test_insert_merges_same_layer_and_model() -> None:
    """Two Faults with the same (layer, model) key merge into one Fault
    holding every site, rather than overwriting or duplicating entries."""
    round = FaultRound()
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (1, 0, 0))))

    assert len(round) == 1
    merged = next(iter(round.values()))
    assert len(merged) == 2


@pytest.mark.neuron
def test_insert_keeps_different_models_separate() -> None:
    """Faults on the same layer but different models stay as distinct
    entries, even when their target/method/args coincide."""
    round = FaultRound()
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))
    round.insert(Fault(StuckNeuron(0.), FaultSite('SF1', (0, 0, 0))))

    assert len(round) == 2


@pytest.mark.neuron
def test_extract_removes_only_the_named_site() -> None:
    """extract() removes exactly the sites of the given Fault, leaving the
    rest of that layer/model's Fault intact."""
    round = FaultRound()
    site_a, site_b = FaultSite('SF1', (0, 0, 0)), FaultSite('SF1', (1, 0, 0))
    round.insert(Fault(DeadNeuron(), [site_a, site_b]))

    round.extract(Fault(DeadNeuron(), site_a))

    remaining = round[('SF1', DeadNeuron())]
    assert remaining.sites == {site_b}


@pytest.mark.neuron
def test_grouped_and_fault_map_consistent_after_insert() -> None:
    """After insert(), grouped and fault_map both reflect the new fault."""
    round = FaultRound()
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))

    assert round.any_neuronal('SF1')
    assert ('SF1', FaultTarget.OUTPUT) in round.grouped


@pytest.mark.neuron
def test_grouped_and_fault_map_consistent_after_extract_to_empty() -> None:
    """Extracting a Fault's only site clears it from both grouped and
    fault_map, not just from the round's own dict."""
    round = FaultRound()
    site = FaultSite('SF1', (0, 0, 0))
    round.insert(Fault(DeadNeuron(), site))

    round.extract(Fault(DeadNeuron(), site))

    assert ('SF1', DeadNeuron()) not in round
    assert ('SF1', FaultTarget.OUTPUT) not in round.grouped
    assert not round.any_neuronal('SF1')


@pytest.mark.neuron
@pytest.mark.synapse
def test_grouped_and_fault_map_consistent_after_clear() -> None:
    """clear() empties grouped and fault_map along with the round itself."""
    round = FaultRound()
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))
    round.insert(Fault(DeadSynapse(), FaultSite('SF1', (0, 0, 0, 0))))

    round.clear()

    assert len(round) == 0
    assert not round.grouped
    assert not round.fault_map


@pytest.mark.neuron
def test_fault_map_row_width_matches_fault_target_count() -> None:
    """Every layer's fault_map row has one slot per FaultTarget member,
    regardless of how many targets that layer actually carries."""
    round = FaultRound()
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))

    assert len(round.fault_map['SF1']) == len(FaultTarget)


@pytest.mark.neuron
@pytest.mark.synapse
def test_optimized_carries_grouped_and_fault_map(
        layers_info: LayersInfo
) -> None:
    """optimized() carries both grouped and fault_map into the returned
    OptimizedFaultRound, not just the round's own faults."""
    round = FaultRound()
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))
    round.insert(Fault(DeadSynapse(), FaultSite('SF2', (0, 0, 0, 0))))

    oround = round.optimized(layers_info)

    assert oround.grouped.keys() == round.grouped.keys()
    assert oround.fault_map.keys() == round.fault_map.keys()


@pytest.mark.neuron
@pytest.mark.synapse
def test_optimized_fault_map_ordered_by_layer_index(
        layers_info: LayersInfo
) -> None:
    """optimized()'s fault_map is ordered by each layer's position in the
    network, earliest layer first, regardless of insertion order."""
    round = FaultRound()
    round.insert(Fault(DeadSynapse(), FaultSite('SF2', (0, 0, 0, 0))))
    round.insert(Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0))))

    oround = round.optimized(layers_info)

    assert list(oround.fault_map.keys()) == ['SF1', 'SF2']
