"""Tier 0 — Campaign.validate(): the last line of defense between whatever
a caller builds by hand and the fault hooks that index into real tensors.
"""


import pytest

import spikefi as sfi
from spikefi.fault import Fault, FaultSite
from spikefi.models import DeadNeuron, ParametricNeuron
from spikefi.utils.layer import LayersInfo


@pytest.mark.neuron
def test_out_of_bounds_site_is_dropped(campaign_stub: sfi.Campaign) -> None:
    """A site whose position falls outside the layer's own shape is
    removed, counted as an invalid site rather than a dropped fault."""
    fault = Fault(DeadNeuron(), FaultSite('SF1', (99, 0, 0)))

    valid, n_invalid, n_dropped = campaign_stub.validate([fault])

    assert valid == []
    assert n_invalid == 1
    assert n_dropped == 0


@pytest.mark.neuron
def test_negative_index_is_accepted_and_normalized(
        campaign_stub: sfi.Campaign,
        layers_info: LayersInfo
) -> None:
    """A negative position is valid (Python-style indexing from the end)
    and is normalized to its positive equivalent as a replaced site, not a
    mutated one, so the set's hash bucket stays correct."""
    shape = layers_info.shapes_neu['SF1']
    fault = Fault(DeadNeuron(), FaultSite('SF1', (-1, 0, 0)))

    valid, n_invalid, n_dropped = campaign_stub.validate([fault])

    assert n_invalid == 0
    assert n_dropped == 0
    normalized_site = next(iter(valid[0].sites))
    assert normalized_site.position == (shape[0] - 1, 0, 0)
    assert normalized_site in valid[0].sites  # set rehashed on the new position


@pytest.mark.neuron
def test_site_less_fault_is_rejected(campaign_stub: sfi.Campaign) -> None:
    """A Fault with no defined sites is dropped outright: an empty
    unroll() would otherwise silently index the whole tensor."""
    fault = Fault(DeadNeuron(), [])

    valid, n_invalid, n_dropped = campaign_stub.validate([fault])

    assert valid == []
    assert n_dropped == 1


@pytest.mark.parametric
def test_unsupported_parametric_param_name_drops_the_fault(
        campaign_stub: sfi.Campaign
) -> None:
    """A parametric fault naming a parameter the campaign's own neuron
    dict does not have is dropped, since there is nothing to perturb."""
    fault = Fault(
        ParametricNeuron('not_a_real_param', 1.5), FaultSite('SF1', (0, 0, 0))
    )

    valid, n_invalid, n_dropped = campaign_stub.validate([fault])

    assert valid == []
    assert n_dropped == 1


@pytest.mark.neuron
@pytest.mark.parametric
def test_validate_counts_are_accurate(campaign_stub: sfi.Campaign) -> None:
    """The returned counts add up to exactly what was dropped and why,
    across a mix of valid, invalid-site and unsupported faults."""
    valid_fault = Fault(DeadNeuron(), FaultSite('SF1', (0, 0, 0)))
    invalid_site_fault = Fault(DeadNeuron(), FaultSite('SF1', (99, 0, 0)))
    unsupported_fault = Fault(
        ParametricNeuron('nope', 1.5), FaultSite('SF1', (0, 0, 0))
    )

    valid, n_invalid, n_dropped = campaign_stub.validate(
        [valid_fault, invalid_site_fault, unsupported_fault]
    )

    assert len(valid) == 1
    assert n_invalid == 1
    assert n_dropped == 1


@pytest.mark.neuron
def test_inject_warns_on_invalid_sites(campaign_stub: sfi.Campaign) -> None:
    """inject() surfaces validate()'s drops as a RuntimeWarning, so a
    caller who fed it a bad site finds out instead of silently losing it."""
    fault = Fault(DeadNeuron(), FaultSite('SF1', (99, 0, 0)))

    with pytest.warns(RuntimeWarning, match='invalid site'):
        campaign_stub.inject(fault)
