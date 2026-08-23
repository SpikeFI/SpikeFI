"""Tier 0 — Campaign's sampling machinery: reproducibility, exclusion
bookkeeping across all three _sample_positions strategies, and the
_unrank_pos <-> product() bijection it relies on.
"""


from itertools import product
import random

import pytest

import spikefi as sfi
from spikefi.core import Campaign
from spikefi.fault import Fault
from spikefi.models import DeadNeuron


# One layer, (K, L, M, N) = (1, 4, 1, 1): 4 positions, the neuron-fault
# padding convention (K fixed to 1).
LAY_DIMS = [('SF1', (1, 4, 1, 1))]


@pytest.mark.neuron
def test_sample_positions_reproducible_with_explicit_rng() -> None:
    """Two calls seeded with equivalent explicit rng objects draw the exact
    same positions, in the same order."""
    drawn_a = Campaign._sample_positions(LAY_DIMS, {}, 3, random.Random(42))
    drawn_b = Campaign._sample_positions(LAY_DIMS, {}, 3, random.Random(42))
    assert drawn_a == drawn_b


@pytest.mark.neuron
def test_sample_positions_unaffected_by_global_seed() -> None:
    """The global random module's seed has no effect: an explicit rng is
    the only source of randomness _sample_positions consults."""
    rng_a = random.Random(7)
    random.seed(123)
    drawn_a = Campaign._sample_positions(LAY_DIMS, {}, 3, rng_a)

    rng_b = random.Random(7)
    random.seed(999)
    drawn_b = Campaign._sample_positions(LAY_DIMS, {}, 3, rng_b)

    assert drawn_a == drawn_b


@pytest.mark.neuron
def test_sample_positions_is_capacity_aware() -> None:
    """Requesting more positions than exist returns only what the space
    actually has, rather than erroring or looping forever."""
    drawn = Campaign._sample_positions(LAY_DIMS, {}, 100, random.Random(0))
    assert len(drawn) == 4


@pytest.mark.neuron
def test_sample_positions_no_duplicates_within_a_call() -> None:
    """A single call never draws the same (layer, position) twice."""
    drawn = Campaign._sample_positions(LAY_DIMS, {}, 4, random.Random(1))
    assert len(set(drawn)) == len(drawn)


@pytest.mark.neuron
def test_sample_positions_no_exclusion_branch() -> None:
    """With nothing excluded, every draw stays within the declared space."""
    drawn = Campaign._sample_positions(LAY_DIMS, {}, 4, random.Random(2))
    all_positions = {
        ('SF1',) + p for p in product(range(1), range(4), range(1), range(1))
    }
    assert set(drawn) <= all_positions


@pytest.mark.neuron
def test_sample_positions_mostly_free_rejection_branch() -> None:
    """With most of the space still free, exclusion is honoured via the
    rejection-sampling branch: no excluded position is ever drawn. Seeded so
    an excluded draw is actually rolled (and must be skipped), not merely
    never attempted by chance."""
    excluded = {'SF1': {(0, 0, 0, 0)}}
    drawn = Campaign._sample_positions(LAY_DIMS, excluded, 3, random.Random(0))

    positions = [d[1:] for d in drawn]
    assert (0, 0, 0, 0) not in positions
    assert len(set(positions)) == len(positions)
    assert len(drawn) == 3


@pytest.mark.neuron
def test_sample_positions_mostly_excluded_enumeration_branch() -> None:
    """With most of the space excluded, the true remainder is enumerated
    directly rather than rejection-sampled."""
    excluded = {'SF1': {(0, i, 0, 0) for i in range(3)}}  # only 1 of 4 free
    drawn = Campaign._sample_positions(LAY_DIMS, excluded, 1, random.Random(4))

    assert drawn == [('SF1', 0, 3, 0, 0)]


@pytest.mark.neuron
def test_unrank_pos_is_bijective_with_product() -> None:
    """_unrank_pos inverts the same linearization product() would produce
    over the combined position space, on a small enough space to check
    exhaustively."""
    lay_dims = [('SF1', (1, 2, 1, 1)), ('SF2', (1, 3, 1, 1))]
    lay_sizes = [1 * 2, 1 * 3]

    all_positions = [
        (lay_name,) + p
        for lay_name, dims in lay_dims
        for p in product(*(range(d) for d in dims))
    ]
    unranked = [
        Campaign._unrank_pos(lay_dims, lay_sizes, idx)
        for idx in range(sum(lay_sizes))
    ]

    assert unranked == all_positions
    assert len(set(unranked)) == len(unranked)


@pytest.mark.neuron
def test_inject_warns_on_sampling_shortfall(campaign_stub: sfi.Campaign) -> None:
    """Requesting more random sites than a layer has room for discards the
    surplus and surfaces it as a RuntimeWarning, rather than sampling with
    replacement or silently under-delivering."""
    fault = Fault.multiple_random_absolute(DeadNeuron(), 5, layers='SF1')

    with pytest.warns(RuntimeWarning, match='had no available position'):
        campaign_stub.inject(fault)
