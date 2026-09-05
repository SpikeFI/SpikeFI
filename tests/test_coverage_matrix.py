"""Coverage matrix: a census, not a semantics tier. One generic oracle,
swept over every concrete FaultModel x layer type x structural case, so an
untested model becomes a collection-time failure instead of a silent gap.
Living outside the tier directories, it declares gpu per test rather than
receiving it from the collection hook.
"""


from collections.abc import Callable
import inspect
import random

import pytest
import torch
from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
import spikefi.hooks as sfh
from spikefi.models import (
    BitflippedSynapse, DeadNeuron, DeadSynapse, IntegrationFaultNeuron,
    mul_value, ParametricNeuron, ParametricNeuronFaultModel, PerturbedSynapse,
    RandomFaultModel, RandomNeuron, RandomParametricNeuron, RandomSynapse,
    RefractoryFaultNeuron, SaturatedNeuron, SaturatedSynapse, StuckNeuron,
    StuckSynapse, ThresholdFaultNeuron,
)
from spikefi.utils.quantization import qargs_from_range

from nets import NetSpec


# --- Registry: every concrete FaultModel gets one deterministic factory ---

FAULT_MODEL_REGISTRY: dict[type, Callable[[], sff.FaultModel]] = {
    DeadNeuron: lambda: DeadNeuron(),
    SaturatedNeuron: lambda: SaturatedNeuron(),
    StuckNeuron: lambda: StuckNeuron(0.3),
    ParametricNeuronFaultModel: lambda: ParametricNeuronFaultModel('theta', mul_value, 2.0),
    ParametricNeuron: lambda: ParametricNeuron('theta', 2.0),
    IntegrationFaultNeuron: lambda: IntegrationFaultNeuron(2.0),
    RefractoryFaultNeuron: lambda: RefractoryFaultNeuron(2.0),
    ThresholdFaultNeuron: lambda: ThresholdFaultNeuron(2.0),
    DeadSynapse: lambda: DeadSynapse(),
    SaturatedSynapse: lambda: SaturatedSynapse(-1.0, 1.0),
    StuckSynapse: lambda: StuckSynapse(0.5),
    PerturbedSynapse: lambda: PerturbedSynapse(1.5),
    BitflippedSynapse: lambda: BitflippedSynapse(
        0, *qargs_from_range(-2.0, 2.0, torch.qint8), torch.qint8
    ),
}

# RandomNeuron/RandomParametricNeuron/RandomSynapse are RandomFaultModel
# __new__ factories, not FaultModel subclasses themselves (they hand back a
# concrete instance of one), so they are registered and discovered separately.
RANDOM_MODEL_REGISTRY: dict[type, Callable[[], sff.FaultModel]] = {
    RandomNeuron: lambda: RandomNeuron(),
    RandomParametricNeuron: lambda: RandomParametricNeuron(),
    RandomSynapse: lambda: RandomSynapse(),
}


def _discover_fault_model_classes() -> set[type]:
    """Every concrete FaultModel subclass spikefi.models exports."""
    return {
        cls for _, cls in inspect.getmembers(sfi.fm, inspect.isclass)
        if issubclass(cls, sff.FaultModel) and cls is not sff.FaultModel
    }


def _discover_random_model_classes() -> set[type]:
    """Every RandomFaultModel __new__ factory spikefi.models exports."""
    return {
        cls for _, cls in inspect.getmembers(sfi.fm, inspect.isclass)
        if issubclass(cls, RandomFaultModel) and cls is not RandomFaultModel
    }


def test_registry_covers_every_concrete_fault_model() -> None:
    """A newly added FaultModel subclass fails this test until a
    deterministic factory is registered for it in FAULT_MODEL_REGISTRY."""
    assert _discover_fault_model_classes() == set(FAULT_MODEL_REGISTRY)


def test_random_model_registry_covers_every_random_variant() -> None:
    """Same completeness guarantee for the three RandomFaultModel __new__
    factories, discovered and registered separately from FAULT_MODEL_REGISTRY
    since they are not FaultModel subclasses themselves."""
    assert _discover_random_model_classes() == set(RANDOM_MODEL_REGISTRY)


def test_random_model_draws_are_always_registered_concrete_classes() -> None:
    """Every RandomFaultModel factory can only ever hand back an instance of
    a concrete model this file's registry already covers, checked over
    enough seeded draws to hit every branch of each one's model_choices."""
    random.seed(0)
    for random_cls, factory in RANDOM_MODEL_REGISTRY.items():
        for _ in range(30):
            drawn = factory()
            assert type(drawn) in FAULT_MODEL_REGISTRY, (
                f'{random_cls.__name__} drew an unregistered model '
                f'{type(drawn).__name__}.'
            )


# --- Layer-type sweeps: (net fixture, faulty layer[, following layer]) ---

_OUTPUT_LAYER_CASES = [
    pytest.param('dense_net', 'SF1', 'SF2', id='dense'),
    pytest.param('conv_net', 'SC1', 'SP1', id='conv-through-pool'),
    pytest.param('dense_net', 'SF2', 'tail', id='output-layer-through-tail'),
]

_WEIGHT_LAYER_CASES = [
    pytest.param('dense_net', 'SF1', id='dense'),
    pytest.param('conv_net', 'SC1', id='conv'),
    pytest.param('dense_net', 'SF2', id='output-layer'),
]

_PARAMETER_LAYER_CASES = [
    pytest.param('dense_net', 'SF1', id='dense'),
    pytest.param('conv_net', 'SC1', id='conv'),
    pytest.param('dense_net', 'SF2', id='output-layer'),
]

_OUTPUT_MODEL_CLASSES = [DeadNeuron, SaturatedNeuron, StuckNeuron]
_WEIGHT_MODEL_CLASSES = [
    DeadSynapse, SaturatedSynapse, StuckSynapse, PerturbedSynapse, BitflippedSynapse
]
_PARAMETER_MODEL_CLASSES = [
    ParametricNeuronFaultModel, ParametricNeuron, IntegrationFaultNeuron,
    RefractoryFaultNeuron, ThresholdFaultNeuron,
]


# --- Shared helpers: site search (the non-vacuity guard, derived rather
# than hand-written) and following-layer input capture ---

def _find_output_site(
        activity: Tensor,
        model: sff.FaultModel,
        exclude: frozenset[tuple[int, int, int]] = frozenset()
) -> tuple[int, int, int]:
    """Scans channel-major positions of a golden OUTPUT activity tensor for
    one where model.perturb() actually changes the value, so the delivery
    oracle never runs on a site the fault happens to be a no-op at."""
    _, C, H, W, _ = activity.shape
    for c in range(C):
        for h in range(H):
            for w in range(W):
                site = (c, h, w)
                if site in exclude:
                    continue
                golden_val = activity[:, c, h, w, :]
                if not torch.all(golden_val == model.perturb(golden_val)):
                    return site
    raise AssertionError(f'No site has a visible effect for {model.get_name()}.')


def _find_weight_site(
        weight: Tensor,
        model: sff.FaultModel
) -> tuple[int, int, int, int]:
    """Scans a golden weight tensor for one position where model.perturb()
    actually changes the value, so the delivery oracle never runs on a
    weight the fault happens to be a no-op at. Only the first 4 dims are
    addressable by a synapse FaultSite; weight carries a trailing size-1
    dim beyond those (see LayersInfo.shapes_syn)."""
    O, I, H, W = weight.shape[:4]
    for o in range(O):
        for i in range(I):
            for h in range(H):
                for w in range(W):
                    site = (o, i, h, w)
                    golden_val = weight[site]
                    if golden_val != model.perturb(golden_val):
                        return site
    raise AssertionError(f'No site has a visible effect for {model.get_name()}.')


def _capture_following_input(
        cmpn: sfi.Campaign,
        following_name: str,
        x: Tensor,
        round_idx: int = 0
) -> Tensor:
    """Runs `round_idx` and returns the exact tensor the following layer
    received as input, captured via a pre-hook registered after the
    fault pre-hook so it observes the already-perturbed value."""
    cmpn._pre_run(sfi.CampaignOptimization.O0)
    captured = {}
    handle = getattr(cmpn.faulty, following_name).register_forward_pre_hook(
        lambda _, inputs: captured.__setitem__('in', inputs[0].clone())
    )
    cmpn.r_idx_ref.r = round_idx
    cmpn.faulty(x)
    handle.remove()
    return captured['in']


def _amplify_all(net: nn.Module, factor: float = 3.0) -> None:
    """Scales every weighted layer up so a change at one layer reliably
    keeps firing all the way to the network's final output within the tiny
    nets' 16-bin window: the PARAMETER oracle's search needs a real,
    propagated difference at the end of a possibly multi-layer chain, not
    just local activity."""
    with torch.no_grad():
        for module in net.modules():
            if getattr(module, 'weight', None) is not None:
                module.weight.mul_(factor)


def _amplify(net: nn.Module, layer_name: str, factor: float = 10.0) -> None:
    """Scales a layer's own weight up so it fires reliably within the tiny
    nets' 16-bin window, regardless of a particular seed's default-init
    draw: a search for a non-vacuous site is only meaningful over an
    actually-active signal."""
    with torch.no_grad():
        getattr(net, layer_name).weight.mul_(factor)


# --- Part 2: the generic perturb-consistency oracle ---

@pytest.mark.gpu
@pytest.mark.neuron
@pytest.mark.parametrize(
    'net_fixture_name, faulty_layer, following_layer', _OUTPUT_LAYER_CASES
)
@pytest.mark.parametrize(
    'model_cls', _OUTPUT_MODEL_CLASSES, ids=lambda c: c.__name__
)
def test_output_fault_matches_perturb_at_exactly_its_site(
        request: pytest.FixtureRequest,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]],
        net_fixture_name: str,
        faulty_layer: str,
        following_layer: str,
        model_cls: type
) -> None:
    """An OUTPUT fault's site in the following layer's input equals exactly
    model.perturb(golden_value); every other site stays bit-identical to
    golden -- for every neuron model, on dense, conv (through a
    non-injectable pool), and the output layer (through the tail
    Identity)."""
    net: NetSpec = request.getfixturevalue(net_fixture_name)
    _amplify(net.net, faulty_layer)
    cmpn = make_campaign(net.net, net.shape_in, slayer)
    x = fixed_input(net.shape_in)
    golden_signal = golden_activity(cmpn, x)[faulty_layer]

    model = FAULT_MODEL_REGISTRY[model_cls]()
    site = _find_output_site(golden_signal, model)
    idx = (slice(None), *site, slice(None))
    perturbed_val = model.perturb(golden_signal[idx])
    assert not torch.all(golden_signal[idx] == perturbed_val), (
        f'perturb() is a no-op at site {site}; would not prove delivery.'
    )

    expected = golden_signal.clone()
    expected[idx] = perturbed_val

    fault = sff.Fault(model, sff.FaultSite(faulty_layer, site))
    cmpn.inject(fault, round_idx=0)
    captured_input = _capture_following_input(cmpn, following_layer, x)

    assert torch.equal(captured_input, expected)


@pytest.mark.gpu
@pytest.mark.synapse
@pytest.mark.parametrize('net_fixture_name, faulty_layer', _WEIGHT_LAYER_CASES)
@pytest.mark.parametrize(
    'model_cls', _WEIGHT_MODEL_CLASSES, ids=lambda c: c.__name__
)
def test_weight_fault_matches_perturb_at_exactly_its_site(
        request: pytest.FixtureRequest,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        net_fixture_name: str,
        faulty_layer: str,
        model_cls: type
) -> None:
    """A WEIGHT fault's site equals exactly model.perturb(golden_weight);
    every other weight stays bit-identical to golden, and the weight is
    restored to golden after the forward pass -- for every synapse model,
    on dense, conv, and the output layer."""
    net: NetSpec = request.getfixturevalue(net_fixture_name)
    cmpn = make_campaign(net.net, net.shape_in, slayer)
    x = fixed_input(net.shape_in)

    layer = getattr(cmpn.golden, faulty_layer)
    golden_weight = layer.weight.detach().clone()

    model = FAULT_MODEL_REGISTRY[model_cls]()
    site = _find_weight_site(golden_weight, model)
    perturbed_val = model.perturb(golden_weight[site])
    assert golden_weight[site] != perturbed_val, (
        f'perturb() is a no-op at site {site}; would not prove delivery.'
    )

    expected_weight = golden_weight.clone()
    expected_weight[site] = perturbed_val

    fault = sff.Fault(model, sff.FaultSite(faulty_layer, site))
    cmpn.inject(fault, round_idx=0)
    cmpn._pre_run(sfi.CampaignOptimization.O0)

    captured = {}
    handle = getattr(cmpn.faulty, faulty_layer).register_forward_pre_hook(
        lambda _, __: captured.__setitem__(
            'w', getattr(cmpn.faulty, faulty_layer).weight.detach().clone()
        )
    )
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)
    handle.remove()

    assert torch.equal(captured['w'], expected_weight)
    assert torch.equal(getattr(cmpn.faulty, faulty_layer).weight.detach(), golden_weight)


def _inject_parametric_at_first_visible_site(
        cmpn: sfi.Campaign,
        x: Tensor,
        layer_name: str,
        following_layer: str,
        model: sff.FaultModel,
        golden_following_input: Tensor,
        layer_shape: tuple[int, int, int]
) -> tuple[sff.FaultModel, Tensor]:
    """Scans every position of the faulty layer for one where the
    perturbed parameter actually changes the following layer's input, so
    the delivery oracle never runs on a site the fault happens not to
    touch -- checked one hop downstream rather than at the network's final
    output, since a small parameter change can be masked by pooling or a
    downstream decision boundary several layers further on. Leaves the
    winning fault injected in round 0 on return.

    FaultRound.insert() deep-copies the model of every Fault it stores, so
    the installed model actually carrying the populated `flayer` is fetched
    back from the round rather than assumed to be the caller's own `model`
    object.
    """
    C, H, W = layer_shape
    for c in range(C):
        for h in range(H):
            for w in range(W):
                site = (c, h, w)
                cmpn.inject(sff.Fault(model, sff.FaultSite(layer_name, site)), round_idx=0)
                faulty_following_input = _capture_following_input(cmpn, following_layer, x)
                if not torch.equal(faulty_following_input, golden_following_input):
                    installed = cmpn.rounds[0].grouped[(layer_name, sff.FaultTarget.PARAMETER)][0]
                    return installed.model, faulty_following_input
                cmpn.eject(round_idx=0)
    raise AssertionError(f'No site has a visible effect for {model.get_name()}.')


@pytest.mark.gpu
@pytest.mark.parametric
@pytest.mark.parametrize('net_fixture_name, faulty_layer', _PARAMETER_LAYER_CASES)
@pytest.mark.parametrize(
    'model_cls', _PARAMETER_MODEL_CLASSES, ids=lambda c: c.__name__
)
def test_parameter_fault_matches_param_method_and_isolates_campaign_slayer(
        request: pytest.FixtureRequest,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        net_fixture_name: str,
        faulty_layer: str,
        model_cls: type
) -> None:
    """A PARAMETER fault's dummy layer carries exactly
    param_method(golden_value, *param_args) for its own parameter; the
    campaign's own slayer.neuron is untouched; and the following layer's
    input differs from golden at the fault's site -- for every parametric
    model, on dense, conv, and the output layer.

    FaultModel.perturb() is not the right oracle here: for a
    ParametricNeuronFaultModel it evaluates set_value(original, *self.args)
    with self.args seeded to an empty tuple at construction (see
    ParametricNeuronFaultModel.__init__), so it returns () rather than a
    perturbed parameter value. The parameter's own transform lives in
    param_method/param_args instead (see param_perturb()), so that is what
    is checked against.
    """
    net: NetSpec = request.getfixturevalue(net_fixture_name)
    _amplify_all(net.net)
    cmpn = make_campaign(net.net, net.shape_in, slayer)
    x = fixed_input(net.shape_in)

    model = FAULT_MODEL_REGISTRY[model_cls]()
    param_name = model.param_name
    golden_param_value = cmpn.slayer.neuron[param_name]
    perturbed_expected = model.param_method(golden_param_value, *model.param_args)
    assert golden_param_value != perturbed_expected, (
        f"param_method() is a no-op on '{param_name}'; would not prove delivery."
    )

    following_layer = cmpn.layers_info.get_following(faulty_layer)
    golden_following_input = _capture_following_input(cmpn, following_layer, x)
    layer_shape = cmpn.layers_info.shapes_neu[faulty_layer]
    installed_model, faulty_following_input = _inject_parametric_at_first_visible_site(
        cmpn, x, faulty_layer, following_layer, model, golden_following_input, layer_shape
    )

    assert installed_model.flayer.neuron[param_name] == perturbed_expected
    assert cmpn.slayer.neuron[param_name] == golden_param_value
    assert not torch.equal(faulty_following_input, golden_following_input)


# --- Part 3: structural cases ---

@pytest.mark.gpu
@pytest.mark.neuron
def test_two_faults_on_the_same_layer_are_each_delivered_independently(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """Two OUTPUT faults on different sites of the same layer each land
    exactly at their own site, independent of one another and of every
    untouched site."""
    _amplify(dense_net.net, 'SF1')
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_sf1 = golden_activity(cmpn, x)['SF1']

    model_a, model_b = DeadNeuron(), StuckNeuron(0.4)
    site_a = _find_output_site(golden_sf1, model_a)
    site_b = _find_output_site(golden_sf1, model_b, exclude=frozenset({site_a}))

    expected = golden_sf1.clone()
    idx_a = (slice(None), *site_a, slice(None))
    idx_b = (slice(None), *site_b, slice(None))
    expected[idx_a] = model_a.perturb(golden_sf1[idx_a])
    expected[idx_b] = model_b.perturb(golden_sf1[idx_b])

    faults = [
        sff.Fault(model_a, sff.FaultSite('SF1', site_a)),
        sff.Fault(model_b, sff.FaultSite('SF1', site_b)),
    ]
    cmpn.inject(faults, round_idx=0)
    captured_input = _capture_following_input(cmpn, 'SF2', x)

    assert torch.equal(captured_input, expected)


@pytest.mark.gpu
@pytest.mark.neuron
def test_two_faults_on_different_layers_are_each_delivered_independently(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """Two OUTPUT faults on different layers of the same round are each
    delivered at their own following layer -- the second checked against
    its own true, already-SF1-perturbed input rather than a stale golden
    baseline, since that is the input SF2 actually computes from."""
    _amplify(dense_net.net, 'SF1')
    _amplify(dense_net.net, 'SF2')
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_sf1 = golden_activity(cmpn, x)['SF1']

    model_a = DeadNeuron()
    site_a = _find_output_site(golden_sf1, model_a)
    idx_a = (slice(None), *site_a, slice(None))
    expected_sf2_input = golden_sf1.clone()
    expected_sf2_input[idx_a] = model_a.perturb(golden_sf1[idx_a])

    fault_a = sff.Fault(model_a, sff.FaultSite('SF1', site_a))
    cmpn.inject(fault_a, round_idx=0)

    # SF2's own true signal once SF1's fault has already propagated into
    # it, with SF2 not yet faulted -- the baseline the SF2 fault below is
    # checked against.
    tail_baseline = _capture_following_input(cmpn, 'tail', x)

    model_b = StuckNeuron(0.4)
    site_b = _find_output_site(tail_baseline, model_b)
    idx_b = (slice(None), *site_b, slice(None))
    expected_tail_input = tail_baseline.clone()
    expected_tail_input[idx_b] = model_b.perturb(tail_baseline[idx_b])

    fault_b = sff.Fault(model_b, sff.FaultSite('SF2', site_b))
    cmpn.inject(fault_b, round_idx=0)

    sf2_input_both = _capture_following_input(cmpn, 'SF2', x)
    tail_input_both = _capture_following_input(cmpn, 'tail', x)

    assert torch.equal(sf2_input_both, expected_sf2_input)
    assert torch.equal(tail_input_both, expected_tail_input)


@pytest.mark.gpu
@pytest.mark.neuron
@pytest.mark.parametric
def test_mixed_output_and_parameter_faults_on_the_earliest_layer(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor],
        golden_activity: Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]
) -> None:
    """A round mixing an OUTPUT and a PARAMETER fault on the same
    (earliest) layer delivers both: the OUTPUT site matches perturb()
    exactly, and the PARAMETER fault's dummy layer carries the
    independently-computed param_method() value, without either
    corrupting the other's site."""
    _amplify(dense_net.net, 'SF1')
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    x = fixed_input(dense_net.shape_in)
    golden_sf1 = golden_activity(cmpn, x)['SF1']

    output_model = DeadNeuron()
    site_out = _find_output_site(golden_sf1, output_model)
    idx_out = (slice(None), *site_out, slice(None))
    expected_sf2_input_out = golden_sf1[idx_out].clone()
    perturbed_out = output_model.perturb(golden_sf1[idx_out])
    assert not torch.all(expected_sf2_input_out == perturbed_out), (
        f'perturb() is a no-op at site {site_out}; would not prove delivery.'
    )

    param_model = ThresholdFaultNeuron(2.0)
    golden_theta = cmpn.slayer.neuron['theta']
    perturbed_theta = param_model.param_method(golden_theta, *param_model.param_args)
    assert golden_theta != perturbed_theta, (
        "param_method() is a no-op on 'theta'; would not prove delivery."
    )
    # A distinct channel from the OUTPUT fault's, so the two sites do not
    # overlap on the same layer.
    site_param = next(c for c in range(golden_sf1.shape[1]) if c != site_out[0])

    fault_out = sff.Fault(output_model, sff.FaultSite('SF1', site_out))
    fault_param = sff.Fault(param_model, sff.FaultSite('SF1', (site_param, 0, 0)))
    cmpn.inject([fault_out, fault_param], round_idx=0)

    captured_input = _capture_following_input(cmpn, 'SF2', x)

    # FaultRound.insert() deep-copies each Fault's model, so the installed
    # PARAMETER model actually carrying the populated `flayer` is fetched
    # back from the round rather than assumed to be `param_model` itself.
    installed_param_model = cmpn.rounds[0].grouped[('SF1', sff.FaultTarget.PARAMETER)][0].model

    assert torch.all(captured_input[idx_out] == perturbed_out)
    assert installed_param_model.flayer.neuron['theta'] == perturbed_theta
    assert cmpn.slayer.neuron['theta'] == golden_theta


@pytest.mark.gpu
@pytest.mark.neuron
def test_shared_following_layer_with_differing_shapes_lands_on_intended_layer_only(
        shared_dropout_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """Two injectables of different output shape (4 vs 6) feeding the same
    shared dropout module: the neuron pre-hook identifies its own
    invocation of that module by position. An SF1-only fault lands
    exactly on SF2's own input, and SF2's own downstream through the same
    shared module is exactly what SF2 itself computes from that
    already-perturbed input -- not additionally touched by SF1's fault."""
    _amplify(shared_dropout_net.net, 'SF1')
    cmpn = make_campaign(shared_dropout_net.net, shared_dropout_net.shape_in, slayer)
    x = fixed_input(shared_dropout_net.shape_in)

    with torch.no_grad():
        golden_sf1 = shared_dropout_net.net.slayer.spike(
            shared_dropout_net.net.slayer.psp(shared_dropout_net.net.SF1(x))
        )

    model = StuckNeuron(0.4)
    site = _find_output_site(golden_sf1, model)
    idx = (slice(None), *site, slice(None))
    expected_sf2_input = golden_sf1.clone()
    expected_sf2_input[idx] = model.perturb(golden_sf1[idx])

    fault = sff.Fault(model, sff.FaultSite('SF1', site))
    cmpn.inject(fault, round_idx=0)

    cmpn._pre_run(sfi.CampaignOptimization.O0)
    captured = []
    handle = cmpn.faulty.drop.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].clone())
    )
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)
    handle.remove()

    assert len(captured) == 2, 'Expected exactly two invocations of the shared drop module.'
    assert torch.equal(captured[0], expected_sf2_input)

    with torch.no_grad():
        expected_sf2_own_output = shared_dropout_net.net.slayer.spike(
            shared_dropout_net.net.slayer.psp(
                shared_dropout_net.net.SF2(expected_sf2_input)
            )
        )
    assert torch.equal(captured[1], expected_sf2_own_output)


@pytest.mark.gpu
@pytest.mark.neuron
def test_shared_following_layer_with_equal_shapes_lands_on_intended_layer_only(
        same_shape_shared_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """Two injectables of EQUAL output shape (4 vs 4) feeding the same shared
    dropout module -- the case no shape comparison can separate. An SF1-only
    fault must land on SF1's own invocation of that module and leave SF2's
    own invocation computed purely from SF2's (already-SF1-perturbed) input,
    so the pre-hook has to identify its invocation by position in the
    forward pass rather than by the shape of what it receives."""
    cmpn = make_campaign(same_shape_shared_net.net, same_shape_shared_net.shape_in, slayer)
    x = fixed_input(same_shape_shared_net.shape_in)

    with torch.no_grad():
        golden_sf1 = same_shape_shared_net.net.slayer.spike(
            same_shape_shared_net.net.slayer.psp(same_shape_shared_net.net.SF1(x))
        )

    model = SaturatedNeuron()
    site = _find_output_site(golden_sf1, model)
    idx = (slice(None), *site, slice(None))
    expected_sf2_input = golden_sf1.clone()
    expected_sf2_input[idx] = model.perturb(golden_sf1[idx])

    fault = sff.Fault(model, sff.FaultSite('SF1', site))
    cmpn.inject(fault, round_idx=0)

    cmpn._pre_run(sfi.CampaignOptimization.O0)
    captured = []
    handle = cmpn.faulty.drop.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].clone())
    )
    cmpn.r_idx_ref.r = 0
    cmpn.faulty(x)
    handle.remove()

    with torch.no_grad():
        expected_sf2_own_output = same_shape_shared_net.net.slayer.spike(
            same_shape_shared_net.net.slayer.psp(
                same_shape_shared_net.net.SF2(expected_sf2_input)
            )
        )
    assert torch.equal(captured[1], expected_sf2_own_output)


@pytest.mark.gpu
@pytest.mark.neuron
def test_direct_neuron_hooks_also_land_on_the_intended_invocation_only(
        same_shape_shared_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        fixed_input: Callable[..., Tensor]
) -> None:
    """The same invocation guarantee on the other hook path: a net returned
    by load_net() carries direct hooks and its own position reference, with
    no Campaign behind it, so it has to disambiguate a shared module's two
    invocations on its own exactly as the dispatching hooks do."""
    cmpn = make_campaign(same_shape_shared_net.net, same_shape_shared_net.shape_in, slayer)
    x = fixed_input(same_shape_shared_net.shape_in)

    with torch.no_grad():
        golden_sf1 = same_shape_shared_net.net.slayer.spike(
            same_shape_shared_net.net.slayer.psp(same_shape_shared_net.net.SF1(x))
        )

    model = SaturatedNeuron()
    site = _find_output_site(golden_sf1, model)
    idx = (slice(None), *site, slice(None))
    expected_sf2_input = golden_sf1.clone()
    expected_sf2_input[idx] = model.perturb(golden_sf1[idx])
    assert not torch.equal(expected_sf2_input, golden_sf1), (
        'perturb() is a no-op at the chosen site; would not prove delivery.'
    )

    cmpn.inject(sff.Fault(model, sff.FaultSite('SF1', site)), round_idx=0)
    device = next(same_shape_shared_net.net.parameters()).device
    loaded = sfi.Campaign.load_net(
        cmpn.save_net(0), same_shape_shared_net.net, device
    )

    # Guard: this is genuinely the direct-hook path, not the dispatching one.
    assert any(
        isinstance(h, sfh.DirectNeuronPerturbPreHook)
        for h in loaded.drop._forward_pre_hooks.values()
    )

    captured = []
    handle = loaded.drop.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].clone())
    )
    loaded(x)
    handle.remove()

    with torch.no_grad():
        expected_sf2_own_output = same_shape_shared_net.net.slayer.spike(
            same_shape_shared_net.net.slayer.psp(
                same_shape_shared_net.net.SF2(expected_sf2_input)
            )
        )

    assert len(captured) == 2, 'Expected exactly two invocations of the shared drop module.'
    assert torch.equal(captured[0], expected_sf2_input)
    assert torch.equal(captured[1], expected_sf2_own_output)
