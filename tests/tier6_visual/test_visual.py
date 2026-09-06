"""Tier 6 — visual.py correctness: _data_mapping's structure and its layer
filter, O0/O4 plot-data agreement, the rename parameter, the export/save/load
round-trip, run_train-result rendering, and heat's documented shape choice.
Every assertion targets the data a plot is built from, never pixels.
"""


from collections.abc import Callable, Generator
from contextlib import contextmanager
from math import prod
import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.fault as sff
from spikefi.models import DeadNeuron, DeadSynapse, SaturatedNeuron
import spikefi.utils.io as sfio
import spikefi.visual as sfv

from nets import NetSpec


@contextmanager
def _spy_fig_paths() -> Generator[list[str], None, None]:
    """Wraps the make_fig_filepath name bound inside spikefi.visual's own
    namespace (visual.py imports the function directly, so patching
    spikefi.utils.io's copy would not intercept any call made from bar(),
    heat(), etc.) to record every path a plotting call actually resolved to,
    without ever asserting on a fixed filename."""
    captured: list[str] = []
    original = sfv.make_fig_filepath

    def _wrapped(*args: object, **kwargs: object) -> str:
        path = original(*args, **kwargs)
        captured.append(path)
        return path

    sfv.make_fig_filepath = _wrapped
    try:
        yield captured
    finally:
        sfv.make_fig_filepath = original


@pytest.mark.visual
def test_data_mapping_maps_faults_to_campaign_and_round_indices(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """_data_mapping keys its result by (layer, fault_model) and maps each
    key to {campaign index: [round indices]}, and a round's own injected
    site is traceable back through the round index the mapping returns for
    its key."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    site_sf1 = sff.FaultSite('SF1', (0, 0, 0))
    cmpn.inject(sff.Fault(DeadNeuron(), site_sf1), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF2', (0, 0, 0, 0))))
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))))

    _, test_loader = tiny_loaders(dense_net.shape_in)
    cmpn.run(test_loader, es_tol=0, progress_mode='silent')
    data = cmpn.export()

    data_map = sfv._data_mapping(data)
    assert data_map, 'No (layer, fault_model) keys were produced; not a real check.'
    for key, cmpn_dict in data_map.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], str) and isinstance(key[1], sff.FaultModel)
        assert isinstance(cmpn_dict, dict)
        for cmpn_idx, r_idxs in cmpn_dict.items():
            assert isinstance(cmpn_idx, int) and isinstance(r_idxs, list)

    key_sf1_dn = ('SF1', DeadNeuron())
    key_sf2_ds = ('SF2', DeadSynapse())
    assert sorted(data_map[key_sf1_dn][0]) == [0, 2]
    assert data_map[key_sf2_ds][0] == [1]

    # The site round 0 was actually built from, traced back through the
    # round index _data_mapping returned for its (layer, fault_model) key.
    assert site_sf1 in data.rounds[0][key_sf1_dn].sites
    assert 0 in data_map[key_sf1_dn][0]


@pytest.mark.visual
def test_data_mapping_skips_rounds_with_multiple_faults(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """A round carrying more than one (layer, fault_model) key is excluded
    from the mapping entirely, since heat/bar/plot each read a single
    performance value per round and cannot attribute it to one key."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))))
    cmpn.inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF2', (0, 0, 0, 0))), round_idx=1)

    # Guard: round 1 must actually be the multi-fault round under test.
    assert len(cmpn.rounds[0]) == 1
    assert len(cmpn.rounds[1]) > 1

    _, test_loader = tiny_loaders(dense_net.shape_in)
    cmpn.run(test_loader, es_tol=0, progress_mode='silent')
    data = cmpn.export()

    data_map = sfv._data_mapping(data)
    assert data_map, 'No single-fault round was mapped; not a real check.'

    all_round_idxs = {
        r for cmpn_dict in data_map.values()
        for r_idxs in cmpn_dict.values()
        for r in r_idxs
    }
    assert 0 in all_round_idxs
    assert 1 not in all_round_idxs


@pytest.mark.optimization
@pytest.mark.visual
def test_data_mapping_filters_on_faulty_layer_not_late_start_name(
        three_layer_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """_data_mapping's layer filter compares against key[0], the layer the
    fault actually sits on -- not the round's late-start name, which for a
    hard neuronal-only round is the layer *after* the faulty one."""
    cmpn = make_campaign(three_layer_net.net, three_layer_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (0, 0, 0))), round_idx=0)

    _, test_loader = tiny_loaders(three_layer_net.shape_in)
    cmpn.run(test_loader, es_tol=0, opt=sfi.CampaignOptimization.O2, progress_mode='silent')
    data = cmpn.export()

    # Guard: the scenario the test relies on actually holds -- late start
    # advanced past the faulty layer to a genuinely different one.
    late_start_name = data.layers_info.get_following('SF2')
    assert late_start_name in data.rgroups
    assert 'SF2' not in data.rgroups
    assert late_start_name != 'SF2'

    unfiltered = sfv._data_mapping(data)
    assert unfiltered, 'No mapping at all; not a real check.'

    filtered_faulty_layer = sfv._data_mapping(data, layer='SF2')
    assert filtered_faulty_layer, 'Filtering on the actual faulty layer returned nothing.'

    filtered_late_start_name = sfv._data_mapping(data, layer=late_start_name)
    assert not filtered_late_start_name, (
        'Filtering on the late-start (non-faulty) layer name returned a result.'
    )


@pytest.mark.serialization
@pytest.mark.visual
def test_export_save_load_round_trip_matches_live_plot_data(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]],
        artifact_name: str
) -> None:
    """export() -> CampaignData.save() -> CampaignData.load() must feed
    heat()/plot() the same _data_mapping and the same per-round accuracy
    values as plotting the live exported data."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadSynapse(), sff.FaultSite('SF2', (0, 0, 0, 0))))
    _, test_loader = tiny_loaders(dense_net.shape_in)
    cmpn.run(test_loader, es_tol=0, progress_mode='silent')

    data_live = cmpn.export()
    # save()'s own rename=True logic resolves this same path internally;
    # computed here (a pure lookup, nothing is written by it) so the test
    # never assumes a clean output directory.
    expected_fpath = sfio.make_res_filepath(artifact_name + '.pkl', rename=True)
    data_live.save(artifact_name)
    assert os.path.exists(expected_fpath)

    data_loaded = sfi.CampaignData.load(expected_fpath)

    map_live = sfv._data_mapping(data_live)
    map_loaded = sfv._data_mapping(data_loaded)
    assert map_live, 'Empty mapping; not a real check.'
    assert map_live == map_loaded

    accu_live = [round(p.testing.maxAccuracy, 6) for p in data_live.performance]
    accu_loaded = [round(p.testing.maxAccuracy, 6) for p in data_loaded.performance]
    assert accu_live, 'No performance data; not a real check.'
    assert accu_live == accu_loaded


@pytest.mark.training
@pytest.mark.training
@pytest.mark.visual
def test_plot_train_renders_run_train_result_and_writes_a_file(
        dense_net: NetSpec,
        slayer: spikeLayer,
        net_params: dict,
        device: torch.device,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """plot_train() renders a run_train() result's per-round training curve
    against a valid x_range without error and writes a file."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))))
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (1, 0, 0))))
    train_loader, test_loader = tiny_loaders(dense_net.shape_in)
    spike_loss = snn.loss(net_params).to(device)
    cmpn.run_train(
        1, train_loader, test_loader, spike_loss,
        lambda params: torch.optim.Adam(params, lr=1e-2), progress_mode='silent'
    )
    data = cmpn.export()
    # x_range[0] == 0 marks round 0 (the empty round) as the golden
    # baseline plot_train() draws separately; one entry per round.
    x_range = range(0, len(data.rounds))

    with _spy_fig_paths() as paths:
        fig = sfv.plot_train(data, x_range)

    assert fig is not None
    assert len(paths) == 1
    assert os.path.exists(paths[0])


@pytest.mark.visual
def test_colormap_renders_and_writes_a_file() -> None:
    """colormap() renders the standalone accuracy color scale without error
    and writes a file."""
    with _spy_fig_paths() as paths:
        fig = sfv.colormap()

    assert fig is not None
    assert len(paths) == 1
    assert os.path.exists(paths[0])


@pytest.mark.visual
def test_heat_reshape_returns_the_largest_divisor_pair_near_sqrt_n_times_ratio() -> None:
    """_heat_reshape starts from a = floor(sqrt(N * ratio)) and decrements
    until it divides N, returning (a, N / a) -- so the pair it returns is
    always the largest such a no bigger than that starting point."""
    # N=12, ratio=1.0: floor(sqrt(12)) = 3, and 3 already divides 12.
    assert sfv._heat_reshape(12, 1.0) == (3, 4)
    # N=12, ratio=4.0: floor(sqrt(48)) = 6, and 6 already divides 12.
    assert sfv._heat_reshape(12, 4.0) == (6, 2)
    # N=17 (prime), ratio=1.0: floor(sqrt(17)) = 4; 4 and 3 and 2 don't
    # divide 17, so it falls all the way down to the trivial pair (1, 17).
    assert sfv._heat_reshape(17, 1.0) == (1, 17)


@pytest.mark.visual
def test_heat_uses_documented_shape_for_a_complete_single_layer_neuron_sweep(
        conv_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """When a layer's fault count N equals prod(its neuron shape) -- a
    complete single-layer sweep -- heat() with preserve_dim=True must use
    the documented (H*W, C) plot shape, not the generic _heat_reshape
    fallback. The net's neuron shape (C=2, H=4, W=8) has three distinct
    dimensions, so the assertion below also pins the H/W axis order."""
    cmpn = make_campaign(conv_net.net, conv_net.shape_in, slayer)
    cmpn.inject_complete(DeadNeuron(), layer_names='SC1')
    _, test_loader = tiny_loaders(conv_net.shape_in)
    cmpn.run(test_loader, es_tol=0, progress_mode='silent')
    data = cmpn.export()

    shape = data.layers_info.shapes_neu['SC1']
    n_faults = prod(shape)
    key = ('SC1', DeadNeuron())
    # Guard: the sweep is actually complete (N == prod(shape)), the
    # precondition the documented (H*W, C) shape branch relies on.
    assert len(sfv._data_mapping(data, layer='SC1')[key][0]) == n_faults

    figs = sfv.heat(data, layer='SC1', fault_model=DeadNeuron(), preserve_dim=True, to_save=False)
    assert figs, 'heat() produced no figures; not a real check.'

    plotted_shape = figs[0].axes[0].images[0].get_array().shape
    assert plotted_shape == (shape[1] * shape[2], shape[0])

    expected_plot_shape = (shape[1] * shape[2], shape[0])
    actual_plot_shape = figs[0].axes[0].images[0].get_array().shape
    assert actual_plot_shape == expected_plot_shape
    # Guard against the fallback coincidentally matching: the generic
    # reshape's own arithmetic must land on a visibly different shape.
    assert sfv._heat_reshape(n_faults, 1.0) != expected_plot_shape


@pytest.mark.visual
def test_title_names_the_single_shared_model_across_several_layers(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """When a figure covers several (layer, model) keys that all carry the
    same model, its filename identifies that model and its leading argument.
    A friendly name supersedes it, since that already names the model, and
    keys carrying different models make the figure comparative instead."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF2', (0, 0, 0))))
    _, test_loader = tiny_loaders(dense_net.shape_in)
    cmpn.run(test_loader, es_tol=0, progress_mode='silent')
    data = cmpn.export()

    data_map = sfv._data_mapping(data)
    # Guard: the shared-model branch is the one actually under test -- more
    # than one key, every one of them carrying the same model.
    assert len(data_map) > 1
    assert len({fm for _, fm in data_map}) == 1

    model = next(iter(data_map))[1]
    title = sfv._title(data, data_map, None, 'scatter', None, 'svg')
    assert f'_{model.get_name()}{int(model.args[0])}' in title

    friendly = sfv._title(data, data_map, 'dead', 'scatter', None, 'svg')
    assert model.get_name() not in friendly
    assert '_dead' in friendly


@pytest.mark.visual
def test_title_marks_a_figure_comparative_when_its_models_differ(
        dense_net: NetSpec,
        slayer: spikeLayer,
        make_campaign: Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign],
        tiny_loaders: Callable[..., tuple[DataLoader, DataLoader]]
) -> None:
    """Keys carrying more than one distinct model make the figure a
    comparison, so its filename says so rather than naming any one model."""
    cmpn = make_campaign(dense_net.net, dense_net.shape_in, slayer)
    cmpn.inject(sff.Fault(DeadNeuron(), sff.FaultSite('SF1', (0, 0, 0))), round_idx=0)
    cmpn.then_inject(sff.Fault(SaturatedNeuron(), sff.FaultSite('SF1', (1, 0, 0))))
    _, test_loader = tiny_loaders(dense_net.shape_in)
    cmpn.run(test_loader, es_tol=0, progress_mode='silent')
    data = cmpn.export()

    data_map = sfv._data_mapping(data)
    assert len({fm for _, fm in data_map}) > 1, 'Not a multi-model mapping; not a real check.'

    assert '_comparative' in sfv._title(data, data_map, None, 'scatter', None, 'svg')
