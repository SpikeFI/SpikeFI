# This file is part of SpikeFI.
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6

# SpikeFI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SpikeFI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


from collections.abc import Iterable
from cycler import cycler
from difflib import SequenceMatcher
from itertools import cycle
from math import prod, sqrt
import numpy as np
import re
import torch

import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.patheffects as path_effects

from spikefi.core import CampaignData
import spikefi.fault as sff
from spikefi.utils.io import make_fig_filepath


CMAP = 'jet'
CPAL = [
    "#02580E", "#9C7720", "#104280", "#B7312C", "#DC996C", "#5F1B08", "#FFD19E"
]


def _data_mapping(
        cmpns_data: CampaignData | Iterable[CampaignData],
        layer: str = None,
        fault_model: sff.FaultModel = None
) -> dict[tuple[str, sff.FaultModel], dict[int, list[int]]]:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    # structure -> { (layer, fault model): { campaign index, [round index] } }
    data_map: dict[
        tuple[
            str, sff.FaultModel
        ],
        dict[
            int,
            list[int]
        ]
    ] = {}

    for cmpn_idx, cmpn_data in enumerate(cmpns_data):
        for lay, r_idxs in cmpn_data.rgroups.items():
            for r in r_idxs:
                round = cmpn_data.rounds[r]
                if len(round) > 1 or (layer and lay != layer):
                    continue
                key = next(iter(round.keys()))

                if fault_model and fault_model not in key:
                    continue

                data_map.setdefault(key, {})
                data_map[key].setdefault(cmpn_idx, [])
                data_map[key][cmpn_idx].append(r)

    return data_map


def _earth_palette() -> None:
    mpl.rcParams['axes.prop_cycle'] = cycler(color=CPAL)


def _heat_reshape(N: int, R: float) -> tuple[int, int]:
    a = int(sqrt(N * R))
    while N % a:
        a -= 1

    return (a, int(N / a))


def _title(
        cmpns_data: CampaignData | Iterable[CampaignData],
        data_map: dict,
        model_friendly: str,
        plot_type: str,
        title_suffix: str,
        format: str
) -> str:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    title_def = ""

    if len(data_map) == 1 and plot_type == "heat":
        fm = next(iter(data_map.keys()))[1]
        title_def = "_" + str(int(fm.args[0]))
    elif len(data_map) > 1:
        model = next(iter(data_map.keys()))[1]
        one_m = True
        for _, fm in data_map.keys():
            one_m &= fm == model

        if one_m:
            if model_friendly:
                title_def = ""
            else:
                f"_{model.get_name()}{int(fm.args[0])}"
        else:
            title_def = "_comparative"

    if model_friendly:
        model_friendly = "_" + model_friendly.strip('_')
    if title_suffix:
        title_suffix = "_" + title_suffix.strip('_')

    cmpn_name = cmpns_data[0].name
    for cmpn_data in cmpns_data:
        match = SequenceMatcher(
            None, cmpn_name, cmpn_data.name
        ).find_longest_match()
        cmpn_name = cmpn_name[match.a:match.a + match.size]

    return (
        f"{cmpn_name.strip('_')}"
        + f"{model_friendly or ''}"
        + f"{title_def}_"
        + f"{plot_type}"
        + f"{title_suffix or ''}."
        + f"{format.strip('.')}"
    )


def bar(
        cmpns_data: CampaignData | Iterable[CampaignData],
        model_friendly: str = None,
        fig_size: tuple[float, float] = None,
        title_suffix: str = None,
        format: str = 'svg',
        to_save: bool = True
) -> Figure:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    data_map = _data_mapping(cmpns_data)

    offset_mult: dict[str, int] = {}
    layers = sorted(list(set(key[0] for key in data_map.keys())))

    fig, ax = plt.subplots(figsize=fig_size)
    colormap = plt.get_cmap(CMAP)
    width = 3 / len(data_map)
    space = width / 5

    for (lay, fm), cmpn_dict in data_map.items():
        offset_mult.setdefault(lay, 0)

        N = 0
        perf = []
        for cmpn_idx, r_idxs in cmpn_dict.items():
            N += len(r_idxs)
            for r in r_idxs:
                test_stats = cmpns_data[cmpn_idx].performance[r].testing
                perf.append(test_stats.maxAccuracy * 100.0)
        perf = np.array(perf)

        groups_freq = np.bincount(perf.round().astype(int), minlength=101)
        groups_cent = groups_freq / N * 100.0

        bottom = 0.0
        for i in range(101):
            offset = (width + space) * offset_mult[lay]
            b = ax.bar(layers.index(lay) + offset, groups_cent[i], width,
                       bottom=bottom, color=colormap(i / 100.0))

            bottom += groups_cent[i]

        bar_labels = ax.bar_label(
            b,
            labels=[fm.get_name()[:4] + "."],
            rotation=90,
            color='white',
            padding=-35
        )

        for bl in bar_labels:
            bl.set_path_effects([
                path_effects.Stroke(linewidth=1.25, foreground='gray'),
                path_effects.Normal()
            ])

        offset_mult[lay] += 1

    ax.set_ylabel("Faults (%)")
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel('Layers')
    ax.set_xticks(
        [
            i + (width / 2 + space / 2) * (offset_mult[lay] - 1)
            for i, lay in enumerate(layers)
        ],
        layers
    )

    if to_save:
        plot_path = make_fig_filepath(
            _title(
                cmpns_data,
                data_map,
                model_friendly,
                "bar",
                title_suffix,
                format
            )
        )
        plt.savefig(plot_path, bbox_inches='tight', transparent=False)

    return fig


def colormap(format: str = 'svg', to_save: bool = True) -> Figure:
    fig = plt.figure()
    fig.set_size_inches(fig.get_figwidth(), 1)

    cax = fig.add_axes([.05, .55, .9, .25])
    norm = mpl.colors.Normalize(0, 100)

    cbar = mpl.colorbar.Colorbar(
        cax,
        cmap=CMAP,
        norm=norm,
        orientation='horizontal',
        ticks=range(0, 101, 10)
    )
    cbar.set_ticks(range(0, 100), minor=True)
    plt.xlabel("Classification Accuracy (%)")

    if to_save:
        plot_path = make_fig_filepath(
            fname="colormap." + format.removeprefix('.')
        )
        plt.savefig(plot_path, transparent=True)

    return fig


def heat(
        cmpns_data: CampaignData | Iterable[CampaignData],
        layer: str = None, fault_model: sff.FaultModel = None,
        preserve_dim: bool = False,
        ratio: float = 1.0,
        max_area: int = 512**2,
        show_axes: bool = True,
        model_friendly: str = None,
        fig_size: tuple[float, float] = None,
        title_suffix: str = None,
        format: str = 'svg',
        to_save: bool = True
) -> list[Figure]:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    figs = []
    data_map = _data_mapping(cmpns_data, layer, fault_model)
    for (lay, fm), cmpn_dict in data_map.items():
        N = 0
        perf = []
        local_cmpns_data = []
        for cmpn_idx, r_idxs in cmpn_dict.items():
            N += len(r_idxs)
            local_cmpns_data.append(cmpns_data[cmpn_idx])

            for r in r_idxs:
                test_stats = cmpns_data[cmpn_idx].performance[r].testing
                perf.append(test_stats.maxAccuracy)
        perf = np.array(perf, dtype=np.float32)

        if N > max_area:
            print(
                "Cannot plot heat map for the following pair "
                "of layer and fault model:"
            )
            print((lay, fm))
            print(f"Reason: too many faults (>{max_area}).")
            continue

        is_syn = fm.is_synaptic()
        if is_syn:
            shape = local_cmpns_data[0].layers_info.shapes_syn[lay]
        else:
            shape = local_cmpns_data[0].layers_info.shapes_neu[lay]

        if N != prod(shape):
            plot_shape = (1, N)
        else:
            if is_syn:
                if shape[2] == 1 and shape[3] == 1:
                    plot_shape = (shape[0], shape[1])
                else:
                    plot_shape = (shape[0] * shape[1], shape[2] * shape[3])
            else:
                plot_shape = (shape[1] * shape[2], shape[0])

        if (
            not preserve_dim
            or plot_shape[0] > sqrt(max_area)
            or plot_shape[1] > sqrt(max_area)
        ):
            plot_shape = _heat_reshape(N, ratio or 1.0)

        fig = plt.figure(str((lay, fm)), figsize=fig_size)

        hx = int(plot_shape[0] / 100.) + 1
        wx = int(plot_shape[1] / 100.) + 1
        fig.set_size_inches(wx * fig.get_figwidth(), hx * fig.get_figheight())

        pos = plt.imshow(
            perf.reshape(*plot_shape),
            cmap=CMAP,
            origin='lower',
            vmin=0.,
            vmax=1.,
            interpolation='nearest',
            extent=[0, plot_shape[1], 0, plot_shape[0]]
        )

        pos.axes.set_xticks(
            [1] + np.arange(10, plot_shape[1] + 1, 10).tolist()
        )
        pos.axes.set_xticklabels(
            [1] + np.arange(10, plot_shape[1] + 1, 10).tolist()
        )
        pos.axes.set_xticks(np.arange(1, plot_shape[1]), minor=True)

        pos.axes.set_yticks(
            [1] + np.arange(10, plot_shape[0] + 1, 10).tolist()
        )
        pos.axes.set_yticklabels(
            [1] + np.arange(10, plot_shape[0] + 1, 10).tolist()
        )
        pos.axes.set_yticks(np.arange(1, plot_shape[0]), minor=True)

        pos.axes.tick_params(axis='both', which='both', length=0)
        pos.axes.grid(which='both', linestyle='-')

        if not show_axes:
            pos.axes.set_xticklabels([])
            pos.axes.set_yticklabels([])

        if to_save:
            plot_path = make_fig_filepath(
                _title(
                    local_cmpns_data,
                    {(lay, fm): cmpn_dict},
                    model_friendly, "heat",
                    title_suffix,
                    format
                )
            )
            plt.savefig(plot_path, bbox_inches='tight', transparent=False)

        figs.append(fig)

    return figs


def plot(
        cmpns_data: CampaignData | Iterable[CampaignData],
        layer: str = None,
        legend_loc: str = "lower right",
        xlabel: str = '',
        model_friendly: str = None,
        fig_size: tuple[float, float] = None,
        title_suffix: str = None,
        format: str = 'svg',
        to_save: bool = True
) -> Figure:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    data_map = _data_mapping(cmpns_data, layer)
    data_map_sorted = dict(
        sorted(
            data_map.items(),
            key=lambda item: item[0][0]
        )
    )
    curves: dict[str, list[tuple[float, np.array[float]]]] = {}

    for (lay, fm), cmpn_dict in data_map_sorted.items():
        curves.setdefault(lay, [])
        perf = []
        for cmpn_idx, r_idxs in cmpn_dict.items():
            for r in r_idxs:
                test_stats = cmpns_data[cmpn_idx].performance[r].testing
                perf.append(test_stats.maxAccuracy)

        a = fm.param_args[0] * 100. if fm.is_parametric() else fm.args[0]
        curves[lay].append((a, np.array(perf) * 100.))

    fig = plt.figure(figsize=fig_size)
    markers = cycle(["D", "s", "*", "X", "p", "^"])
    _earth_palette()

    for lay in curves.keys():
        curves[lay].sort(key=lambda tup: tup[0])

        x, y, y_l, y_h = [], [], [], []
        for pair in curves[lay]:
            x.append(pair[0])
            y.append(pair[1].mean())
            y_l.append(pair[1].min())
            y_h.append(pair[1].max())

        plt.fill_between(x, y_h, y_l, alpha=0.2)
        plt.plot(x, y, marker=next(markers), label=lay)

        plt.ylim((0, 100))
        plt.yticks(range(0, 101, 10))
        plt.ylabel("Classification accuracy (%)")

        plt.xlim(np.min(x), np.max(x))
        if np.min(x) not in list(plt.xticks()[0]):
            plt.xticks(list(plt.xticks()[0]) + [np.min(x)])
        plt.xlabel(xlabel)

        plt.minorticks_on()
        plt.grid(visible=True, which='major', axis='y', alpha=0.5)
        plt.grid(visible=True, which='minor', axis='y', linestyle='dotted')

    plt.legend(loc=legend_loc)

    if to_save:
        plot_path = make_fig_filepath(
            _title(
                cmpns_data,
                data_map,
                model_friendly,
                "scatter",
                title_suffix,
                format
            )
        )
        plt.savefig(plot_path, bbox_inches='tight', transparent=False)

    return fig


def plot_train(
        cmpns_data: CampaignData | Iterable[CampaignData],
        x_range: range,
        fig_size: tuple[float, float] = None,
        title_suffix: str = None,
        format: str = 'svg',
        to_save: bool = True
) -> Figure:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    accu = [
        [
            perf.training.maxAccuracy
            for perf in cmpn_data.performance
        ] for cmpn_data in cmpns_data
    ]
    mean_accu = np.array(accu).mean(axis=0)

    x_slice = slice(None) if x_range[0] else slice(1, None, None)

    fig = plt.figure(figsize=fig_size)
    _earth_palette()
    if not x_range[0]:
        plt.axhline(y=mean_accu[0] * 100., label='Golden', color=CPAL[1])
    plt.plot(
        x_range[x_slice],
        mean_accu[x_slice] * 100.,
        marker='d',
        linestyle='-',
        linewidth=0.7,
        label='Faulty'
    )
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
    plt.ylim([60, 100])
    plt.xlim([x_range[x_slice][0], x_range[-1]])
    plt.xlabel("Dead neurons (%)")
    plt.ylabel("Mean classification accuracy (%)")
    plt.legend()

    if to_save:
        if title_suffix:
            title_suffix = "_" + title_suffix.strip('_')
        common_name = re.sub(r'(_net)\d+', r'\1', cmpns_data[0].name)

        plot_path = make_fig_filepath(
            f"{common_name}_mean{title_suffix or ''}.{format.strip('.')}"
        )
        plt.savefig(plot_path, bbox_inches='tight', transparent=False)

    return fig


def learning_curve(
        cmpns_data: CampaignData | Iterable[CampaignData],
        fig_size: tuple[float, float] = None,
        title_suffix: str = None,
        format: str = 'svg',
        to_save: bool = True
) -> list[Figure]:
    if isinstance(cmpns_data, CampaignData):
        cmpns_data = [cmpns_data]

    figs = list()
    for cmpn_data in cmpns_data:
        n_rounds = len(cmpn_data.rounds)
        fig = plt.figure(figsize=fig_size)

        for r, perf in enumerate(cmpn_data.performance):
            epochs = len(perf.training.accuracyLog)

            plt.plot(
                range(1, epochs + 1),
                torch.Tensor(perf.training.accuracyLog) * 100.,
                label='Training' + (f' (round {r})' if n_rounds > 1 else '')
            )
            if all(accu for accu in perf.testing.accuracyLog):
                plt.plot(
                    range(1, epochs + 1),
                    torch.Tensor(perf.testing.accuracyLog) * 100.,
                    label='Testing' + (f' (round {r})' if n_rounds > 1 else '')
                )

            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel("Accuracy (%)")
            plt.xlim((1, epochs))
            plt.ylim((0., 100.))

            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            if to_save:
                if title_suffix:
                    title_suffix = "_" + title_suffix.strip('_')

                plot_path = make_fig_filepath(
                    f"{cmpn_data.name}_learning{title_suffix or ''}."
                    + f"{format.strip('.')}"
                )
                plt.savefig(plot_path, bbox_inches='tight', transparent=False)

        figs.append(fig)

    return figs
