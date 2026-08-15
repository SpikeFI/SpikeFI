import random
from time import time
from typing import Literal

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import demo
from spikefi.fault import Fault, FaultSite
from spikefi.models import DeadNeuron
from spikefi.utils.io import make_fig_filepath

from .benchmark import Benchmark, OverheadData
from .io import build_csv_fname, build_fig_fname, FnameInfo, parse_fname


class BenchmarkA(Benchmark):
    LAYERS = ['SC1', 'SC2', 'SC3', 'SF4a', 'SF4b']

    def __init__(
        self,
        casestudy: demo.SUPPORTED_CASE_STUDIES,
        session_id: int,
        n_reps: int,
        n_warmup: int,
        layers: list[str],
        batch_size: int,
        golden_on: bool,
        use_synthetic: bool,
        syn_n_batches: int,
        syn_n_time_bins: int,
        subset_div: int = 1
    ) -> None:
        super().__init__(
            casestudy, session_id, n_reps, n_warmup,
            use_synthetic, syn_n_batches, subset_div
        )

        self.layers = layers
        self.batch_size = batch_size
        self.golden_on = golden_on
        self.syn_n_time_bins = syn_n_time_bins

    @classmethod
    def _kwargs_from_csv(
        cls,
        info: FnameInfo,
        df: pd.DataFrame,
        faulty: pd.DataFrame
    ) -> dict[str, object]:
        return dict(
            casestudy=info.casestudy,
            session_id=info.session_id,
            n_reps=info.n_reps,
            n_warmup=int((df.kind == 'warmup').sum()),
            layers=set(faulty['layer']),
            batch_size=int(df['batch_size'].iloc[0]),
            golden_on=bool((df.kind == 'golden').any()),
            use_synthetic=info.use_synthetic,
            syn_n_batches=info.n_batches,
            syn_n_time_bins=int(df['time_bins'].iloc[0])
        )

    def _render_plot(
        self,
        ovh: OverheadData,
        t_col: str,
        *,
        show_err: bool = False,
        show_gsf: bool = False,
        fig_scale: float = 1.5,
        group_w: float = 0.8
    ) -> Figure:
        layers = [layer for layer in self.LAYERS if layer in self.layers]
        ftypes = [
            ftype for ftype, entry in self.FTYPES.items()
            if entry['active'] and ftype in ovh.ftypes_active
        ]

        OVH = ovh.abs_ovh[t_col]
        STD = ovh.abs_std[t_col]
        y_label = 'Overhead per Batch (µs)'

        if show_gsf and ('golden_spikefi', '-', '-') not in OVH.index:
            print("No 'golden_spikefi' entries in the data; disabling show_gsf.")
            show_gsf = False

        bar_w = group_w / len(ftypes)
        x_labels = (['Golden\n(SpikeFI)'] if show_gsf else []) + layers
        x_ticks = np.arange(len(x_labels))
        layer_x = x_ticks[1:] if show_gsf else x_ticks
        ftype_offsets = (np.arange(len(ftypes)) - (len(ftypes) - 1) / 2) * bar_w
        err_kw = {'ecolor': '0.2', 'elinewidth': 0.8, 'capsize': 2, 'capthick': 0.8}
        n_legend_items = len(ftypes) + (1 if show_gsf else 0)

        fig, ax = plt.subplots(figsize=(fig_scale * 4.2, fig_scale * 2.9))

        if show_gsf:
            gsf_err = STD.loc[('golden_spikefi', '-', '-')] if show_err else None
            ax.bar(x_ticks[0], OVH.loc[('golden_spikefi', '-', '-')], bar_w,
                   yerr=gsf_err, color='0.55', label='Golden (SpikeFI)',
                   error_kw=err_kw if show_err else {})

        for i, ftype in enumerate(ftypes):
            vals = [OVH.loc[('faulty', ftype, lay)] for lay in layers]
            errs = [STD.loc[('faulty', ftype, lay)] for lay in layers] if show_err else None
            ax.bar(layer_x + ftype_offsets[i], vals, bar_w, yerr=errs,
                   color=self.FTYPES[ftype]['color'], label=self.FTYPES[ftype]['name'],
                   error_kw=err_kw if show_err else {})

        ax.axhline(0, color='0.3', linewidth=0.8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(r'Faulty Layer ($l_f$)')
        ax.set_ylabel(y_label)
        ax.margins(y=0.12)
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  ncol=min(n_legend_items, 3), fontsize=8, handletextpad=0.4,
                  columnspacing=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def run(self) -> None:
        print("Preparing benchmarks...")
        self.prepare()

        n_time_bins = (
            self.syn_n_time_bins if self.use_synthetic
            else int(demo.net_params['simulation']['tSample'] / demo.net_params['simulation']['Ts'])
        )

        test_loader = DataLoader(
            dataset=self.build_dataset(self.batch_size, self.syn_n_time_bins),
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )

        trials = [(fm, layer, rep)
                  for fm in self.fmodels
                  for layer in self.layers
                  for rep in range(self.n_reps)]
        random.shuffle(trials)

        golden_every = len(self.layers) * len(self.fmodels)
        schedule: list[tuple] = [('warmup', None, w) for w in range(self.n_warmup)]
        for i, trial in enumerate(trials):
            if self.golden_on and i % golden_every == 0:
                golden_rep = i // golden_every
                schedule += [('golden', None, golden_rep), ('golden_spikefi', None, golden_rep)]
            schedule.append(trial)

        print(f"Benchmarking {len(schedule)} trials of {len(test_loader)} "
              f"batches of {self.batch_size} sample(s) each...")

        self.rows = []
        t_session = time()
        for i, (ftype, layer, rep) in enumerate(schedule):
            t_rel = time() - t_session

            row = dict(trial=i, t_rel=round(t_rel, 3))

            if ftype == 'warmup':
                t_exec, t_wall, t_setup, _ = self._run_campaign(
                    Fault(DeadNeuron(), FaultSite(self.layers[0])),
                    'warmup', test_loader
                )
                row |= dict(kind='warmup', ftype='-', layer=self.layers[0],
                            batch_size=self.batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)
            elif ftype == 'golden':
                t_exec, _ = self._run_golden(test_loader)
                row |= dict(kind='golden', ftype='-', layer='-',
                            batch_size=self.batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=0.0, t_exec=t_exec, t_wall=t_exec)
            elif ftype == 'golden_spikefi':
                t_exec, t_wall, t_setup, _ = self._run_campaign(None, 'golden_spikefi', test_loader)
                row |= dict(kind='golden_spikefi', ftype='-', layer='-',
                            batch_size=self.batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)
            else:
                fault = Fault(self.fmodels[ftype](self.net, layer), FaultSite(layer))
                t_exec, t_wall, t_setup, _ = self._run_campaign(
                    fault, f'{ftype}_{layer}_r{rep}', test_loader
                )
                row |= dict(kind='faulty', ftype=ftype, layer=layer,
                            batch_size=self.batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)

            self.rows.append(row)
            print(f"[{i:4d}] {row['kind']:16s} {row['ftype']:18s} {row['layer']:5s} "
                  f"{t_exec:7.3f} s")

        self.csv_fname = build_csv_fname(FnameInfo(
            benchmark='A',
            casestudy=self.casestudy,
            use_synthetic=self.use_synthetic,
            n_reps=self.n_reps,
            n_batches=len(test_loader),
            session_id=self.session_id
        ))
        self.save_results(self.csv_fname)

    def plot(
        self,
        t_col: str,
        save_as: str = 'png',
        *,
        show_err: bool = False,
        show_gsf: bool = False,
        include_wall_overhead: bool = False,
        golden_exec: float | None = None,
        golden_source: Literal['csv', 'arg'] = 'csv'
    ) -> Figure:
        ovh = self._compute_overhead(golden_exec=golden_exec, golden_source=golden_source)

        plot_col = 't_setup_wall' if include_wall_overhead and t_col == 't_setup' else t_col
        fig = self._render_plot(ovh, plot_col, show_err=show_err, show_gsf=show_gsf)

        fig_path = make_fig_filepath(build_fig_fname(parse_fname(self._csv_fpath()), t_col, save_as))
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved plot to {fig_path}")
        return fig
