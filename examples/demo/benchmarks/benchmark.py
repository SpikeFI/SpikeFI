from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import os
from time import time
from typing import Callable, Literal

from matplotlib import colormaps
from matplotlib.figure import Figure
import pandas as pd
import slayerSNN as snn
import torch
from tonic import transforms
from torch.utils.data import DataLoader, Dataset, Subset

import demo
from spikefi.core import Campaign
from spikefi.fault import Fault, FaultModel
from spikefi.models import (
    BitflippedSynapse, DeadNeuron, DeadSynapse, ThresholdFaultNeuron
)
from spikefi.utils.io import make_out_filepath
from spikefi.utils.quantization import qargs_from_tensor

from .io import FnameInfo, parse_fname
from .synthetic import SyntheticSpikeDataset


def bitflipped_synapse(net: torch.nn.Module, lay_name: str) -> BitflippedSynapse:
    qdtype = torch.quint8
    W = getattr(net, lay_name).weight
    scale, zero_point = qargs_from_tensor(W, qdtype)

    return BitflippedSynapse(3, scale, zero_point, qdtype)


FaultModelFactory = Callable[[torch.nn.Module, str], FaultModel]

COLORS = colormaps['Paired'].colors


@dataclass
class OverheadData:
    means: pd.DataFrame  # 't_setup', 't_setup_wall', 't_exec' per (kind, ftype, layer), pre-normalization
    abs_ovh: pd.DataFrame  # 't_setup', 't_setup_wall', 't_exec'
    abs_std: pd.DataFrame  # 't_setup', 't_setup_wall', 't_exec'
    ftypes_active: set[str]


class Benchmark(ABC):
    FTYPES = {
        'neuron_hard': {
            'name': 'Neuron (hard)',
            'color': COLORS[1],
            'model': lambda net, lay: DeadNeuron(),
            'active': True
        },
        'synapse_hard': {
            'name': 'Synapse (hard)',
            'color': COLORS[3],
            'model': lambda net, lay: DeadSynapse(),
            'active': True
        },
        'neuron_param': {
            'name': 'Neuron (param.)',
            'color': COLORS[0],
            'model': lambda net, lay: ThresholdFaultNeuron(0.8),
            'active': True
        },
        'synapse_param': {
            'name': 'Synapse (param.)',
            'color': COLORS[2],
            'model': lambda net, lay: bitflipped_synapse(net, lay),
            'active': True
        }
    }

    def __init__(
        self,
        *,
        casestudy: demo.SUPPORTED_CASE_STUDIES,
        session_id: int,
        n_reps: int,
        n_warmup: int = 3,
        use_synthetic: bool = True,
        syn_n_batches: int,
        subset_div: int = 1
    ) -> None:
        self.casestudy = casestudy
        self.session_id = session_id
        self.n_reps = n_reps
        self.n_warmup = n_warmup
        self.use_synthetic = use_synthetic
        self.syn_n_batches = syn_n_batches
        self.subset_div = subset_div

        self.net: torch.nn.Module | None = None
        self.rows: list[dict] = []
        self.csv_fname: str | None = None
        self.overhead: OverheadData | None = None

    @property
    def fmodels(self) -> dict[str, FaultModelFactory]:
        return {ftype: entry['model'] for ftype, entry in self.FTYPES.items() if entry['active']}

    def _csv_fpath(self) -> str:
        if self.csv_fname is None:
            raise RuntimeError("No results to plot yet; call run() first.")

        return make_out_filepath(self.csv_fname)

    @classmethod
    def from_csv(cls, fpath: str) -> 'Benchmark':
        info = parse_fname(fpath)
        df = pd.read_csv(fpath)
        faulty = df[df.kind == 'faulty']

        bm = cls(**cls._kwargs_from_csv(info, df, faulty))
        bm.rows = df.to_dict('records')
        bm.csv_fname = os.path.basename(fpath)

        return bm

    @classmethod
    @abstractmethod
    def _kwargs_from_csv(
        cls,
        info: FnameInfo,
        df: pd.DataFrame,
        faulty: pd.DataFrame
    ) -> dict[str, object]:
        ...

    def prepare(self) -> None:
        demo.prepare(casestudy=self.casestudy)
        self.net = demo.get_net(
            os.path.join(demo.DEMO_DIR, 'models', demo.get_fnetname())
        )

    def _run_golden(self, test_loader: DataLoader) -> tuple[float, float]:
        stats = snn.utils.stats()
        torch.cuda.synchronize()
        t0 = time()
        with torch.no_grad():
            for input, label in test_loader:
                label = label.to(demo.device)
                output = self.net(input.to(demo.device))

                predict = output.sum(dim=(2, 3, 4)).argmax(dim=1)
                stats.testing.correctSamples += (predict == label).sum().item()
                stats.testing.numSamples += label.size(0)

        torch.cuda.synchronize()
        t_exec = time() - t0
        stats.update()

        return t_exec, stats.testing.accuracyLog[-1]

    def _run_campaign(
        self,
        fault: Fault | None,
        name: str,
        test_loader: DataLoader
    ) -> tuple[float, float, float, float]:
        t_setup = time()

        cmpn = Campaign(self.net, demo.shape_in, self.net.slayer, name=name)
        if fault is not None:
            cmpn.inject(fault)

        t_setup = time() - t_setup

        torch.cuda.synchronize()
        t_wall = time()
        cmpn.run(test_loader, progress_mode='silent')
        torch.cuda.synchronize()
        t_wall = time() - t_wall

        accu = cmpn.performance[0].testing.accuracyLog[-1]
        t_exec = cmpn.duration
        del cmpn

        return t_exec, t_wall, t_setup, accu

    def _compute_overhead(
        self,
        *,
        golden_exec: float | None = None,
        golden_source: Literal['csv', 'arg'] = 'csv'
    ) -> OverheadData:
        fpath = self._csv_fpath()
        info = parse_fname(fpath)

        D = pd.read_csv(fpath)
        # 't_setup_wall' folds the portion of wall-clock time not captured by 't_exec'
        # (e.g. data loading) into the setup time, as an alternative view of setup overhead.
        D = D.assign(t_setup_wall=D['t_setup'] + (D['t_wall'] - D['t_exec']))

        cols = ['t_setup', 't_setup_wall', 't_exec']
        grp = D.groupby(['kind', 'ftype', 'layer'], sort=False, observed=True)[cols]
        M, S = grp.mean(), grp.std()

        if golden_source == 'csv':
            if not self.golden_on:
                raise RuntimeError(
                    "No 'golden' entries in the CSV to normalize overhead against. "
                    "Pass golden_exec together with golden_source='arg' to supply one explicitly."
                )
            gold = M.loc[('golden', '-', '-')]
        elif golden_source == 'arg':
            if golden_exec is None:
                raise RuntimeError("golden_source='arg' but no golden_exec was provided.")
            gold = pd.Series({'t_setup': 0.0, 't_setup_wall': 0.0, 't_exec': golden_exec})
        else:
            raise ValueError(f"Unknown golden_source: {golden_source!r}; expected 'csv' or 'arg'.")

        gold_exec = gold['t_exec']

        abs_scale = gold_exec * 1e6 / info.n_batches
        abs_ovh = (M - gold) / gold_exec * abs_scale
        abs_std = S / gold_exec * abs_scale
        if golden_source == 'csv':
            abs_ovh = abs_ovh.drop(('golden', '-', '-'))
            abs_std = abs_std.drop(('golden', '-', '-'))

        ftypes_active = set(abs_ovh.loc['faulty'].index.get_level_values('ftype'))

        self.overhead = OverheadData(means=M, abs_ovh=abs_ovh, abs_std=abs_std, ftypes_active=ftypes_active)
        return self.overhead

    @property
    def n_classes(self) -> int:
        if hasattr(self.net, "classes_out"):
            return self.net.classes_out

        fc_layers = [lay for lay in self.net.children() if isinstance(lay, snn.slayer._denseLayer)]
        return fc_layers[-1].out_channels

    def build_dataset(self, batch_size: int, n_time_bins: int | None = None) -> Dataset:
        if self.use_synthetic:
            return SyntheticSpikeDataset(
                n_samples=batch_size * self.syn_n_batches,
                shape_in=demo.shape_in,
                n_time_bins=n_time_bins,
                n_classes=self.n_classes
            )

        ds = demo.get_cached_dataset(
            train=False,
            transform=transforms.Denoise(filter_time=10000)
        )
        if self.subset_div > 1:
            ds = Subset(ds, range(int(len(ds) / int(self.subset_div))))

        return ds

    def save_results(self, fname: str) -> None:
        fpath = make_out_filepath(fname)
        with open(fpath, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"\nWrote {len(self.rows)} rows to {fpath}")

    @abstractmethod
    def run(self) -> None:
        ...

    @abstractmethod
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
        ...
