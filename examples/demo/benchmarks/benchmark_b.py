import random
from time import time
from typing import Literal

from matplotlib.figure import Figure
import pandas as pd
from torch.utils.data import DataLoader

import demo
from spikefi.fault import Fault, FaultSite
from spikefi.models import DeadNeuron

from .benchmark import Benchmark
from .io import build_csv_fname, FnameInfo


class BenchmarkB(Benchmark):
    def __init__(
        self,
        casestudy: demo.SUPPORTED_CASE_STUDIES,
        session_id: int,
        n_reps: int,
        n_warmup: int,
        layer: str,
        nt_sweep: list[tuple[int, int]],
        golden_on: bool,
        use_synthetic: bool,
        syn_n_batches: int,
        subset_div: int = 1
    ) -> None:
        super().__init__(
            casestudy, session_id, n_reps, n_warmup,
            use_synthetic, syn_n_batches, subset_div
        )

        self.layer = layer
        self.nt_sweep = nt_sweep
        self.golden_on = golden_on

    @classmethod
    def _kwargs_from_csv(
        cls,
        info: FnameInfo,
        df: pd.DataFrame,
        faulty: pd.DataFrame
    ) -> dict[str, object]:
        nt_sweep = sorted(set(zip(df['batch_size'], df['time_bins'])))
        n_warmup = int((df.kind == 'warmup').sum() / len(nt_sweep))

        return dict(
            casestudy=info.casestudy,
            session_id=info.session_id,
            n_reps=info.n_reps,
            n_warmup=n_warmup,
            layer=faulty['layer'].iloc[0],
            nt_sweep=nt_sweep,
            golden_on=bool((df.kind == 'golden').any()),
            use_synthetic=info.use_synthetic,
            syn_n_batches=info.n_batches
        )

    def _get_loader(
        self,
        loader_cache: dict[tuple[int, int], DataLoader],
        pair: tuple[int, int]
    ) -> DataLoader:
        if pair not in loader_cache:
            batch_size, n_time_bins = pair
            loader_cache[pair] = DataLoader(
                dataset=self.build_dataset(batch_size, n_time_bins),
                shuffle=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True
            )
        return loader_cache[pair]

    def run(self) -> None:
        print("Preparing benchmarks...")
        self.prepare()

        nt_sweep = self.nt_sweep

        # Every (batch_size, n_time_bins) pair needs its own dataset/loader, so trials are
        # built and shuffled across all pairs at once (not per pair). This spreads any
        # systemic drift (e.g. thermal throttling) evenly across both fault types and
        # N/T configurations, instead of letting it bias whichever pair happens to run
        # later in a fixed sweep order. Loaders are cached per pair (built once, on first
        # use) so full cross-pair shuffling doesn't repeatedly pay dataset-generation cost.
        trials = [(pair, ftype, rep)
                  for pair in nt_sweep
                  for ftype in self.fmodels
                  for rep in range(self.n_reps)]
        random.shuffle(trials)

        golden_every = len(nt_sweep) * len(self.fmodels)
        schedule: list[tuple] = []
        for i, trial in enumerate(trials):
            if self.golden_on and i % golden_every == 0:
                golden_rep = i // golden_every
                pair = trial[0]
                schedule += [(pair, 'golden', golden_rep), (pair, 'golden_spikefi', golden_rep)]
            schedule.append(trial)

        loader_cache: dict[tuple[int, int], DataLoader] = {}
        warmed_up: set[tuple[int, int]] = set()

        self.rows = []
        t_session = time()
        trial_i = 0

        for pair, ftype, rep in schedule:
            batch_size, n_time_bins = pair
            test_loader = self._get_loader(loader_cache, pair)

            if pair not in warmed_up:
                warmed_up.add(pair)
                print(f"\n=== N={batch_size}  T={n_time_bins}  "
                      f"(n_samples={batch_size * self.syn_n_batches}) ===")

                for w in range(self.n_warmup):
                    t_rel = time() - t_session
                    trial_i += 1

                    t_exec, t_wall, t_setup, _ = self._run_campaign(
                        Fault(DeadNeuron(), FaultSite(self.layer)),
                        'warmup', test_loader
                    )

                    row = dict(trial=trial_i, t_rel=round(t_rel, 3))
                    row |= dict(kind='warmup', ftype='-', layer=self.layer,
                                batch_size=batch_size, time_bins=n_time_bins,
                                rep=w, t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)
                    self.rows.append(row)
                    print(f"[{trial_i:4d}] N={batch_size:3d} T={n_time_bins:5d} "
                          f"{row['kind']:16s} {row['ftype']:14s} {t_exec:7.4f} s")

            t_rel = time() - t_session
            trial_i += 1

            row = dict(trial=trial_i, t_rel=round(t_rel, 3))

            if ftype == 'golden':
                t_exec, _ = self._run_golden(test_loader)
                row |= dict(kind='golden', ftype='-', layer='-',
                            batch_size=batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=0.0, t_exec=t_exec, t_wall=t_exec)
            elif ftype == 'golden_spikefi':
                t_exec, t_wall, t_setup, _ = self._run_campaign(None, 'golden_spikefi', test_loader)
                row |= dict(kind='golden_spikefi', ftype='-', layer='-',
                            batch_size=batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)
            else:
                fault = Fault(self.fmodels[ftype](self.net, self.layer), FaultSite(self.layer))
                t_exec, t_wall, t_setup, _ = self._run_campaign(
                    fault, f'{ftype}_r{rep}', test_loader
                )
                row |= dict(kind='faulty', ftype=ftype, layer=self.layer,
                            batch_size=batch_size, time_bins=n_time_bins,
                            rep=rep, t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)

            self.rows.append(row)
            print(f"[{trial_i:4d}] N={batch_size:3d} T={n_time_bins:5d} "
                  f"{row['kind']:16s} {row['ftype']:14s} {t_exec:7.4f} s")

        self.csv_fname = build_csv_fname(FnameInfo(
            benchmark='B',
            casestudy=self.casestudy,
            use_synthetic=True,
            n_reps=self.n_reps,
            n_batches=self.syn_n_batches,
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
        raise NotImplementedError("BenchmarkB.plot is not implemented yet.")
