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
        *,
        casestudy: demo.SUPPORTED_CASE_STUDIES,
        layer: str,
        n_reps: int,
        session_id: int,
        t_sweep: list[int],
        b_sweep: list[int],
        cross_check: list[tuple[int, int]],
        n_warmup: int = 3,
        use_synthetic: bool = True,
        syn_n_batches: int,
        subset_div: int = 1
    ) -> None:
        super().__init__(
            casestudy=casestudy, session_id=session_id, n_reps=n_reps,
            n_warmup=n_warmup, use_synthetic=use_synthetic,
            syn_n_batches=syn_n_batches, subset_div=subset_div
        )

        self.layer = layer
        self.t_sweep = t_sweep
        self.b_sweep = b_sweep
        self.cross_check = cross_check

    @classmethod
    def _kwargs_from_csv(
        cls,
        info: FnameInfo,
        df: pd.DataFrame,
        faulty: pd.DataFrame
    ) -> dict[str, object]:
        return dict(
            casestudy=info.casestudy,
            layer=faulty['layer'].iloc[0],
            n_reps=info.n_reps,
            session_id=info.session_id,
            t_sweep=sorted(set(df['time_bins'])),
            b_sweep=sorted(set(df['batch_size'])),
            cross_check=[],
            use_synthetic=info.use_synthetic,
            syn_n_batches=info.n_batches
        )

    def run(self) -> None:
        print("Preparing benchmarks...")
        self.prepare()

        main_sweep = [(b, t) for b in self.b_sweep for t in self.t_sweep]
        sweep = main_sweep + self.cross_check

        self.rows = []
        t_session = time()
        trial_i = 0

        for batch_size, n_time_bins in sweep:
            print(f"\n=== N={batch_size}  T={n_time_bins}  "
                  f"(n_samples={batch_size * self.syn_n_batches}) ===")

            test_loader = DataLoader(
                dataset=self.build_dataset(batch_size, n_time_bins),
                shuffle=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True
            )

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

            configs = [('golden_spikefi', None)] + list(self.fmodels.items())
            for ftype, fmodel in configs:
                for rep in range(self.n_reps):
                    t_rel = time() - t_session
                    trial_i += 1

                    row = dict(trial=trial_i, t_rel=round(t_rel, 3))

                    if ftype == 'golden_spikefi':
                        t_exec, t_wall, t_setup, _ = self._run_campaign(None, 'golden_spikefi', test_loader)
                        row |= dict(kind='golden_spikefi', ftype='-', layer='-',
                                    batch_size=batch_size, time_bins=n_time_bins, rep=rep)
                    else:
                        fault = Fault(fmodel(self.net, self.layer), FaultSite(self.layer))
                        t_exec, t_wall, t_setup, _ = self._run_campaign(
                            fault, f'{ftype}_r{rep}', test_loader
                        )
                        row |= dict(kind='faulty', ftype=ftype, layer=self.layer,
                                    batch_size=batch_size, time_bins=n_time_bins, rep=rep)

                    row |= dict(t_setup=t_setup, t_exec=t_exec, t_wall=t_wall)
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
