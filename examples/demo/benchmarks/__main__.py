import argparse

from .benchmark_a import BenchmarkA
from .benchmark_b import BenchmarkB


def run_a() -> None:
    bm = BenchmarkA(
        casestudy='nmnist_cnn',
        session_id=3,
        n_reps=40,
        n_warmup=3,
        layers=['SC1', 'SC2', 'SC3', 'SF4a', 'SF4b'],
        batch_size=1,
        golden_on=True,
        use_synthetic=True,
        syn_n_batches=1000,
        syn_n_time_bins=50
    )
    bm.run()


def run_b() -> None:
    nt_sweep = [(4, 94), (4, 375), (4, 1500), (16, 94), (16, 375), (16, 1500)]

    bm = BenchmarkB(
        casestudy='nmnist_cnn',
        session_id=1,
        n_reps=10,
        n_warmup=3,
        layer='SC1',
        nt_sweep=nt_sweep,
        golden_on=True,
        use_synthetic=True,
        syn_n_batches=20
    )
    bm.run()


BENCHMARKS = {
    'A': run_a,
    'B': run_b
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SpikeFI demo benchmark.")
    parser.add_argument('benchmark', choices=sorted(BENCHMARKS), help="Which benchmark to run.")
    args = parser.parse_args()

    BENCHMARKS[args.benchmark]()


if __name__ == '__main__':
    main()
