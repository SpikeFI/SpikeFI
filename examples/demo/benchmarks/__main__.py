import argparse

from .benchmark_a import BenchmarkA
from .benchmark_b import BenchmarkB


def run_a() -> None:
    bm = BenchmarkA(
        casestudy='nmnist_cnn',
        layers=['SC1', 'SC2', 'SC3', 'SF4a', 'SF4b'],
        batch_size=1,
        n_reps=40,
        golden_on=True,
        session_id=3,
        use_synthetic=True,
        syn_n_batches=1000,
        syn_n_time_bins=50
    )
    bm.run()


def run_b() -> None:
    t_sweep = [10, 25, 50, 100, 200, 375, 750, 1500, 3000, 6000]
    b_sweep = []
    cross_check = [
        (4, 375), (16, 94),
        (4, 1500), (16, 375)
    ]

    bm = BenchmarkB(
        casestudy='nmnist_cnn',
        layer='SC1',
        syn_n_batches=20,
        n_reps=10,
        session_id=1,
        t_sweep=t_sweep,
        b_sweep=b_sweep,
        cross_check=cross_check
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
