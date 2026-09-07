"""Subparser registry gluing each concrete Experiment to the command line."""

import argparse
from collections.abc import Callable
from typing import get_args

import demo

from .benchmark_a import BenchmarkA
from .benchmark_b import BenchmarkB
from .context import CaseStudyContext
from .experiment import EXPERIMENTS
# Importing experiments here is what runs each experiment module's
# @register decorator and populates EXPERIMENTS as a side effect.
from . import experiments  # noqa: F401


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


LEGACY_BENCHMARKS: dict[str, Callable[[], None]] = {
    'A': run_a,
    'B': run_b
}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--casestudy', choices=get_args(demo.SUPPORTED_CASE_STUDIES), required=True,
        help="Case study to run the experiment against."
    )
    parser.add_argument('--session', type=int, default=1, help="Session id stamped into output filenames.")
    parser.add_argument('--tags', nargs='*', default=(), help="Extra filename tags.")
    parser.add_argument('--seed', type=int, default=0, help="Seed for shuffling the trial schedule.")
    parser.add_argument('--n-warmup', type=int, default=1, help="Warmup batches run per distinct data shape.")
    parser.add_argument('--no-shuffle', action='store_true', help="Do not shuffle the trial schedule.")
    parser.add_argument('--out-dir', default=None, help="Write output files here instead of examples/out.")
    parser.add_argument('--dry-run', action='store_true', help="Print the schedule without running it.")
    parser.add_argument(
        '--resume', action='store_true', help="Skip trials already recorded in the output trials CSV."
    )
    parser.add_argument(
        '--run-unresolvable', action='store_true',
        help="Run cells even where the resolvability precondition is not met."
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level CLI parser: one subcommand per registered
    Experiment plus the legacy 'A'/'B' benchmark entry points.
    """

    parser = argparse.ArgumentParser(description="Run a SpikeFI demo benchmark.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    for name, cls in sorted(EXPERIMENTS.items()):
        sub = subparsers.add_parser(name, help=f"Run the '{name}' experiment.")
        _add_common_arguments(sub)
        cls.add_arguments(sub)

    for name in sorted(LEGACY_BENCHMARKS):
        subparsers.add_parser(name, help=f"Run legacy benchmark {name}.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in LEGACY_BENCHMARKS:
        LEGACY_BENCHMARKS[args.command]()
        return

    experiment_cls = EXPERIMENTS[args.command]
    context = CaseStudyContext(args.casestudy)
    experiment = experiment_cls.from_args(context, args)

    path = experiment.run(dry_run=args.dry_run, resume=args.resume, run_unresolvable=args.run_unresolvable)
    print(path)


if __name__ == '__main__':
    main()
