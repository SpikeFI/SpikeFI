"""Shared scheduling and execution driver every benchmark experiment runs
through: a plan of TrialSpec cells, a resolvability precondition on each,
resumable CSV output and a provenance sidecar, all on top of the concrete
run_slayer_inference/run_spikefi primitives in runners.py.
"""

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, fields
import math
import os
import random
import time

import pandas as pd
import torch

from spikefi.utils.io import make_out_filepath

from . import naming, records
from .context import CaseStudyContext
from .measure import Resolvability, Stopwatch
from .runners import TrialOutcome


def _normalize_key(value: object) -> str:
    """
    Renders a value the same way whether it came straight from a TrialSpec
    or was round-tripped through a CSV column, so resume can match keys
    regardless of pandas' coercions (e.g. a column with any blank cell
    reads back as float64, so 30 and 30.0 must normalize identically).
    """

    if value is None or value is pd.NA:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    # bool is an int subclass, so this check must come before the int branch.
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))

    return str(value)


@dataclass(frozen=True)
class TrialSpec:
    """
    One planned unit of work. `kind` names what the trial is (e.g.
    'golden', 'faulty'); every other field is an optional configuration
    axis that stays None where an experiment does not have it. Field
    names deliberately match records.TrialRecord's, so a spec maps onto a
    CSV row with no translation table.
    """

    kind: str
    ftype: str | None = None
    layer: str | None = None
    n_faults: int | None = None
    opt: str | None = None
    es_tol: float | None = None
    n_rounds: int | None = None
    batch_size: int | None = None
    time_bins: int | None = None
    n_samples: int | None = None
    n_batches: int | None = None
    synthetic: bool | None = None
    spike_prob: float | None = None
    subset_div: int | None = None
    rep: int | None = None
    fault_seed: int | None = None
    probe: str | None = None
    pair_id: int | None = None

    def key(self) -> tuple[str, ...]:
        """Hashable identity used to match a spec against a CSV row already
        on disk when resuming, immune to pandas' round-trip coercions.
        """

        return tuple(_normalize_key(getattr(self, name)) for name in _SPEC_FIELDS)

    def as_record_fields(self) -> dict[str, object]:
        """This spec's fields as a dict, ready to splat into a TrialRecord."""

        return {name: getattr(self, name) for name in _SPEC_FIELDS}


_SPEC_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TrialSpec))


class Experiment(ABC):
    """
    Shared driver for a benchmark experiment: builds a schedule of
    TrialSpec cells, checks each against a resolvability precondition,
    resumes a partially completed run, warms up the GPU once per data
    shape, then executes and records every remaining trial.
    """

    name: str = ''

    def __init__(
            self,
            context: CaseStudyContext,
            session: int = 1,
            tags: tuple[str, ...] = (),
            seed: int = 0,
            n_warmup: int = 1,
            shuffle: bool = True,
            out_dir: str | None = None
    ) -> None:
        self.context: CaseStudyContext = context
        self.session: int = session
        self.tags: tuple[str, ...] = tags
        self.seed: int = seed
        self.n_warmup: int = n_warmup
        self.shuffle: bool = shuffle
        self.out_dir: str | None = out_dir

    @abstractmethod
    def plan(self) -> list[TrialSpec]:
        """The full trial schedule. Must be deterministic: resume matches
        on TrialSpec.key(), so a plan that varies between processes cannot
        be resumed against.
        """
        ...

    @abstractmethod
    def execute(self, spec: TrialSpec) -> TrialOutcome:
        """Runs the one trial spec describes and returns its outcome."""
        ...

    def resolvability(self, spec: TrialSpec) -> Resolvability | None:
        """Precondition check for whether spec's expected signal can clear
        the timing-noise floor. None means no precondition is declared for
        this spec, i.e. it is always run.
        """

        return None

    def estimate_seconds(self, spec: TrialSpec, t_batch: float | None) -> float | None:
        """
        Plain-inference lower bound on spec's wall time: n_batches rounds
        of one pass each, at t_batch seconds per pass. Counts neither
        per-round setup cost nor the effect of optimization levels on a
        fault campaign, so a real trial's wall time is generally higher
        than this. Returns None when either input is unknown.
        """

        if spec.n_batches is None or t_batch is None:
            return None

        n_rounds = spec.n_rounds if spec.n_rounds is not None else 1
        return spec.n_batches * max(n_rounds, 1) * t_batch

    def config(self) -> dict[str, object]:
        """Resolved configuration recorded in the run's sidecar JSON."""

        return {
            'casestudy': self.context.casestudy,
            'session': self.session,
            'tags': list(self.tags),
            'seed': self.seed,
            'n_warmup': self.n_warmup,
            'shuffle': self.shuffle,
            'experiment': self.name,
            'n_trials': len(self.plan())
        }

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for a subclass to register its own CLI arguments. No-op by
        default.
        """

        return None

    def run(
            self,
            dry_run: bool = False,
            resume: bool = False,
            run_unresolvable: bool = False
    ) -> str:
        """
        Runs every planned trial through execute() and writes the trials
        and rounds CSVs plus a provenance sidecar JSON next to them.
        Returns the trials CSV path.

        t_rel restarts at zero for a resumed process rather than
        continuing the original run's elapsed clock, since the two
        processes do not share a start time.
        """

        scheduled = self._ordered_schedule()
        verdicts = [(spec, self.resolvability(spec)) for spec in scheduled]
        dropped = [(spec, v) for spec, v in verdicts if v is not None and not v.ok]
        kept = [spec for spec, v in verdicts if run_unresolvable or v is None or v.ok]

        trials_path, rounds_path, sidecar_path = self._output_paths()
        existing_keys, next_trial = self._existing_state(trials_path, resume)
        hypothetical_skips = sum(1 for spec in kept if spec.key() in existing_keys)

        final_schedule = kept
        if resume and existing_keys:
            final_schedule = [spec for spec in kept if spec.key() not in existing_keys]

        t_batch_by_shape = self._run_warmups(final_schedule)

        if dry_run:
            self._print_dry_run(
                scheduled=scheduled, dropped=dropped, final_schedule=final_schedule,
                resume=resume, hypothetical_skips=hypothetical_skips, t_batch_by_shape=t_batch_by_shape
            )
            return trials_path

        for spec, verdict in dropped:
            print(f"Dropping unresolvable cell {spec.kind}/{spec.ftype}/{spec.layer}: {verdict.explain()}")

        naming.write_sidecar(sidecar_path, self.config(), naming.gather_provenance())
        self._execute_schedule(final_schedule, trials_path, rounds_path, next_trial, append=resume)

        return trials_path

    def _ordered_schedule(self) -> list[TrialSpec]:
        """
        Groups plan() by pair_id (a None pair_id is its own singleton
        group) and shuffles the GROUPS, not the specs, with
        random.Random(self.seed) when shuffle is enabled, then flattens
        them back out preserving each group's internal order. Grouping
        keeps a golden trial adjacent to the faulty trial it will be
        differenced against, while shuffling spreads thermal drift
        across the schedule.
        """

        specs = self.plan()

        groups: dict[object, list[TrialSpec]] = {}
        order: list[object] = []
        n_singletons = 0
        for spec in specs:
            if spec.pair_id is None:
                group_key: object = ('__singleton__', n_singletons)
                n_singletons += 1
            else:
                group_key = spec.pair_id

            if group_key not in groups:
                groups[group_key] = []
                order.append(group_key)
            groups[group_key].append(spec)

        ordered_groups = [groups[key] for key in order]
        if self.shuffle:
            random.Random(self.seed).shuffle(ordered_groups)

        return [spec for group in ordered_groups for spec in group]

    def _output_paths(self) -> tuple[str, str, str]:
        trials_fname = naming.build_csv_name(
            self.name, self.context.casestudy, session=self.session, shape='trials', tags=self.tags
        )
        rounds_fname = naming.build_csv_name(
            self.name, self.context.casestudy, session=self.session, shape='rounds', tags=self.tags
        )
        sidecar_fname = naming.build_sidecar_name(
            self.name, self.context.casestudy, session=self.session, tags=self.tags
        )

        return (
            self._resolve_path(trials_fname),
            self._resolve_path(rounds_fname),
            self._resolve_path(sidecar_fname)
        )

    def _resolve_path(self, fname: str) -> str:
        if self.out_dir is not None:
            return os.path.join(self.out_dir, fname)

        return make_out_filepath(fname)

    def _existing_state(self, trials_path: str, resume: bool) -> tuple[set[tuple[str, ...]], int]:
        """
        Reads an existing trials CSV, if resume is requested and the file
        is there, into the set of already-recorded spec keys and the next
        unused trial id. Returns an empty set and trial id 1 otherwise, so
        callers can use the returned skip count purely as a hypothetical
        preview even when resume is False.
        """

        if not resume or not os.path.exists(trials_path):
            return set(), 1

        existing = records.load_records(trials_path)
        if len(existing) == 0:
            return set(), 1

        keys = {
            tuple(_normalize_key(row[field]) for field in _SPEC_FIELDS)
            for _, row in existing.iterrows()
        }
        next_trial = int(existing['trial'].max()) + 1

        return keys, next_trial

    def _run_warmups(self, specs: list[TrialSpec]) -> dict[tuple[int, int], float]:
        """
        Runs one generic forward-pass warmup at each distinct
        (batch_size, time_bins) shape appearing in specs, in schedule
        order, and returns the measured mean per-batch seconds for each
        shape. Skipped entirely when CUDA is unavailable or n_warmup is 0,
        since there is then nothing to measure or warm up.
        """

        t_batch_by_shape: dict[tuple[int, int], float] = {}
        if self.n_warmup == 0 or not torch.cuda.is_available():
            return t_batch_by_shape

        seen: set[tuple[int, int]] = set()
        for spec in specs:
            if spec.batch_size is None or spec.time_bins is None:
                continue

            shape = (spec.batch_size, spec.time_bins)
            if shape in seen:
                continue
            seen.add(shape)

            t_batch_by_shape[shape] = self._warmup(spec.batch_size, spec.time_bins)

        return t_batch_by_shape

    def _warmup(self, batch_size: int, time_bins: int) -> float:
        """Warms this data shape and returns the per-batch seconds measured
        after it is warm.

        The loader carries one batch more than n_warmup, and only that
        final batch is timed: the passes that pay for CUDA context setup
        and kernel autotuning are the ones being discarded, so their cost
        does not land in the number the schedule estimate is built from.
        """

        loader = self.context.synthetic_loader(
            batch_size=batch_size, n_time_bins=time_bins, n_batches=self.n_warmup + 1
        )
        batches = list(loader)

        with torch.no_grad():
            for input, _ in batches[:-1]:
                self.context.net(input.to(self.context.device))

            with Stopwatch() as sw:
                self.context.net(batches[-1][0].to(self.context.device))

        return sw.elapsed

    def _print_dry_run(
            self,
            scheduled: list[TrialSpec],
            dropped: list[tuple[TrialSpec, Resolvability]],
            final_schedule: list[TrialSpec],
            resume: bool,
            hypothetical_skips: int,
            t_batch_by_shape: dict[tuple[int, int], float]
    ) -> None:
        print("Config:")
        for key, value in self.config().items():
            print(f"  {key}: {value}")

        print(f"\nSchedule ({len(final_schedule)} trial(s)):")
        for i, spec in enumerate(final_schedule):
            fields_str = ', '.join(
                f"{name}={value}" for name, value in spec.as_record_fields().items()
                if name != 'kind' and value is not None
            )
            print(f"  [{i}] {spec.kind} {fields_str}")

        print(f"\nTrial count: {len(final_schedule)}")

        print("\nResolvability:")
        any_verdict = False
        for spec in scheduled:
            verdict = self.resolvability(spec)
            if verdict is None:
                continue
            any_verdict = True
            print(f"  {spec.kind}/{spec.ftype}/{spec.layer}: {verdict.explain()}")
        if not any_verdict:
            print("  No precondition declared for any scheduled cell.")
        if dropped:
            print(f"  {len(dropped)} cell(s) excluded as unresolvable.")

        if resume:
            print(f"\nResume would skip {hypothetical_skips} already-recorded trial(s).")

        estimates = [
            self.estimate_seconds(spec, t_batch_by_shape.get((spec.batch_size, spec.time_bins)))
            for spec in final_schedule
        ]
        known = [e for e in estimates if e is not None]
        unknown_count = len(estimates) - len(known)
        total = sum(known)
        unknown_note = f"; {unknown_count} trial(s) have an unknown estimate" if unknown_count else ""
        print(
            f"\nEstimated total wall time (plain-inference lower bound): "
            f"{total:.1f}s over {len(known)} trial(s){unknown_note}."
        )

    def _reconcile_rounds(self, rounds_path: str, next_trial: int, append: bool) -> None:
        """
        Brings the rounds CSV back in line with the trials CSV, which is
        the authority on which trials are recorded.

        A resumed run reassigns trial ids from next_trial onwards, so any
        round row at or beyond that id belongs to a trial whose own row
        did not survive and is about to be re-run under the same id;
        keeping it would orphan or duplicate it. A run that is not
        resuming rewrites the trials file from scratch, so a rounds file
        left by an earlier run describes trials that no longer exist.
        """

        if not os.path.exists(rounds_path):
            return

        if not append:
            os.remove(rounds_path)
            return

        existing = records.load_records(rounds_path, fields=records.ROUND_FIELDS)
        survivors = existing[existing['trial'] < next_trial]
        if len(survivors) == len(existing):
            return

        with records.RecordWriter(rounds_path, records.ROUND_FIELDS) as writer:
            for _, row in survivors.iterrows():
                writer.write({field: row[field] for field in records.ROUND_FIELDS})

    def _execute_schedule(
            self,
            final_schedule: list[TrialSpec],
            trials_path: str,
            rounds_path: str,
            next_trial: int,
            append: bool
    ) -> None:
        self._reconcile_rounds(rounds_path, next_trial, append)

        trial_writer = records.RecordWriter(trials_path, records.TRIAL_FIELDS, append=append)
        round_writer: records.RecordWriter | None = None

        t0 = time.perf_counter()
        trial_id = next_trial
        try:
            for spec in final_schedule:
                outcome = self.execute(spec)
                t_rel = time.perf_counter() - t0

                mem = outcome.mem
                row: dict[str, object] = {
                    'session': self.session,
                    'trial': trial_id,
                    't_rel': t_rel,
                    'experiment': self.name,
                    'casestudy': self.context.casestudy,
                    **spec.as_record_fields(),
                    't_setup': outcome.t_setup,
                    't_exec': outcome.t_exec,
                    't_wall': outcome.t_wall,
                    'accuracy': outcome.accuracy,
                    'n_critical': outcome.n_critical,
                    'mem_alloc_base': mem.alloc_base if mem is not None else None,
                    'mem_alloc_peak': mem.alloc_peak if mem is not None else None,
                    'mem_alloc_end': mem.alloc_end if mem is not None else None,
                    'mem_reserved_peak': mem.reserved_peak if mem is not None else None,
                    'oom': outcome.oom
                }
                trial_writer.write(row)

                if outcome.rounds:
                    if round_writer is None:
                        round_writer = records.RecordWriter(rounds_path, records.ROUND_FIELDS, append=append)
                    for round_outcome in outcome.rounds:
                        round_writer.write({
                            'session': self.session,
                            'trial': trial_id,
                            'round': round_outcome.round,
                            'layer': round_outcome.layer,
                            'ftype': round_outcome.ftype,
                            'n_faults': round_outcome.n_faults,
                            'accuracy': round_outcome.accuracy,
                            'n_critical': round_outcome.n_critical
                        })

                print(f"[{trial_id}] {spec.kind} done in {outcome.t_wall:.3f}s (t_rel={t_rel:.1f}s)")
                trial_id += 1
        finally:
            trial_writer.close()
            if round_writer is not None:
                round_writer.close()
