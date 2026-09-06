import csv
from dataclasses import dataclass, fields as dataclass_fields, MISSING
import os

import pandas as pd

# Shared column vocabulary for every benchmark CSV shape. Order is the
# authority: load_records() returns frames with columns in this order, and
# RecordWriter.write() validates rows against it.
TRIAL_FIELDS = (
    # identity
    'session', 'trial', 't_rel', 'experiment', 'casestudy', 'kind', 'pair_id',
    # configuration
    'ftype', 'layer', 'n_faults', 'opt', 'es_tol', 'n_rounds', 'batch_size',
    'time_bins', 'n_samples', 'n_batches', 'synthetic', 'spike_prob',
    'subset_div', 'rep', 'fault_seed', 'probe',
    # outcome
    't_setup', 't_exec', 't_wall', 'accuracy', 'n_critical',
    'mem_alloc_base', 'mem_alloc_peak', 'mem_alloc_end', 'mem_reserved_peak', 'oom'
)
ROUND_FIELDS = ('session', 'trial', 'round', 'layer', 'ftype', 'n_faults', 'accuracy', 'n_critical')

# SAMPLE_FIELDS: a third, *_samples.csv row shape for per-sample outcomes
# belongs here once an experiment needs it. Not yet done.


@dataclass
class TrialRecord:
    """
    One row of a *_trials.csv: a single timed unit of work, whatever the
    experiment. The identity fields are required because a row is
    unattributable without them; every configuration and outcome field is
    optional and stays None where the experiment does not have that axis or
    the probe was not run, which is what lets one vocabulary carry overhead
    trials and tolerance sweeps in the same file shape.

    `kind` names what was timed (e.g. 'golden', 'golden_spikefi',
    'faulty'), and `pair_id` links the rows of a paired comparison such as
    a golden/faulty couple. Memory fields come from a MemoryProbe and are
    None when it was not attached; `oom` marks a trial abandoned to an
    out-of-memory error, whose timings are then meaningless.
    """

    session: int
    trial: int
    t_rel: float
    experiment: str
    casestudy: str
    kind: str
    pair_id: int | None = None
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
    t_setup: float | None = None
    t_exec: float | None = None
    t_wall: float | None = None
    accuracy: float | None = None
    n_critical: int | None = None
    mem_alloc_base: int | None = None
    mem_alloc_peak: int | None = None
    mem_alloc_end: int | None = None
    mem_reserved_peak: int | None = None
    oom: bool = False

    def as_row(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in TRIAL_FIELDS}


@dataclass
class RoundRecord:
    """
    One row of a *_rounds.csv: the per-round outcome of a campaign, keyed
    back to its trial by (session, trial). Every field is required — a
    round always knows the faults it ran and the accuracy they produced —
    so this shape carries the accuracy detail that a TrialRecord aggregates
    away, without widening the trial vocabulary.
    """

    session: int
    trial: int
    round: int
    layer: str
    ftype: str
    n_faults: int
    accuracy: float
    n_critical: int

    def as_row(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in ROUND_FIELDS}


def _required_fields(record_cls: type) -> tuple[str, ...]:
    """Names the record's fields with no default: the ones a row is unattributable without."""

    return tuple(f.name for f in dataclass_fields(record_cls) if f.default is MISSING)


# Which columns a CSV of each row shape must physically carry. Derived from
# the records themselves so the identity contract is stated once.
REQUIRED_FIELDS: dict[tuple[str, ...], tuple[str, ...]] = {
    TRIAL_FIELDS: _required_fields(TrialRecord),
    ROUND_FIELDS: _required_fields(RoundRecord)
}


def _drop_partial_last_row(path: str, fields: tuple[str, ...]) -> None:
    """
    Truncates path back to the end of its last complete row when the final
    line does not carry one field per column, as a process killed mid-write
    would leave behind, and ensures the file ends with a newline so a
    subsequent append starts on its own line.
    """

    with open(path, newline='') as f:
        lines = f.readlines()

    if not lines:
        return

    last_row = next(csv.reader([lines[-1]]))
    if len(last_row) != len(fields):
        lines = lines[:-1]

    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'

    with open(path, 'w', newline='') as f:
        f.writelines(lines)


class RecordWriter:
    """
    Writes TrialRecord/RoundRecord rows to a CSV opened against an explicit
    field tuple (TRIAL_FIELDS or ROUND_FIELDS). The header is written on
    open and every row is flushed immediately: a run interrupted mid-way
    still leaves every completed trial on disk, unlike buffering all rows
    in memory and deriving fieldnames from the first one at the very end.

    With append=True against an existing, non-empty file, the file's header
    is validated to match fields, any partial final row left by a process
    killed mid-write is dropped, and writing resumes without repeating the
    header. append=True against a missing or empty file behaves exactly
    like append=False.
    """

    def __init__(self, path: str, fields: tuple[str, ...], append: bool = False) -> None:
        self.path: str = path
        self.fields: tuple[str, ...] = tuple(fields)

        do_append = append and os.path.exists(path) and os.path.getsize(path) > 0
        if do_append:
            with open(path, newline='') as f:
                header = next(csv.reader(f), [])
            if tuple(header) != self.fields:
                raise ValueError(
                    f"Cannot append to {path!r}: existing header {header} does not "
                    f"match expected fields {list(self.fields)}."
                )

            _drop_partial_last_row(path, self.fields)

        self._file = open(path, 'a' if do_append else 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=list(self.fields))
        if not do_append:
            self._writer.writeheader()
        self._file.flush()

    def write(self, record: 'TrialRecord | RoundRecord | dict[str, object]') -> None:
        row = record.as_row() if isinstance(record, (TrialRecord, RoundRecord)) else dict(record)

        field_set = set(self.fields)
        row_keys = set(row)
        unknown = row_keys - field_set
        missing = field_set - row_keys
        if unknown or missing:
            raise KeyError(
                f"Record does not match the field tuple for {self.path!r}: "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> 'RecordWriter':
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_records(path: str, fields: tuple[str, ...] = TRIAL_FIELDS) -> pd.DataFrame:
    """
    Loads a benchmark CSV into a DataFrame with columns in `fields` order.
    Optional columns may be absent and are added as NA, so a file written
    before a field existed still loads; the identity columns of the row
    shape may not, since a row without them cannot be attributed to a run.
    Pass fields=ROUND_FIELDS to load a *_rounds.csv file instead. Raises a
    clear error naming the file and the offending column(s) otherwise.
    """

    # 'round_trip' is the only float parser that reproduces the written
    # digits exactly; the default trades the last ~3 ulp for speed.
    raw = pd.read_csv(path, float_precision='round_trip')
    columns = set(raw.columns)

    if not columns <= set(fields):
        unknown = columns - set(fields)
        raise ValueError(
            f"{path!r} does not match the schema for fields={list(fields)}; "
            f"unrecognized column(s): {sorted(unknown)}"
        )

    missing = set(REQUIRED_FIELDS.get(tuple(fields), ())) - columns
    if missing:
        raise ValueError(
            f"{path!r} is missing identity column(s) {sorted(missing)} required "
            f"by the schema for fields={list(fields)}"
        )

    for field in fields:
        if field not in raw.columns:
            raw[field] = pd.NA

    return raw[list(fields)]
