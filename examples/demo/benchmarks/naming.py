from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
import json
import os
import re
import socket
import subprocess
from typing import get_args, Literal

import torch

import demo
import spikefi

TAG_RE = re.compile(r'^[a-z0-9]+$')

_CASESTUDIES = sorted(get_args(demo.SUPPORTED_CASE_STUDIES), key=len, reverse=True)
_CASESTUDY_ALT = '|'.join(_CASESTUDIES)

NAME_RE = re.compile(
    r'^sfi_(?P<experiment>[a-z0-9]+)_(?P<casestudy>' + _CASESTUDY_ALT + r')'
    r'(?:_(?P<tags>[a-z0-9]+(?:_[a-z0-9]+)*))?'
    r'_S(?P<session>\d+)_(?P<shape>trials|rounds)\.csv$'
)


@dataclass
class NameInfo:
    experiment: str
    casestudy: str
    tags: tuple[str, ...]
    session: int
    shape: Literal['trials', 'rounds']


def _validate_tags(tags: tuple[str, ...]) -> None:
    for tag in tags:
        if not TAG_RE.match(tag):
            raise ValueError(f"Invalid tag {tag!r}: tags must match {TAG_RE.pattern!r}")


def _validate_name_parts(
    experiment: str,
    casestudy: str,
    tags: tuple[str, ...]
) -> None:
    if not TAG_RE.match(experiment):
        raise ValueError(f"Invalid experiment token {experiment!r}: must match {TAG_RE.pattern!r}")
    if casestudy not in _CASESTUDIES:
        raise ValueError(f"Unknown case study {casestudy!r}: expected one of {_CASESTUDIES}")
    _validate_tags(tags)


def _stem(
    experiment: str,
    casestudy: str,
    tags: tuple[str, ...],
    session: int,
    shape: Literal['trials', 'rounds']
) -> str:
    _validate_name_parts(experiment, casestudy, tags)
    if shape not in ('trials', 'rounds'):
        raise ValueError(f"Unknown shape {shape!r}: expected 'trials' or 'rounds'")

    tag_part = ''.join(f'_{tag}' for tag in tags)
    return f'sfi_{experiment}_{casestudy}{tag_part}_S{session}_{shape}'


def build_csv_name(
        experiment: str,
        casestudy: str,
        *,
        session: int,
        shape: Literal['trials', 'rounds'],
        tags: tuple[str, ...] = ()
) -> str:
    return f'{_stem(experiment, casestudy, tags, session, shape)}.csv'


def build_sidecar_name(
        experiment: str,
        casestudy: str,
        *,
        session: int,
        tags: tuple[str, ...] = ()
) -> str:
    # A sidecar describes a whole run (resolved config, fault-plan digest,
    # provenance), not one row shape, so it carries no shape segment and no
    # single sidecar is tied to either the _trials.csv or the _rounds.csv
    # of the same run.
    _validate_name_parts(experiment, casestudy, tags)

    tag_part = ''.join(f'_{tag}' for tag in tags)
    return f'sfi_{experiment}_{casestudy}{tag_part}_S{session}.json'


def build_fig_name(
        experiment: str,
        casestudy: str,
        *,
        session: int,
        shape: Literal['trials', 'rounds'],
        tags: tuple[str, ...] = (),
        t_col: str,
        ext: str = 'png'
) -> str:
    # Kept as a separate statement (rather than nested inside the f-string
    # below) because a same-quote nested f-string expression is PEP 701 and
    # only parses on Python 3.12+, while this project targets >=3.11.
    suffix = t_col.removeprefix('t_')
    return f'{_stem(experiment, casestudy, tags, session, shape)}_C{suffix}.{ext}'


def parse_name(fpath: str) -> NameInfo:
    fname = os.path.basename(fpath)
    m = NAME_RE.match(fname)
    if m is None:
        raise ValueError(f"Could not parse benchmark name: {fname!r}")

    tags = tuple(m['tags'].split('_')) if m['tags'] else ()

    return NameInfo(
        experiment=m['experiment'],
        casestudy=m['casestudy'],
        tags=tags,
        session=int(m['session']),
        shape=m['shape']
    )


def gather_provenance() -> dict[str, object]:
    """
    Collects the run's provenance for the sidecar JSON. Each field is
    gathered independently and degrades to None on failure (missing git
    binary, no CUDA device, package not installed, ...) so that provenance
    collection can never be the reason a benchmark run crashes.
    """

    provenance: dict[str, object] = {}

    try:
        provenance['spikefi_version'] = spikefi.__version__
    except (AttributeError, PackageNotFoundError):
        provenance['spikefi_version'] = None

    provenance['torch_version'] = torch.__version__
    provenance['cuda_version'] = torch.version.cuda

    try:
        provenance['device_name'] = (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        )
    except RuntimeError:
        provenance['device_name'] = None

    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            check=True,
            text=True
        )
        provenance['git_commit'] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        provenance['git_commit'] = None

    provenance['timestamp'] = datetime.now(timezone.utc).isoformat()

    try:
        provenance['hostname'] = socket.gethostname()
    except OSError:
        provenance['hostname'] = None

    return provenance


def write_sidecar(
    path: str,
    config: dict[str, object],
    provenance: dict[str, object]
) -> None:
    payload = {
        'config': config,
        # Slot for a fault-plan content hash, filled in by the caller once available.
        'fault_plan_digest': None,
        'provenance': provenance
    }

    with open(path, 'w') as sidecar_file:
        json.dump(payload, sidecar_file, indent=2, default=str)
