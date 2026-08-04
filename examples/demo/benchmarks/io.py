from dataclasses import dataclass
import os
import re

from demo import SUPPORTED_CASE_STUDIES

FNAME_RE = re.compile(
    r'benchmark_(?P<benchmark>[A-Z])_(?P<casestudy>nmnist_cnn|nmnist_mlp|gesture)_'
    r'(?P<synthetic>synthetic_)?'
    r'(?:R(?P<n_reps>\d+)_)?'
    r'B(?P<n_batches>\d+)_'
    r'S(?P<session_id>\d+)'
    r'(?:\.csv|_\w+\.\w+)$'
)


@dataclass
class FnameInfo:
    benchmark: str
    casestudy: SUPPORTED_CASE_STUDIES
    use_synthetic: bool
    n_reps: int
    n_batches: int
    session_id: int


def parse_fname(fpath: str) -> FnameInfo:
    fname = os.path.basename(fpath)
    m = FNAME_RE.search(fname)
    if m is None:
        raise ValueError(f"Could not parse benchmark fname: {fpath!r}")

    return FnameInfo(
        benchmark=m['benchmark'],
        casestudy=m['casestudy'],
        use_synthetic=m['synthetic'] is not None,
        n_reps=int(m['n_reps']),
        n_batches=int(m['n_batches']),
        session_id=int(m['session_id'])
    )


def build_fname_stem(info: FnameInfo) -> str:
    synthetic = 'synthetic_' if info.use_synthetic else ''

    return (
        f'benchmark_{info.benchmark}_{info.casestudy}_{synthetic}'
        f'R{info.n_reps}_B{info.n_batches}_S{info.session_id}'
    )


def build_csv_fname(info: FnameInfo) -> str:
    return f'{build_fname_stem(info)}.csv'


def build_fig_fname(info: FnameInfo, t_col: str, ext: str = 'png') -> str:
    return f'{build_fname_stem(info)}_C{t_col.removeprefix('t_')}.{ext}'
