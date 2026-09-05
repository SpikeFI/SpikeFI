# This file is part of SpikeFI.
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6

# SpikeFI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SpikeFI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import os
import re


OUT_DIR = 'out'
RES_DIR = os.path.join(OUT_DIR, 'res')
FIG_DIR = os.path.join(OUT_DIR, 'fig')
NET_DIR = os.path.join(OUT_DIR, 'net')


# fname includes file extension

def make_filepath(
        fname: str,
        parentdir: str | None = None,
        rename: bool = False
) -> str:
    parentdir = parentdir or OUT_DIR
    os.makedirs(parentdir, exist_ok=True)

    return os.path.join(
        parentdir,
        rename_if_multiple(fname, parentdir) if rename else fname
     )


def make_out_filepath(fname: str, rename: bool = False) -> str:
    return make_filepath(fname, OUT_DIR, rename)


def make_fig_filepath(fname: str, rename: bool = False) -> str:
    return make_filepath(fname, FIG_DIR, rename)


def make_res_filepath(fname: str, rename: bool = False) -> str:
    return make_filepath(fname, RES_DIR, rename)


def make_net_filepath(fname: str, rename: bool = False) -> str:
    return make_filepath(fname, NET_DIR, rename)


def calculate_trial(fname: str, parentdir: str = OUT_DIR) -> int:
    if not os.path.isdir(parentdir):
        return 0

    fname, extension = os.path.splitext(fname)
    # A file collides only if its name is exactly fname, optionally
    # followed by a ' (N)' trial marker.
    trial_re = re.compile(rf"^{re.escape(fname)}(?: \((\d+)\))?$")

    trials = []
    for f in os.listdir(parentdir):
        if not f.endswith(extension):
            continue
        match = trial_re.match(f.removesuffix(extension))
        if match:
            trials.append(int(match.group(1) or 0))

    return max(trials) + 1 if trials else 0


def rename_if_multiple(fname: str, parentdir: str = OUT_DIR) -> str:
    t = calculate_trial(fname, parentdir)
    if t == 0:
        return fname

    fname, extension = os.path.splitext(fname)

    return fname + f" ({t})" + extension
