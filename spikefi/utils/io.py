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


def make_filepath(filename: str, parentdir: str = '') -> str:
    os.makedirs(parentdir, exist_ok=True)
    return os.path.join(parentdir, filename)


def make_out_filepath(filename: str) -> str:
    return make_filepath(filename, OUT_DIR)


def make_fig_filepath(filename: str) -> str:
    return make_filepath(filename, FIG_DIR)


def make_res_filepath(filename: str) -> str:
    return make_filepath(filename, RES_DIR)


def make_net_filepath(filename: str) -> str:
    return make_filepath(filename, NET_DIR)


def calculate_trial(filename: str, parentdir: str) -> int:
    if not os.path.isdir(parentdir):
        return 0

    fname, extension = os.path.splitext(filename)
    fnames = [f.removesuffix(extension) for f in os.listdir(parentdir)
              if fname in f and f.endswith(extension)]

    if not fnames:
        return 0

    trial_matches = [re.search(r' \(\d+\)$', f) for f in fnames]

    return max([int(m.group().strip(' ()')) if m else 0 for m in trial_matches]) + 1


def rename_if_multiple(filename: str, parentdir: str) -> str:
    t = calculate_trial(filename, parentdir)
    if t == 0:
        return filename

    fname, extension = os.path.splitext(filename)

    return fname + f" ({t})" + extension
