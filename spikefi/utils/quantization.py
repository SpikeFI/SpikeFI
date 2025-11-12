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


from numpy import clip
import torch
from torch import Tensor


def qiinfo(dtype: torch.dtype) -> torch.iinfo:
    if dtype is torch.quint8:
        return torch.iinfo(torch.uint8)

    if dtype is torch.qint8:
        return torch.iinfo(torch.int8)

    if dtype is torch.qint32:
        return torch.iinfo(torch.int32)

    return torch.iinfo(dtype)


def qargs_exact(xmin: float, xmax: float, qmin: int, qmax: int) -> tuple[float, int]:
    assert xmin != xmax, "Min and max values cannot be equal"

    scale = (xmax - xmin) / (qmax - qmin)
    zero_point = int(clip(qmin - xmin / scale, qmin, qmax))

    return scale, zero_point


def qargs_precision(xmin: float, xmax: float, p: int) -> tuple[float, int]:
    return qargs_exact(xmin, xmax, 0, 2**p - 1)


def qargs_from_range(xmin: float, xmax: float, dtype: torch.dtype) -> tuple[float, int]:
    info = qiinfo(dtype)
    qmin = info.min
    qmax = info.max

    scale, zero_point = qargs_exact(xmin, xmax, qmin, qmax)

    return scale, zero_point


def qargs_from_tensor(x: Tensor, dtype: torch.dtype) -> tuple[float, int]:
    return qargs_from_range(x.min().item(), x.max().item(), dtype)
