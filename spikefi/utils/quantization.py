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


import torch
from torch import Tensor


def q2i_dtype(qdtype: torch.dtype) -> torch.dtype:
    if qdtype is torch.quint8:
        idtype = torch.uint8
    elif qdtype is torch.qint8:
        idtype = torch.int8
    elif qdtype is torch.qint32:
        idtype = torch.int32
    else:
        raise AssertionError('The desired data type of returned tensor has to be'
                             'one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32')

    return idtype


def qargs_from_tensor(x: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor, torch.dtype]:
    return qargs_from_range(x.min(), x.max(), dtype)


def qargs_from_range(xmin: float | Tensor, xmax: float | Tensor,
                     dtype: torch.dtype) -> tuple[Tensor, Tensor, torch.dtype]:
    dt_info = torch.iinfo(dtype)
    qmin = dt_info.min
    qmax = dt_info.max

    scale, zero_point = qargs_exact(xmin, xmax, qmin, qmax)

    return scale, zero_point, dtype


def qargs_exact(xmin: float | Tensor, xmax: float | Tensor,
                qmin: int, qmax: int) -> tuple[Tensor, Tensor]:
    xmin = torch.as_tensor(xmin, dtype=torch.float32)
    xmax = torch.as_tensor(xmax, dtype=torch.float32)

    assert xmin.shape == xmax.shape, 'Tensors shape mismatch'

    scale = (xmax - xmin) / (qmax - qmin)
    zero_point = torch.clip(qmin - xmin / scale, qmin, qmax).int()

    return scale, zero_point


def qargs_precision(xmin: float | Tensor, xmax: float | Tensor,
                    p: int) -> tuple[Tensor, Tensor]:
    return qargs_exact(xmin, xmax, 0, 2**p-1)
