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


def quant_args_from_range(xmin: float | Tensor, xmax: float | Tensor,
                          dtype: torch.dtype) -> tuple[Tensor, Tensor, torch.dtype]:
    if not torch.is_tensor(xmin):
        xmin = torch.tensor(xmin)
    if not torch.is_tensor(xmax):
        xmax = torch.tensor(xmax)
    xmin = xmin.float()
    xmax = xmax.float()

    assert xmin.size() == xmax.size()

    dt_info = torch.iinfo(dtype)
    qmin = dt_info.min
    qmax = dt_info.max

    scale = ((xmax - xmin) / (qmax - qmin))
    zero_point = torch.clip(qmin - xmin / scale, qmin, qmax).int()

    return scale, zero_point, dtype


def quantize_exact(xmin: float | Tensor, xmax: float | Tensor,
                   bit_precision: int) -> tuple[Tensor, Tensor]:
    if not torch.is_tensor(xmin):
        xmin = torch.tensor(xmin)
    if not torch.is_tensor(xmax):
        xmax = torch.tensor(xmax)
    xmin = xmin.float()
    xmax = xmax.float()

    assert xmin.size() == xmax.size()

    qmin = int(0)
    qmax = int(2**bit_precision - 1)

    scale = ((xmax - xmin) / (qmax - qmin))
    zero_point = torch.clip(qmin - xmin / scale, qmin, qmax)

    return scale, zero_point
