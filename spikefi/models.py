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


from collections.abc import Callable
from copy import deepcopy
import random

import torch
from torch import Tensor

from slayerSNN.slayer import spikeLayer

from spikefi.fault import FaultModel, FaultSite, FaultTarget
import spikefi.utils.quantization as qua


# Fault Model Functions
def set_value(_, value: float | Tensor) -> float | Tensor:
    return value


def add_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
    return original + value


def mul_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
    return original * value


def qua_value(original: float | Tensor, scale: float | Tensor, zero_point: int | Tensor, dtype: torch.dtype) -> Tensor:
    return torch.dequantize(torch.quantize_per_tensor(original, scale, zero_point, dtype))


# LSB: bit 0
# MSB: bit N-1
def bfl_value(original: float | Tensor, bit: int, scale: float | Tensor, zero_point: int | Tensor, dtype: torch.dtype) -> Tensor:
    idt_info = torch.iinfo(dtype)
    assert bit >= 0 and bit < idt_info.bits, 'Invalid bit position to flip'

    # q = torch.quantize_per_tensor(original, scale, zero_point, qdtype).int_repr()
    # return torch.dequantize(q ^ 2 ** bit)

    q = torch.round(original / scale + zero_point).type(dtype) ^ (2 ** bit)
    return scale * (q - zero_point)


# Mother class for parametric faults
class ParametricFaultModel(FaultModel):
    def __init__(self, param_name: str, param_method: Callable[..., float | Tensor], *param_args) -> None:
        super().__init__(FaultTarget.PARAMETER, set_value, dict())
        self.method.__name__ = 'set_value'

        self.param_name = param_name
        self.param_method = param_method
        self.param_args = param_args

        self.param_original = None
        self.param_perturbed = None
        self.flayer = None

    def __repr__(self) -> str:
        return super().__repr__() + "\n" \
            + f"  - Parameter Name: '{self.param_name}'\n" \
            + f"  - Parameter Method: {self.param_method.__name__}\n" \
            + f"  - Parameter Arguments: {self.param_args}"

    def __str__(self) -> str:
        return super().__str__() + f" | Parametric: '{self.param_name}', {self.param_method.__name__}"

    def _key(self) -> tuple:
        return self.target, self.method, self.param_name, self.param_method, self.param_args

    def is_param_perturbed(self) -> bool:
        return self.flayer is not None

    def param_perturb(self, slayer: spikeLayer, device: torch.device) -> None:
        self.param_original = slayer.neuron[self.param_name]
        self.param_perturbed = self.param_method(self.param_original, *self.param_args)

        dummy = deepcopy(slayer)
        dummy.neuron[self.param_name] = self.param_perturbed

        self.flayer = spikeLayer(dummy.neuron, dummy.simulation, fullRefKernel=True).to(device)

    def param_restore(self) -> None:
        self.flayer.neuron[self.param_name] = self.param_original
        self.param_original = None
        self.param_perturbed = None

        return self.flayer.neuron[self.param_name]

    def perturb(self, original: float | Tensor, site: FaultSite) -> float | Tensor:
        return super().perturb(original, site, self.args[0].pop(site))


# Neuron fault models
class DeadNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.OUTPUT, set_value, 0.)


class SaturatedNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.OUTPUT, set_value, 1.)


class ParametricNeuron(ParametricFaultModel):
    def __init__(self, param_name: str, percentage: float = None) -> None:
        rho = percentage if percentage is not None else random.uniform(0.1, 3.0)
        super().__init__(param_name, mul_value, rho)


# Synapse fault models
class DeadSynapse(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.WEIGHT, set_value, 0.)


class SaturatedSynapse(FaultModel):
    def __init__(self, satu: float = 10.0) -> None:
        super().__init__(FaultTarget.WEIGHT, set_value, satu)


class PerturbedSynapse(FaultModel):
    def __init__(self, percentage: float = None):
        rho = percentage if percentage is not None else random.uniform(0.1, 3.0)
        super().__init__(FaultTarget.WEIGHT, mul_value, rho)


class BitflippedSynapse(FaultModel):
    def __init__(self, bit: int, wmin: float, wmax: float, quant_dtype: torch.dtype) -> None:
        super().__init__(FaultTarget.WEIGHT, bfl_value, bit, *qua.quant_args_from_range(wmin, wmax, quant_dtype))
