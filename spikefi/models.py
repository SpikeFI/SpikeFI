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
from dataclasses import dataclass, field
import random
from typing import Any, Iterable, Literal, Optional

import torch
from torch import Tensor

from slayerSNN.slayer import spikeLayer

from spikefi.fault import FaultModel, FaultSite, FaultTarget
from spikefi.utils.quantization import qiinfo


# Fault Model Functions
def set_value(_, value: float | Tensor) -> float | Tensor:
    return value


def add_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
    return original + value


def mul_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
    return original * value


def qua_value(original: Tensor, scale: float, zero_point: int, dtype: torch.dtype) -> Tensor:
    return torch.dequantize(torch.quantize_per_tensor(original, scale, zero_point, dtype))


# LSB: bit 0
# MSB: bit N-1
# Works with torch.quint8, torch.qint8, torch.qint32
def bfl_value(original: Tensor, bit: int | Iterable[int] | Tensor,
              scale: float, zero_point: int, dtype: torch.dtype) -> Tensor:
    if isinstance(bit, int):
        bit = [bit]
    assert all(0 <= b < qiinfo(dtype).bits for b in bit), 'Invalid bit position(s) to flip'

    bfl_mask = 0
    for b in bit:
        bfl_mask |= (1 << b)

    q = torch.quantize_per_tensor(original, scale, zero_point, dtype)
    qi = q.int_repr()
    qif = qi ^ bfl_mask
    qifq = torch._make_per_tensor_quantized_tensor(qif, scale, zero_point)

    return torch.dequantize(qifq)


# Mother class for parametric faults
class ParametricNeuronFaultModel(FaultModel):
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


@dataclass
class RandomModelChoice:
    model_cls: type
    args_factory: Callable[[], tuple[Any, ...]] = field(default=lambda: ())
    select_chance: Optional[float] = None


class RandomFaultModel:
    def __new__(cls, model_choices: list[RandomModelChoice]) -> FaultModel:
        total_weight = 0.0
        none_count = 0
        for c in model_choices:
            if c.select_chance is not None:
                total_weight += c.select_chance
            else:
                none_count += 1

        if total_weight > 1.0:
            none_count = len(model_choices)
            total_weight = 0.0

        # Distribute remaining weight equally among None weights
        remaining_weight = 1.0 - total_weight
        inferred_weight = remaining_weight / none_count if none_count > 0 else 0.0

        weights = [
            c.select_chance if c.select_chance is not None else inferred_weight
            for c in model_choices
        ]

        # Normalize weights to sum 1
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        chosen = random.choices(model_choices, weights=weights, k=1)[0]

        return chosen.model_cls(*chosen.args_factory())


# Neuron fault models
class DeadNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.OUTPUT, set_value, 0.)


class SaturatedNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.OUTPUT, set_value, 1.)


class StuckNeuron(FaultModel):
    def __init__(self, x: float = None):
        _x = x if x is not None else random.uniform(0, 1)
        assert _x >= 0.0 and _x <= 1.0, 'Stuck-at value x needs to be in the range [0, 1]'
        super().__init__(FaultTarget.OUTPUT, set_value, _x)


class RandomNeuron(RandomFaultModel):
    DEF_MODEL_CHOICES = [
        RandomModelChoice(DeadNeuron, select_chance=2/3),
        RandomModelChoice(SaturatedNeuron, select_chance=1/6),
        RandomModelChoice(StuckNeuron, select_chance=1/6)
    ]

    def __new__(cls, model_choices: Optional[list[RandomModelChoice]] = None) -> FaultModel:
        return super().__new__(cls, model_choices or cls.DEF_MODEL_CHOICES)


class ParametricNeuron(ParametricNeuronFaultModel):
    def __init__(self, param_name: str, percentage: float = None) -> None:
        rho = percentage if percentage is not None else random.uniform(0.1, 3.0)
        super().__init__(param_name, mul_value, rho)


class IntegrationFaultNeuron(ParametricNeuron):
    def __init__(self, percentage: float = None) -> None:
        super().__init__('tauSr', percentage)


class RefractoryFaultNeuron(ParametricNeuron):
    def __init__(self, percentage: float = None) -> None:
        super().__init__('tauRef', percentage)


class ThresholdFaultNeuron(ParametricNeuron):
    def __init__(self, percentage: float = None) -> None:
        super().__init__('theta', percentage)


class RandomParametricNeuron(RandomFaultModel):
    DEF_MODEL_CHOICES = [
        RandomModelChoice(ParametricNeuron, lambda: (random.choice(['theta', 'tauSr', 'tauRef']), None))
    ]

    def __new__(cls, model_choices: Optional[list[RandomModelChoice]] = None) -> ParametricNeuronFaultModel:
        return super().__new__(cls, model_choices or cls.DEF_MODEL_CHOICES)


# Synapse fault models
class DeadSynapse(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.WEIGHT, set_value, 0.)


class SaturatedSynapse(FaultModel):
    def __init__(self, Q1: float | Tensor, Q3: float | Tensor,
                 tail: Literal['upper', 'lower'] = 'lower',
                 intensity: Literal['mild', 'extreme'] = 'mild') -> None:
        # Q1 = torch.quantile(weight_matrix, 0.25)
        # Q3 = torch.quantile(weight_matrix, 0.75)
        Q1 = torch.as_tensor(Q1, dtype=torch.float32)
        Q3 = torch.as_tensor(Q3, dtype=torch.float32)

        IQR = Q3 - Q1
        i = 3.0 if intensity == 'extreme' else 1.5

        if tail == 'upper':
            U = Q3 + i * IQR
            satu = U.ceil() + 1
        else:
            L = Q1 - i * IQR
            satu = L.floor() - 1

        super().__init__(FaultTarget.WEIGHT, set_value, satu)


class StuckSynapse(FaultModel):
    def __init__(self, x: float):
        super().__init__(FaultTarget.WEIGHT, set_value, x)


class PerturbedSynapse(FaultModel):
    def __init__(self, percentage: float = None):
        rho = percentage if percentage is not None else random.uniform(0.1, 3.0)
        super().__init__(FaultTarget.WEIGHT, mul_value, rho)


class BitflippedSynapse(FaultModel):
    def __init__(self, bit: int | Iterable[int], scale: float, zero_point: int, dtype: torch.dtype):
        super().__init__(FaultTarget.WEIGHT, bfl_value, bit, scale, zero_point, dtype)


class RandomSynapse(RandomFaultModel):
    DEF_MODEL_CHOICES = [
        RandomModelChoice(DeadSynapse, select_chance=2/3),
        RandomModelChoice(PerturbedSynapse, select_chance=1/3)
    ]

    def __new__(cls, model_choices: Optional[list[RandomModelChoice]] = None) -> FaultModel:
        return super().__new__(cls, model_choices or cls.DEF_MODEL_CHOICES)
