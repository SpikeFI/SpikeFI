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
from math import prod
import random
from typing import Any
from torch import nn, Tensor
from slayerSNN import slayer


class LayersInfo:
    INJECTABLES = (slayer._convLayer, slayer._denseLayer)
    UNSUPPORTED = (slayer._pspLayer, slayer._pspFilter, slayer._delayLayer)

    def __init__(self, shape_in: tuple[int, int, int]) -> None:
        self.shape_in = shape_in
        self.names: set[str] = set()
        self.order: list[str] = []
        self.types: dict[str, type] = {}
        self.injectables: dict[str, bool] = {}
        self.weightables: dict[str, bool] = {}
        self.shapes_neu: dict[str, tuple[int, int, int]] = {}
        self.shapes_syn: dict[str, tuple[int, int, int, int]] = {}

    def __eq__(self, other: object) -> bool:
        isEq = isinstance(other, LayersInfo)

        isEq &= self.shape_in == other.shape_in
        isEq &= self.order == other.order
        isEq &= self.types == other.types
        isEq &= self.shapes_neu == other.shapes_neu
        isEq &= self.shapes_syn == other.shapes_syn

        return isEq

    def __len__(self) -> int:
        return len(self.order)

    def __repr__(self) -> str:
        s = f"Layers info ({len(self.names)} in {len(self.order)} levels): {{\n"
        for lay_idx, lay_name in enumerate(self.order):
            neu = "{:2d} x {:2d} x {:2d}".format(*self.shapes_neu[lay_name])
            syn = "{:2d} x {:2d} x {:2d} x {:2d}".format(*self.shapes_syn[lay_name]) if self.weightables[lay_name] else "-"

            s += f"      #{lay_idx:2d}: '{lay_name}' - {self.types[lay_name].__name__} - {'' if self.injectables[lay_name] else 'non '}injectable\n"
            s += f"        Shapes: neurons {neu} | synapses {syn}\n"
        s += '}'

        return s

    def identify(self, shape: tuple[int, int, int]) -> str | None:
        for n, s in self.shapes_neu.items():
            if s == shape:
                return n
        return None

    def infer(self, name: str, layer: nn.Module, output: Tensor) -> None:
        if not LayersInfo.is_module_supported(layer):
            print('Attention: Unsupported layer type ' + type(layer) + ' found. Potential invalidity of results.')
            return

        is_injectable = LayersInfo.is_module_injectable(layer)
        if is_injectable and name in self.names:
            print('Cannot use an injectable layer more than once in the network.')
            return

        has_weigth = hasattr(layer, 'weight')  # isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d))

        self.names.add(name)
        self.order.append(name)
        self.types[name] = type(layer)
        self.injectables[name] = is_injectable
        self.weightables[name] = has_weigth
        self.shapes_neu[name] = tuple(output.shape[1:4])
        self.shapes_syn[name] = tuple(layer.weight.shape[0:4]) if has_weigth else None

    def index(self, name: str) -> int:
        return self.order.index(name)

    def get_injectables(self) -> list[str]:
        return [inj[0] for inj in self.injectables.items() if inj[1]]

    def get_following(self, injectable_name: str) -> str | None:
        idx = self.index(injectable_name)
        return self.order[idx + 1] if idx < len(self) - 1 else None

    def get_random_inj(self, syn_select: bool) -> str:
        return random.choices(self.get_injectables(), weights=self.get_sizes_inj(syn_select), k=1)[0]

    def get_shape(self, syn_select: bool, name: str) -> tuple[int, ...]:
        return self.shapes_syn[name] if syn_select else self.shapes_neu[name]

    def get_shapes_inj(self, syn_select: bool) -> list[tuple[int, ...]]:
        return [self.get_shape(syn_select, inj) for inj in self.get_injectables()]

    def get_size(self, syn_select: bool, name: str) -> int:
        return prod(self.get_shape(syn_select, name))

    def get_sizes_inj(self, syn_select: bool) -> list[int]:
        return [prod(shape) for shape in self.get_shapes_inj(syn_select)]

    def is_injectable(self, name: str) -> bool:
        return self.injectables.get(name, False)

    def is_weightable(self, name: str) -> bool:
        return self.weightables.get(name, False)

    def is_output(self, name: str) -> bool:
        return name == self.order[-1]

    def infer_hook_wrapper(self, layer_name: str) -> Callable[[nn.Module, tuple[Any, ...], Tensor], None]:
        def infer_hook(layer: nn.Module, _, output: Tensor) -> None:
            self.infer(layer_name, layer, output)

        return infer_hook

    @staticmethod
    def is_module_injectable(layer: nn.Module) -> bool:
        is_inj = False
        for type_ in LayersInfo.INJECTABLES:
            is_inj |= isinstance(layer, type_)

        return is_inj

    @staticmethod
    def is_module_supported(layer: nn.Module) -> bool:
        is_sup = True
        for type_ in LayersInfo.UNSUPPORTED:
            is_sup &= not isinstance(layer, type_)

        return is_sup
