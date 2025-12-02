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


from collections.abc import Callable, Iterable, Sequence
from copy import copy, deepcopy
from enum import auto, Flag
from math import log2, ceil
from functools import reduce
from operator import or_
import random
import re

import torch
from torch import Tensor

from spikefi.utils.layer import LayersInfo


class FaultSite:
    def __init__(
            self,
            layer_name: str | None = None,
            position: tuple[int, ...] = ()
    ) -> None:
        self.layer = layer_name

        if position:
            assert len(position) in (3, 4), (
                'Position must be a tuple of length 3 or 4 '
                'for neuron or synapse faults, respectively.'
            )
        self.position = position or tuple()

    def __bool__(self) -> bool:
        return self is not None and self.is_defined()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FaultSite) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            "Fault Site @ layer '" + (self.layer or '*') + "' ("
            + ", ".join(list(map(FaultSite.pos2str, self.position))) + ")"
        )

    def _key(self) -> tuple:
        if not self.position or len(self.position) == 3:
            pos_key = (-1,) + self.position
        else:
            pos_key = self.position

        return self.layer, pos_key

    def is_defined(self) -> bool:
        return (
            bool(self.layer)
            and self.position is not None
            and len(self.position) in (3, 4)
        )

    def unroll(self, full_5D: bool = False) -> tuple[int | slice, ...]:
        if full_5D:
            return (
                ((slice(None),) + self.position)
                if len(self.position) == 3
                else self.position
            ) + (slice(None),)

        return self.position

    @staticmethod
    def pos2str(x: int) -> str:
        return '*' if x is None else ':' if x == slice(None) else str(x)


class FaultTarget(Flag):
    OUTPUT = Z = auto()     # 1
    WEIGHT = W = auto()     # 2
    PARAMETER = P = auto()  # 4

    @classmethod
    def all(cls) -> 'FaultTarget':
        return reduce(or_, cls)

    def get_index(self) -> int:
        return int(log2(self.value))

    @staticmethod
    def neuronal() -> 'FaultTarget':
        return FaultTarget.OUTPUT | FaultTarget.PARAMETER

    @staticmethod
    def parametric() -> 'FaultTarget':
        return FaultTarget.PARAMETER

    @staticmethod
    def synaptic() -> 'FaultTarget':
        return FaultTarget.WEIGHT


class FaultModel:
    def __init__(
            self,
            target: FaultTarget,
            method: Callable[..., float | Tensor],
            *args
    ) -> None:
        assert len(target) == 1, 'Fault Model must have exactly 1 Fault Target'
        self.target = target
        self.method = method
        self.args = args

        self.original: float | Tensor = None
        self.perturbed: float | Tensor = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FaultModel) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"Fault Model: '{self.get_name()}'\n"
            + f"  - Target: {self.target}\n"
            + f"  - Method: {self.method.__name__}\n"
            + f"  - Arguments: {self.args}"
        )

    def __str__(self) -> str:
        return (
            f"Fault Model '{self.get_name()}': "
            + f"{self.target}, {self.method.__name__}"
        )

    def _key(self) -> tuple:
        return self.target, self.method, self.args

    def get_name(self) -> str:
        name = self.__class__.__name__
        return 'CustomFault' if name == FaultModel.__name__ else name

    def get_name_snake_case(
            self,
            reverse: bool = True,
            delimiter: str = '_'
    ) -> str:
        words = re.findall(r'[A-Z][a-z]*', self.get_name())
        words = [word.lower() for word in words]

        return delimiter.join(reversed(words) if reverse else words)

    def is_neuronal(self) -> bool:
        return self.target in FaultTarget.neuronal()

    def is_parametric(self) -> bool:
        return self.target in FaultTarget.parametric()

    def is_synaptic(self) -> bool:
        return self.target in FaultTarget.synaptic()

    def is_perturbed(self) -> bool:
        return self.original is not None and self.perturbed is not None

    def perturb(self, original: float | Tensor, *new_args) -> float | Tensor:
        return self.method(original, *(new_args or self.args))

    def perturb_store(self, original: float | Tensor) -> float | Tensor:
        self.original = original
        self.perturbed = self.perturb(original)

        return self.perturbed

    def restore(self) -> float | Tensor:
        self.perturbed = None
        original = self.original

        return original

    def store(self, perturbed: float | Tensor) -> None:
        self.perturbed = perturbed

        return

    def unstore(self) -> float | Tensor:
        tore = self.perturbed
        self.perturbed = None

        return tore


class Fault:
    def __init__(
            self,
            model: FaultModel,
            sites: FaultSite | Iterable[FaultSite] = FaultSite()
    ) -> None:
        self.model = model
        self.sites: set[FaultSite] = set()
        self.sites_pending: list[FaultSite] = []

        if isinstance(sites, Iterable):
            self.update_sites(sites)
        else:
            self.add_site(sites)

    def __add__(self, other: 'Fault') -> 'Fault':
        assert self.model == other.model, (
            'Only two Faults with the same Fault Model can be added'
        )

        return Fault(
            deepcopy(self.model),
            self.get_sites(include_pending=True)
            + other.get_sites(include_pending=True)
        )

    def __bool__(self) -> bool:
        return self is not None and bool(self.sites)

    def __contains__(self, site: FaultSite) -> bool:
        return site in self.sites

    def __eq__(self, other: object) -> bool:
        # Equality is checked so as the Faults were injected "as is",
        # therefore it does not take into account the pending Fault Sites,
        # as they are considered random sites and are not yet defined
        return isinstance(other, Fault) and self.sites == other.sites

    def __len__(self) -> int:
        return len(self.sites)

    def __repr__(self) -> str:
        s = f"Fault '{self.model.get_name()}' @ {self.sites or '0 sites'}"
        if self.sites_pending:
            s += f" (+{len(self.sites_pending)} pending)"

        return s

    def __str__(self) -> str:
        s = f"Fault '{self.model.get_name()}' @ "
        if len(self) == 1:
            f = next(iter(self.sites))
            s += str(f).split('@ ')[-1]
        else:
            s += f"{len(self)} sites"

        if self.sites_pending:
            s += f" (+{len(self.sites_pending)} pending)"

        return s

    def add_site(self, site: FaultSite) -> None:
        if site.is_defined():
            self.sites.add(site)
        else:
            self.sites_pending.append(site)

    def breakdown(self) -> list['Fault']:
        separated_faults = []
        for s in self.get_sites(include_pending=True):
            separated_faults.append(Fault(deepcopy(self.model), s))

        return separated_faults

    # Sorts faults by their fault model and
    # merges multiple faults of same model
    @staticmethod
    def buildup(faults: Iterable['Fault']) -> list['Fault']:
        merged = {}
        for f in faults:
            if f.model in merged:
                merged[f.model] += f
            else:
                merged[f.model] = f

        return list(merged.values())

    def has_site(self, site: FaultSite) -> bool:
        return site in self.sites

    def get_sites(self, include_pending: bool = False) -> list[FaultSite]:
        return (
            list(self.sites) + (self.sites_pending if include_pending else [])
        )

    def is_complete(self) -> bool:
        return not self.sites_pending

    def is_multiple(self) -> bool:
        return len(self) > 1

    def unroll(self) -> tuple[Tensor, ...]:
        # Grouped unrolled site indices per dimension
        unrolled_per_dim = zip(*(s.position for s in self.sites))
        return tuple(
            torch.tensor(v, dtype=torch.int64) for v in unrolled_per_dim
        )

    @staticmethod
    def multiple_random_absolute(
        model: FaultModel,
        sites_num: int,
        layers: Sequence[str] | str | None = None,
        layer_sizes: Sequence[int] | None = None
    ) -> 'Fault':
        if isinstance(layers, str):
            ran_lay = layers
        elif isinstance(layers, Sequence) and len(layers) == 1:
            ran_lay = layers[0]
        else:
            ran_lay = None

        sites = []
        for _ in range(sites_num):
            if not layers:
                sites.append(FaultSite())
                continue

            ran_lay = ran_lay or (
                random.choices(layers, weights=layer_sizes, k=1)[0]
            )
            sites.append(FaultSite(ran_lay))

        return Fault(model, sites)

    @staticmethod
    def multiple_random_percent(
        model: FaultModel,
        fault_density: float | Sequence[float],
        layers: Sequence[str],
        layer_sizes: Sequence[int]
    ) -> 'Fault':
        assert len(layer_sizes) == len(layers), (
            'For every layer provided there needs to be a layer_size as well.'
        )
        if isinstance(fault_density, Sequence):
            assert len(fault_density) == len(layers), (
                'For every layer provided there needs to be a fault_density.'
            )
        else:
            fault_density = [fault_density] * len(layers)

        faults = []
        for i, layer in enumerate(layers):
            sites_num = ceil(layer_sizes[i] * fault_density[i])
            faults.append(
                Fault.multiple_random_absolute(model, sites_num, layer)
            )

        return Fault.buildup(faults)[0]

    def refresh(self, discard_duplicates: bool = False) -> None:
        newly_defined = []
        for s in self.sites_pending:
            if not s.is_defined():
                continue

            if self.sites.isdisjoint({s}) or discard_duplicates:
                newly_defined.append(s)
            self.sites.add(s)

        for s in newly_defined:
            self.sites_pending.remove(s)

    def update_sites(self, sites: Iterable[FaultSite]) -> None:
        for s in sites:
            self.add_site(s)


class FaultRound(dict):  # dict[tuple[str, FaultModel], Fault]
    def __init__(self, *args, **kwargs) -> None:
        # Indicate faulty layers and their fault targets
        self.fault_map: dict[str, list[bool]] = {}

        # Construct from an Iterable of Faults
        if (
            'faults' in kwargs
            or (
                bool(args)
                and isinstance(args[0], Iterable)
                and all(isinstance(el, Fault) for el in args[0])
            )
        ):
            super().__init__()
            self.insert_many(kwargs.get('faults', args[0]))
            return

        # Construct from another dict
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return self._info(verbose=True)

    def __str__(self) -> str:
        return self._info(verbose=False)

    def _info(self, verbose: bool) -> str:
        sfunc = repr if verbose else str
        s = 'Fault Round: {'
        for key, fault in self.items():
            s += f"\n  '{key[0]}': {sfunc(fault)}"

        return s + '\n}'

    def any(
            self,
            layer_name: str,
            target: FaultTarget = FaultTarget.all()
    ) -> bool:
        layer_map = self.fault_map.get(layer_name, [False] * 3)
        return any(layer_map[t.get_index()] for t in target)

    def any_neuronal(self, layer_name: str) -> bool:
        return self.any(layer_name, FaultTarget.neuronal())

    def any_parametric(self, layer_name: str) -> bool:
        return self.any(layer_name, FaultTarget.parametric())

    def any_synaptic(self, layer_name: str) -> bool:
        return self.any(layer_name, FaultTarget.synaptic())

    # Exclusive any
    def xany(
            self,
            layer_name: str,
            target: FaultTarget = FaultTarget.all()
    ) -> bool:
        return (
            self.any(layer_name, target)
            and not self.any(layer_name, ~target)
        )

    def xany_neuronal(self, layer_name: str) -> bool:
        return self.xany(layer_name, FaultTarget.neuronal())

    def xany_parametric(self, layer_name: str) -> bool:
        return self.xany(layer_name, FaultTarget.parametric())

    def xany_synaptic(self, layer_name: str) -> bool:
        return self.xany(layer_name, FaultTarget.synaptic())

    def extract(self, fault: Fault) -> None:
        if not fault:
            return

        for s in copy(fault.sites):
            key = (s.layer, fault.model)
            f = self.get(key)
            if f is not None:
                f.sites.discard(s)
                if not f:
                    del self[key]

    def extract_many(self, faults: Iterable[Fault]) -> None:
        for f in faults:
            self.extract(f)

    def insert(self, fault: Fault) -> None:
        if not fault:
            return

        for s in fault.sites:
            key = (s.layer, fault.model)
            self.setdefault(key, Fault(deepcopy(fault.model), []))
            self[key].add_site(s)

            self.fault_map.setdefault(s.layer, [False] * 3)
            self.fault_map[s.layer][fault.model.target.get_index()] = True

    def insert_many(self, faults: Iterable[Fault]) -> None:
        for f in faults:
            self.insert(f)

    def get_faults(
            self,
            target: FaultTarget = FaultTarget.all()
    ) -> list[Fault]:
        return [self[k] for k in self if k[1].target in target]

    def get_neuronal(self) -> list[Fault]:
        return self.get_faults(FaultTarget.neuronal())

    def get_parametric(self) -> list[Fault]:
        return self.get_faults(FaultTarget.parametric())

    def get_synaptic(self) -> list[Fault]:
        return self.get_faults(FaultTarget.synaptic())

    def search(
            self,
            layer_name: str,
            target: FaultTarget = FaultTarget.all()
    ) -> list[Fault]:
        return [
            self[k]
            for k in self
            if k[0] == layer_name and k[1].target in target
        ]

    def search_neuronal(self, layer_name: str) -> list[Fault]:
        return self.search(layer_name, FaultTarget.neuronal())

    def search_parametric(self, layer_name: str) -> list[Fault]:
        return self.search(layer_name, FaultTarget.parametric())

    def search_synaptic(self, layer_name: str) -> list[Fault]:
        return self.search(layer_name, FaultTarget.synaptic())

    def optimized(
            self,
            layers_info: LayersInfo,
            late_start_en: bool = True,
            early_stop_en: bool = True
    ) -> 'OptimizedFaultRound':
        oround = OptimizedFaultRound(
            self, layers_info, late_start_en, early_stop_en
        )
        oround.fault_map = deepcopy(self.fault_map)

        return oround


# Optimized Fault Round applies only in faults after training
class OptimizedFaultRound(FaultRound):
    def __init__(
            self,
            round: FaultRound,
            layers_info: LayersInfo,
            late_start_en: bool = True,
            early_stop_en: bool = True
    ) -> None:
        # Sort round's faults in ascending order of faults appearance
        # (late-start layer first)
        super().__init__(
            FaultRound(
                sorted(
                    round.items(),
                    key=lambda item: layers_info.index(item[0][0])
                )
            )
        )
        self.fault_map = dict(
            sorted(
                self.fault_map.items(),
                key=lambda item: layers_info.index(item[0])
            )
        )

        self.is_out_faulty = any(layers_info.is_output(key[0]) for key in self)

        # The following code is useful only in the "evaluate optimized" method

        self.neuronal_only = True
        self.parametric_only = True
        self.synaptic_only = True
        for key in self:
            fm: FaultModel = key[1]
            self.neuronal_only &= fm.is_neuronal()
            self.parametric_only &= fm.is_parametric()
            self.synaptic_only &= fm.is_synaptic()

        # Late-Start and Early-Stop layers are the first and last ones
        # to contain a fault, respectively
        # For a single fault, the late-start and early-stop layers are the same
        self.late_start_en = late_start_en
        self.late_start_name = layers_info.order[0]
        self.late_start_idx = 0
        self.early_stop_en = early_stop_en and not self.is_out_faulty
        self.early_stop_name = None
        self.early_stop_idx = None

        # For an empty round (golden inference), the late-start layer is None
        # and its idx is equal to the layer number
        if not self:
            self.late_start_en = True
            self.late_start_idx = len(layers_info)
            self.early_stop_en = False
            return

        if not self.late_start_en and not self.early_stop_en:
            return

        round_iter = iter(self)

        self.late_start_name = next(round_iter, (None,))[0]
        self.late_start_idx = (
            layers_info.index(self.late_start_name)
            if self.late_start_name
            else None
        )

        if self.early_stop_en:
            self.early_stop_name = self.late_start_name
            for key in round_iter:
                self.early_stop_name = key[0]
            self.early_stop_idx = (
                layers_info.index(self.early_stop_name)
                if self.early_stop_name
                else None
            )

            # Early-Stop is meaningful only when at least
            # the 2 last layers are fault-free
            if self.early_stop_idx >= len(layers_info) - 2:
                self.early_stop_en = False
                self.early_stop_name = None
                self.early_stop_idx = None

        if not self.late_start_en:
            self.late_start_name = layers_info.order[0]
            self.late_start_idx = 0
            return

        # If there are only neuronal faults, late-start layer can be increased
        # because the fault effect is evaluated on the next layer's pre-hook
        if self.neuronal_only and not self.parametric_only:
            self.late_start_idx += 1
            if self.late_start_idx < len(layers_info):
                self.late_start_name = layers_info.order[self.late_start_idx]
            else:
                self.late_start_name = None
