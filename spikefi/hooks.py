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


from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

import spikefi.fault as sff


class RoundIndex:
    def __init__(self, r: int = 0):
        self.r = r


# Where the optimized forward currently is in LayersInfo.order, shared by
# reference with the neuron pre-hooks. A module used more than once by the
# network (a shared dropout, say) occupies one position per invocation, so
# a position identifies which of those invocations is running.
class LayerPosition:
    def __init__(self, p: int = -1):
        self.p = p


# Pre-hook on the layer succeeding a faulty layer: overwrites the
# incoming spikes at each fault site with its perturbed value.
def _perturb_neuron_spikes(
        faults: list[sff.Fault],
        prev_spikes_out: Tensor,
        layer_name: str
) -> None:
    if not faults:
        return

    for fault in faults:
        idx = (slice(None), *fault.unroll(), slice(None))
        fspike_out = fault.model.unstore()

        if fspike_out is not None:
            fm_args = (fspike_out,)
        elif fault.model.is_parametric():
            # A neuron parametric fault must always have a value stashed.
            raise RuntimeError(
                f"No stashed value for parametric fault {fault.model} "
                f"on layer '{layer_name}'"
            )
        else:
            fm_args = fault.model.args

        prev_spikes_out[idx] = fault.model.perturb(
            prev_spikes_out[idx], *fm_args
        )


# Forward hook on a faulty layer: evaluates each parametric fault's dummy
# neuron layer on the fault site and stashes the result, for the neuron
# perturb pre-hook to pick up on the following layer's pre-hook.
def _stash_parametric_spikes(
        faults: list[sff.Fault],
        spikes_out: Tensor
) -> None:
    if not faults:
        return

    for fault in faults:
        idx = (slice(None), *fault.unroll(), slice(None))

        # Evaluate the dummy layer only on the fault sites
        val_site = spikes_out[idx]
        b, s, d = val_site.shape

        flayer = fault.model.flayer
        fspike_out = flayer.spike(flayer.psp(val_site.reshape(b, s, 1, 1, d)))
        # Left attached: the following layer's pre-hook writes this value
        # straight into its input, so gradient can reach the real synaptic
        # weights that produced val_site during training. Different than
        # a hard neuron fault, whose set_value() ignores its input and is
        # fan-in-frozen by construction.
        fault.model.store(fspike_out.reshape(b, s, d), detach=False)


# Base for all layer-scoped fault hook classes, declaring what they share:
# the faulty layer they serve and the faults active on it.
class FaultHook(ABC):
    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(layer='{self.layer_name}')"

    def __bool__(self) -> bool:
        return bool(self._active_faults())

    @abstractmethod
    def _active_faults(self) -> list[sff.Fault]:
        ...

    @abstractmethod
    def __call__(self, *args) -> None:
        ...


# Base for dispatching fault hook classes: they serve every round touching a
# given dispatching point from one object. Used only in post-training FI,
# where a single faulty net evaluates every round in turn and a hook must
# resolve "whose faults, right now" dynamically.
class DispatchingFaultHook(FaultHook):
    def __init__(
            self,
            actual_round_idx: RoundIndex,
            rounds_ref: Sequence[sff.FaultRound],
            layer_name: str,
            target: sff.FaultTarget
    ) -> None:
        super().__init__(layer_name)
        self.actual_round_idx = actual_round_idx
        self.rounds_ref = rounds_ref
        self.target = target

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(layer='{self.layer_name}', "
            f"target={self.target})"
        )

    # Returns the faults of the active round this hook is responsible for
    def _active_faults(self) -> list[sff.Fault]:
        round = self.rounds_ref[self.actual_round_idx.r]
        return round.grouped.get((self.layer_name, self.target), [])


# Used only in pre-training FI, each round has its own faulty net, so there is
# exactly one round a hook here will ever need to serve. Holding the Fault objects
# directly makes a net returned by run_train() self-contained, ensuring that
# inference runs correctly even after the Campaign that built it is gone.
class DirectFaultHook(FaultHook):
    def __init__(self, faults: list[sff.Fault], layer_name: str) -> None:
        super().__init__(layer_name)
        # Ordinary reference to faults keeps them alive even without the
        # original Campaign object where they derive from.
        self.faults = faults

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(layer='{self.layer_name}', "
            f"faults={len(self.faults)})"
        )

    def _active_faults(self) -> list[sff.Fault]:
        return self.faults


class DispatchingNeuronPerturbPreHook(DispatchingFaultHook):
    def __init__(
            self,
            actual_round_idx: RoundIndex,
            rounds_ref: Sequence[sff.FaultRound],
            layer_name: str,
            position_ref: LayerPosition,
            following_pos: int
    ) -> None:
        super().__init__(
            actual_round_idx, rounds_ref, layer_name,
            sff.FaultTarget.neuronal()
        )
        self.position_ref = position_ref
        self.following_pos = following_pos

    # Neuronal faults span two targets, which are grouped separately
    def _active_faults(self) -> list[sff.Fault]:
        grouped = self.rounds_ref[self.actual_round_idx.r].grouped
        return (
            grouped.get((self.layer_name, sff.FaultTarget.OUTPUT), [])
            + grouped.get((self.layer_name, sff.FaultTarget.PARAMETER), [])
        )

    def __call__(self, _, inputs: tuple[Tensor, ...]) -> None:
        # The module this pre-hook sits on may be invoked several times per
        # forward pass; only the invocation directly following the faulty
        # layer carries that layer's spikes.
        if self.position_ref.p != self.following_pos:
            return

        _perturb_neuron_spikes(
            self._active_faults(), inputs[0], self.layer_name
        )


class DirectNeuronPerturbPreHook(DirectFaultHook):
    def __init__(
            self,
            faults: list[sff.Fault],
            layer_name: str,
            position_ref: LayerPosition,
            following_pos: int
    ) -> None:
        super().__init__(faults, layer_name)
        self.position_ref = position_ref
        self.following_pos = following_pos

    def __call__(self, _, inputs: tuple[Tensor, ...]) -> None:
        if self.position_ref.p != self.following_pos:
            return

        _perturb_neuron_spikes(
            self._active_faults(), inputs[0], self.layer_name
        )


class DispatchingNeuronParametricHook(DispatchingFaultHook):
    def __init__(
            self,
            actual_round_idx: RoundIndex,
            rounds_ref: Sequence[sff.FaultRound],
            layer_name: str
    ) -> None:
        super().__init__(
            actual_round_idx, rounds_ref, layer_name,
            sff.FaultTarget.PARAMETER
        )

    def __call__(self, _, __, spikes_out: Tensor) -> None:
        _stash_parametric_spikes(self._active_faults(), spikes_out)


class DirectNeuronParametricHook(DirectFaultHook):
    def __call__(self, _, __, spikes_out: Tensor) -> None:
        _stash_parametric_spikes(self._active_faults(), spikes_out)


# Pre-hook on a faulty layer: writes each fault's cached perturbed
# weight value in before the forward pass. Post-training FI only.
class DispatchingSynapsePerturbPreHook(DispatchingFaultHook):
    def __init__(
            self,
            actual_round_idx: RoundIndex,
            rounds_ref: Sequence[sff.FaultRound],
            layer_name: str
    ) -> None:
        super().__init__(
            actual_round_idx, rounds_ref, layer_name,
            sff.FaultTarget.WEIGHT
        )

    def __call__(self, layer: nn.Module, *args) -> None:
        faults = self._active_faults()
        if not faults:
            return

        with torch.no_grad():
            for fault in faults:
                all_ind = fault.unroll()
                layer.weight[all_ind] = fault.model.perturbed


# Forward hook on a faulty layer: restores each fault's original
# weight value after the forward pass to isolate each fault round.
# Pairs with DispatchingSynapsePerturbPreHook. Post-training FI only.
class DispatchingSynapseRestoreHook(DispatchingFaultHook):
    def __init__(
            self,
            actual_round_idx: RoundIndex,
            rounds_ref: Sequence[sff.FaultRound],
            layer_name: str
    ) -> None:
        super().__init__(
            actual_round_idx, rounds_ref, layer_name,
            sff.FaultTarget.WEIGHT
        )

    def __call__(self, layer: nn.Module, *args) -> None:
        faults = self._active_faults()
        if not faults:
            return

        with torch.no_grad():
            for fault in faults:
                all_ind = fault.unroll()
                layer.weight[all_ind] = fault.model.restore()


# Optimizer step-post-hook for persistent synapse faults in run_train():
# persistent synapse faults are re-applied after every optimizer step so
# they aren't trained away. One hook per round, aggregating faults across
# every layer that round touches, as (layer, fault) pairs.
#   - Within the campaign's training, the fault is delivered correctly:
#     every optimizer.step() is immediately followed by this hook
#     re-clamping the weight to the fault value, which is carried by
#     state_dict() in the faulty net after training.
#   - The term "persistent" concerns the duration of this campaign's
#     training only. If training is resumed later in a bare training loop,
#     or in another campaign that doesn't inject this same fault, then
#     ordinary gradient updates train the fault away over time.
# The 'Direct' in the name denotes the fault resolution strategy it shares
# with DirectFaultHook, not inheritance from it: being bound to the optimizer
# and spanning every layer of the round, it is not a layer-scoped FaultHook.
class DirectSynapsePersistentOptimizerHook:
    def __init__(self, faulty: nn.Module, round: sff.FaultRound) -> None:
        self.persistent_syn_faults: list[tuple[nn.Module, sff.Fault]] = [
            (getattr(faulty, layer_name), fault)
            for (layer_name, target), faults in round.grouped.items()
            if target == sff.FaultTarget.WEIGHT
            for fault in faults
            if fault.model.persistent
        ]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(persistent_syn_faults={len(self.persistent_syn_faults)})"
        )

    def __bool__(self) -> bool:
        return bool(self.persistent_syn_faults)

    def __call__(self, optimizer: Optimizer, args: tuple, kwargs: dict) -> None:
        # Fires synchronously right after optimizer.step() returns, so
        # persistent faults are never observed in a drifted state.
        with torch.no_grad():
            for layer, fault in self.persistent_syn_faults:
                all_ind = fault.unroll()
                layer.weight[all_ind] = fault.model.perturb(layer.weight[all_ind])
