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


from collections.abc import Callable, Iterable
from copy import deepcopy
from enum import Enum
from glob import glob
from importlib.metadata import version
from itertools import product
from math import prod
import numpy as np
import pickle
import random
from threading import Thread
from types import MethodType
from typing import Literal, Optional
import warnings

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer
from slayerSNN.utils import stats as spikeStats

import spikefi.fault as sff
import spikefi.utils.io as sfio
from spikefi.utils.layer import LayersInfo
from spikefi.utils.progress import CampaignProgress, refresh_progress_job


__version__ = version("spikefi")


class CampaignOptimization(Enum):
    FB = O0 = 0     # Loop-Nesting 1: Per fault, per batch
    BF = O1 = 1     # Loop-Nesting 2: Per batch, per fault
    LS = O2 = 2     # Late Start (implies loop-nesting 2)
    ES = O3 = 3     # Early Stop (implies loop-nesting 2)
    FO = O4 = 4     # Fully Optimized (all opts combined)


class Campaign:
    def __init__(
            self,
            net: nn.Module,
            shape_in: tuple[int, int, int],
            slayer: spikeLayer,
            name: str = 'sfi-campaign',
            device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.slayer = deepcopy(slayer)
        self.faulty = None
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.golden = deepcopy(net).to(self.device)
        setattr(self.golden, 'tail', torch.nn.Identity())
        self.golden.eval()

        self.layers_info = LayersInfo(shape_in)
        self.infer_layers_info()

        # Assign optimized forward function to golden network
        self.golden.forward = MethodType(
            Campaign._forward_opt_wrapper(self.layers_info, self.slayer),
            self.golden
        )

        self.r_idx_ref = RoundIndex(0)
        self.duration = 0.
        self.rounds: list[sff.FaultRound] = [sff.FaultRound()]
        self.orounds: list[sff.OptimizedFaultRound] = []
        self.rgroups: dict[str, list[int]] = {}
        # handles dict format: layer - neuron/synapse - pre-hook/hook
        self.handles: dict[str, list[list[list[RemovableHandle]]]] = {}
        self.performance: list[spikeStats] = []
        # One TrainingSynapseHook per round, index-aligned with
        # self.rounds; only populated by run_train().
        self.training_syn_hooks: list['TrainingSynapseHook'] = []

    def __repr__(self) -> str:
        s = 'FI Campaign:\n'
        s += f"  - Name: '{self.name}'\n"
        s += f"  - Network: '{self.golden.__class__.__name__}':\n"
        s += f"  - {str(self.layers_info).replace('}', '  }')}\n"
        s += f"  - Rounds ({len(self.rounds)}): {{\n"

        rounds_num = len(self.rounds)
        show_less_rounds = rounds_num > 10

        def indented(s):
            return s.replace('\n', '\n      ')

        for r in range(5) if show_less_rounds else range(rounds_num):
            s += f"      #{r}: {indented(str(self.rounds[r]))}\n"

        if show_less_rounds:
            s += "\n       ...\n\n"
            for r in range(rounds_num - 5, rounds_num):
                s += f"      #{r}: {indented(str(self.rounds[r]))}\n"

        s += '  }'

        return s

    def infer_layers_info(self) -> None:
        handles = []
        for name, child in self.golden.named_children():
            hook = self.layers_info.infer_hook_wrapper(name)
            handle = child.register_forward_hook(hook)
            handles.append(handle)

        dummy_input = torch.rand(
            (1, *self.layers_info.shape_in, 1)
        ).to(self.device)

        out = self.golden(dummy_input)
        self.golden.tail(out)

        for handle in handles:
            handle.remove()

    def inject(
            self,
            faults: sff.Fault | Iterable[sff.Fault],
            round_idx: int = -1,
            rng: random.Random | None = None
    ) -> list[sff.Fault]:
        assert (
            -len(self.rounds) <= round_idx < len(self.rounds)
        ), f'Invalid round index {round_idx}'

        if isinstance(faults, sff.Fault):
            faults = [faults]

        # Merge newly injected faults with the ones already injected
        # in this round to ensure the uniqueness and validity of all
        # faults in the entire fault round.
        round_faults = [*self.rounds[round_idx].get_faults(), *faults]

        round_faults = sff.Fault.buildup(round_faults)
        round_faults, n_unplaced = self.define_random(round_faults, rng=rng)
        round_faults, n_invalid_sites, n_dropped_faults = self.validate(round_faults)

        # Neither step raises on a shortfall; report it here instead.
        messages = []
        if n_unplaced:
            messages.append(
                f"{n_unplaced} fault site(s) had no available position and were discarded"
            )
        if n_invalid_sites or n_dropped_faults:
            parts = []
            if n_invalid_sites:
                parts.append(f"{n_invalid_sites} invalid site(s)")
            if n_dropped_faults:
                parts.append(f"{n_dropped_faults} unsupported fault(s)")
            messages.append(f"{' and '.join(parts)} were dropped during validation")
        if messages:
            warnings.warn('; '.join(messages) + '.', RuntimeWarning)

        self.rounds[round_idx].clear()
        self.rounds[round_idx].insert_many(round_faults)

        return round_faults

    def define_random(
            self,
            faults: sff.Fault | Iterable[sff.Fault],
            rng: random.Random | None = None
    ) -> tuple[list[sff.Fault], int]:
        if isinstance(faults, sff.Fault):
            faults = [faults]
        faults = list(faults)
        rng = rng or random

        # Neuron and synapse positions are disjoint spaces: resolve separately.
        for is_syn in (False, True):
            group = [f for f in faults if f.model.is_synaptic() == is_syn]  # faults of this kind
            if not group:
                continue

            layer_fixed: dict[str, list[sff.FaultSite]] = {}  # layer -> its position-only pending sites
            layer_free: list[sff.FaultSite] = []  # pending sites needing both layer and position
            for f in group:
                for s in f.sites_pending:
                    if s.layer:
                        layer_fixed.setdefault(s.layer, []).append(s)
                    else:
                        layer_free.append(s)

            if not layer_fixed and not layer_free:
                continue

            eligible_layers = self.layers_info.get_injectables()  # candidate layers for layer_free
            needed_layers = set(layer_fixed) | set(eligible_layers)  # layers whose shape is needed
            dims_by_layer: dict[str, tuple[int, int, int, int]] = {}  # layer -> (K, L, M, N) sizes
            for lay_name in needed_layers:
                shape = self.layers_info.get_shape(is_syn, lay_name)
                dims_by_layer[lay_name] = (
                    shape[0] if is_syn else 1,
                    shape[0 + is_syn],
                    shape[1 + is_syn],
                    shape[2 + is_syn]
                )

            # Positions already taken this round, per layer, by any fault of this kind.
            excluded: dict[str, set[tuple[int, ...]]] = {lay: set() for lay in needed_layers}
            for f in group:
                for s in f.sites:
                    pos4 = s.position if is_syn else (0,) + s.position  # pad neuron position to 4-tuple
                    excluded.setdefault(s.layer, set()).add(pos4)

            # Layer-fixed sites: sample within their own layer only.
            for lay_name, sites in layer_fixed.items():
                drawn = Campaign._sample_positions(
                    [(lay_name, dims_by_layer[lay_name])],
                    {lay_name: excluded[lay_name]},
                    len(sites), rng
                )  # positions assigned to as many sites as the layer still has room for
                for site, d in zip(sites, drawn):
                    pos4 = d[1:]
                    site.position = pos4[1:] if not is_syn else pos4
                    excluded[lay_name].add(pos4)
                # Any sites past len(drawn) stay undefined and are discarded below.

            # Layer-free sites: sample from all eligible layers combined.
            if layer_free:
                lay_dims = [(lay, dims_by_layer[lay]) for lay in eligible_layers]  # combined pool
                excluded_free = {lay: excluded[lay] for lay in eligible_layers}
                drawn = Campaign._sample_positions(
                    lay_dims, excluded_free, len(layer_free), rng
                )  # layer + position assigned to as many sites as the combined pool has room for
                for site, d in zip(layer_free, drawn):
                    site.layer = d[0]
                    pos4 = d[1:]
                    site.position = pos4[1:] if not is_syn else pos4
                # Any sites past len(drawn) stay undefined and are discarded below.

        # Discard whatever couldn't be placed; report only how many.
        n_unplaced = 0
        for f in faults:
            defined = [s for s in f.sites_pending if s.is_defined()]
            n_unplaced += len(f.sites_pending) - len(defined)
            f.sites_pending[:] = defined

            f.refresh(discard_duplicates=True)
            assert f.is_complete()

        return faults, n_unplaced

    def validate(
            self,
            faults: sff.Fault | Iterable[sff.Fault]
    ) -> tuple[list[sff.Fault], int, int]:
        if isinstance(faults, sff.Fault):
            faults = [faults]

        valid_faults = []
        n_invalid_sites = 0  # sites removed for a bad layer/out-of-bounds position
        n_dropped_faults = 0  # whole faults removed for an unsupported parametric target
        for f in faults:
            if (
                f.model.is_parametric()
                and f.model.param_name not in self.slayer.neuron
            ):
                n_dropped_faults += 1
                continue

            is_syn = f.model.is_synaptic()
            to_remove = set()

            for s in f.sites:  # Validate only the defined fault sites
                v = self.layers_info.is_injectable(s.layer)
                if v:
                    shape = self.layers_info.get_shape(is_syn, s.layer)
                    for i in range(len(s.position)):
                        v &= -shape[i] <= s.position[i] < shape[i]

                if not v:
                    to_remove.add(s)

            n_invalid_sites += len(to_remove)
            f.sites.difference_update(to_remove)
            if f:
                valid_faults.append(f)

        return valid_faults, n_invalid_sites, n_dropped_faults

    def then_inject(
            self,
            faults: sff.Fault | Iterable[sff.Fault],
            rng: random.Random | None = None
    ) -> list[sff.Fault]:
        self.rounds.append(sff.FaultRound())
        return self.inject(faults, -1, rng=rng)

    def inject_complete(
            self,
            fault_model: sff.FaultModel,
            layer_names: str | Iterable[str] | None = None,
            fault_sampling_k: int | None = None,
            rng: random.Random | None = None
    ) -> list[sff.Fault]:
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        if layer_names:
            # Keep only injectable layers
            # (could skip that and remove non-injectable layers
            # at the validation step but this way it is faster,
            # as the invalid faults are not even created)
            lay_names_inj = [lay_name for lay_name in layer_names
                             if self.layers_info.is_injectable(lay_name)]
        else:
            lay_names_inj = self.layers_info.get_injectables()

        if self.rounds and not len(self.rounds[-1]):
            self.rounds.pop(-1)

        is_syn = fault_model.is_synaptic()
        rng = rng or random

        # Per-layer (K, L, M, N) dimension sizes, in the same order used
        # by product() below, so that a single linear index into the
        # combined position space of all targeted layers can be mapped
        # back to a (layer, position) pair on demand (see _unrank_pos),
        # without ever materializing the full Cartesian product.
        lay_dims: list[tuple[str, tuple[int, int, int, int]]] = []
        for lay_name in lay_names_inj:
            lay_shape = self.layers_info.get_shape(is_syn, lay_name)
            lay_dims.append((lay_name, (
                lay_shape[0] if is_syn else 1,
                lay_shape[0 + is_syn],
                lay_shape[1 + is_syn],
                lay_shape[2 + is_syn]
            )))

        lay_sizes = [prod(dims) for _, dims in lay_dims]
        total_size = sum(lay_sizes)

        if fault_sampling_k is not None and fault_sampling_k < total_size:
            inj_pos = Campaign._sample_positions(lay_dims, {}, fault_sampling_k, rng)
        else:
            inj_pos = [
                (lay_name,) + p
                for lay_name, dims in lay_dims
                for p in product(*(range(dim) for dim in dims))
            ]

        inj_faults = []
        for p in inj_pos:
            site = sff.FaultSite(p[0], p[1:] if is_syn else p[2:])
            fault = sff.Fault(fault_model, site)
            inj_faults.append(self.then_inject(fault, rng=rng))

        return inj_faults

    @staticmethod
    def _unrank_pos(
            lay_dims: list[tuple[str, tuple[int, int, int, int]]],
            lay_sizes: list[int],
            idx: int
    ) -> tuple:
        # Walk layer blocks, sized lay_sizes[i], subtracting each one
        # until idx lands in and is re-based to be local to its layer.
        for (lay_name, dims), size in zip(lay_dims, lay_sizes):
            if idx < size:
                break
            idx -= size

        # Mixed-radix decode of idx into (k, l, m, n), matching
        # product(K, L, M, N)'s order (N fastest, K slowest).
        position = []
        for dim in reversed(dims):
            idx, coord = divmod(idx, dim)
            position.append(coord)
        position.reverse()

        return (lay_name,) + tuple(position)

    # Draws k unique (layer, position) tuples from the combined position
    # space of lay_dims, skipping whatever is already in excluded per
    # layer. Never materializes the full space unless it actually has to:
    # picks the cheapest of three strategies based on how much of that
    # space is excluded (see the branches below).
    @staticmethod
    def _sample_positions(
            lay_dims: list[tuple[str, tuple[int, int, int, int]]],
            excluded: dict[str, set[tuple[int, ...]]],
            k: int,
            rng: random.Random
    ) -> list[tuple]:
        lay_sizes = [prod(dims) for _, dims in lay_dims]  # per-layer position counts
        total_size = sum(lay_sizes)
        total_excluded = sum(len(excluded.get(lay, ())) for lay, _ in lay_dims)
        remaining = total_size - total_excluded
        k = min(k, max(remaining, 0))
        if k <= 0:
            return []

        if total_excluded == 0:
            # No exclusions: sample indices directly, no materialization needed.
            return [
                Campaign._unrank_pos(lay_dims, lay_sizes, idx)
                for idx in rng.sample(range(total_size), k)
            ]

        if remaining >= total_size / 2:
            # Mostly free space: rejection sampling converges fast.
            chosen: set[tuple[str, tuple]] = set()  # (layer, position) already drawn this call
            result: list[tuple] = []
            while len(result) < k:
                idx = rng.randrange(total_size)
                unranked = Campaign._unrank_pos(lay_dims, lay_sizes, idx)
                lay_name, pos = unranked[0], unranked[1:]
                if pos in excluded.get(lay_name, ()) or (lay_name, pos) in chosen:
                    continue
                chosen.add((lay_name, pos))
                result.append(unranked)
            return result

        # Mostly excluded space: enumerate the true remainder instead.
        pool = [
            (lay_name,) + p
            for lay_name, dims in lay_dims
            for p in product(*(range(d) for d in dims))
            if p not in excluded.get(lay_name, ())
        ]
        return rng.sample(pool, k)

    def eject(
            self,
            faults: sff.Fault | Iterable[sff.Fault] | None = None,
            round_idx: int | None = None
    ) -> None:
        if isinstance(faults, sff.Fault):
            faults = [faults]

        # Eject from a specific round
        if round_idx is not None:
            # Eject indicated faults from the round
            if faults:
                self.rounds[round_idx].extract_many(faults)
            # Eject all faults from the round, i.e., remove the round itself
            if not faults or not self.rounds[round_idx]:
                self.rounds.pop(round_idx)
        # Eject from all rounds
        else:
            # Eject indicated faults from any round the might exist
            if faults:
                for r in self.rounds:
                    r.extract_many(faults)
                    if not r:
                        self.rounds.pop(r)
            # Eject all faults from all rounds, i.e., all the rounds themselves
            else:
                self.rounds.clear()

        if not self.rounds:
            self.rounds.append(sff.FaultRound())

    def reset(self) -> None:
        if not self.rounds:
            self.rounds = [sff.FaultRound()]

        # Reset fault round variables
        self.r_idx_ref = RoundIndex(0)
        self.orounds.clear()
        self.rgroups.clear()
        self.handles.clear()
        self.performance.clear()
        self.training_syn_hooks.clear()

        # Create faulty version of network
        self.faulty = deepcopy(self.golden)
        self.faulty.forward = MethodType(
            Campaign._forward_opt_wrapper(self.layers_info, self.slayer),
            self.faulty
        )

    def run_train(
            self,
            epochs: int,
            train_loader: DataLoader,
            test_loader: DataLoader,
            spike_loss: snn.loss,
            optimizer_factory: Callable[[Iterable], Optimizer],
            scheduler_factory: Callable[[Optimizer], LRScheduler] | None = None,
            progress_mode: Literal[
                'verbose', 'table', 'pbar', 'silent'
            ] | None = None
    ) -> list[nn.Module]:
        # Initialize and refresh progress
        self.progress = CampaignProgress(
            len(train_loader), len(self.rounds), epochs, progress_mode
        )
        self._progress_thread = self._start_progress_thread(progress_mode)

        self.faulties = self._pre_run_train()

        self.progress.timer()
        for r_idx, faulty in enumerate(self.faulties):
            self.r_idx_ref.r = r_idx
            self.progress.step_round()
            self._evaluate_train(
                faulty,
                epochs,
                train_loader,
                test_loader,
                spike_loss,
                optimizer_factory,
                scheduler_factory
            )
        self.progress.timer()

        self._post_run(update_stats=False)

        return self.faulties

    def _pre_run_train(self) -> list[nn.Module]:
        self.reset()
        faulties = []

        for r, round in enumerate(self.rounds):
            _faulty = deepcopy(self.faulty)
            faulties.append(_faulty)
            self.performance.append(spikeStats())

            self._perturb_net(r, round, _faulty, training=True)

        return faulties

    def run(
            self,
            test_loader: DataLoader,
            spike_loss: snn.loss | None = None,
            es_tol: int = 0,
            opt: CampaignOptimization = CampaignOptimization.FO,
            progress_mode: Literal[
                'verbose', 'table', 'pbar', 'silent'
            ] | None = None
    ) -> Tensor | None:
        self._pre_run(opt)

        # Initialize and refresh progress
        self.progress = CampaignProgress(
            len(test_loader), len(self.rounds), mode=progress_mode
        )
        self._progress_thread = self._start_progress_thread(progress_mode)

        # Decide optimization level
        if len(self.rounds) <= 1:
            self.progress.step_round()
            evaluate_method = self._evaluate_single
            opt = CampaignOptimization.O0
        else:
            if opt.value >= CampaignOptimization.O2.value:
                evaluate_method = self._evaluate_optimized
            elif opt == CampaignOptimization.O1:
                evaluate_method = self._evaluate_O1
            else:
                evaluate_method = self._evaluate_O0

        # Evaluate faults' effects
        with torch.no_grad():
            eval_args = (test_loader, spike_loss)
            if opt.value >= CampaignOptimization.O3.value:
                eval_args += (es_tol,)

            self.progress.timer()
            N_critical = evaluate_method(*eval_args)
            self.progress.timer()

        self._post_run()

        return N_critical

    def _start_progress_thread(
            self,
            progress_mode: Literal[
                'verbose', 'table', 'pbar', 'silent'
            ] | None
    ) -> Thread | None:
        # In silent mode skip the polling thread entirely to avoid contending
        # for self.progress' lock against every step() call for nothing.
        if self.progress.mode == 'silent':
            return None

        thread = Thread(
            target=refresh_progress_job,
            args=(self.progress, 0.1, progress_mode, ),
            daemon=True)
        thread.start()

        return thread

    def _pre_run(self, opt: CampaignOptimization) -> None:
        self.reset()

        late_start_en = (
            opt == CampaignOptimization.LS
            or opt == CampaignOptimization.FO
        )
        early_stop_en = (
            opt == CampaignOptimization.ES
            or opt == CampaignOptimization.FO
        )

        for r, round in enumerate(self.rounds):
            # Create optimized fault rounds from rounds
            oround = round.optimized(
                self.layers_info, late_start_en, early_stop_en
            )
            self.orounds.append(oround)

            # Group fault rounds per earliest faulty layer
            self.rgroups.setdefault(oround.late_start_name, list())
            self.rgroups[oround.late_start_name].append(r)

            # Register fault (pre-)hooks for all fault rounds
            self._perturb_net(r, oround, self.faulty, training=False)

            # Create statistics for fault rounds
            self.performance.append(spikeStats())

        # Sort fault round groups in ascending order of group's earliest layer
        self.rgroups = dict(
            sorted(
                self.rgroups.items(),
                key=lambda item: (
                    -1 if item[0] is None else self.layers_info.index(item[0])
                )
            )
        )

    def _perturb_net(
            self,
            r: int,
            round: sff.FaultRound,
            faulty: nn.Module,
            training: bool
    ) -> None:
        ind_neu = sff.FaultTarget.Z.get_index()  # 0
        ind_syn = sff.FaultTarget.W.get_index()  # 1
        persistent_faults: list[tuple[nn.Module, sff.Fault]] = []

        for layer_name in self.layers_info.get_injectables():
            self.handles.setdefault(layer_name, [[[], []], [[], []]])
            layer = getattr(faulty, layer_name)

            # Neuronal faults
            if round.any_neuronal(layer_name):
                # Parametric faults (subset of neuronal faults)
                if round.any_parametric(layer_name):
                    param_faults = round.search_parametric(layer_name)
                    for fault in param_faults:
                        # Create parametric faults' dummy layers
                        fault.model.param_perturb(self.slayer, self.device)

                    # Register parametric neuron fault hooks
                    # (on the faulty layer)
                    hook = NeuronHook(
                        hook_type='param',
                        faults=param_faults,
                        active_round_idx=r,
                        actual_round_idx=self.r_idx_ref,
                        layer_shape=self.layers_info.shapes_neu[layer_name]
                    )

                    self.handles[layer_name][ind_neu][1].append(
                        layer.register_forward_hook(hook)
                    )

                # Neuronal faults for last layer are evaluated on
                # a 'tail' layer that does nothing
                following_layer = getattr(
                    faulty, self.layers_info.get_following(layer_name)
                )

                # Register neuron fault pre-hooks
                # (on the layer succeeding the faulty layer)
                pre_hook = NeuronHook(
                    hook_type='neuron',
                    faults=round.search_neuronal(layer_name),
                    active_round_idx=r,
                    actual_round_idx=self.r_idx_ref,
                    layer_shape=self.layers_info.shapes_neu[layer_name]
                )

                self.handles[layer_name][ind_neu][0].append(
                    following_layer.register_forward_pre_hook(pre_hook)
                )

            # Synaptic faults
            if round.any_synaptic(layer_name):
                syn_faults = round.search_synaptic(layer_name)

                if not training:
                    # Inference: perturb before every forward pass and
                    # restore after it.
                    pre_hook, hook = SynapseHook.generate(
                        faults=syn_faults,
                        active_round_idx=r,
                        actual_round_idx=self.r_idx_ref
                    )

                    self.handles[layer_name][ind_syn][0].append(
                        layer.register_forward_pre_hook(pre_hook)
                    )

                    # Store the perturbed synapse weight in the cache
                    for fault in syn_faults:
                        fault.model.perturb_store(layer.weight[fault.unroll()], clean=True)

                    self.handles[layer_name][ind_syn][1].append(
                        layer.register_forward_hook(hook)
                    )
                else:
                    # Training: every synaptic fault is applied once,
                    # right now, as the network's initial faulty state.
                    # For "soft" synapse faults (persistent=False),
                    # this is their only application.
                    with torch.no_grad():
                        for fault in syn_faults:
                            all_ind = fault.unroll()
                            layer.weight[all_ind] = fault.model.perturb(layer.weight[all_ind])

                            # Persistent faults are collected here so a
                            # TrainingSynapseHook attached to the round's
                            # optimizer keeps re-enforcing them.
                            if fault.model.persistent:
                                persistent_faults.append((layer, fault))

        if training:
            self.training_syn_hooks.append(TrainingSynapseHook(persistent_faults))

    def _post_run(self, update_stats: bool = True):
        self.duration = self.progress.get_duration_sec()

        if self._progress_thread is not None:
            self._progress_thread.join()
        else:
            # In silent mode close the bar directly
            with self.progress._lock:
                self.progress.pbar.close()
        del self._progress_thread

        # Update fault rounds statistics
        if update_stats:
            for stats in self.performance:
                stats.update()

        # Replace erroneous 'None' elements in statistics (slayer issue)
        for stats in self.performance:
            stats.training.accuracyLog = [
                x or 0. for x in stats.training.accuracyLog
            ]
            stats.testing.accuracyLog = [
                x or 0. for x in stats.testing.accuracyLog
            ]

    def _evaluate_train(
            self,
            faulty: nn.Module,
            epochs: int,
            train_loader: DataLoader,
            test_loader: DataLoader,
            spike_loss: snn.loss,
            optimizer_factory: Callable[[Iterable], Optimizer],
            scheduler_factory: Callable[[Optimizer], LRScheduler] | None = None,
    ) -> None:
        stat = self.performance[self.r_idx_ref.r].training

        optimizer_ = optimizer_factory(faulty.parameters())
        has_scheduler = scheduler_factory is not None
        scheduler_ = scheduler_factory(optimizer_) if has_scheduler else None

        training_hook = self.training_syn_hooks[self.r_idx_ref.r]
        if training_hook:
            optimizer_.register_step_post_hook(training_hook)

        for epoch in range(epochs):
            self.progress.step_epoch()

            # Training loop
            faulty.train()
            for input, label in train_loader:
                self.progress.step_batch()

                output = faulty.forward(input.to(self.device))

                loss, _ = self._advance_performance(
                    output, label.to(self.device), spike_loss, training=True
                )

                optimizer_.zero_grad()
                loss.backward()
                optimizer_.step()

                self.progress.set_train(stat.loss(), stat.accuracy())
                self.progress.step()

            if has_scheduler:
                scheduler_.step()

            # Testing loop
            faulty.eval()
            with torch.inference_mode():
                for input, label in test_loader:
                    output = faulty.forward(input.to(self.device))

                    loss, _ = self._advance_performance(
                        output, label.to(self.device), spike_loss, training=False
                    )

            self.performance[self.r_idx_ref.r].update()

            # Save the trained network instance with the best testing accuracy
            accu = np.asarray(self.performance[self.r_idx_ref.r].testing.accuracyLog, dtype=float)
            rev_i = np.nanargmax(accu[::-1])
            last_max_epoch = accu.size - 1 - rev_i

            if last_max_epoch == epoch:
                best_state_dict = deepcopy(faulty.state_dict())

        # At the end, restore the best model into net
        if best_state_dict is not None:
            faulty.load_state_dict(best_state_dict)

        faulty.eval()

    def _evaluate_single(
            self,
            test_loader: DataLoader,
            spike_loss: snn.loss | None = None
    ) -> None:
        for input, label in test_loader:
            self.progress.step_batch()

            output = self.faulty(input.to(self.device))

            self._advance_performance(
                output, label.to(self.device), spike_loss
            )
            self.progress.step()

    def _evaluate_O0(
            self,
            test_loader: DataLoader,
            spike_loss: snn.loss | None = None
    ) -> None:
        # For each fault round group
        for round_group in self.rgroups.values():
            # For each fault round
            for r_idx in round_group:
                self.r_idx_ref.r = r_idx
                self.progress.step_round()
                self._evaluate_single(test_loader, spike_loss)

    def _evaluate_O1(
            self,
            test_loader: DataLoader,
            spike_loss: snn.loss | None = None
    ) -> None:
        # For each batch
        for input, label in test_loader:
            self.progress.step_batch()

            # For each fault round group
            for round_group in self.rgroups.values():
                # For each fault round
                for r_idx in round_group:
                    self.r_idx_ref.r = r_idx
                    self.progress.step_round()

                    output = self.faulty(input.to(self.device))

                    self._advance_performance(
                        output, label.to(self.device), spike_loss
                    )
                    self.progress.step()

    def _evaluate_optimized(
            self,
            test_loader: DataLoader,
            spike_loss: snn.loss | None = None,
            es_tol: int = 0
    ) -> Tensor:
        N_critical = torch.zeros(len(self.rounds), dtype=torch.int, device=self.device)

        # For each batch
        for input, label in test_loader:
            label = label.to(self.device)
            self.progress.step_batch()

            # Store golden spikes
            golden_spikes = [input.to(self.device)]
            for layer_idx in range(len(self.layers_info)):
                golden_spikes.append(
                    self.golden(golden_spikes[layer_idx], layer_idx, layer_idx)
                )
            golden_pred = golden_spikes[-1].sum(dim=(2, 3, 4)).argmax(dim=1)

            # For each fault round group
            for round_group in self.rgroups.values():
                # For each fault round
                for r_idx in round_group:
                    self.r_idx_ref.r = r_idx
                    self.progress.step_round()

                    oround = self.orounds[r_idx]
                    ls_idx = oround.late_start_idx
                    es_idx = oround.early_stop_idx

                    if not oround.early_stop_en:
                        output = self.faulty(golden_spikes[ls_idx], ls_idx)
                    else:
                        # Early stop optimization
                        early_stop_next_out = self.faulty(
                            golden_spikes[ls_idx], ls_idx, es_idx + 1
                        )
                        early_stop = torch.sum(
                            early_stop_next_out.ne(golden_spikes[es_idx + 2]),
                            dim=(1, 2, 3, 4)
                        ) <= es_tol

                        # Replace output only for the
                        # affected samples of the batch
                        output = torch.zeros(
                            golden_spikes[-1].size()
                        ).to(self.device)
                        output[early_stop] = golden_spikes[-1][early_stop]
                        if torch.any(~early_stop):
                            output[~early_stop] = self.faulty(
                                early_stop_next_out[~early_stop], es_idx + 2
                            )

                    pred = output.sum(dim=(2, 3, 4)).argmax(dim=1)
                    N_critical[r_idx] += torch.sum(
                        (golden_pred == label) & (pred != label)
                    )

                    self._advance_performance(
                        output, label, spike_loss
                    )
                    self.progress.step()

        return N_critical

    def _advance_performance(
            self,
            output: Tensor,
            label: Tensor,
            spike_loss: snn.loss | None = None,
            training: bool = False
    ) -> tuple[Tensor, int] | None:
        if spike_loss is not None:
            # One-hot vector for labels: target[b, label[b], 0, 0, 0] = 1
            target = (
                torch.zeros_like(output[..., :1])
                .scatter_(1, label.view(-1, 1, 1, 1, 1), 1.0)
            )

            loss = spike_loss.numSpikes(output, target)

        with torch.no_grad():
            perf = self.performance[self.r_idx_ref.r]
            stat = perf.training if training else perf.testing

            predict = output.sum(dim=(2, 3, 4)).argmax(dim=1)
            correct = (predict == label).sum().item()
            batch_s = label.size(0)

            stat.correctSamples += correct
            stat.numSamples += batch_s

            if spike_loss is not None:
                stat.lossSum += loss.detach().item()

        if spike_loss is not None:
            return loss, target

        return None

    def export(self) -> 'CampaignData':
        return CampaignData(self)

    def save(self, fname: str | None = None) -> None:
        self.export().save(fname)

    def save_net(
            self,
            net: nn.Module | None = None,
            fname: str | None = None
    ) -> None:
        to_save = net or self.faulty
        if not to_save:
            return

        torch.save(
            to_save.state_dict(),
            sfio.make_net_filepath((fname or self.name) + '.pt', rename=True)
        )

    @staticmethod
    def load(
        fpath: str,
        unpickler_type: type[pickle.Unpickler] | None = None
    ) -> 'Campaign':
        return CampaignData.load(fpath, unpickler_type).restore()

    @staticmethod
    def load_many(
        pathname: str,
        unpickler_type: type[pickle.Unpickler] | None = None
    ) -> list['Campaign']:
        return [
            cmpn_data.restore()
            for cmpn_data in CampaignData.load_many(pathname, unpickler_type)
        ]

    @staticmethod
    def _forward_opt_wrapper(
        layers_info: LayersInfo,
        slayer: spikeLayer
    ) -> Callable[[Tensor, Optional[int], Optional[int]], Tensor]:
        def forward_opt(
                self: nn.Module,
                spikes_in: Tensor,
                start_layer_idx: int = None,
                end_layer_idx: int = None
        ) -> Tensor:
            start_idx = 0 if start_layer_idx is None else start_layer_idx
            end_idx = (
                (len(layers_info) - 1)
                if end_layer_idx is None else end_layer_idx
            )

            if start_idx < 0:
                start_idx = len(layers_info) + start_idx
            if end_idx < 0:
                end_idx = len(layers_info) + end_idx

            subject_layers = [
                lay_name for lay_idx, lay_name in enumerate(layers_info.order)
                if start_idx <= lay_idx <= end_idx
            ]

            spikes = torch.clone(spikes_in)
            for layer_name in subject_layers:
                layer = getattr(self, layer_name)
                spikes = layer(spikes)

                # Skip dropout and tail layers from calling slayer functions
                if (
                    layers_info.types[layer_name]
                    in (snn.slayer._dropoutLayer, nn.Identity)
                ):
                    continue

                spikes = slayer.spike(slayer.psp(spikes))

            return spikes

        return forward_opt


class RoundIndex:
    def __init__(self, r: int = 0):
        self.r = r


class NeuronHook:
    def __init__(
            self,
            hook_type: Literal['neuron', 'param'],
            faults: Iterable[sff.Fault],
            active_round_idx: int,
            actual_round_idx: RoundIndex,
            layer_shape: tuple[int, int, int]
    ) -> None:
        self.hook_type = hook_type
        self.faults = faults
        self.active_round_idx = active_round_idx
        self.actual_round_idx = actual_round_idx
        self.layer_shape = layer_shape

    def __call__(self, _, *args) -> None:
        if self.actual_round_idx.r != self.active_round_idx:
            return

        if self.hook_type == "neuron":
            self._neuron_pre_hook(_, args[0])
        elif self.hook_type == "param":
            self._neuron_param_hook(_, *args)

    def _neuron_pre_hook(self, _, inputs: tuple[Tensor, ...]) -> None:
        prev_spikes_out = inputs[0]
        # Verify that the pre-hook attached on shared
        # dropout layers is executed after the faulty layer
        if prev_spikes_out.shape[1:4] != self.layer_shape:
            return

        for fault in self.faults:
            idx = (slice(None), *fault.unroll(), slice(None))
            fspike_out = fault.model.unstore()
            fm_args = (
                (fspike_out,)
                if fspike_out is not None
                else fault.model.args
            )

            prev_spikes_out[idx] = fault.model.perturb(
                prev_spikes_out[idx], *fm_args
            )

    def _neuron_param_hook(self, _, __, spikes_out: Tensor) -> None:
        for fault in self.faults:
            idx = (slice(None), *fault.unroll(), slice(None))

            # Evaluate the dummy layer only on the fault sites
            val_site = spikes_out[idx]
            b, s, d = val_site.shape

            flayer = fault.model.flayer
            fspike_out = flayer.spike(flayer.psp(val_site.reshape(b, s, 1, 1, d)))
            fault.model.store(fspike_out.reshape(b, s, d))


class SynapseHook:
    def __init__(
            self,
            hook_type: Literal['pre', 'post'],
            faults: Iterable[sff.Fault],
            active_round_idx: int,
            actual_round_idx: RoundIndex
    ) -> None:
        self.hook_type = hook_type
        self.faults = faults
        self.active_round_idx = active_round_idx
        self.actual_round_idx = actual_round_idx

    def __call__(self, layer: nn.Module, *args) -> None:
        if self.actual_round_idx.r != self.active_round_idx:
            return

        for fault in self.faults:
            all_ind = fault.unroll()
            with torch.no_grad():
                if self.hook_type == "pre":
                    # This is only used in inference, so a cached
                    # perturbed value is assumed (stored while
                    # attaching the synapse hook), since the
                    # weights remain unchanged.
                    layer.weight[all_ind] = fault.model.perturbed
                elif self.hook_type == "post":
                    layer.weight[all_ind] = fault.model.restore()

    @staticmethod
    def generate(
        faults: Iterable[sff.Fault],
        active_round_idx: int,
        actual_round_idx: RoundIndex
    ) -> tuple['SynapseHook', 'SynapseHook']:
        common_args = (faults, active_round_idx, actual_round_idx)

        pre_hook = SynapseHook('pre', *common_args)
        hook = SynapseHook('post', *common_args)

        return pre_hook, hook


class TrainingSynapseHook:
    def __init__(self, persistent_faults: list[tuple[nn.Module, sff.Fault]]) -> None:
        self.persistent_faults: list[tuple[nn.Module, sff.Fault]] = persistent_faults

    def __bool__(self) -> bool:
        return bool(self.persistent_faults)

    def __call__(self, optimizer: Optimizer, args: tuple, kwargs: dict) -> None:
        # Fires synchronously right after optimizer.step() returns, so
        # persistent faults are never observed in a drifted state.
        with torch.no_grad():
            for layer, fault in self.persistent_faults:
                all_ind = fault.unroll()
                layer.weight[all_ind] = fault.model.perturb(layer.weight[all_ind])


# CampaignData is essential for the (de)serialization of Campaign
# objects and the easy handling of results with the spikefi visual
# module functions.
class CampaignData:
    def __init__(self, campaign: Campaign) -> None:
        self.version = __version__
        self.name = campaign.name

        self.golden = deepcopy(campaign.golden).to('cpu')
        self.slayer = deepcopy(campaign.slayer)
        self.device = deepcopy(campaign.device)

        self.layers_info = deepcopy(campaign.layers_info)

        # Restore default forward function to golden network
        self.golden.forward = MethodType(
            type(self.golden).forward, self.golden
        )

        self.duration = campaign.duration
        self.rounds = deepcopy(campaign.rounds)
        self.orounds = deepcopy(campaign.orounds)
        self.rgroups = deepcopy(campaign.rgroups)
        self.performance = deepcopy(campaign.performance)

    # Restoring a Campaign from its Campaign Data relies on reconstructing
    # the Campaign object (i.e., calling Campaign.__init__()), which is
    # not directly applicable in the case of using __get/setstate__ with
    # a dictionary containing the Campaign Data object in Campaign.
    def restore(self) -> Campaign:
        campaign = Campaign(
            self.golden, self.layers_info.shape_in, self.slayer, self.name
        )
        campaign.rounds = self.rounds

        return campaign

    def save(self, fname: str | None = None) -> None:
        with open(
            sfio.make_res_filepath(
                (fname or self.name) + '.pkl', rename=True
            ), 'wb'
        ) as pkl:
            pickle.dump(self, pkl)

    @staticmethod
    def load(
        fpath: str,
        unpickler_type: type[pickle.Unpickler] | None = None
    ) -> 'CampaignData':
        with open(fpath, 'rb') as pkl:
            if unpickler_type is None:
                return pickle.load(pkl)
            return unpickler_type(pkl).load()

    @staticmethod
    def load_many(
        pathname: str,
        unpickler_type: type[pickle.Unpickler] | None = None
    ) -> list['CampaignData']:
        return [
            CampaignData.load(fpath, unpickler_type)
            for fpath in glob(pathname)
        ]
