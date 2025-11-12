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
from itertools import cycle, product
import pickle
import random
from threading import Thread
from types import MethodType
from typing import Optional
from warnings import warn

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer
from slayerSNN.utils import stats as spikeStats

import spikefi.fault as ff
import spikefi.utils.io as sfi_io
from spikefi.utils.layer import LayersInfo
from spikefi.utils.progress import CampaignProgress, refresh_progress_job


VERSION = version("spikefi")


class CampaignOptimization(Enum):
    FB = O0 = 0     # Loop-Nesting 1: Per fault, per batch
    BF = O1 = 1     # Loop-Nesting 2: Per batch, per fault
    LS = O2 = 2     # Late Start (implies loop-nesting 2)
    ES = O3 = 3     # Early Stop (implies loop-nesting 2)
    FO = O4 = 4     # Fully Optimized (all opts combined)


class Campaign:
    def __init__(self, net: nn.Module, shape_in: tuple[int, int, int], slayer: spikeLayer, name: str = 'sfi-campaign') -> None:
        self.name = name
        self.golden = net
        setattr(self.golden, 'tail', torch.nn.Identity())
        self.golden.eval()
        self.faulty = None
        self.slayer = slayer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layers_info = LayersInfo(shape_in)
        self.infer_layers_info()

        # Assign optimized forward function to golden network
        self.golden.forward = MethodType(Campaign._forward_opt_wrapper(self.layers_info, self.slayer), self.golden)

        self.r_idx = 0
        self.duration = 0.
        self.rounds: list[ff.FaultRound] = [ff.FaultRound()]
        self.orounds: list[ff.OptimizedFaultRound] = []
        self.rgroups: dict[str, list[int]] = {}
        self.handles: dict[str, list[list[list[RemovableHandle]]]] = {}  # layer - neuron/synapse - pre-hook/hook
        self.performance: list[spikeStats] = []

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

        dummy_input = torch.rand((1, *self.layers_info.shape_in, 1)).to(self.device)
        out = self.golden(dummy_input)
        self.golden.tail(out)

        for handle in handles:
            handle.remove()

    def inject(self, faults: ff.Fault | Iterable[ff.Fault], round_idx: int = -1) -> list[ff.Fault]:
        assert -len(self.rounds) <= round_idx < len(self.rounds), f'Invalid round index {round_idx}'

        if isinstance(faults, ff.Fault):
            faults = [faults]

        self.define_random(faults)
        inj_faults = self.validate(faults)
        inj_faults = ff.Fault.buildup(inj_faults)

        self.rounds[round_idx].insert_many(inj_faults)

        return inj_faults

    def define_random(self, faults: ff.Fault | Iterable[ff.Fault]) -> Iterable[ff.Fault]:
        if isinstance(faults, ff.Fault):
            faults = [faults]

        # Uniqueness of fault sites for multiple faults is guaranteed for all sites
        # within the same fault object
        for f in faults:
            is_syn = f.model.is_synaptic()
            l_count = {}
            l_pos = {}
            l_pos_iter = {}

            for s in f.sites_pending:
                if not s.layer:
                    s.layer = self.layers_info.get_random_inj(is_syn)

                l_count.setdefault(s.layer, 0)
                l_count[s.layer] += 1

            for lay, l_site_num in l_count.items():
                ranges = [range(dim) for dim in self.layers_info.get_shape(is_syn, lay)]
                all_comb = list(product(*ranges))

                pos_excl = [p for p in [fs.position for fs in f.sites if fs.layer == lay]]
                pos_comb = [p for p in all_comb if p not in pos_excl]

                if l_site_num >= len(pos_comb):
                    l_pos[lay] = pos_comb
                else:
                    l_pos[lay] = random.sample(pos_comb, l_site_num)

                l_pos_iter[lay] = cycle(l_pos[lay])

            for s in f.sites_pending:
                s.position = next(l_pos_iter[s.layer])

            f.refresh(discard_duplicates=True)
            assert f.is_complete()

        return faults

    def validate(self, faults: ff.Fault | Iterable[ff.Fault]) -> list[ff.Fault]:
        if isinstance(faults, ff.Fault):
            faults = [faults]

        valid_faults = []
        for f in faults:
            if f.model.is_parametric() and f.model.param_name not in self.slayer.neuron:
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

            f.sites.difference_update(to_remove)
            if f:
                valid_faults.append(f)

        return valid_faults

    def then_inject(self, faults: ff.Fault | Iterable[ff.Fault]) -> list[ff.Fault]:
        self.rounds.append(ff.FaultRound())
        return self.inject(faults, -1)

    def inject_complete(self, fault_model: ff.FaultModel, layer_names: str | Iterable[str] = [], fault_sampling_k: int = None) -> list[ff.Fault]:
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        if layer_names:
            # Keep only injectable layers
            # (could skip that and remove non-injectable layers at the validation step
            #  but this way it is faster as the invalid faults are not even created)
            lay_names_inj = [lay_name for lay_name in layer_names
                             if self.layers_info.is_injectable(lay_name)]
        else:
            lay_names_inj = self.layers_info.get_injectables()

        if self.rounds and not len(self.rounds[-1]):
            self.rounds.pop(-1)

        is_syn = fault_model.is_synaptic()

        inj_pos = []
        for lay_name in lay_names_inj:
            lay_shape = self.layers_info.get_shape(is_syn, lay_name)

            K = range(lay_shape[0] if is_syn else 1)
            L = range(lay_shape[0 + is_syn])
            M = range(lay_shape[1 + is_syn])
            N = range(lay_shape[2 + is_syn])

            inj_pos += [(lay_name,) + p for p in product(K, L, M, N)]

        if fault_sampling_k is not None and fault_sampling_k <= len(inj_pos):
            inj_pos = random.sample(inj_pos, fault_sampling_k)

        inj_faults = []
        for p in inj_pos:
            inj_faults.append(self.then_inject(
                [ff.Fault(fault_model, ff.FaultSite(p[0], (p[1] if is_syn else slice(None), *p[2:])))]
            ))

        return inj_faults

    def eject(self, faults: ff.Fault | Iterable[ff.Fault] = None, round_idx: int = None) -> None:
        if isinstance(faults, ff.Fault):
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
            self.rounds.append(ff.FaultRound())

    def reset(self) -> None:
        if not self.rounds:
            self.rounds = [ff.FaultRound()]

        # Reset fault round variables
        self.r_idx = 0
        self.orounds.clear()
        self.rgroups.clear()
        self.handles.clear()
        self.performance.clear()

        # Create faulty version of network
        self.faulty = deepcopy(self.golden)
        self.faulty.forward = MethodType(Campaign._forward_opt_wrapper(self.layers_info, self.slayer), self.faulty)

    def run_train(self, epochs: int, train_loader: DataLoader,
                  optimizer: Optimizer, spike_loss: snn.loss,
                  progress_mode: str = '') -> list[nn.Module]:
        # Initialize and refresh progress
        self.progress = CampaignProgress(len(train_loader), len(self.rounds), epochs)
        self._progress_thread = Thread(target=refresh_progress_job, args=(self.progress, 0.1, progress_mode,), daemon=True)
        self._progress_thread.start()

        self.faulties = self._pre_run_train()

        self.progress.timer()
        for self.r_idx, faulty in enumerate(self.faulties):
            self.progress.step_round()
            self._evaluate_train(faulty, epochs, train_loader, optimizer, spike_loss)
        self.progress.timer()

        self._post_run(update_stats=False)

        return self.faulties

    def _pre_run_train(self) -> list[nn.Module]:
        self.reset()
        faulties = []

        ind_neu = ff.FaultTarget.Z.get_index()  # 0
        ind_syn = ff.FaultTarget.W.get_index()  # 1

        for round in self.rounds:
            _faulty = deepcopy(self.faulty)
            faulties.append(_faulty)
            self.performance.append(spikeStats())

            # Register neuron faults (pre-)hooks
            for layer_name in self.layers_info.get_injectables():
                self.handles.setdefault(layer_name, [[[], []], [[], []]])
                layer = getattr(self.faulty, layer_name)

                if round.any_neuronal(layer_name):
                    if not self.layers_info.is_output(layer_name):
                        following_layer = getattr(_faulty, self.layers_info.get_following(layer_name))

                        # Register neuron fault pre-hooks
                        pre_hook = self._neuron_hook_wrapper(round.search_neuronal(layer_name), is_out_layer=False)
                        self.handles[layer_name][ind_neu][0].append(following_layer.register_forward_pre_hook(pre_hook))
                    else:
                        out_layer = getattr(_faulty, self.layers_info.get_injectables()[-1])

                        # Register neuron fault hook for output layer only
                        hook = self._neuron_hook_wrapper(faults=round.search_neuronal(layer_name), is_out_layer=True)
                        self.handles[layer_name][ind_neu][1].append(out_layer.register_forward_hook(hook))

                    # Parametric faults (subset of neuronal faults)
                    if round.any_parametric(layer_name):
                        for fault in round.search_parametric(layer_name):
                            # Create parametric faults' dummy layers
                            fault.model.param_perturb(self.slayer, self.device)

                # Synaptic faults
                if round.any_synaptic(layer_name):
                    pre_hook, _ = self._synapse_hook_wrapper(round.search_synaptic(layer_name))
                    self.handles[layer_name][ind_syn][0].append(layer.register_forward_pre_hook(pre_hook))

        return faulties

    def run(self, test_loader: DataLoader, spike_loss: snn.loss = None, es_tol: int = 0,
            opt: CampaignOptimization = CampaignOptimization.FO,
            progress_mode: str = '') -> Tensor | None:
        self._pre_run(opt)

        # Initialize and refresh progress
        self.progress = CampaignProgress(len(test_loader), len(self.rounds))
        self._progress_thread = Thread(target=refresh_progress_job, args=(self.progress, 0.1, progress_mode,), daemon=True)
        self._progress_thread.start()

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

    def _pre_run(self, opt: CampaignOptimization) -> None:
        self.reset()

        late_start_en = opt == CampaignOptimization.LS or opt == CampaignOptimization.FO
        early_stop_en = opt == CampaignOptimization.ES or opt == CampaignOptimization.FO

        for r, round in enumerate(self.rounds):
            # Create optimized fault rounds from rounds
            oround = round.optimized(self.layers_info, late_start_en, early_stop_en)
            self.orounds.append(oround)

            # Group fault rounds per earliest faulty layer
            self.rgroups.setdefault(oround.late_start_name, list())
            self.rgroups[oround.late_start_name].append(r)

            # Register fault (pre-)hooks for all fault rounds
            self._perturb_net(oround)

            # Create statistics for fault rounds
            self.performance.append(spikeStats())

        # Sort fault round groups in ascending order of group's earliest layer
        self.rgroups = dict(sorted(self.rgroups.items(), key=lambda item: -1 if item[0] is None else self.layers_info.index(item[0])))

    def _perturb_net(self, round: ff.FaultRound) -> None:
        ind_neu = ff.FaultTarget.Z.get_index()  # 0
        ind_syn = ff.FaultTarget.W.get_index()  # 1

        for layer_name in self.layers_info.get_injectables():
            self.handles.setdefault(layer_name, [[[], []], [[], []]])
            layer = getattr(self.faulty, layer_name)

            # Neuronal faults
            if round.any_neuronal(layer_name):
                # Parametric faults (subset of neuronal faults)
                if round.any_parametric(layer_name):
                    for fault in round.search_parametric(layer_name):
                        # Create parametric faults' dummy layers
                        fault.model.param_perturb(self.slayer, self.device)

                        hook = self._neuron_param_hook_wrapper(round.search_parametric(layer_name))
                        self.handles[layer_name][ind_neu][1] = layer.register_forward_hook(hook)

                # TODO: Check 'tail' layer pre-hook and REMOVE hook for output layer
                # Neuronal faults for last layer are evaluated on a 'tail' layer that does nothing
                if not self.layers_info.is_output(layer_name):
                    following_layer = getattr(self.faulty, self.layers_info.get_following(layer_name))

                    # Register neuron fault pre-hooks
                    pre_hook = self._neuron_hook_wrapper(round.search_neuronal(layer_name))
                    self.handles[layer_name][ind_neu][0].append(following_layer.register_forward_pre_hook(pre_hook))

            # Synaptic faults
            if round.any_synaptic(layer_name):
                # Register synapse (perturb) pre-hook and (restore) hook
                pre_hook, hook = self._synapse_hook_wrapper(round.search_synaptic(layer_name))
                self.handles[layer_name][ind_syn][0].append(layer.register_forward_pre_hook(pre_hook))
                self.handles[layer_name][ind_syn][1].append(layer.register_forward_hook(hook))

    def _post_run(self, update_stats: bool = True):
        self.duration = self.progress.get_duration_sec()

        self._progress_thread.join()
        del self._progress_thread

        # Update fault rounds statistics
        if update_stats:
            for stats in self.performance:
                stats.update()

        # Replace erroneous 'None' elements in statistics (slayer issue)
        for stats in self.performance:
            stats.training.accuracyLog = [x or 0. for x in stats.training.accuracyLog]
            stats.testing.accuracyLog = [x or 0. for x in stats.testing.accuracyLog]

    def _evaluate_train(self, faulty: nn.Module, epochs: int, train_loader: DataLoader, optimizer: Optimizer, spike_loss: snn.loss) -> None:
        stat = self.performance[self.r_idx].training

        # Create a new optimizer for network to be trained using the provided optimizer type and parameters
        optimizer_class = type(optimizer)
        optimizer_params = {k: v for param_group in optimizer.param_groups for k, v in param_group.items() if k != 'params'}
        optimizer_ = optimizer_class(faulty.parameters(), **optimizer_params)

        for _ in range(epochs):
            self.progress.step_epoch()
            faulty.train()

            for _, (_, input, target, label) in enumerate(train_loader):
                self.progress.step_batch()

                target = target.to(self.device)
                output = faulty.forward(input.to(self.device))

                loss = spike_loss.numSpikes(output, target)
                optimizer_.zero_grad()
                loss.backward()
                optimizer_.step()

                self._advance_performance(output, target, label, spike_loss, training=True)

                self.progress.set_train(stat.loss(), stat.accuracy())
                self.progress.step()

            self.performance[self.r_idx].update()

        # Final set of faulty synapses (if any)
        # Hooks are not removed from the final network
        faulty.forward(input.to(self.device))

        faulty.eval()

    def _evaluate_single(self, test_loader: DataLoader, spike_loss: snn.loss = None) -> None:
        for _, (_, input, target, label) in enumerate(test_loader):
            self.progress.step_batch()
            output = self.faulty(input.to(self.device))

            self._advance_performance(output, target.to(self.device), label, spike_loss)
            self.progress.step()

    def _evaluate_O0(self, test_loader: DataLoader, spike_loss: snn.loss = None) -> None:
        for round_group in self.rgroups.values():  # For each fault round group
            for self.r_idx in round_group:  # For each fault round
                self.progress.step_round()
                self._evaluate_single(test_loader, spike_loss)

    def _evaluate_O1(self, test_loader: DataLoader, spike_loss: snn.loss = None) -> None:
        for _, (_, input, target, label) in enumerate(test_loader):  # For each batch
            self.progress.step_batch()

            for round_group in self.rgroups.values():  # For each fault round group
                for self.r_idx in round_group:  # For each fault round
                    self.progress.step_round()
                    output = self.faulty(input.to(self.device))

                    self._advance_performance(output, target.to(self.device), label, spike_loss)
                    self.progress.step()

    def _evaluate_optimized(self, test_loader: DataLoader, spike_loss: snn.loss = None, es_tol: int = 0) -> Tensor:
        N_critical = torch.zeros(len(self.rounds), dtype=torch.int)

        for _, (_, input, target, label) in enumerate(test_loader):  # For each batch
            self.progress.step_batch()

            # Store golden spikes
            golden_spikes = [input.to(self.device)]
            for layer_idx in range(len(self.layers_info)):
                golden_spikes.append(self.golden(golden_spikes[layer_idx], layer_idx, layer_idx))
            golden_prediction = snn.predict.getClass(golden_spikes[-1])

            for round_group in self.rgroups.values():  # For each fault round group
                for self.r_idx in round_group:  # For each fault round
                    self.progress.step_round()

                    oround = self.orounds[self.r_idx]
                    ls_idx = oround.late_start_idx
                    es_idx = oround.early_stop_idx

                    if not oround.early_stop_en:
                        output = self.faulty(golden_spikes[ls_idx], ls_idx)
                    else:
                        # Early stop optimization
                        early_stop_next_out = self.faulty(golden_spikes[ls_idx], ls_idx, es_idx + 1)
                        early_stop = torch.sum(early_stop_next_out.ne(golden_spikes[es_idx + 2]), dim=(1, 2, 3, 4)) <= es_tol

                        # Replace output only for the affected samples of the batch
                        output = torch.zeros(golden_spikes[-1].size()).to(self.device)
                        output[early_stop] = golden_spikes[-1][early_stop]
                        if torch.any(~early_stop):
                            output[~early_stop] = self.faulty(early_stop_next_out[~early_stop], es_idx + 2)

                    prediction = snn.predict.getClass(output)
                    N_critical[self.r_idx] += torch.sum((golden_prediction == label) & (prediction != label))

                    self._advance_performance(output, target.to(self.device), label, spike_loss)
                    self.progress.step()

        return N_critical

    def _advance_performance(self, output: Tensor, target: Tensor, label: Tensor, spike_loss: snn.loss = None, training: bool = False) -> None:
        perf = self.performance[self.r_idx]
        stat = perf.training if training else perf.testing

        stat.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
        stat.numSamples += len(label)
        if spike_loss:
            stat.lossSum += spike_loss.numSpikes(output, target).cpu().item()

    def export(self) -> 'CampaignData':
        return CampaignData(VERSION, self)

    def save(self, fname: str = None) -> None:
        self.export().save(fname)

    def save_net(self, net: nn.Module = None, fname: str = None) -> None:
        to_save = net or self.faulty
        if not to_save:
            return

        torch.save(to_save.state_dict(), sfi_io.make_net_filepath((fname or self.name) + '.pt', rename=True))

    @staticmethod
    def load(fpath: str, unpickler_type: type[pickle.Unpickler] = None) -> 'Campaign':
        return CampaignData.load(fpath, unpickler_type).build()

    @staticmethod
    def load_many(pathname: str, unpickler_type: type[pickle.Unpickler] = None) -> list['Campaign']:
        return CampaignData.load_many(pathname, unpickler_type).build()

    @staticmethod
    def _forward_opt_wrapper(layers_info: LayersInfo, slayer: spikeLayer) -> Callable[[Tensor, Optional[int], Optional[int]], Tensor]:
        def forward_opt(self: nn.Module, spikes_in: Tensor, start_layer_idx: int = None, end_layer_idx: int = None) -> Tensor:
            start_idx = 0 if start_layer_idx is None else start_layer_idx
            end_idx = (len(layers_info) - 1) if end_layer_idx is None else end_layer_idx

            if start_idx < 0:
                start_idx = len(layers_info) + start_idx
            if end_idx < 0:
                end_idx = len(layers_info) + end_idx

            subject_layers = [lay_name for lay_idx, lay_name in enumerate(layers_info.order)
                              if start_idx <= lay_idx <= end_idx]

            spikes = torch.clone(spikes_in)
            for layer_name in subject_layers:
                layer = getattr(self, layer_name)
                spikes = layer(spikes)

                # Skip dropout and tail layers from calling slayer functions
                if layers_info.types[layer_name] in (snn.slayer._dropoutLayer, nn.Identity):
                    continue

                spikes = slayer.spike(slayer.psp(spikes))

            return spikes

        return forward_opt

    def _neuron_hook_wrapper(self, faults: list[ff.Fault]) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def neuron_pre_hook(_, args: tuple[Tensor, ...]) -> None:
            prev_spikes_out = args[0]
            for fault in faults:  # or self.orounds[self.r_idx].search_neuronal(layer_name)
                all_ind = fault.unroll()
                param_args = (fault.model.unstore(), )

                prev_spikes_out[:, *all_ind, :] = fault.model.perturb(
                    prev_spikes_out[:, *all_ind, :],
                    *(param_args if param_args != (None,) else fault.model.args))

        return neuron_pre_hook

    def _neuron_param_hook_wrapper(self, faults: list[ff.Fault]) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def neuron_param_hook(_, __, spikes_out: Tensor) -> None:
            for fault in faults:
                all_ind = fault.unroll()

                flayer = fault.model.flayer
                fspike_out = flayer.spike(flayer.psp(spikes_out))
                fault.model.store(fspike_out[:, *all_ind, :])

        return neuron_param_hook

    def _synapse_hook_wrapper(self, faults: list[ff.Fault]) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def _synapse_hook_core(to_perturb: bool, layer: nn.Module) -> None:
            for fault in faults:  # or self.orounds[self.r_idx].search_synaptic(layer_name):
                all_ind = fault.unroll()
                with torch.no_grad():
                    if to_perturb:
                        layer.weight[*all_ind] = fault.model.perturb_store(layer.weight[*all_ind])
                    else:
                        layer.weight[*all_ind] = fault.model.restore()

        # Perturb weights for layer's synapse faults
        def synapse_pre_hook(layer: nn.Module, _) -> None:
            _synapse_hook_core(to_perturb=True, layer=layer)

        # Restore weights after layer's synapse faults evaluation
        def synapse_hook(layer: nn.Module, _, __) -> None:
            _synapse_hook_core(to_perturb=False, layer=layer)

        return synapse_pre_hook, synapse_hook


class CampaignData:
    def __init__(self, version: str, campaign: Campaign) -> None:
        self.version = version

        has_pbar = hasattr(campaign, 'progress')
        if has_pbar:
            pbar = campaign.progress.pbar
            campaign.progress.pbar = None

        cmpn = deepcopy(campaign)

        if has_pbar:
            campaign.progress.pbar = pbar

        self.name = cmpn.name

        self.golden = cmpn.golden
        self.slayer = cmpn.slayer
        self.device = cmpn.device

        self.layers_info = cmpn.layers_info

        # Restore default forward function to golden network
        self.golden.forward = MethodType(type(self.golden).forward, self.golden)

        self.duration = cmpn.duration
        self.rounds = cmpn.rounds
        self.orounds = cmpn.orounds
        self.rgroups = cmpn.rgroups
        self.performance = cmpn.performance

    def build(self) -> Campaign:
        campaign = Campaign(self.golden, self.layers_info.shape_in, self.slayer, self.name)
        campaign.layers_info = self.layers_info
        campaign.duration = self.duration
        campaign.rounds = self.rounds
        campaign.orounds = self.orounds
        campaign.rgroups = self.rgroups
        campaign.performance = self.performance

        if self.version != VERSION:
            warn('The loaded campaign object was created with a different version of the SpikeFI framework.', DeprecationWarning)

        return campaign

    def save(self, fname: str = None) -> None:
        with open(sfi_io.make_res_filepath((fname or self.name) + '.pkl', rename=True), 'wb') as pkl:
            pickle.dump(self, pkl)

    @staticmethod
    def load(fpath: str, unpickler_type: type[pickle.Unpickler] = None) -> 'CampaignData':
        with open(fpath, 'rb') as pkl:
            if unpickler_type is None:
                return pickle.load(pkl)
            return unpickler_type(pkl).load()

    @staticmethod
    def load_many(pathname: str, unpickler_type: type[pickle.Unpickler] = None) -> list['CampaignData']:
        tore = []
        for f in glob(pathname):
            with open(f, 'rb') as pkl:
                o = pickle.load(pkl) if unpickler_type is None else unpickler_type(pkl).load()
                tore.append(o)

        return tore
