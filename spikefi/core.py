import copy
import random
from typing import Any, Callable, List, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

import slayerSNN as snn
from slayerSNN.slayer import _convLayer, _denseLayer, spikeLayer

from .fault import Fault, FaultModel, FaultRound, FaultSite


class Campaign:
    def __init__(self, net: nn.Module, shape_in: Tuple[int, int, int], slayer: spikeLayer) -> None:
        self.golden_net = net
        self.faulty_net = copy.deepcopy(net)
        self.faulty_net.eval()
        self.device = next(net.parameters()).device
        self.slayer = slayer

        self.layers = {}
        self.names_lay = []
        self.names_inj = []
        self.shapes_lay = {}
        self.shapes_wei = {}
        self.injectables = {}
        self._assume_layer_info(shape_in)

        self.rounds = [FaultRound(self.names_inj)]

    def __repr__(self) -> str:
        # TODO: Implement
        pass

    def _assume_layer_info(self, shape_in: Tuple[int, int, int]) -> None:
        handles = []
        for name, child in self.faulty_net.named_children():
            h = child.register_forward_hook(self._layer_info_hook_wrapper(name))
            handles.append(h)

        dummy_input = torch.rand((1, *shape_in, 1)).to(self.device)
        self.faulty_net.forward(dummy_input)

        for h in handles:
            h.remove()

    def _layer_info_hook_wrapper(self, layer_name: str) -> Callable[[nn.Module, Tuple[Any, ...], Tensor], None]:
        def layer_info_hook(layer: nn.Module, _, output: Tensor) -> None:
            self.layers[layer_name] = layer
            self.names_lay.append(layer_name)

            self.shapes_lay[layer_name] = tuple(output.shape[1:4])
            self.shapes_wei[layer_name] = tuple(layer.weight.shape[0:4])

            if type(layer) in [_convLayer, _denseLayer]:
                self.injectables[layer_name] = layer
                self.names_inj.append(layer_name)

        return layer_info_hook

    def inject(self, faults: List[Fault], round_idx: int = -1) -> List[Fault]:
        assert -len(self.rounds) <= round_idx < len(self.rounds)

        inj_faults = self._assert_injected_faults(faults)
        self.define_random_faults(inj_faults)

        self.rounds[round_idx].insert(inj_faults)

        return inj_faults

    # TODO: Assert parameter names for parametric faults
    def _assert_injected_faults(self, faults: List[Fault]) -> List[Fault]:
        valid_faults = []
        for f in faults:
            fc = copy.deepcopy(f)
            is_syn = fc.model.is_synaptic()
            shapes = self.shapes_wei if is_syn else self.shapes_lay

            for s in fc.sites:
                v = s.is_random() or s.layer in self.injectables
                if v:
                    if is_syn:
                        v &= not s.dim0 or -shapes[s.layer][0] <= s.dim0 < shapes[s.layer][0]

                    chw = s.get_chw()
                    for i in range(3):
                        v &= not chw[i] or chw[i] <= chw[i] < shapes[s.layer][i+is_syn]

                if not v:
                    fc.sites.remove(s)

            if fc.sites:
                valid_faults.append(fc)

        return valid_faults

    def define_random_faults(self, faults: List[Fault]) -> List[Fault]:
        for f in faults:
            is_syn = f.model.is_synaptic()
            shapes = self.shapes_wei if is_syn else self.shapes_lay

            for s in f.sites:
                if s.is_random():
                    s.layer = random.choice(self.names_inj)
                    s.set_chw(None)
                    if is_syn:
                        s.dim0 = None

                if s.dim0 is None:
                    s.dim0 = random.randrange(shapes[s.layer][0]) if is_syn else slice(None)

                new_chw = []
                for dim, val in enumerate(s.get_chw()):
                    new_chw.append(val if val is not None else random.randrange(shapes[s.layer][dim+is_syn]))
                s.set_chw(tuple(new_chw))

        return faults

    def then_inject(self, faults: List[Fault]) -> List[Fault]:
        self.rounds.append(FaultRound(self.names_inj))

        return self.inject(faults, -1)

    def eject(self, faults: List[Fault] = None, round_idx: int = None) -> None:
        # Eject from a specific round
        if round_idx:
            # Eject indicated faults from the round
            if faults:
                self.rounds[round_idx].remove(faults)
            # Eject all faults from the round, i.e., remove the round itself
            if not faults or not self.rounds[round_idx]:
                self.rounds.pop(round_idx)
        # Eject from all rounds
        else:
            # Eject indicated faults from any round the might exist
            if faults:
                for r in self.rounds:
                    r.remove(faults)
                    if not r:
                        self.rounds.pop(r)
            # Eject all faults from all rounds, i.e., all the rounds themselves
            else:
                self.rounds.clear()

        if not self.rounds:
            self.rounds.append(FaultRound(self.names_inj))

    # TODO: Store round index in self to avoid passing the round as a parameter? (probably not compatible with optimizations)
    # FIXME: Cuda out of memory issues
    def run(self, test_loader: DataLoader) -> None:
        for r in self.rounds:
            handles = self._register_hooks(r)

            self._perturb_synap(r)
            self._perturb_param(r)

            self._evaluate(r, test_loader)

            self._restore_param(r)
            self._restore_synap(r)

            for h in handles:
                h.remove()

    def _register_hooks(self, round: FaultRound) -> List[RemovableHandle]:
        handles = []
        if not round.has_neuron_faults():
            return []

        # Neuron fault pre-hooks
        # Neuron faults for last layer are evaluated directly on network's output (in _evaluate method)
        for i, lay_name in enumerate(self.names_lay[:-1]):
            if round.has_neuron_faults(lay_name):
                # Register the pre-hook on the next layer of the faulty one
                # to inject at its input, i.e., the output of the faulty layer
                next_lay = self.layers[self.names_lay[i+1]]
                h = next_lay.register_forward_pre_hook(self._neuron_pre_hook_wrapper(round, lay_name))
                handles.append(h)

        # Parametric fault pre-hooks
        for lay_name, lay in self.injectables.items():
            if round.has_parametric_faults(lay_name):
                h = lay.register_forward_pre_hook(self._parametric_pre_hook_wrapper(round, lay_name))
                handles.append(h)

        return handles

    def _perturb_synap(self, round: FaultRound) -> None:
        for f in round.get_synapse_faults():
            for s in f.sites:
                # Perturb weights for synapse faults (no pre-hook needed)
                lay = self.injectables[s.layer]
                with torch.no_grad():
                    lay.weight[s.unroll()] = f.model.perturb(lay.weight[s.unroll()], s.key())

    def _restore_synap(self, round: FaultRound) -> None:
        for f in round.get_synapse_faults():
            for s in f.sites:
                # Restore weights after the end of the round (no hook needed)
                original = f.model.restore(s.key())
                if original:
                    self.injectables[s.layer].weight[s.unroll()] = original

    def _perturb_param(self, round: FaultRound) -> None:
        # Create a dummy layer for each parametric fault
        for f in round.get_parametric_faults():
            flayer = copy.deepcopy(self.slayer)
            flayer.neuron[f.model.param_name] = f.model.perturb_param(flayer.neuron[f.model.param_name])
            f.model.flayer = flayer

    def _restore_param(self, round: FaultRound) -> None:
        # Destroy dummy layers
        for f in round.get_parametric_faults():
            f.model.flayer = None

    def _evaluate(self, round: FaultRound, test_loader: DataLoader) -> None:
        is_out_faulty = round.has_neuron_faults(self.names_lay[-1])
        out_neuron_callable = self._neuron_pre_hook_wrapper(round, self.names_lay[-1])

        for b, (input, target, label) in enumerate(test_loader):
            input = input.to(self.device)
            target = target.to(self.device)

            output = self.faulty_net.forward(input)
            if is_out_faulty:
                out_neuron_callable(self.layers[self.names_lay[-1]], (output,))

            round.stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
            round.stats.testing.numSamples += len(label)
            round.stats.print(0, b)  # TODO: Make output more indicative

        round.stats.update()

    def run_complete(self, test_loader: DataLoader, fault_model: FaultModel, layers: List[str] = None) -> None:
        self.rounds.clear()

        is_syn = fault_model.is_synaptic()
        shapes = self.shapes_wei if is_syn else self.shapes_lay

        lay_names_inj = [lay_name for lay_name in layers if lay_name in self.names_inj] if layers else self.names_inj

        for lay_name in lay_names_inj:
            lay_shape = shapes[lay_name]
            for k in range(lay_shape[0] if is_syn else 1):
                for c in range(lay_shape[0+is_syn]):
                    for h in range(lay_shape[1+is_syn]):
                        for w in range(lay_shape[2+is_syn]):
                            self.then_inject(
                                [Fault(fault_model, [FaultSite(lay_name, k if is_syn else None, (c, h, w))])])

        self.run(test_loader)

    def _neuron_pre_hook_wrapper(self, round: FaultRound, prev_layer_name: str) -> Callable[[nn.Module, Tuple[Any, ...]], None]:
        def neuron_pre_hook(_, args: Tuple[Any, ...]) -> None:
            prev_spikes_out = args[0]
            for f in round.get_neuron_faults(prev_layer_name):
                for s in f.sites:
                    prev_spikes_out[s.unroll()] = f.model.perturb(prev_spikes_out[s.unroll()])

        return neuron_pre_hook

    def _parametric_pre_hook_wrapper(self, round: FaultRound, layer_name: str) -> Callable[[nn.Module, Tuple[Any, ...]], None]:
        def parametric_pre_hook(layer: nn.Module, args: Tuple[Any, ...]) -> None:
            spikes_in = args[0]

            # Temporarily clear layer's pre-hooks, so that they are not called recursively
            pre_hooks = layer._forward_pre_hooks.copy()
            layer._forward_pre_hooks.clear()

            for f in round.get_parametric_faults(layer_name):
                flayer = f.model.flayer
                fspike_out = flayer.spike(flayer.psp(layer(spikes_in)))
                f.model.args = (fspike_out[f.sites[0].unroll()],)

            layer._forward_pre_hooks.update(pre_hooks)

        return parametric_pre_hook