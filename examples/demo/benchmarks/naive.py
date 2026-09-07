"""A hand-rolled, independently derived reference implementation of fault
injection, standing outside SpikeFI's own Campaign/LayersInfo/hook-dispatch
machinery so a bug shared across every SpikeFI optimization level is still
visible when compared against this baseline.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from slayerSNN.slayer import spikeLayer
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from spikefi.fault import FaultModel

from .faults import FaultPlan, FaultSpec
from .measure import Stopwatch


@dataclass
class NaiveRoundResult:
    """Outcome of one fault round run by NaiveCampaign."""

    accuracy: float
    n_critical: int | None = None
    outputs: Tensor | None = None  # concatenated per-sample network output, on the device


@dataclass
class _PreparedFault:
    """One fault of a round with everything that does not depend on the
    data already resolved: its model, and for a parametric neuron fault
    the dummy neuron layer that model implies.
    """

    spec: FaultSpec
    model: FaultModel
    flayer: spikeLayer | None = None


class NaiveCampaign:
    """Runs a fault plan against a network's own forward() method, applying
    each fault type's numeric perturbation by hand rather than through any
    SpikeFI hook-dispatch mechanism.

    The network's layer execution order and each layer's successor are
    discovered from a live forward pass rather than taken from LayersInfo,
    so the two implementations never share a source of truth for "what
    runs after what".
    """

    def __init__(
            self,
            net: torch.nn.Module,
            slayer: spikeLayer,
            shape_in: tuple[int, int, int],
            device: torch.device
    ) -> None:
        self.device: torch.device = device
        self.net: torch.nn.Module = deepcopy(net).to(device)
        self.net.eval()
        self.slayer: spikeLayer = deepcopy(slayer).to(device)

        self.duration: float = 0.0
        self.duration_wall: float = 0.0

        self.order: list[str] = self._discover_order(shape_in)
        # Maps a layer name to the name of the module that runs right after
        # it, or None when it is the last one a forward pass invokes.
        self.following: dict[str, str | None] = {
            name: (self.order[i + 1] if i + 1 < len(self.order) else None)
            for i, name in enumerate(self.order)
        }

    def _discover_order(self, shape_in: tuple[int, int, int]) -> list[str]:
        """Runs one dummy forward pass with a hook on every direct child
        module and records the order in which those hooks fire. This is
        the baseline's own, independent notion of "the module following
        layer X" — never LayersInfo's.
        """

        order: list[str] = []

        def make_hook(name: str) -> Callable[..., None]:
            def hook(_module: torch.nn.Module, _inputs: tuple, _output: Tensor) -> None:
                order.append(name)

            return hook

        handles = [
            child.register_forward_hook(make_hook(name))
            for name, child in self.net.named_children()
        ]

        dummy_input = torch.rand((1, *shape_in, 1), device=self.device)
        with torch.no_grad():
            self.net(dummy_input)

        for handle in handles:
            handle.remove()

        seen: set[str] = set()
        for name in order:
            if name in seen:
                raise NotImplementedError(
                    f"Module '{name}' fired its forward hook more than once during "
                    "discovery; a module invoked several times per forward would need "
                    "per-invocation disambiguation that this baseline does not implement."
                )
            seen.add(name)

        return order

    def run(
            self,
            loader: DataLoader,
            plan: FaultPlan,
            model_factories: dict[str, Callable[[str], FaultModel]],
            compute_critical: bool = False,
            collect_outputs: bool = False
    ) -> list[NaiveRoundResult]:
        """Runs every round of plan in turn, each round's faults applied to
        self.net and then removed again before the next round starts, so a
        round only ever sees the faults it was assigned.

        Rounds are the outer loop and batches the inner one, deliberately:
        a naive, hand-rolled fault injector has no reason to notice that
        batching outside the fault loop saves repeated golden work, so
        inverting the nesting would silently adopt SpikeFI's own batching
        insight and defeat the point of an independent baseline.
        """

        golden_predictions: list[Tensor] = []
        if compute_critical:
            with torch.inference_mode():
                for input, _label in loader:
                    output = self.net(input.to(self.device))
                    golden_predictions.append(output.sum(dim=(2, 3, 4)).argmax(dim=1))

        prepared = self._prepare(plan, model_factories)

        results: list[NaiveRoundResult] = []
        with Stopwatch() as sw_wall:
            with Stopwatch() as sw_exec:
                for round_idx in range(plan.n_rounds):
                    results.append(self._run_round(
                        loader,
                        prepared.get(round_idx, []),
                        golden_predictions if compute_critical else None,
                        collect_outputs
                    ))
            self.duration = sw_exec.elapsed
        self.duration_wall = sw_wall.elapsed

        return results

    def _prepare(
            self,
            plan: FaultPlan,
            model_factories: dict[str, Callable[[str], FaultModel]]
    ) -> dict[int, list[_PreparedFault]]:
        """Resolves every fault's model, and every parametric fault's dummy
        neuron layer, before the timed region opens.

        Both describe what a fault *is* rather than the act of injecting
        it, and both are resolved once per campaign on the SpikeFI side,
        outside the region its own duration covers. Building them inside
        the round loop would charge this baseline for work its counterpart
        is not charged for, and the per-round cost is real: the bitflipped
        synapse model derives its quantization arguments from the whole
        weight tensor.
        """

        prepared: dict[int, list[_PreparedFault]] = {}
        for spec in plan.specs:
            model = model_factories[spec.ftype](spec.layer)
            flayer = self._build_flayer(model) if model.is_parametric() else None
            prepared.setdefault(spec.round_idx, []).append(_PreparedFault(spec, model, flayer))

        return prepared

    def _build_flayer(self, model: FaultModel) -> spikeLayer:
        """The dummy neuron layer a parametric fault implies, derived from
        the model's own parameter name, method and arguments rather than
        from anything SpikeFI builds.
        """

        neuron = dict(self.slayer.neuron)
        neuron[model.param_name] = model.param_method(neuron[model.param_name], *model.param_args)

        return spikeLayer(neuron, self.slayer.simulation).to(self.device)

    def _run_round(
            self,
            loader: DataLoader,
            faults: list[_PreparedFault],
            golden_predictions: list[Tensor] | None,
            collect_outputs: bool
    ) -> NaiveRoundResult:
        handles, restore_ops, post_ops = self._apply_round(faults)

        try:
            correct = 0
            total = 0
            n_critical = 0 if golden_predictions is not None else None
            outputs: list[Tensor] = []

            with torch.inference_mode():
                for batch_idx, (input, label) in enumerate(loader):
                    label = label.to(self.device)
                    output = self.net(input.to(self.device))
                    for post_op in post_ops:
                        post_op(output)

                    pred = output.sum(dim=(2, 3, 4)).argmax(dim=1)
                    correct += (pred == label).sum().item()
                    total += label.size(0)

                    if golden_predictions is not None:
                        golden_pred = golden_predictions[batch_idx]
                        n_critical += torch.sum((golden_pred == label) & (pred != label)).item()

                    if collect_outputs:
                        # Kept on the device and concatenated only once the
                        # round is over: a host transfer per batch would
                        # synchronize inside the region being timed.
                        outputs.append(output.detach().clone())
        finally:
            for handle in handles:
                handle.remove()
            for restore_op in restore_ops:
                restore_op()

        return NaiveRoundResult(
            accuracy=correct / total,
            n_critical=n_critical,
            outputs=torch.cat(outputs) if collect_outputs else None
        )

    def _apply_round(
            self,
            faults: list[_PreparedFault]
    ) -> tuple[list[RemovableHandle], list[Callable[[], None]], list[Callable[[Tensor], None]]]:
        """Injects every fault of one round: weight faults are perturbed in
        place right away and their restoration deferred to restore_ops;
        neuron faults (hard or parametric) are wired as hooks, deferred to
        post_ops only when their faulty layer is the last one executed.
        """

        handles: list[RemovableHandle] = []
        restore_ops: list[Callable[[], None]] = []
        post_ops: list[Callable[[Tensor], None]] = []

        for prepared in faults:
            spec = prepared.spec
            model = prepared.model
            layer = getattr(self.net, spec.layer)

            if model.is_synaptic():
                handle_restore = self._apply_synapse_fault(layer, spec.position, model)
                restore_ops.append(handle_restore)
            elif model.is_parametric():
                handles_added, post_op = self._apply_parametric_fault(layer, spec, prepared.flayer)
                handles.extend(handles_added)
                if post_op is not None:
                    post_ops.append(post_op)
            elif model.is_neuronal():
                handle_added, post_op = self._apply_hard_neuron_fault(spec, model)
                if handle_added is not None:
                    handles.append(handle_added)
                if post_op is not None:
                    post_ops.append(post_op)
            else:
                raise NotImplementedError(
                    f"Fault model {model} targets neither weight nor neuron output "
                    "nor a neuron parameter; this baseline has no injection path for it."
                )

        return handles, restore_ops, post_ops

    def _apply_synapse_fault(
            self,
            layer: torch.nn.Module,
            position: tuple[int, ...],
            model: FaultModel
    ) -> Callable[[], None]:
        with torch.no_grad():
            original = layer.weight[position].clone()
            layer.weight[position] = model.perturb(original)

        def restore() -> None:
            with torch.no_grad():
                layer.weight[position] = original

        return restore

    def _apply_hard_neuron_fault(
            self,
            spec: FaultSpec,
            model: FaultModel
    ) -> tuple[RemovableHandle | None, Callable[[Tensor], None] | None]:
        full_idx = (slice(None), *spec.position, slice(None))
        following_name = self.following[spec.layer]

        if following_name is None:
            def post_op(output: Tensor) -> None:
                output[full_idx] = model.perturb(output[full_idx])

            return None, post_op

        following_layer = getattr(self.net, following_name)

        def pre_hook(_module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
            inputs[0][full_idx] = model.perturb(inputs[0][full_idx])

        handle = following_layer.register_forward_pre_hook(pre_hook)
        return handle, None

    def _apply_parametric_fault(
            self,
            layer: torch.nn.Module,
            spec: FaultSpec,
            flayer: spikeLayer
    ) -> tuple[list[RemovableHandle], Callable[[Tensor], None] | None]:
        full_idx = (slice(None), *spec.position, slice(None))
        stash: list[Tensor] = []

        def forward_hook(_module: torch.nn.Module, _inputs: tuple, output: Tensor) -> None:
            site = output[full_idx]
            batch, time_bins = site.shape
            reshaped = site.reshape(batch, 1, 1, 1, time_bins)
            fspike_out = flayer.spike(flayer.psp(reshaped))
            stash[:] = [fspike_out.reshape(batch, time_bins)]

        handles = [layer.register_forward_hook(forward_hook)]
        following_name = self.following[spec.layer]

        if following_name is None:
            def post_op(output: Tensor) -> None:
                output[full_idx] = stash[0]

            return handles, post_op

        following_layer = getattr(self.net, following_name)

        def pre_hook(_module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
            inputs[0][full_idx] = stash[0]

        handles.append(following_layer.register_forward_pre_hook(pre_hook))
        return handles, None
