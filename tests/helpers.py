"""Shared test helpers: precondition assertions for the non-vacuity oracle,
hand-built fault mutants for the differential oracle, and layer-invocation
probes for the optimization work-counting tests.
"""


from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import nn, Tensor


# --- Precondition assertions (non-vacuity: a fault must have room to act) ---

def assert_active(spikes: Tensor, index: tuple) -> None:
    """DeadNeuron precondition: the site must actually fire at least once."""
    assert spikes[index].sum() > 0, (
        f'Site {index} never fires; DeadNeuron would be a no-op here.'
    )


def assert_not_saturated(spikes: Tensor, index: tuple, n_time_bins: int) -> None:
    """SaturatedNeuron precondition: the site must not already fire at every
    time bin, or the fault would be a no-op."""
    assert spikes[index].sum() < n_time_bins, (
        f'Site {index} is already saturated; SaturatedNeuron would be a no-op here.'
    )


def assert_differs(golden_value: float | Tensor, fault_value: float | Tensor) -> None:
    """set_value precondition: the golden value must differ from the
    fault's target value, or the fault would be indistinguishable from a
    no-op."""
    assert golden_value != fault_value, (
        f'Golden value {golden_value} already equals the fault value '
        f'{fault_value}; the fault would be a no-op here.'
    )


# --- Hand-built mutants (differential oracle: reproduce the effect without
# SpikeFI's own machinery, so the two paths can be compared bit-for-bit) ---

def hand_mutate_weight(
        layer: nn.Module,
        index: tuple,
        value: float | Tensor
) -> Tensor:
    """Directly overwrites a weight site, bypassing SpikeFI entirely.
    Returns the original value so the caller can restore it."""
    with torch.no_grad():
        original = layer.weight[index].clone()
        layer.weight[index] = value
    return original


@contextmanager
def hand_mutate_neuron_output(
        module: nn.Module,
        index: tuple,
        value: float | Tensor
) -> Generator[None]:
    """Registers a raw forward hook that overwrites a neuron output site,
    bypassing SpikeFI's hook classes entirely, for the same differential
    comparison as hand_mutate_weight but on the OUTPUT target. The
    overwrite is active only for the duration of the `with` block."""
    def _hook(_: nn.Module, __: tuple, output: Tensor) -> None:
        output[index] = value

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# --- Layer/work probes (Tier 3: an optimization must actually skip work,
# not merely produce the right answer) ---

@contextmanager
def count_invocations(
        net: nn.Module,
        layer_names: list[str]
) -> Generator[dict[str, int]]:
    """Counts how many times each named layer is actually invoked during
    whatever forward passes run inside the `with` block, so a claimed
    optimization (e.g. late start skipping early layers) can be checked
    directly rather than inferred from the output alone."""
    counts: dict[str, int] = {name: 0 for name in layer_names}
    handles = []

    for name in layer_names:
        def _hook(_: nn.Module, __: tuple, ___: Tensor, name: str = name) -> None:
            counts[name] += 1

        handles.append(getattr(net, name).register_forward_hook(_hook))

    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()
