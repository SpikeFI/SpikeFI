"""Shared test helpers: precondition assertions for the non-vacuity oracle,
hand-built fault mutants for the differential oracle, layer-invocation
probes for the optimization work-counting tests, and round-execution
helpers for Tiers 1-3.
"""


from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import spikefi as sfi


# --- Round execution (post-training FI: pull a round's raw output tensor
# out directly, since Campaign.run() only exposes aggregate stats) ---

def run_round(
        campaign: sfi.Campaign,
        round_idx: int,
        x: Tensor,
        opt: sfi.CampaignOptimization = sfi.CampaignOptimization.O0
) -> Tensor:
    """Runs `x` through campaign.faulty for one already-injected round, via
    the same private _pre_run()/faulty() path Campaign.run() itself uses,
    so the raw output tensor can be inspected directly instead of only the
    aggregate accuracy/loss stats run() exposes."""
    campaign._pre_run(opt)
    campaign.r_idx_ref.r = round_idx
    return campaign.faulty(x)


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


def capture_invocation_widths(
        campaign: sfi.Campaign,
        layer_names: list[str],
        test_loader: DataLoader,
        **run_kwargs: Any
) -> dict[str, list[int]]:
    """Like count_invocations, but for a full campaign.run() call, and
    recording the batch width of each invocation rather than only how many
    there were: an optimization that drops samples from a forward pass is
    told apart from one that keeps them by how wide the call is, not by how
    often it happens. campaign.faulty only exists once run()'s own _pre_run()
    has built it, so the hooks are attached by wrapping _pre_run itself,
    right after it returns, rather than by hooking a net that does not yet
    exist."""
    widths: dict[str, list[int]] = {name: [] for name in layer_names}
    handles = []

    original_pre_run = campaign._pre_run

    def _patched_pre_run(opt: sfi.CampaignOptimization) -> None:
        original_pre_run(opt)
        for name in layer_names:
            def _hook(_: nn.Module, inputs: tuple, __: Tensor, name: str = name) -> None:
                widths[name].append(inputs[0].shape[0])

            handles.append(getattr(campaign.faulty, name).register_forward_hook(_hook))

    campaign._pre_run = _patched_pre_run
    try:
        campaign.run(test_loader, **run_kwargs)
    finally:
        campaign._pre_run = original_pre_run
        for handle in handles:
            handle.remove()

    return widths


def count_invocations_during_run(
        campaign: sfi.Campaign,
        layer_names: list[str],
        test_loader: DataLoader,
        **run_kwargs: Any
) -> dict[str, int]:
    """How many times each named layer was invoked over a whole
    campaign.run() call, regardless of how wide each invocation was."""
    return {
        name: len(invocations)
        for name, invocations in capture_invocation_widths(
            campaign, layer_names, test_loader, **run_kwargs
        ).items()
    }


def capture_run_outputs(
        campaign: sfi.Campaign,
        test_loader: DataLoader,
        **run_kwargs: Any
) -> list[Tensor]:
    """Runs campaign.run() while intercepting _advance_performance to
    collect each round's raw output tensor, batch by batch, since run()
    itself only exposes aggregate accuracy/loss stats. Returns one
    concatenated tensor per round, in round order."""
    captured: dict[int, list[Tensor]] = {}
    original_advance = campaign._advance_performance

    def _spy(
            output: Tensor,
            label: Tensor,
            spike_loss: Any = None,
            training: bool = False,
            predict: Tensor | None = None
    ) -> Any:
        captured.setdefault(campaign.r_idx_ref.r, []).append(output.clone())
        return original_advance(output, label, spike_loss, training, predict)

    campaign._advance_performance = _spy
    try:
        campaign.run(test_loader, **run_kwargs)
    finally:
        campaign._advance_performance = original_advance

    return [torch.cat(captured[r], dim=0) for r in sorted(captured)]
