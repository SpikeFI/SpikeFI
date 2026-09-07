"""Concrete execution primitives every benchmark experiment calls: a plain
SLAYER inference pass, a SpikeFI campaign run and a naive-baseline campaign
run, all reduced to a single TrialOutcome (plus, for a campaign, one
RoundOutcome per fault round).
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import gc
from hashlib import sha256

import slayerSNN as snn
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from spikefi.core import Campaign, CampaignOptimization
from spikefi.fault import Fault

from .catalog import FTYPES
from .context import CaseStudyContext
from .faults import FaultPlan, FaultSpec
from .measure import MemoryProbe, MemStats, Stopwatch
from .naive import NaiveCampaign, NaiveRoundResult


@dataclass
class RoundOutcome:
    round: int
    layer: str
    ftype: str
    n_faults: int
    accuracy: float
    n_critical: int | None = None
    out_hash: str | None = None


def output_digest(outputs: Tensor) -> str:
    """Stable digest of a concatenated network output tensor, used to
    compare two independent implementations by construction: both sides
    hash the same way, so equal outputs always hash equal and a
    difference in a single element of a single sample is caught.

    The output tensor rather than the predicted classes, because the
    argmax is far too lossy to certify anything: on a sparse spiking
    network most neurons never fire for a given input, so a fault lands
    on silence and thousands of injected faults can leave every
    prediction, and therefore every accuracy, exactly at its golden
    value. The output spike train still moves, and that is what makes a
    comparison between two implementations meaningful.
    """

    raw = outputs.detach().to(torch.float32).cpu().numpy().tobytes()
    return sha256(raw).hexdigest()[:16]


@dataclass
class TrialOutcome:
    t_setup: float
    t_exec: float
    t_wall: float
    accuracy: float | None = None
    n_critical: int | None = None
    mem: MemStats | None = None
    oom: bool = False
    rounds: tuple[RoundOutcome, ...] = ()


def run_slayer_inference(
        context: CaseStudyContext,
        loader: DataLoader,
        probe_memory: bool = False
) -> TrialOutcome:
    """Plain SLAYER forward pass over loader, with no Campaign, hooks or
    other SpikeFI code on the path: the reference timing and accuracy a
    campaign's own overhead is measured against.
    """

    stats = snn.utils.stats()
    probe = MemoryProbe(context.device) if probe_memory else nullcontext()

    with probe, Stopwatch() as sw, torch.no_grad():
        for input, label in loader:
            label = label.to(context.device)
            output = context.net(input.to(context.device))

            predict = output.sum(dim=(2, 3, 4)).argmax(dim=1)
            stats.testing.correctSamples += (predict == label).sum().item()
            stats.testing.numSamples += label.size(0)

    stats.update()

    return TrialOutcome(
        t_setup=0.0,
        t_exec=sw.elapsed,
        t_wall=sw.elapsed,
        accuracy=stats.testing.accuracyLog[-1],
        n_critical=None,
        mem=probe.stats if probe_memory else None,
        oom=False,
        rounds=()
    )


def run_spikefi(
        context: CaseStudyContext,
        loader: DataLoader,
        plan: FaultPlan | None = None,
        opt: CampaignOptimization = CampaignOptimization.FO,
        es_tol: int = 0,
        compute_critical: bool = False,
        probe_memory: bool = False,
        collect_outputs: bool = False,
        name: str = 'benchmark'
) -> TrialOutcome:
    """Builds a Campaign over context.net, injects plan (or runs the single
    empty round Campaign.__init__ already creates, when plan is None), runs
    it, and reduces the result to one TrialOutcome plus one RoundOutcome per
    fault round.

    Campaign.run forces opt to O0 and takes the _evaluate_single path
    whenever the campaign holds one round or fewer, so a single-round trial
    (plan is None, or a plan with n_rounds == 1) executes identically no
    matter which optimization level was requested here.

    With collect_outputs, every round's per-batch output tensors are
    captured as the campaign scores them (see _install_output_capture) and
    each RoundOutcome.out_hash is filled from their digest. Those tensors
    are held in device memory until the run ends, so it is meant for the
    small validation runs that compare implementations, not for a full
    measurement sweep.
    """

    t_setup_sw = Stopwatch()
    probe = MemoryProbe(context.device) if probe_memory else nullcontext()

    cmpn: Campaign | None = None
    try:
        with probe:
            with t_setup_sw:
                cmpn = Campaign(
                    context.net, context.shape_in, context.net.slayer, name=name, device=context.device
                )

                if plan is not None:
                    model_factories = {
                        key: partial(ftype.factory, context.net) for key, ftype in FTYPES.items()
                    }
                    rounds: list[list[Fault]] = plan.to_rounds(model_factories)
                    for r, faults in enumerate(rounds):
                        if r == 0:
                            cmpn.inject(faults, 0)
                        else:
                            cmpn.then_inject(faults)
                        _assert_sites_injected(cmpn, r, len(faults))

            sink: dict[int, list[Tensor]] | None = None
            restore_capture: Callable[[], None] | None = None
            if collect_outputs:
                sink, restore_capture = _install_output_capture(cmpn)
            try:
                n_critical_tensor = cmpn.run(
                    loader, es_tol=es_tol, opt=opt, compute_critical=compute_critical, progress_mode='silent'
                )
            finally:
                if restore_capture is not None:
                    restore_capture()

        out_hashes: dict[int, str] | None = None
        if sink is not None:
            out_hashes = {
                r: output_digest(torch.cat(outs)) for r, outs in sink.items()
            }

        rounds_out = _build_round_outcomes(cmpn, plan, n_critical_tensor, out_hashes)

        # A mean or a sum across rounds would be a derived quantity that the
        # per-round rows already carry the truth for, so the trial-level
        # accuracy/criticality are only meaningful for a single-round trial.
        single_round = len(rounds_out) == 1
        trial_accuracy = rounds_out[0].accuracy if single_round else None
        trial_n_critical = rounds_out[0].n_critical if single_round else None

        return TrialOutcome(
            t_setup=t_setup_sw.elapsed,
            t_exec=cmpn.duration,
            t_wall=cmpn.duration_wall,
            accuracy=trial_accuracy,
            n_critical=trial_n_critical,
            mem=probe.stats if probe_memory else None,
            oom=False,
            rounds=rounds_out
        )
    except torch.cuda.OutOfMemoryError:
        return TrialOutcome(t_setup=t_setup_sw.elapsed, t_exec=0.0, t_wall=0.0, oom=True)
    finally:
        # Hooks hold references to the faulty net through closures, so a
        # bare del is not enough to free it; gc.collect() breaks those
        # cycles before the CUDA cache is released back to the allocator.
        del cmpn
        gc.collect()
        torch.cuda.empty_cache()


def _assert_sites_injected(cmpn: Campaign, round_idx: int, n_expected: int) -> None:
    """Confirms a round holds exactly the fault sites it was given.

    Campaign.inject validates what it is handed and discards anything it
    cannot place, reporting the shortfall as a warning rather than an
    error, so a round can quietly run with fewer faults than the trial
    goes on to record. Sites are counted rather than faults because
    Fault.buildup merges faults sharing a model into one multi-site fault.
    """

    n_injected = sum(len(fault.sites) for fault in cmpn.rounds[round_idx].get_faults())
    if n_injected != n_expected:
        raise RuntimeError(
            f"Round {round_idx} of campaign '{cmpn.name}' holds {n_injected} fault "
            f"site(s) after injection, expected {n_expected}; the campaign would "
            "measure a different fault load than the one recorded."
        )


def _install_output_capture(cmpn: Campaign) -> tuple[dict[int, list[Tensor]], Callable[[], None]]:
    """Shadows Campaign._advance_performance on this one campaign instance
    so every batch's output tensor is captured, per round, as the campaign
    itself scores it.

    _advance_performance is the single point every optimization level
    funnels its output through before any per-round bookkeeping can
    scatter it: run() calls _pre_run() -> reset(), which rebuilds the
    faulty net fresh, discarding any hook registered ahead of time, and at
    O3/O4 the final output is assembled inside _evaluate_optimized with
    golden rows spliced in for early-stopped samples and never returned by
    a single forward call.

    Captured tensors stay on the device and are hashed only once the run
    is over: moving each to host memory as it arrives would synchronize on
    every batch and distort the very durations the run is measuring.

    This depends on _advance_performance's current name and keyword
    signature (output, label, spike_loss, training, predict) as a private
    implementation detail of Campaign, not a public contract: a rename or
    reshape there breaks this loudly (AttributeError/TypeError at call
    time) rather than silently capturing nothing.
    """

    sink: dict[int, list[Tensor]] = {r: [] for r in range(len(cmpn.rounds))}
    original = cmpn._advance_performance

    def capture(
            output: Tensor,
            label: Tensor,
            spike_loss: snn.loss | None = None,
            training: bool = False,
            predict: Tensor | None = None
    ) -> tuple[Tensor, int] | None:
        sink[cmpn.r_idx_ref.r].append(output.detach().clone())
        return original(output, label, spike_loss, training, predict=predict)

    cmpn._advance_performance = capture

    def restore() -> None:
        del cmpn._advance_performance

    return sink, restore


def _build_round_outcomes(
        cmpn: Campaign,
        plan: FaultPlan | None,
        n_critical_tensor: torch.Tensor | None,
        out_hashes: dict[int, str] | None = None
) -> tuple[RoundOutcome, ...]:
    """Reduces a finished campaign's per-round performance and criticality
    into one RoundOutcome per round, pairing each round back to the layer,
    fault type and fault count that produced it via the fault plan's specs.
    """

    n_rounds = len(cmpn.rounds)

    if plan is None:
        accuracy = cmpn.performance[0].testing.accuracyLog[-1]
        n_critical = int(n_critical_tensor[0].item()) if n_critical_tensor is not None else None
        out_hash = out_hashes.get(0) if out_hashes is not None else None
        return (
            RoundOutcome(
                round=0, layer='', ftype='', n_faults=0,
                accuracy=accuracy, n_critical=n_critical, out_hash=out_hash
            ),
        )

    specs_by_round: dict[int, list[FaultSpec]] = {}
    for spec in plan.specs:
        specs_by_round.setdefault(spec.round_idx, []).append(spec)

    outcomes = []
    for r in range(n_rounds):
        specs = specs_by_round.get(r, [])
        layers = sorted({s.layer for s in specs})
        ftype = specs[0].ftype if specs else ''

        accuracy = cmpn.performance[r].testing.accuracyLog[-1]
        n_critical = int(n_critical_tensor[r].item()) if n_critical_tensor is not None else None
        out_hash = out_hashes.get(r) if out_hashes is not None else None

        outcomes.append(RoundOutcome(
            round=r,
            layer='+'.join(layers),
            ftype=ftype,
            n_faults=len(specs),
            accuracy=accuracy,
            n_critical=n_critical,
            out_hash=out_hash
        ))

    return tuple(outcomes)


def run_naive(
        context: CaseStudyContext,
        loader: DataLoader,
        plan: FaultPlan,
        compute_critical: bool = False,
        probe_memory: bool = False,
        collect_outputs: bool = False
) -> TrialOutcome:
    """Runs plan through NaiveCampaign, the suite's independently derived
    fault-injection baseline, and reduces the result to the same
    TrialOutcome/RoundOutcome shape run_spikefi produces, so a naive and a
    SpikeFI trial over the same plan can be compared row for row.
    """

    t_setup_sw = Stopwatch()
    probe = MemoryProbe(context.device) if probe_memory else nullcontext()

    campaign: NaiveCampaign | None = None
    try:
        with probe:
            with t_setup_sw:
                campaign = NaiveCampaign(
                    context.net, context.net.slayer, context.shape_in, context.device
                )

            model_factories = {
                key: partial(ftype.factory, context.net) for key, ftype in FTYPES.items()
            }
            results = campaign.run(
                loader, plan, model_factories,
                compute_critical=compute_critical, collect_outputs=collect_outputs
            )

        rounds_out = _build_naive_round_outcomes(plan, results)

        # A mean or a sum across rounds would be a derived quantity that the
        # per-round rows already carry the truth for, so the trial-level
        # accuracy/criticality are only meaningful for a single-round trial.
        single_round = len(rounds_out) == 1
        trial_accuracy = rounds_out[0].accuracy if single_round else None
        trial_n_critical = rounds_out[0].n_critical if single_round else None

        return TrialOutcome(
            t_setup=t_setup_sw.elapsed,
            t_exec=campaign.duration,
            t_wall=campaign.duration_wall,
            accuracy=trial_accuracy,
            n_critical=trial_n_critical,
            mem=probe.stats if probe_memory else None,
            oom=False,
            rounds=rounds_out
        )
    except torch.cuda.OutOfMemoryError:
        return TrialOutcome(t_setup=t_setup_sw.elapsed, t_exec=0.0, t_wall=0.0, oom=True)
    finally:
        # Hooks hold references to the net through closures, so a bare del
        # is not enough to free it; gc.collect() breaks those cycles
        # before the CUDA cache is released back to the allocator.
        del campaign
        gc.collect()
        torch.cuda.empty_cache()


def _build_naive_round_outcomes(
        plan: FaultPlan,
        results: list[NaiveRoundResult]
) -> tuple[RoundOutcome, ...]:
    """Reduces a finished NaiveCampaign's per-round results into one
    RoundOutcome per round, pairing each round back to the layer, fault
    type and fault count that produced it via the fault plan's specs.
    """

    specs_by_round: dict[int, list[FaultSpec]] = {}
    for spec in plan.specs:
        specs_by_round.setdefault(spec.round_idx, []).append(spec)

    outcomes = []
    for r, result in enumerate(results):
        specs = specs_by_round.get(r, [])
        layers = sorted({s.layer for s in specs})
        ftype = specs[0].ftype if specs else ''
        out_hash = output_digest(result.outputs) if result.outputs is not None else None

        outcomes.append(RoundOutcome(
            round=r,
            layer='+'.join(layers),
            ftype=ftype,
            n_faults=len(specs),
            accuracy=result.accuracy,
            n_critical=result.n_critical,
            out_hash=out_hash
        ))

    return tuple(outcomes)
