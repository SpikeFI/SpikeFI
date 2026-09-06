from dataclasses import dataclass
from math import sqrt
import time

import torch


class Stopwatch:
    """
    Context manager measuring the wall-clock duration of the enclosed
    block. CUDA work is synchronized immediately before both the start and
    the stop timestamp, so queued GPU kernels are actually finished before
    either reading is taken and 'elapsed' reflects real execution time
    rather than kernel-launch time.
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> 'Stopwatch':
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0


@dataclass
class MemStats:
    alloc_base: int
    alloc_peak: int
    alloc_end: int
    reserved_peak: int

    @property
    def delta_peak(self) -> int:
        return self.alloc_peak - self.alloc_base


class MemoryProbe:
    """
    Context manager reporting the CUDA memory a block of code actually
    consumed, as a MemStats available on 'stats' once the block exits.

    torch.cuda.reset_peak_memory_stats() resets the peak tracker to the
    CURRENT allocation, not to zero, so 'alloc_base' is captured right
    after the reset and subtracted back out via MemStats.delta_peak;
    without that, the reported peak would describe the whole process
    rather than the probed block. Meant to wrap setup and execution
    together, since SpikeFI's dominant allocations (deepcopies in
    Campaign.__init__/_pre_run) happen before run() is ever called.
    Reports zeroed stats when CUDA is unavailable, so CPU-only smoke
    tests do not crash.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device: torch.device | None = device
        self.stats: MemStats | None = None
        self._enabled: bool = False
        self._alloc_base: int = 0

    def __enter__(self) -> 'MemoryProbe':
        self._enabled = torch.cuda.is_available()
        if self._enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._alloc_base = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._enabled:
            torch.cuda.synchronize(self.device)
            self.stats = MemStats(
                alloc_base=self._alloc_base,
                alloc_peak=torch.cuda.max_memory_allocated(self.device),
                alloc_end=torch.cuda.memory_allocated(self.device),
                reserved_peak=torch.cuda.max_memory_reserved(self.device)
            )
        else:
            self.stats = MemStats(alloc_base=0, alloc_peak=0, alloc_end=0, reserved_peak=0)


@dataclass
class Resolvability:
    """
    Precondition check for whether a planned trial configuration can
    actually resolve 'expected_overhead_s' against timer noise, before
    spending the time to run it. 'ratio' compares the accumulated signal
    across a campaign (n_batches * expected_overhead_s) against the
    standard error of the paired-difference noise floor implied by a
    single pass's duration, its relative timing jitter and how many
    repetitions will be averaged. A configuration below 'threshold'
    produces a number that looks like data but is not, and must not be run.
    """

    n_batches: int
    expected_overhead_s: float
    t_pass_s: float
    sigma_rel: float
    n_reps: int
    threshold: float = 5.0

    @property
    def ratio(self) -> float:
        signal = self.n_batches * self.expected_overhead_s
        noise = self.t_pass_s * self.sigma_rel / sqrt(self.n_reps)
        return signal / noise

    @property
    def ok(self) -> bool:
        return self.ratio > self.threshold

    def explain(self) -> str:
        verdict = 'resolvable' if self.ok else 'NOT resolvable'

        if self.ok:
            return (
                f"ratio={self.ratio:.2f} > threshold={self.threshold:.1f}: {verdict} "
                f"(margin {self.ratio / self.threshold:.2f}x)"
            )

        # ratio scales linearly with n_batches but only with sqrt(n_reps), so
        # closing the same gap via n_reps alone always demands a steeper,
        # squared increase; whichever knob needs the larger absolute bump is
        # reported as the binding term, since that is the harder one to fix.
        needed_scale = self.threshold / self.ratio
        n_batches_needed = self.n_batches * needed_scale
        n_reps_needed = self.n_reps * needed_scale ** 2
        binding = (
            'n_reps' if (n_reps_needed - self.n_reps) > (n_batches_needed - self.n_batches)
            else 'n_batches'
        )

        return (
            f"ratio={self.ratio:.2f} <= threshold={self.threshold:.1f}: {verdict} "
            f"(binding term: {binding}; need n_batches>={n_batches_needed:.0f} "
            f"or n_reps>={n_reps_needed:.0f} to resolve)"
        )
