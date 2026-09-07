"""Equivalence gate certifying the naive fault-injection baseline against
SpikeFI's own optimization levels.

For each fault type, the same fault plan is run six ways: the naive
baseline plus SpikeFI at O0 through O4. Every variant must agree exactly,
round by round, on accuracy and on a hash of the output tensors it
produced. Timing the same six runs is also where the naive-versus-O1 cost
ratio comes from, so criticality is never computed here: it would add a
golden pass into SpikeFI's timed region for a comparison this gate has no
use for.

Agreement alone is not evidence, which is why the fault load per round is
high and why finalize() insists the rounds actually differ from one
another. A spiking network is sparse enough that most neurons never fire
for a given input, so a fault landing on one of them changes nothing at
all: with a single fault per round, every round of every variant returns
the golden output, and a baseline that injected nothing whatsoever would
agree with SpikeFI perfectly.
"""

import argparse
import math
import os

from spikefi.core import CampaignOptimization

from .. import catalog, faults, records
from ..context import CaseStudyContext
from ..experiment import Experiment, register, TrialSpec
from ..runners import run_naive, run_spikefi, TrialOutcome

# Every variant a fault type is checked across, in a fixed order so a
# schedule group's trials are always laid out the same way.
_VARIANTS: tuple[str, ...] = ('naive', 'O0', 'O1', 'O2', 'O3', 'O4')

# How many of the first disagreements finalize() names in its error, so a
# gate failing across many rounds still raises a readable message.
_MAX_NAMED_DISAGREEMENTS: int = 5


@register
class Equivalence(Experiment):
    """Runs every fault type through all six variants under one shared
    fault plan per fault type, then checks their agreement in finalize().
    """

    name: str = 'equivalence'

    def __init__(
            self,
            context: CaseStudyContext,
            session: int = 1,
            tags: tuple[str, ...] = (),
            seed: int = 0,
            n_warmup: int = 1,
            shuffle: bool = True,
            out_dir: str | None = None,
            n_rounds: int = 10,
            batch_size: int = 8,
            n_samples: int = 32,
            faults_per_round: int = 500,
            fault_seed: int = 0,
            ftypes: tuple[str, ...] = ()
    ) -> None:
        super().__init__(
            context=context, session=session, tags=tags, seed=seed, n_warmup=n_warmup,
            shuffle=shuffle, out_dir=out_dir
        )
        self.n_rounds: int = n_rounds
        self.batch_size: int = batch_size
        self.n_samples: int = n_samples
        self.faults_per_round: int = faults_per_round
        self.fault_seed: int = fault_seed
        self.ftypes: tuple[str, ...] = ftypes

        # Lazily filled by _fault_plan(), keyed so every variant of a given
        # fault type is handed the exact same FaultPlan object.
        self._fault_plans: dict[tuple[str, int, int], faults.FaultPlan] = {}

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--n-rounds', type=int, default=10, help="Fault rounds sampled per fault type.")
        parser.add_argument('--batch-size', type=int, default=8, help="Batch size for every variant.")
        parser.add_argument('--n-samples', type=int, default=32, help="Resident dataset size for every variant.")
        parser.add_argument(
            '--faults-per-round', type=int, default=500,
            help="Fault sites injected per round; enough of them that a round actually perturbs the network."
        )
        parser.add_argument('--fault-seed', type=int, default=0, help="Seed the fault plans are sampled with.")
        parser.add_argument(
            '--ftypes', nargs='*', default=(), choices=list(catalog.FTYPES),
            help="Fault types to check; defaults to every fault type in the catalogue."
        )

    @classmethod
    def from_args(cls, context: CaseStudyContext, args: argparse.Namespace) -> 'Equivalence':
        return cls(
            context=context,
            session=args.session,
            tags=tuple(args.tags),
            seed=args.seed,
            n_warmup=args.n_warmup,
            shuffle=not args.no_shuffle,
            out_dir=args.out_dir,
            n_rounds=args.n_rounds,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            faults_per_round=args.faults_per_round,
            fault_seed=args.fault_seed,
            ftypes=tuple(args.ftypes)
        )

    def plan(self) -> list[TrialSpec]:
        """One group of six TrialSpecs per fault type, sharing a pair_id so
        _ordered_schedule keeps them adjacent through the shuffle: the six
        variants of a fault type then run under the same thermal
        conditions. Touches neither the GPU nor any data, since config()
        calls this too.
        """

        ftypes = self.ftypes or tuple(catalog.FTYPES)

        specs: list[TrialSpec] = []
        for pair_id, ftype in enumerate(ftypes):
            for variant in _VARIANTS:
                specs.append(TrialSpec(
                    kind='faulty', opt=variant, ftype=ftype, n_faults=self.faults_per_round, es_tol=0,
                    n_rounds=self.n_rounds, batch_size=self.batch_size,
                    time_bins=self.context.n_time_bins, n_samples=self.n_samples,
                    n_batches=math.ceil(self.n_samples / self.batch_size),
                    synthetic=False, fault_seed=self.fault_seed, pair_id=pair_id
                ))

        return specs

    def execute(self, spec: TrialSpec) -> TrialOutcome:
        loader = self.context.resident_loader(batch_size=spec.batch_size, n_samples=spec.n_samples)
        plan = self._fault_plan(spec.ftype)

        if spec.opt == 'naive':
            return run_naive(
                self.context, loader, plan,
                compute_critical=False, probe_memory=False, collect_outputs=True
            )

        return run_spikefi(
            self.context, loader, plan=plan, opt=CampaignOptimization[spec.opt], es_tol=0,
            compute_critical=False, probe_memory=False, collect_outputs=True
        )

    def _fault_plan(self, ftype: str) -> faults.FaultPlan:
        """The fault plan for ftype, sampled once per (ftype, fault_seed)
        and cached so every variant of that fault type faces the identical
        plan object rather than a fresh, independently drawn one.
        """

        key = (ftype, self.fault_seed, self.faults_per_round)
        if key not in self._fault_plans:
            # Synapse fault types carry a 4-dimensional position (out, in,
            # kernel_h, kernel_w); neuron ones carry a 3-dimensional one.
            syn_select = ftype.startswith('synapse')
            layers = self.context.injectable_layers()
            shapes = {lay: self.context.layer_shape(lay, syn_select) for lay in layers}

            self._fault_plans[key] = faults.sample_plan(
                casestudy=self.context.casestudy, ftype=ftype, layers=layers, shapes=shapes,
                n_rounds=self.n_rounds, seed=self.fault_seed,
                faults_per_round=self.faults_per_round, syn_select=syn_select
            )

        return self._fault_plans[key]

    def finalize(self, trials_path: str, rounds_path: str) -> None:
        """Checks that every variant of every fault type agrees, round by
        round, on accuracy and on output hash, and that the rounds were
        distinguishable in the first place. Raises if either fails.

        The rounds file's own `ftype` column already describes the fault
        (identical across variants, since all six share one FaultPlan); it
        is the trials file that carries `opt`, the variant name, so the two
        are joined on (session, trial) to bring `opt` onto each round row.
        Accuracy is compared exactly, not with a tolerance, because
        bit-exact agreement is precisely what this gate asserts, and
        records.load_records reads floats with 'round_trip' precision so
        the written and reloaded digits match.
        """

        if not os.path.exists(rounds_path):
            raise RuntimeError(
                f"Equivalence gate has nothing to check: no rounds file at {rounds_path!r}. "
                "Every trial either failed or recorded no rounds."
            )

        trials = records.load_records(trials_path)
        rounds = records.load_records(rounds_path, fields=records.ROUND_FIELDS)
        if len(rounds) == 0:
            raise RuntimeError(f"Equivalence gate has nothing to check: {rounds_path!r} holds no rows.")

        joined = rounds.merge(trials[['session', 'trial', 'opt']], on=['session', 'trial'], how='left')

        disagreements: list[str] = []
        verdict_lines: list[str] = []
        all_agree = True

        for ftype, ftype_rows in joined.groupby('ftype'):
            ftype_agrees = True
            n_rounds = 0

            for round_idx, round_rows in ftype_rows.groupby('round'):
                n_rounds += 1

                # Agreement among whatever happens to be present is not
                # evidence: a round carrying one variant, or none carrying
                # an output hash, would otherwise pass this gate trivially.
                missing = [v for v in _VARIANTS if v not in set(round_rows['opt'])]
                if missing:
                    ftype_agrees = False
                    disagreements.append(f"{ftype} round {round_idx}: no row for variant(s) {missing}")

                unhashed = sorted(round_rows[round_rows['out_hash'].isna()]['opt'])
                if unhashed:
                    ftype_agrees = False
                    disagreements.append(
                        f"{ftype} round {round_idx}: no output hash from variant(s) {unhashed}"
                    )

                accuracies = dict(zip(round_rows['opt'], round_rows['accuracy']))
                if len(set(accuracies.values())) > 1:
                    ftype_agrees = False
                    disagreements.append(f"{ftype} round {round_idx}: accuracy differs across {accuracies}")

                with_hash = round_rows[round_rows['out_hash'].notna()]
                hashes = dict(zip(with_hash['opt'], with_hash['out_hash']))
                if len(set(hashes.values())) > 1:
                    ftype_agrees = False
                    disagreements.append(f"{ftype} round {round_idx}: out_hash differs across {hashes}")

            # How many of this fault type's rounds were actually
            # distinguishable from one another. With every round returning
            # the same output, the variants agree no matter what any of
            # them does, so this gate would certify a baseline that never
            # injected anything. Raising the fault load is the fix.
            n_distinct = joined[joined['ftype'] == ftype]['out_hash'].nunique()
            if n_distinct < 2:
                ftype_agrees = False
                disagreements.append(
                    f"{ftype}: all {n_rounds} round(s) produced one and the same output, so this "
                    "fault load cannot distinguish an implementation from one that injects nothing"
                )

            n_variants = ftype_rows['opt'].nunique()
            verdict = 'agree' if ftype_agrees else 'DISAGREE'
            verdict_lines.append(
                f"  {ftype}: {n_rounds} round(s) x {n_variants} variant(s), "
                f"{n_distinct} distinct output(s) -> {verdict}"
            )
            all_agree = all_agree and ftype_agrees

        print("Equivalence verdict:")
        for line in verdict_lines:
            print(line)
        print('All fault types agree.' if all_agree else 'DISAGREEMENT DETECTED.')

        if not all_agree:
            named = disagreements[:_MAX_NAMED_DISAGREEMENTS]
            raise RuntimeError(
                f"Equivalence gate failed with {len(disagreements)} disagreement(s), "
                f"first {len(named)}: " + '; '.join(named)
            )
