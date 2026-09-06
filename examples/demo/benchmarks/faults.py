from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
import random
from typing import Any

from spikefi.fault import Fault, FaultModel, FaultSite


# Bump only if the on-disk JSON layout changes in a way that
# breaks round-tripping older plan files.
SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class FaultSpec:
    ftype: str
    layer: str
    position: tuple[int, ...]
    round_idx: int

    def __post_init__(self) -> None:
        # A neuron site is 3 dimensional (channel, height, width), a
        # synapse site is 4 dimensional (out, in, kernel_h, kernel_w),
        # matching FaultSite's own layout assertion.
        if len(self.position) not in (3, 4):
            raise ValueError(
                f"Position {self.position} for layer '{self.layer}' must "
                "have 3 (neuron) or 4 (synapse) dimensions, got "
                f"{len(self.position)}."
            )


@dataclass
class FaultPlan:
    casestudy: str
    seed: int
    specs: tuple[FaultSpec, ...]

    @property
    def n_rounds(self) -> int:
        return max((s.round_idx for s in self.specs), default=-1) + 1

    def prefix(self, n_rounds: int) -> 'FaultPlan':
        # Rounds are appended strictly in draw order during sampling, so
        # keeping every spec whose round_idx is below n_rounds yields the
        # exact same specs a plan sampled for only n_rounds would hold.
        return FaultPlan(
            casestudy=self.casestudy,
            seed=self.seed,
            specs=tuple(s for s in self.specs if s.round_idx < n_rounds)
        )

    def digest(self) -> str:
        canonical = json.dumps(
            self._to_dict(), sort_keys=True, separators=(',', ':')
        )
        return sha256(canonical.encode('utf-8')).hexdigest()[:12]

    def save(self, path: str) -> str:
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: str) -> 'FaultPlan':
        with open(path) as f:
            data = json.load(f)

        version = data.get('version')
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported fault plan schema version {version}, "
                f"expected {SCHEMA_VERSION}."
            )

        specs = tuple(
            FaultSpec(
                ftype=s['ftype'],
                layer=s['layer'],
                position=tuple(s['position']),
                round_idx=s['round_idx']
            )
            for s in data['specs']
        )

        return cls(casestudy=data['casestudy'], seed=data['seed'], specs=specs)

    def to_rounds(
            self,
            model_factories: dict[str, Callable[[str], FaultModel]]
    ) -> list[list[Fault]]:
        """Groups this plan's specs into per-round Fault lists, building
        each fault's model via model_factories[spec.ftype](spec.layer).

        model_factories takes only a layer name, not a network, unlike
        catalog.FTYPES[key].factory, which takes (net, layer_name). This
        keeps faults.py free of any catalogue or network import, so
        callers must bind the network first, e.g.:
            {k: partial(ft.factory, net) for k, ft in FTYPES.items()}
        """
        rounds: list[list[Fault]] = [[] for _ in range(self.n_rounds)]
        for spec in self.specs:
            model = model_factories[spec.ftype](spec.layer)
            site = FaultSite(spec.layer, spec.position)
            rounds[spec.round_idx].append(Fault(model, site))

        return rounds

    def _to_dict(self) -> dict[str, Any]:
        return {
            'casestudy': self.casestudy,
            'seed': self.seed,
            'version': SCHEMA_VERSION,
            'specs': [
                {
                    'ftype': s.ftype,
                    'layer': s.layer,
                    'position': list(s.position),
                    'round_idx': s.round_idx
                }
                for s in self.specs
            ]
        }


def validate_position(
        layer: str,
        position: tuple[int, ...],
        shape: tuple[int, ...]
) -> None:
    if len(position) != len(shape):
        raise ValueError(
            f"Position {position} for layer '{layer}' has "
            f"{len(position)} dimensions, expected {len(shape)} to match "
            f"layer shape {shape}."
        )

    for dim, (p, bound) in enumerate(zip(position, shape)):
        if not (0 <= p < bound):
            raise ValueError(
                f"Position {position} for layer '{layer}' is out of "
                f"bounds at dimension {dim}: {p} not in [0, {bound})."
            )


def sample_plan(
        casestudy: str,
        ftype: str,
        layers: Sequence[str],
        shapes: dict[str, tuple[int, ...]],
        n_rounds: int,
        seed: int,
        faults_per_round: int = 1,
        syn_select: bool = False
) -> FaultPlan:
    expected_arity = 4 if syn_select else 3
    for layer in layers:
        shape = shapes[layer]
        if len(shape) != expected_arity:
            raise ValueError(
                f"Layer '{layer}' shape {shape} has arity {len(shape)}, "
                f"expected {expected_arity} for "
                f"{'synapse' if syn_select else 'neuron'} faults."
            )

    # Combined position space across all eligible layers, sized so a
    # single uniform draw over [0, total_size) lands on a layer with
    # probability proportional to that layer's own size.
    sizes: list[int] = [prod(shapes[layer]) for layer in layers]
    total_size: int = sum(sizes)

    if faults_per_round > total_size:
        raise ValueError(
            f"faults_per_round ({faults_per_round}) exceeds the combined "
            f"position space across all eligible layers ({total_size}); "
            "distinct positions cannot be drawn without replacement."
        )

    # Bounds the without-replacement rejection loop below: as faults_per_round
    # approaches total_size, draws increasingly collide with positions
    # already seen, so a generous but finite attempt budget turns a
    # pathologically crowded space into a clear error instead of a hang.
    max_attempts: int = 100 * faults_per_round

    rng = random.Random(seed)
    specs: list[FaultSpec] = []
    for round_idx in range(n_rounds):
        seen: set[tuple[str, tuple[int, ...]]] = set()
        drawn: list[tuple[str, tuple[int, ...]]] = []
        attempts = 0
        while len(drawn) < faults_per_round:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not draw {faults_per_round} distinct positions "
                    f"for round {round_idx} of case study '{casestudy}' "
                    f"after {max_attempts} attempts; the position space "
                    f"({total_size}) is too crowded relative to "
                    "faults_per_round."
                )

            idx = rng.randrange(total_size)
            attempts += 1
            layer, position = _unrank(layers, shapes, sizes, idx)
            key = (layer, position)
            if key in seen:
                continue
            seen.add(key)
            drawn.append(key)

        for layer, position in drawn:
            validate_position(layer, position, shapes[layer])
            specs.append(FaultSpec(ftype, layer, position, round_idx))

    return FaultPlan(casestudy, seed, tuple(specs))


def _unrank(
        layers: Sequence[str],
        shapes: dict[str, tuple[int, ...]],
        sizes: Sequence[int],
        idx: int
) -> tuple[str, tuple[int, ...]]:
    # Walk layer blocks, sized sizes[i], subtracting each one until idx
    # lands in and is re-based to be local to its layer.
    for layer, size in zip(layers, sizes):
        if idx < size:
            break
        idx -= size

    # Mixed-radix decode of idx into the layer's own shape, matching
    # the row-major order Campaign._unrank_pos and _sample_positions use.
    shape = shapes[layer]
    position = []
    for dim in reversed(shape):
        idx, coord = divmod(idx, dim)
        position.append(coord)
    position.reverse()

    return layer, tuple(position)
