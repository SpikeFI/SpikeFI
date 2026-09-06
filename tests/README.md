# SpikeFI test suite

Quick reference for the suite: what each tier proves, how to run a slice of it, and how test
artifacts are named.

## Tiers

| Tier | Directory | Proves | CPU/GPU | Tests | Target runtime |
|---|---|---|---|---|---|
| 0 | `tier0_units/` | pure functions & data structures | CPU | 62 | ~0.5 s |
| 1 | `tier1_semantics/` | fault semantics (exact/differential/cross-path oracles) | GPU | 25 | ~0.8 s |
| 2 | `tier2_propagation/` | a fault's local effect reaches the next layer | GPU | 4 | ~0.6 s |
| 3 | `tier3_invariants/` | metamorphic invariants: O0-O4 agreement, round isolation, `eject()` | GPU | 25 | ~1.2 s |
| 4 | `tier4_training/` | training mode: `run_train`, self-containment, persistent-fault semantics, best-epoch checkpointing | GPU | 14 | ~2.5 s |
| 5 | `tier5_serialization/` | `save`/`load`/`export`, `save_net`/`load_net`, copy semantics | GPU | 15 | ~2.5 s |
| 6 | `tier6_visual/` | `visual.py` correctness (data mapping, titles) | GPU | 10 | ~2.6 s |

`test_coverage_matrix.py` sits directly under `tests/` and is tier-agnostic: parametrized over
*(fault model) x (layer type)* so a new model can't arrive untested. Its `FAULT_MODEL_REGISTRY` maps
every concrete `FaultModel` to a deterministic factory, and a completeness test compares that map
against what `inspect` discovers in `spikefi.models` — a newly added model fails it until one
registry line is added, which is what then sweeps it through the generic `perturb()` oracle. Its own
48 tests run in ~0.8 s.

Every tier but Tier 0 needs CUDA (slayer's spike/psp kernels are CUDA-only) and is skipped
automatically when unavailable. Living outside the tier directories, the coverage
matrix gets no marker from the collection hook and declares `gpu` per test instead, so its
pure-Python registry checks still run on a CPU-only machine. The suite currently runs **203 tests**
(155 across the tiers above + 48 in the coverage matrix) in **~5 s** on GPU (measured on a
Quadro RTX 4000).

## Running

```bash
pytest                                   # everything (GPU tiers skip if no CUDA)
pytest -m tier0                          # one tier
pytest -m "tier1 or tier2"               # several tiers
pytest -m "not gpu"                      # CPU-only — what CI runs
pytest -m "synapse and not tier0"        # one family, GPU depth only
pytest tests/tier1_semantics             # a tier by path
pytest tests/tier1_semantics/test_neuron_semantics.py::test_dead_neuron_exact_value
pytest tests/test_coverage_matrix.py     # the coverage matrix
pytest --collect-only -q -m tier4        # list what a selection would run
```

Tier markers (`tier0` … `tier6`, `gpu`) are applied automatically from a test's directory — no test
file declares its own tier. A tier-agnostic test sitting directly under `tests/` matches no tier
directory and so declares `gpu` for itself where it actually needs one; the CUDA skip keys off that
marker either way. Family markers (`neuron`, `synapse`, `parametric`, `optimization`,
`training`, `serialization`, `visual`) are declared explicitly per test, since a fault family is
exercised across several tiers (a synapse fault, for instance, appears in tiers 0, 1, 3, 4 and 5).

## Fixtures

Defined in `conftest.py`, with the net classes themselves in `nets.py`.

- `net_params`, `device`, `slayer` — shared simulation config and the one `spikeLayer` instance every
  net fixture builds its layers from.
- `dense_net`, `three_layer_net`, `conv_net`, `shared_dropout_net`, `same_shape_shared_net` — tiny
  synthetic `NetSpec(net, shape_in)` fixtures covering the structural cases the suite depends on: a
  plain injectable-injectable chain, a three-injectable chain (the shortest net on which a round can
  fault two layers and still leave early stop its two fault-free trailing layers, so late start and
  early stop land on different layers), a non-injectable (pool) between two injectables, and two
  variants of a shared dropout module (differing and equal output shape) for the neuron perturb
  pre-hook's `layer_shape` guard. Each seeds `torch.manual_seed()` right before construction, since
  `slayer.dense()`/`.conv()` initialize weights from the global torch RNG rather than a passed-in
  generator — otherwise a test's exact-value/differential assertions could depend on which
  neurons happen to be active, differing from run to run.
- `fixed_input`, `tiny_loaders` — factories for seeded spike tensors and a train/test `DataLoader`
  pair, so no test touches disk or `tonic`/N-MNIST.
- `make_campaign` — builds a `Campaign` named after the calling test (see *Artifacts* below).
- `golden_activity` — per-layer golden activations for a campaign's input, used to search for sites
  satisfying a fault's precondition instead of hard-coding indices.

`helpers.py` holds precondition assertions (`assert_active`, `assert_not_saturated`,
`assert_differs`), a hand-built fault mutant for the differential oracle (`hand_mutate_weight`),
layer-invocation probes for the Tier 3 work-counting tests (`capture_invocation_widths` for a whole
`Campaign.run()` call, since `campaign.faulty` only exists once `_pre_run()` has built it, recording
each invocation's batch width so an optimization that drops samples is told apart from one that keeps
them; and `count_invocations_during_run`, its how-many-times-only wrapper), and two round-execution
helpers: `run_round`, which drives a single already-injected round through `Campaign`'s private
`_pre_run()`/`faulty()` path to return its raw output tensor, and `capture_run_outputs`, which
intercepts `_advance_performance` during a full `run()` call to return every round's raw,
batch-concatenated output tensor — since `Campaign.run()` itself only exposes aggregate
accuracy/loss stats.

`tier0_units/conftest.py` adds two CPU-only fixtures local to that tier: `layers_info`, a
`LayersInfo` populated by calling `dense_net`'s layers directly instead of through its own forward
(which would route through the CUDA-only `slayer.psp()`/`spike()`), and `campaign_stub`, a `Campaign`
built without `__init__` so `validate()`/`inject()` can be exercised without the GPU-only forward
pass `__init__` performs to infer layer shapes.

## Artifacts

Every file the suite writes goes to `tests/out/` (auto-redirected there by the autouse
`tests_out_dir` fixture) and is named after the test that wrote it: the `make_campaign` factory
builds each `Campaign` with `name=artifact_name`, so `Campaign.save()`, `save_net`, and
`visual`'s figure titles all inherit the test's own name for free. Nothing here is cleaned up between
runs; collisions are resolved by the framework's own `rename=True` default, not by deleting anything.
No test may assert on a fixed filename or on the directory's contents — only on the path a `save`
call returns, or on a file it just wrote.
