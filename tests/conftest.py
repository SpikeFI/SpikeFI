"""Shared fixtures for the SpikeFI test suite: tiny synthetic nets, seeded
inputs/datasets, a campaign factory, and the tier/gpu auto-marking and
output-artifact-redirection machinery every test relies on. See
tests/README.md for usage.
"""


from collections.abc import Callable, Iterator
from pathlib import Path
import re

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from slayerSNN.slayer import spikeLayer

import spikefi as sfi
import spikefi.utils.io as sfio

from nets import NetSpec


# --- Tier/gpu auto-marking ---

# Maps each tier directory to its marker name, so a test's tier is derived
# from where it lives rather than declared by hand in every file.
_TIER_DIRS: dict[str, str] = {
    'tier0_units': 'tier0',
    'tier1_semantics': 'tier1',
    'tier2_propagation': 'tier2',
    'tier3_invariants': 'tier3',
    'tier4_training': 'tier4',
    'tier5_serialization': 'tier5',
    'tier6_visual': 'tier6',
}


def pytest_collection_modifyitems(
        config: pytest.Config,
        items: list[pytest.Item]
) -> None:
    """Derives each test's tier marker from its directory and applies the
    gpu marker (skipped when CUDA is unavailable) to everything outside
    tier0_units/, so no individual test file needs its own pytestmark line."""
    tests_dir = Path(__file__).resolve().parent
    cuda_available = torch.cuda.is_available()
    skip_gpu = pytest.mark.skip(reason='requires CUDA, not available')

    for item in items:
        rel_dir = item.path.resolve().relative_to(tests_dir).parts[0]

        tier_marker = _TIER_DIRS.get(rel_dir)
        if tier_marker is not None:
            item.add_marker(getattr(pytest.mark, tier_marker))

        if rel_dir != 'tier0_units':
            item.add_marker(pytest.mark.gpu)
            if not cuda_available:
                item.add_marker(skip_gpu)


# --- Output artifacts ---

@pytest.fixture(scope='session', autouse=True)
def tests_out_dir() -> Iterator[None]:
    """Retargets spikefi.utils.io's output directories at tests/out/, so the
    suite never writes into (or depends on) the package's own out/."""
    tests_out = Path(__file__).resolve().parent / 'out'
    tests_out.mkdir(exist_ok=True)

    originals = {
        name: getattr(sfio, name)
        for name in ('OUT_DIR', 'RES_DIR', 'FIG_DIR', 'NET_DIR')
    }
    # RES_DIR/FIG_DIR/NET_DIR are derived from OUT_DIR at import time but
    # read as module globals at call time, so all four must be patched here.
    sfio.OUT_DIR = str(tests_out)
    sfio.RES_DIR = str(tests_out / 'res')
    sfio.FIG_DIR = str(tests_out / 'fig')
    sfio.NET_DIR = str(tests_out / 'net')

    yield

    for name, value in originals.items():
        setattr(sfio, name, value)


@pytest.fixture
def artifact_name(request: pytest.FixtureRequest) -> str:
    """Filesystem-safe token derived from the test's own (possibly
    parametrized) name, so every artifact it writes stays distinct across
    cases and traceable back to the test that produced it."""
    name = request.node.name.replace('[', '_').replace(']', '')
    return re.sub(r'[^\w.-]+', '_', name)


# --- Network & simulation parameters ---

@pytest.fixture(scope='session')
def net_params() -> dict:
    """Plain-dict slayer neuron/simulation/training config, ~16 time bins."""
    return {
        'neuron': {
            'type': 'SRMALPHA',
            'theta': 10,
            'tauSr': 10.0,
            'tauRef': 1.0,
            'scaleRef': 2,
            'tauRho': 1,
            'scaleRho': 1,
        },
        'simulation': {'Ts': 1.0, 'tSample': 16, 'nSample': 1},
        'training': {
            'error': {
                'type': 'NumSpikes',
                'tgtSpikeRegion': {'start': 0, 'stop': 16},
                'tgtSpikeCount': {True: 10, False: 1},
            }
        },
    }


@pytest.fixture(scope='session')
def device() -> torch.device:
    """The device every net fixture, campaign, and GPU-tier test runs on."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='session')
def slayer(net_params: dict, device: torch.device) -> spikeLayer:
    """One spikeLayer instance shared by every net fixture: Campaign
    deep-copies whatever slayer it is given per-campaign, so sharing this
    base instance across nets is safe and avoids rebuilding it repeatedly."""
    return spikeLayer(net_params['neuron'], net_params['simulation']).to(device)


# --- Tiny synthetic nets ---

class _DenseNet(nn.Module):
    """Two chained dense layers: the minimal injectable-injectable topology."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SF1: nn.Module = slayer.dense(8, 4)
        self.SF2: nn.Module = slayer.dense(4, 3)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s = self.slayer.spike(self.slayer.psp(self.SF1(spikes_in)))
        return self.slayer.spike(self.slayer.psp(self.SF2(s)))


class _ConvNet(nn.Module):
    """conv -> pool -> dense: conv/dense weight-index-order asymmetry, and a
    non-injectable layer (the pool) sitting between two injectables."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SC1: nn.Module = slayer.conv(1, 2, 3, padding=1)
        self.SP1: nn.Module = slayer.pool(2)
        # dense()'s tuple inFeatures is (W, H, C), the reverse of a tensor's
        # own (C, H, W) shape (e.g. LayersInfo.shapes_neu) - SP1 outputs
        # (C=2, H=3, W=3), so this reverses it to (3, 3, 2).
        self.SF2: nn.Module = slayer.dense((3, 3, 2), 4)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s1 = self.slayer.spike(self.slayer.psp(self.SC1(spikes_in)))
        p1 = self.slayer.spike(self.slayer.psp(self.SP1(s1)))
        return self.slayer.spike(self.slayer.psp(self.SF2(p1)))


class _SharedDropoutNet(nn.Module):
    """Two injectables of *different* output shape feeding the same shared
    dropout module, for the neuron perturb pre-hook's layer_shape guard."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SF1: nn.Module = slayer.dense(8, 4)
        self.SF2: nn.Module = slayer.dense(4, 6)
        self.drop: nn.Module = slayer.dropout(0.0)
        self.SF3: nn.Module = slayer.dense(6, 3)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s1 = self.slayer.spike(self.slayer.psp(self.SF1(spikes_in)))
        d1 = self.drop(s1)
        s2 = self.slayer.spike(self.slayer.psp(self.SF2(d1)))
        d2 = self.drop(s2)
        return self.slayer.spike(self.slayer.psp(self.SF3(d2)))


class _SameShapeSharedNet(nn.Module):
    """Two injectables of *equal* output shape sharing a dropout module: the
    layer_shape guard is shape-only, so cross-contamination would appear
    here if the guard were relied on to fully disambiguate the two."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SF1: nn.Module = slayer.dense(8, 4)
        self.SF2: nn.Module = slayer.dense(4, 4)
        self.drop: nn.Module = slayer.dropout(0.0)
        self.SF3: nn.Module = slayer.dense(4, 3)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s1 = self.slayer.spike(self.slayer.psp(self.SF1(spikes_in)))
        d1 = self.drop(s1)
        s2 = self.slayer.spike(self.slayer.psp(self.SF2(d1)))
        d2 = self.drop(s2)
        return self.slayer.spike(self.slayer.psp(self.SF3(d2)))


# Each net fixture seeds the global torch RNG explicitly right
# before construction to favor reproducibility and ensure that
# weight values do not differ from run to run.
@pytest.fixture
def dense_net(slayer: spikeLayer, device: torch.device) -> NetSpec:
    torch.manual_seed(100)
    return NetSpec(net=_DenseNet(slayer).to(device), shape_in=(8, 1, 1))


@pytest.fixture
def conv_net(slayer: spikeLayer, device: torch.device) -> NetSpec:
    torch.manual_seed(101)
    return NetSpec(net=_ConvNet(slayer).to(device), shape_in=(1, 6, 6))


@pytest.fixture
def shared_dropout_net(slayer: spikeLayer, device: torch.device) -> NetSpec:
    torch.manual_seed(102)
    return NetSpec(net=_SharedDropoutNet(slayer).to(device), shape_in=(8, 1, 1))


@pytest.fixture
def same_shape_shared_net(slayer: spikeLayer, device: torch.device) -> NetSpec:
    torch.manual_seed(103)
    return NetSpec(net=_SameShapeSharedNet(slayer).to(device), shape_in=(8, 1, 1))


# --- Seeded inputs & datasets ---

@pytest.fixture
def fixed_input(device: torch.device) -> Callable[..., Tensor]:
    """Factory returning a seeded spike tensor for a given (shape_in, batch,
    n_time_bins, p): deterministic across calls with the same arguments."""
    def _fixed_input(
            shape_in: tuple[int, int, int],
            batch: int = 4,
            n_time_bins: int = 16,
            p: float = 0.3,
            seed: int = 42
    ) -> Tensor:
        assert batch >= 2, 'fixed_input requires batch >= 2'
        generator = torch.Generator(device=device).manual_seed(seed)
        rand = torch.rand(
            (batch, *shape_in, n_time_bins), device=device, generator=generator
        )
        return (rand < p).float()

    return _fixed_input


@pytest.fixture
def tiny_loaders(device: torch.device) -> Callable[..., tuple[DataLoader, DataLoader]]:
    """Factory building a seeded (train_loader, test_loader) TensorDataset
    pair, so run_train() needs no dataset on disk."""
    def _tiny_loaders(
            shape_in: tuple[int, int, int],
            n_classes: int = 3,
            n_samples: int = 8,
            batch_size: int = 4,
            n_time_bins: int = 16,
            p: float = 0.3,
            seed: int = 7
    ) -> tuple[DataLoader, DataLoader]:
        generator = torch.Generator(device=device).manual_seed(seed)
        x = (torch.rand(
            (n_samples, *shape_in, n_time_bins), device=device, generator=generator
        ) < p).float()
        y = torch.arange(n_samples, device=device) % n_classes

        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    return _tiny_loaders


# --- Campaign factory & activation probe ---

@pytest.fixture
def make_campaign(
        artifact_name: str,
        device: torch.device
) -> Callable[[nn.Module, tuple[int, int, int], spikeLayer], sfi.Campaign]:
    """Factory building a Campaign named after the calling test, so every
    artifact it writes is traceable back to the test that produced it."""
    def _make_campaign(
            net: nn.Module,
            shape_in: tuple[int, int, int],
            slayer: spikeLayer
    ) -> sfi.Campaign:
        return sfi.Campaign(net, shape_in, slayer, name=artifact_name, device=device)

    return _make_campaign


@pytest.fixture
def golden_activity() -> Callable[[sfi.Campaign, Tensor], dict[str, Tensor]]:
    """Factory returning per-layer golden activations for a campaign's
    input, computed via the campaign's own layer-by-layer forward
    (forward_opt) so the result matches exactly what the framework's fault
    hooks observe. Used to search for sites satisfying a fault's precondition
    (e.g. 'already firing', 'not yet saturated') rather than hard-coding
    indices."""
    def _golden_activity(campaign: sfi.Campaign, x: Tensor) -> dict[str, Tensor]:
        activity: dict[str, Tensor] = {}
        spikes = x
        for idx, name in enumerate(campaign.layers_info.order):
            spikes = campaign.golden(spikes, idx, idx)
            activity[name] = spikes
        return activity

    return _golden_activity
