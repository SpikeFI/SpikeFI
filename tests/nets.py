"""NetSpec and its home for tiny synthetic nets"""


from dataclasses import dataclass

from torch import nn, Tensor

from slayerSNN.slayer import spikeLayer


@dataclass
class NetSpec:
    net: nn.Module
    shape_in: tuple[int, int, int]


class DenseNet(nn.Module):
    """Two chained dense layers: the minimal injectable-injectable topology."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SF1: nn.Module = slayer.dense(8, 4)
        self.SF2: nn.Module = slayer.dense(4, 3)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s = self.slayer.spike(self.slayer.psp(self.SF1(spikes_in)))
        return self.slayer.spike(self.slayer.psp(self.SF2(s)))


class ThreeLayerNet(nn.Module):
    """Three chained dense layers: the shortest topology in which a round can
    fault two different layers and still leave the two fault-free trailing
    layers early stop needs, so the early-stop layer and the late-start layer
    are distinct rather than collapsing onto the same one."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SF1: nn.Module = slayer.dense(8, 6)
        self.SF2: nn.Module = slayer.dense(6, 4)
        self.SF3: nn.Module = slayer.dense(4, 3)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s = self.slayer.spike(self.slayer.psp(self.SF1(spikes_in)))
        s = self.slayer.spike(self.slayer.psp(self.SF2(s)))
        return self.slayer.spike(self.slayer.psp(self.SF3(s)))


class ConvNet(nn.Module):
    """conv -> pool -> dense: conv/dense weight-index-order asymmetry, and a
    non-injectable layer (the pool) sitting between two injectables. SC1's
    neuron shape (C=2, H=6, W=10) has three distinct dimensions on purpose."""

    def __init__(self, slayer: spikeLayer) -> None:
        super().__init__()
        self.slayer: spikeLayer = slayer
        self.SC1: nn.Module = slayer.conv(1, 2, 3, padding=1)
        self.SP1: nn.Module = slayer.pool(2)
        # dense()'s tuple inFeatures is (W, H, C), the reverse of a tensor's
        # own (C, H, W) shape (e.g. LayersInfo.shapes_neu) - SP1 outputs
        # (C=2, H=3, W=5), so this reverses it to (5, 3, 2).
        self.SF2: nn.Module = slayer.dense((5, 3, 2), 4)

    def forward(self, spikes_in: Tensor) -> Tensor:
        s1 = self.slayer.spike(self.slayer.psp(self.SC1(spikes_in)))
        p1 = self.slayer.spike(self.slayer.psp(self.SP1(s1)))
        return self.slayer.spike(self.slayer.psp(self.SF2(p1)))


class SharedDropoutNet(nn.Module):
    """Two injectables of *different* output shape feeding the same shared
    dropout module, for the neuron perturb pre-hook's invocation guard."""

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


class SameShapeSharedNet(nn.Module):
    """Two injectables of *equal* output shape sharing a dropout module: no
    comparison of what the shared module receives can tell the two apart, so
    cross-contamination would appear here if the neuron perturb pre-hook did
    not identify its own invocation by position in the forward pass."""

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
