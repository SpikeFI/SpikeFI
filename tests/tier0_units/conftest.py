"""Tier-0-local fixtures: a LayersInfo built without touching CUDA, and a
bare Campaign stub exposing only the attributes validate()/inject() read.
Building a real Campaign forwards through slayer.psp()/spike(), which are
CUDA-only kernels, so Tier 0 constructs what it needs by hand instead.
"""


import pytest
import torch
from torch import nn

import spikefi as sfi
from spikefi.utils.layer import LayersInfo


@pytest.fixture
def layers_info(dense_net) -> LayersInfo:
    """LayersInfo for dense_net's SF1 -> SF2 -> tail chain, populated by
    calling each layer directly instead of through the net's own forward,
    which would route through slayer.psp()/spike()."""
    info = LayersInfo(dense_net.shape_in)
    device = next(dense_net.net.parameters()).device
    x = torch.zeros(1, *dense_net.shape_in, 1, device=device)

    for name in ('SF1', 'SF2'):
        layer = getattr(dense_net.net, name)
        x = layer(x)
        info.infer(name, layer, x)
    info.infer('tail', nn.Identity(), x)

    return info


@pytest.fixture
def campaign_stub(layers_info: LayersInfo, slayer) -> sfi.Campaign:
    """A Campaign built without calling __init__, so validate()/inject()
    can be exercised without the GPU-only forward pass __init__ performs
    to infer layer shapes."""
    stub = sfi.Campaign.__new__(sfi.Campaign)
    stub.layers_info = layers_info
    stub.slayer = slayer
    stub.rounds = [sfi.ff.FaultRound()]
    return stub
