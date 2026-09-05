"""Tier 0 — LayersInfo: the layer bookkeeping every fault site is bounds-
checked against, and its own equality contract.
"""


import pytest
import torch

from slayerSNN import slayer

from spikefi.utils.layer import LayersInfo

from nets import NetSpec


def test_get_following_returns_the_next_layer_and_none_at_the_end(
        layers_info: LayersInfo
) -> None:
    """get_following() names the next layer in the chain, and returns None
    once called on the last layer rather than raising or wrapping around."""
    assert layers_info.get_following('SF1') == 'SF2'
    assert layers_info.get_following('SF2') == 'tail'
    assert layers_info.get_following('tail') is None


def test_infer_records_the_layers_own_neuron_and_synapse_shapes(
        dense_net: NetSpec,
        layers_info: LayersInfo
) -> None:
    """infer() records each layer's own output and weight shapes exactly as
    the layer itself reports them, since these are the coordinate space
    every fault site is later bounds-checked against."""
    device = next(dense_net.net.parameters()).device
    x = torch.zeros(1, *dense_net.shape_in, 1, device=device)
    output = dense_net.net.SF1(x)

    assert layers_info.shapes_neu['SF1'] == tuple(output.shape[1:4])
    assert (
        layers_info.shapes_syn['SF1']
        == tuple(dense_net.net.SF1.weight.shape[0:4])
    )
    assert layers_info.shapes_syn['tail'] is None


def test_infer_rejects_an_unsupported_layer_type() -> None:
    """infer() raises before recording anything when handed a layer type
    SpikeFI cannot faithfully replay, rather than silently learning shapes
    for a layer whose forward pass it does not reproduce."""
    info = LayersInfo((1, 1, 1))
    delay_layer = slayer._delayLayer(1, 1.0)

    with pytest.raises(RuntimeError, match='Unsupported layer type'):
        info.infer('delay', delay_layer, torch.zeros(1))


def test_infer_rejects_a_reused_injectable_layer(
        dense_net: NetSpec
) -> None:
    """infer() raises when called twice with the same name for an
    injectable layer, since an injectable layer must be unique in the
    network for fault sites on it to be unambiguous."""
    info = LayersInfo(dense_net.shape_in)
    device = next(dense_net.net.parameters()).device
    x = torch.zeros(1, *dense_net.shape_in, 1, device=device)
    output = dense_net.net.SF1(x)

    info.infer('SF1', dense_net.net.SF1, output)
    with pytest.raises(RuntimeError, match='more than once'):
        info.infer('SF1', dense_net.net.SF1, output)
