"""Tier 0 — exact known-answer tests for spikefi.models' fault functions:
set_value/add_value/mul_value/qua_value/bfl_value.
"""


import pytest
import torch

from spikefi.models import add_value, bfl_value, mul_value, qua_value, set_value
from spikefi.utils.quantization import qargs_from_range, qiinfo


def _hand_quantize(
        x: torch.Tensor,
        scale: float,
        zero_point: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """Independent round-to-nearest-grid-point oracle, computed without
    calling torch's own quantize/dequantize pair, so a broken qua_value
    can't hide by reproducing its own bug in the check."""
    info = qiinfo(dtype)
    q = torch.clamp(torch.round(x / scale + zero_point), info.min, info.max)
    return (q - zero_point) * scale


@pytest.fixture
def qargs() -> tuple[float, int, torch.dtype]:
    dtype = torch.qint8
    scale, zero_point = qargs_from_range(-2.0, 2.0, dtype)
    return scale, zero_point, dtype


@pytest.mark.neuron
@pytest.mark.synapse
def test_set_value_returns_value_verbatim() -> None:
    """set_value ignores the original and returns the target value exactly."""
    assert set_value(torch.tensor(3.14), 0.0) == 0.0
    assert set_value(1.0, 5.0) == 5.0


@pytest.mark.neuron
@pytest.mark.synapse
def test_add_value_exact_sum() -> None:
    """add_value returns original + value exactly."""
    assert add_value(2.0, 3.0) == 5.0
    assert torch.equal(add_value(torch.tensor([1.0, 2.0]), 1.0), torch.tensor([2.0, 3.0]))


@pytest.mark.neuron
@pytest.mark.synapse
def test_mul_value_exact_product() -> None:
    """mul_value returns original * value exactly."""
    assert mul_value(2.0, 3.0) == 6.0
    assert torch.equal(mul_value(torch.tensor([1.0, 2.0]), 2.0), torch.tensor([2.0, 4.0]))


@pytest.mark.synapse
def test_qua_value_matches_hand_computed_quantization(
        qargs: tuple[float, int, torch.dtype]
) -> None:
    """qua_value's output matches an independently, hand-computed nearest
    grid point, not just SpikeFI's own quantize/dequantize round-trip."""
    scale, zero_point, dtype = qargs
    x = torch.tensor([0.37, -1.1, 1.9])

    expected = _hand_quantize(x, scale, zero_point, dtype)
    assert torch.allclose(qua_value(x, scale, zero_point, dtype), expected)


@pytest.mark.synapse
def test_qua_value_is_idempotent(
        qargs: tuple[float, int, torch.dtype]
) -> None:
    """Quantizing an already-quantized value leaves it unchanged: qua_value
    snaps onto a fixed grid, so a second pass is a no-op."""
    scale, zero_point, dtype = qargs
    x = torch.tensor([0.37, -1.1, 1.9])

    once = qua_value(x, scale, zero_point, dtype)
    twice = qua_value(once, scale, zero_point, dtype)

    assert torch.equal(once, twice)


@pytest.mark.synapse
def test_bfl_value_twice_is_quantizing_involution(
        qargs: tuple[float, int, torch.dtype]
) -> None:
    """bfl_value is an involution on the quantization grid, not on the
    original float: flipping the same bit twice returns qua_value(x), the
    quantized version of x, not x itself."""
    scale, zero_point, dtype = qargs
    x = torch.tensor([0.37, -1.1, 1.9])
    bit = 0

    once = bfl_value(x, bit, scale, zero_point, dtype)
    twice = bfl_value(once, bit, scale, zero_point, dtype)

    assert torch.equal(twice, qua_value(x, scale, zero_point, dtype))


@pytest.mark.synapse
def test_bfl_value_residual_equals_bit_weight(
        qargs: tuple[float, int, torch.dtype]
) -> None:
    """A single-bit flip changes the dequantized value by exactly the
    weight of that bit in the quantization grid: ±scale * 2**bit."""
    scale, zero_point, dtype = qargs
    x = torch.tensor([0.37, -1.1, 1.9])
    bit = 2

    flipped = bfl_value(x, bit, scale, zero_point, dtype)
    residual = (flipped - qua_value(x, scale, zero_point, dtype)).abs()

    assert torch.allclose(residual, torch.full_like(residual, scale * 2 ** bit))


@pytest.mark.synapse
def test_bfl_value_changes_exactly_the_named_bits(
        qargs: tuple[float, int, torch.dtype]
) -> None:
    """The quantized integer representation differs from the original's by
    exactly the flipped bit(s), no others."""
    scale, zero_point, dtype = qargs
    x = torch.tensor([0.37, -1.1, 1.9])
    bits = (0, 2)

    q = torch.quantize_per_tensor(x, scale, zero_point, dtype)
    flipped = bfl_value(x, bits, scale, zero_point, dtype)
    qf = torch.quantize_per_tensor(flipped, scale, zero_point, dtype)

    expected_mask = (1 << 0) | (1 << 2)
    xor = q.int_repr() ^ qf.int_repr()
    assert torch.all(xor == expected_mask)


@pytest.mark.synapse
def test_bfl_value_rejects_out_of_range_bit(
        qargs: tuple[float, int, torch.dtype]
) -> None:
    """A bit index outside the dtype's width raises rather than silently
    wrapping or corrupting the flip mask."""
    scale, zero_point, dtype = qargs
    x = torch.tensor([0.5])

    with pytest.raises(AssertionError):
        bfl_value(x, qiinfo(dtype).bits, scale, zero_point, dtype)
