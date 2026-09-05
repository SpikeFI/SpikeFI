"""Tier 0 — quantization argument derivation: qargs_exact checked against
values computed independently, inline, from the affine quantization
definition rather than against the module's own output.
"""


import torch

from spikefi.utils.quantization import qargs_exact, qargs_from_range, qiinfo


def test_qargs_exact_matches_the_hand_computed_affine_mapping() -> None:
    """qargs_exact's scale and zero point satisfy the affine mapping
    definition, checked against expectations written out from that
    definition rather than from the function itself."""
    xmin, xmax = -1.0, 3.0
    qmin, qmax = -128, 127

    expected_scale = (xmax - xmin) / (qmax - qmin)
    expected_zero_point = round(qmin - xmin / expected_scale)

    scale, zero_point = qargs_exact(xmin, xmax, qmin, qmax)

    assert scale == expected_scale
    assert zero_point == expected_zero_point


def test_qargs_maps_the_range_endpoints_onto_the_dtype_limits() -> None:
    """Quantizing the original range's own endpoints with the returned
    scale and zero point lands them on the dtype's integer limits,
    independently of the formula used to derive those arguments."""
    xmin, xmax = -1.0, 3.0
    dtype = torch.qint8
    info = qiinfo(dtype)

    scale, zero_point = qargs_from_range(xmin, xmax, dtype)
    x = torch.tensor([xmin, xmax])
    q = torch.quantize_per_tensor(x, scale, zero_point, dtype).int_repr()

    assert abs(int(q[0]) - info.min) <= 1
    assert abs(int(q[1]) - info.max) <= 1
