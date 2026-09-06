"""Typed catalogue of the fault types exercised across the benchmark suite.

Uses a frozen dataclass per fault type, so callers get attribute access
and type-checking instead of string-keyed lookups into an untyped dict.
"""

from dataclasses import dataclass
from typing import Callable

from matplotlib import colormaps
import torch

from spikefi.fault import FaultModel
from spikefi.models import BitflippedSynapse, DeadNeuron, DeadSynapse, ThresholdFaultNeuron
from spikefi.utils.quantization import qargs_from_tensor


def bitflipped_synapse(net: torch.nn.Module, lay_name: str) -> BitflippedSynapse:
    qdtype = torch.quint8
    W = getattr(net, lay_name).weight
    scale, zero_point = qargs_from_tensor(W, qdtype)

    return BitflippedSynapse(3, scale, zero_point, qdtype)


COLORS: list[tuple[float, float, float]] = colormaps['Paired'].colors


@dataclass(frozen=True)
class FaultType:
    key: str
    name: str
    color: tuple[float, float, float]
    factory: Callable[[torch.nn.Module, str], FaultModel]
    active: bool = True


FTYPES: dict[str, FaultType] = {
    'neuron_hard': FaultType(
        key='neuron_hard',
        name='Neuron (hard)',
        color=COLORS[1],
        factory=lambda net, lay: DeadNeuron()
    ),
    'synapse_hard': FaultType(
        key='synapse_hard',
        name='Synapse (hard)',
        color=COLORS[3],
        factory=lambda net, lay: DeadSynapse()
    ),
    'neuron_param': FaultType(
        key='neuron_param',
        name='Neuron (param.)',
        color=COLORS[0],
        factory=lambda net, lay: ThresholdFaultNeuron(0.8)
    ),
    'synapse_param': FaultType(
        key='synapse_param',
        name='Synapse (param.)',
        color=COLORS[2],
        factory=bitflipped_synapse
    )
}


def active_ftypes() -> dict[str, FaultType]:
    return {key: ftype for key, ftype in FTYPES.items() if ftype.active}
