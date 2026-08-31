"""NetSpec and its home for tiny synthetic nets"""


from dataclasses import dataclass

from torch import nn


@dataclass
class NetSpec:
    net: nn.Module
    shape_in: tuple[int, int, int]
