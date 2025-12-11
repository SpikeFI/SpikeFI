import os
import slayerSNN as snn
from tonic.datasets import DVSGesture
import torch
from typing import Callable


class GestureDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: str,
            train: bool,
            sampling_time: int,
            sample_length: int,
            transform: Callable | None = None
    ) -> None:
        super().__init__()

        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.n_time_bins = int(sample_length / sampling_time)

        self.dataset = DVSGesture(
            save_to=os.path.abspath(os.path.join(root_dir, '..')),
            train=train,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        events, label = self.dataset[idx]
        sy, sx, sp = DVSGesture.sensor_size

        spikes_in = (
            snn.io.event(
                events['x'], events['y'], events['p'], events['t'] / 1000
            )
            .toSpikeTensor(
                torch.zeros((sp, sy, sx, self.n_time_bins)),
                self.sampling_time
            )
        )

        return spikes_in, label


class GestureShallow(torch.nn.Module):
    def __init__(self, net_params: snn.params) -> None:
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        # Block 0: 2x128x128 -> 2x32x32
        self.SP0 = self.slayer.pool(4)

        # Block 1: 2x32x32 -> 32x16x16
        self.SC1 = self.slayer.conv(2, 32, 5, padding=2, weightScale=10)
        self.SP1 = self.slayer.pool(2)

        # Block 2: 32x16x16 -> 64x8x8
        self.SC2 = self.slayer.conv(32, 64, 3, padding=1, weightScale=20)
        self.SP2 = self.slayer.pool(2)

        # Block 3: 64x8x8 -> 64x8x8
        self.SC3 = self.slayer.conv(64, 64, 3, padding=1, weightScale=20)

        # Block 4: 64x8x8 -> 256 -> 11
        self.SF4a = self.slayer.dense((8, 8, 64), 256)
        self.SF4b = self.slayer.dense(256, 11)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SP0(s_in)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SF4a(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SF4b(s_out)))

        return s_out


class GestureDeep(torch.nn.Module):
    def __init__(self, net_params: snn.params) -> None:
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        # Block 0: 2x128x128 -> 2x64x64
        self.SP0 = self.slayer.pool(2)

        # Block 1: 2x64x64 -> 32x32x32
        self.SC1 = self.slayer.conv(2, 32, 5, padding=2, weightScale=10)
        self.SP1 = self.slayer.pool(2)

        # Block 2: 32x32x32 -> 64x16x16
        self.SC2 = self.slayer.conv(32, 64, 3, padding=1, weightScale=20)
        self.SP2 = self.slayer.pool(2)

        # Block 3: 64x16x16 -> 128x8x8
        self.SC3 = self.slayer.conv(64, 128, 3, padding=1, weightScale=20)
        self.SP3 = self.slayer.pool(2)

        # Block 4: 128x8x8 -> 1024 -> 256 -> 11
        self.SF4a = self.slayer.dense((8, 8, 128), 512)
        self.SD4 = self.slayer.dropout(0.5)
        self.SF4b = self.slayer.dense(512, 11)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SP0(s_in)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP3(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SF4a(s_out)))
        s_out = self.SD4(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF4b(s_out)))

        return s_out

    # def __init__(self, net_params: snn.params) -> None:
    #     super().__init__()

    #     self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])
    #
        # # Block 0: 2x128x128 -> 2x64x64
        # self.SP0 = self.slayer.pool(2)

        # # Block 1: 2x64x64 -> 32x32x32
        # self.SC1a = self.slayer.conv(2, 32, 3, padding=1)
        # self.SC1b = self.slayer.conv(32, 32, 3, padding=1)
        # self.SP1 = self.slayer.pool(2)

        # # Block 2: 32x32x32 -> 64x16x16
        # self.SC2a = self.slayer.conv(32, 64, 3, padding=1)
        # self.SC2b = self.slayer.conv(64, 64, 3, padding=1)
        # self.SP2 = self.slayer.pool(2)

        # # Block 3: 64x16x16 -> 128x8x8
        # self.SC3a = self.slayer.conv(64, 128, 3, padding=1)
        # self.SC3b = self.slayer.conv(128, 128, 3, padding=1)
        # self.SP3 = self.slayer.pool(2)

        # # Block 4: 128x8x8 -> 256x1x1
        # self.SC4 = self.slayer.conv(128, 256, 3, padding=1)

        # # Block 5: 256 - 128 - 11
        # self.SF5a = self.slayer.dense(256, 128)
        # self.SD5 = self.slayer.dropout(0.5)
        # self.SF5b = self.slayer.dense(128, 11)

    # def forward(self, s_in: torch.Tensor) -> torch.Tensor:
    #     s_out = self.slayer.spike(self.slayer.psp(self.SP0(s_in)))

    #     s_out = self.slayer.spike(self.slayer.psp(self.SC1a(s_out)))
    #     s_out = self.slayer.spike(self.slayer.psp(self.SC1b(s_out)))
    #     s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))

    #     s_out = self.slayer.spike(self.slayer.psp(self.SC2a(s_out)))
    #     s_out = self.slayer.spike(self.slayer.psp(self.SC2b(s_out)))
    #     s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))

    #     s_out = self.slayer.spike(self.slayer.psp(self.SC3a(s_out)))
    #     s_out = self.slayer.spike(self.slayer.psp(self.SC3b(s_out)))
    #     s_out = self.slayer.spike(self.slayer.psp(self.SP3(s_out)))

    #     s_out = self.slayer.spike(self.slayer.psp(self.SC4(s_out)))

    #     s_out = self.slayer.spike(self.slayer.psp(self.SF5a(s_out)))
    #     s_out = self.SD5(s_out)
    #     s_out = self.slayer.spike(self.slayer.psp(self.SF5b(s_out)))

    #     return s_out
