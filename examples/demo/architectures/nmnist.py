import os
import slayerSNN as snn
from tonic.datasets import NMNIST
import torch
from typing import Callable


class NmnistDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        train: bool,
        sampling_time: int,
        sample_length: int,
        transform: Callable | None = None
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.n_time_bins = int(sample_length / sampling_time)
        self.train = train
        self.split = 'train' if train else 'test'

        self.dataset = NMNIST(
            save_to=os.path.abspath(os.path.join(root_dir, '..')),
            train=train,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        events, label = self.dataset[idx]
        sy, sx, sp = NMNIST.sensor_size

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


class NmnistCnn(torch.nn.Module):
    def __init__(self, net_params: snn.params) -> None:
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        # Block 1: 2x34x34 -> 24x17x17
        self.SC1 = self.slayer.conv(2, 24, 5, padding=2, weightScale=10)
        self.SP1 = self.slayer.pool(2)

        # Block 2: 24x17x17 -> 48x9x9
        self.SC2 = self.slayer.conv(24, 48, 3, padding=1, weightScale=15)
        self.SP2 = self.slayer.pool(2)

        # Block 3: 48x9x9 -> 96x5x5
        self.SC3 = self.slayer.conv(48, 96, 3, padding=1, weightScale=15)
        self.SP3 = self.slayer.pool(2)

        # Block 4: 96x5x5 -> 256 -> 10
        self.SF4a = self.slayer.dense((5, 5, 96), 256)
        self.SF4b = self.slayer.dense(256, 10)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP3(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SF4a(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SF4b(s_out)))

        return s_out


class NmnistMlp(torch.nn.Module):
    def __init__(self, net_params: snn.params) -> None:
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        # 2x34x34 -> 512 -> 512 -> 10
        self.SF1 = self.slayer.dense(NMNIST.sensor_size, 512, weightScale=10)
        self.SF2 = self.slayer.dense(512, 512, weightScale=10)
        self.SF3 = self.slayer.dense(512, 10, weightScale=12)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_in)))
        s_out = self.slayer.spike(self.slayer.psp(self.SF2(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SF3(s_out)))

        return s_out
