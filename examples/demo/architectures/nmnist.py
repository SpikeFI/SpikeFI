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

        self.SC1 = self.slayer.conv(2, 16, 5, padding=1)
        self.SC2 = self.slayer.conv(16, 32, 3, padding=1)
        self.SC3 = self.slayer.conv(32, 64, 3, padding=1)

        self.SP1 = self.slayer.pool(2)
        self.SP2 = self.slayer.pool(2)

        self.SF1 = self.slayer.dense((8, 8, 64), 10)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_out)))

        return s_out


class NmnistMlp(torch.nn.Module):
    def __init__(self, net_params: snn.params) -> None:
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        self.SF1 = self.slayer.dense(NMNIST.sensor_size, 1000)
        self.SF2 = self.slayer.dense(1000, 10)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_in)))
        s_out = self.slayer.spike(self.slayer.psp(self.SF2(s_out)))

        return s_out
