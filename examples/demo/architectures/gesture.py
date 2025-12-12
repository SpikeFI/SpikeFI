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
            transform: Callable | None = None,
            exclude_other: bool = False
    ) -> None:
        super().__init__()

        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.n_time_bins = int(sample_length / sampling_time)
        self.exclude_other = exclude_other
        self.train = train

        self.dataset = DVSGesture(
            save_to=os.path.abspath(os.path.join(root_dir, '..')),
            train=train,
            transform=transform
        )

        if exclude_other:
            # Label 10 corresponds to the 'other gestures' class
            labels = self.dataset.targets
            self.keep_indices = [
                i for i, l in enumerate(labels) if l != 10
            ]

    def __len__(self) -> int:
        if self.exclude_other:
            return len(self.keep_indices)

        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self.keep_indices[idx] if self.exclude_other else idx
        events, label = self.dataset[real_idx]
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


class GestureNet(torch.nn.Module):
    def __init__(
            self,
            net_params: snn.params,
            exclude_other: bool = False
     ) -> None:
        super().__init__()

        n_out = 10 if exclude_other else 11

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        # Block 0: 2x128x128 -> 2x32x32
        self.SP0 = self.slayer.pool(4)

        # Block 1: 2x32x32 -> 32x16x16
        self.SC1 = self.slayer.conv(2, 32, 5, padding=2, weightScale=10)
        self.SP1 = self.slayer.pool(2)

        # Block 2: 32x16x16 -> 64x8x8
        self.SC2 = self.slayer.conv(32, 64, 3, padding=1, weightScale=20)
        self.SP2 = self.slayer.pool(2)

        # Block 3: 64x8x8 -> 128x4x4
        self.SC3 = self.slayer.conv(64, 128, 3, padding=1, weightScale=20)
        self.SP3 = self.slayer.pool(2)

        # Block 4: 128x4x4 -> 256 -> 11 | 10
        self.SF4a = self.slayer.dense((4, 4, 128), 256)
        self.SF4b = self.slayer.dense(256, n_out)

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        s_out = self.slayer.spike(self.slayer.psp(self.SP0(s_in)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SP3(s_out)))

        s_out = self.slayer.spike(self.slayer.psp(self.SF4a(s_out)))
        s_out = self.slayer.spike(self.slayer.psp(self.SF4b(s_out)))

        return s_out
