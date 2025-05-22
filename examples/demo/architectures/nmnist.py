import torch
import os
import slayerSNN as snn


class NMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, split: str, sampling_time: int, sample_length: int, transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

        self.samples = []
        self.labels = []

        for label in range(10):
            label_dir = os.path.join(self.split_dir, str(label))
            for fname in os.listdir(label_dir):
                if fname.endswith('.bin'):
                    self.samples.append(os.path.join(label_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath = self.samples[idx]
        label = self.labels[idx]

        spikes_in = snn.io.read2Dspikes(fpath) \
            .toSpikeTensor(torch.zeros((2, 34, 34, self.n_time_bins)), self.sampling_time)

        target = torch.zeros((10, 1, 1, 1))
        target[label, ...] = 1

        if self.transform:
            spikes_in = self.transform(spikes_in)

        return idx, spikes_in, target, label


class LeNetNetwork(torch.nn.Module):
    def __init__(self, net_params: snn.params, do_enable=False):
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        self.SC1 = self.slayer.conv(2,   6, 7)
        self.SC2 = self.slayer.conv(6,  16, 5)
        self.SC3 = self.slayer.conv(16, 120, 5)

        self.SP1 = self.slayer.pool(2)
        self.SP2 = self.slayer.pool(2)

        self.SF1 = self.slayer.dense(120, 84)
        self.SF2 = self.slayer.dense(84, 10)

        self.SDC = self.slayer.dropout(0.10 if do_enable else 0.0)
        self.SDF = self.slayer.dropout(0.25 if do_enable else 0.0)

    def forward(self, s_in):
        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))   # 6, 28, 28
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))  # 6, 14, 14

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))  # 16, 10, 10
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))  # 16,  5,  5

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))  # 120, 1, 1

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_out)))  # 84

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF2(s_out)))  # 10

        return s_out


class NMNISTNetwork(torch.nn.Module):
    def __init__(self, net_params: snn.params, do_enable=False):
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        self.SC1 = self.slayer.conv(2, 16, 5, padding=1)
        self.SC2 = self.slayer.conv(16, 32, 3, padding=1)
        self.SC3 = self.slayer.conv(32, 64, 3, padding=1)

        self.SP1 = self.slayer.pool(2)
        self.SP2 = self.slayer.pool(2)

        self.SF1 = self.slayer.dense((8, 8, 64), 10)

        self.SDC = self.slayer.dropout(0.10 if do_enable else 0.0)
        self.SDF = self.slayer.dropout(0.25 if do_enable else 0.0)

    def forward(self, s_in):
        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))   # 16, 32, 32
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))  # 16, 16, 16

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))  # 32, 16, 16
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))  # 32, 8,  8

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))  # 64, 8,  8

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_out)))  # 10

        return s_out
