import numpy as np
import torch
import slayerSNN as snn


class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, samples_file: str, sampling_time: int, sample_length: int):
        super().__init__()

        self.path = data_path
        self.samples = np.loadtxt(samples_file, dtype='str')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        input_index = int(self.samples[index, 0])
        class_label = int(self.samples[index, 1].split("/")[1].split(".")[0])

        spikes_in = snn.io.readNpSpikes(f"{self.path}{self.samples[index, 1]}") \
            .toSpikeTensor(torch.zeros((2, 128, 128, self.n_time_bins)), self.sampling_time)
        desired_class = torch.zeros((11, 1, 1, 1))
        desired_class[class_label, ...] = 1

        return input_index, spikes_in, desired_class, class_label


class GestureNetwork(torch.nn.Module):
    def __init__(self, net_params: snn.params, do_enable=False):
        super().__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])

        self.SC1 = self.slayer.conv(2, 16, 5, padding=2, weightScale=10)
        self.SC2 = self.slayer.conv(16, 32, 3, padding=1, weightScale=50)

        self.SP0 = self.slayer.pool(4)
        self.SP1 = self.slayer.pool(2)
        self.SP2 = self.slayer.pool(2)

        self.SF1 = self.slayer.dense((8, 8, 32), 512)
        self.SF2 = self.slayer.dense(512, 11)

        self.SDC = self.slayer.dropout(0.05 if do_enable else 0.0)
        self.SDF = self.slayer.dropout(0.10 if do_enable else 0.0)

    def forward(self, s_in):
        s_out = self.slayer.spike(self.slayer.psp(self.SP0(s_in)))   # 2,  32, 32

        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_out)))  # 16, 32, 32
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))  # 16, 16, 16

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))  # 32, 16, 16
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))  # 32, 8,  8

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_out)))  # 512

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF2(s_out)))  # 11

        return s_out
