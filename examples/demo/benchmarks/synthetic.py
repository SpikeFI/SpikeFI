import torch
from torch.utils.data import Dataset


class SyntheticSpikeDataset(Dataset):
    """
    Random binary spike tensors shaped like a real event-based dataset
    sample: (*shape_in, n_time_bins), e.g. (2, 34, 34, T) for the N-MNIST
    CNN. Lets the benchmark size (n_samples) and the temporal resolution
    (n_time_bins) be swept independently.
    """

    def __init__(
        self,
        n_samples: int,
        shape_in: tuple[int, int, int],
        n_time_bins: int,
        n_classes: int,
        spike_prob: float = 0.05,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.shape_in = shape_in
        self.n_time_bins = n_time_bins
        self.n_classes = n_classes
        self.spike_prob = spike_prob

        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        shape = (n_samples, *shape_in, n_time_bins)
        self.data = torch.bernoulli(
            torch.full(shape, spike_prob), generator=gen
        )
        self.labels = torch.randint(0, n_classes, (n_samples,), generator=gen)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], int(self.labels[idx])
