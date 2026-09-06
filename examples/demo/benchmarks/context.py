"""Owns everything about one demo case study, so every benchmark in the
suite shares one consistent, verified data and network path.
"""

import gc
import os
from time import time

import slayerSNN as snn
import torch
from tonic import transforms
from torch.utils.data import DataLoader, Dataset, Subset

import demo
from spikefi.core import Campaign
from spikefi.utils.layer import LayersInfo

from .synthetic import SyntheticSpikeDataset

# Chunk size used to verify the resident store against the disk-cached
# loader. Keeping it small bounds the extra memory the verification needs
# to one chunk instead of a second full copy of the subset.
_VERIFY_BATCH_SIZE: int = 8


class ResidentSpikeDataset(Dataset):
    """A subset held entirely in one pinned, pre-scaled float32 CPU tensor,
    so a resident DataLoader serves batches without touching the disk cache
    or converting dtypes on a per-item basis.
    """

    def __init__(self, store: torch.Tensor, labels: torch.Tensor) -> None:
        self.store: torch.Tensor = store
        self.labels: torch.Tensor = labels

    def __len__(self) -> int:
        return self.store.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.store[idx], self.labels[idx]


class CaseStudyContext:
    """Owns the network, its shapes, its injectable layers, and both its
    resident (memory-held) and synthetic data sources for one case study.
    It is the only place a DataLoader over the disk cache is ever built.
    """

    def __init__(
        self,
        casestudy: demo.SUPPORTED_CASE_STUDIES,
        device: torch.device | None = None
    ) -> None:
        demo.prepare(casestudy, device or torch.device('cuda'))

        # demo.prepare() rebinds module-level globals; a later prepare() call
        # for a different case study would silently change them under this
        # context, so snapshot them immediately and read only the snapshot
        # from here on.
        self.casestudy: demo.SUPPORTED_CASE_STUDIES = casestudy
        self.device: torch.device = demo.device
        self.net_params: dict = demo.net_params
        self.shape_in: tuple[int, int, int] = demo.shape_in

        self.net: torch.nn.Module | None = demo.get_net(
            os.path.join(demo.DEMO_DIR, 'models', demo.get_fnetname())
        )
        self.net.eval()

        self.ts: float = self.net_params['simulation']['Ts']
        self.t_sample: float = self.net_params['simulation']['tSample']
        self.n_time_bins: int = int(self.t_sample / self.ts)

        self._layers_info: LayersInfo | None = None
        self._resident_stores: dict[tuple[int, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self.materialisation_seconds: dict[tuple[int, int | None], float] = {}

    @property
    def n_classes(self) -> int:
        if hasattr(self.net, "classes_out"):
            return self.net.classes_out

        fc_layers = [lay for lay in self.net.children() if isinstance(lay, snn.slayer._denseLayer)]
        return fc_layers[-1].out_channels

    @property
    def layer_order(self) -> list[str]:
        return list(self._get_layers_info().order)

    def injectable_layers(self) -> list[str]:
        return self._get_layers_info().get_injectables()

    def layer_shape(self, name: str, syn_select: bool = False) -> tuple[int, ...]:
        return self._get_layers_info().get_shape(syn_select, name)

    def _get_layers_info(self) -> LayersInfo:
        if self._layers_info is None:
            cmpn = Campaign(
                self.net, self.shape_in, self.net.slayer,
                name='layers_info_probe', device=self.device
            )
            self._layers_info = cmpn.layers_info
            del cmpn
            gc.collect()
            torch.cuda.empty_cache()

        return self._layers_info

    def resident_loader(
        self,
        batch_size: int,
        subset_div: int = 1,
        n_samples: int | None = None
    ) -> DataLoader:
        key = (subset_div, n_samples)
        if key not in self._resident_stores:
            self._resident_stores[key] = self._materialize_resident(subset_div, n_samples)

        store, labels = self._resident_stores[key]
        dataset = ResidentSpikeDataset(store, labels)

        return DataLoader(
            dataset=dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )

    def _materialize_resident(
        self,
        subset_div: int,
        n_samples: int | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ds = demo.get_cached_dataset(train=False, transform=transforms.Denoise(filter_time=10000))
        n = n_samples if n_samples is not None else int(len(ds) / int(subset_div))
        subset = Subset(ds, range(n))

        # The disk-cached dataset yields numpy arrays; torch.as_tensor avoids
        # an extra copy when filling the pinned store one sample at a time.
        store = torch.empty((n, *self.shape_in, self.n_time_bins), dtype=torch.float32)
        labels = torch.empty((n,), dtype=torch.long)

        t0 = time()
        for i in range(n):
            sample, label = subset[i]
            store[i] = torch.as_tensor(sample, dtype=torch.float32)
            labels[i] = int(label)
        self.materialisation_seconds[(subset_div, n_samples)] = time() - t0

        store = store.pin_memory()
        labels = labels.pin_memory()

        self._verify_resident(subset, store, labels)

        return store, labels

    def _verify_resident(
        self,
        subset: Subset,
        store: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        # shuffle=False means both the resident store and this disk-cached
        # loader chunk samples in the same order, so comparing chunk by
        # chunk implies an exact match at any batch size, while keeping
        # peak extra memory to one chunk instead of a second full copy of
        # the subset.
        loader = DataLoader(
            dataset=subset, batch_size=_VERIFY_BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=True
        )

        offset = 0
        for loader_input, loader_label in loader:
            n = loader_input.shape[0]
            store_chunk = store[offset:offset + n]
            label_chunk = labels[offset:offset + n]

            if not torch.equal(loader_input, store_chunk) or not torch.equal(loader_label, label_chunk):
                raise RuntimeError(
                    "Resident dataset does not exactly match the disk-cached "
                    f"loader for case study '{self.casestudy}' at sample "
                    f"range [{offset}, {offset + n})."
                )

            offset += n

    def synthetic_loader(
        self,
        batch_size: int,
        n_time_bins: int,
        n_batches: int,
        spike_prob: float = 0.05,
        seed: int = 0
    ) -> DataLoader:
        dataset = SyntheticSpikeDataset(
            n_samples=batch_size * n_batches,
            shape_in=self.shape_in,
            n_time_bins=n_time_bins,
            n_classes=self.n_classes,
            spike_prob=spike_prob,
            seed=seed
        )

        return DataLoader(
            dataset=dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )

    def release(self) -> None:
        self.net = None
        self._resident_stores.clear()
        self._layers_info = None

        gc.collect()
        torch.cuda.empty_cache()
