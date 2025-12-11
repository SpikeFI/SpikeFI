__all__ = [
    "architectures", "Dataset", "Network"
]

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import slayerSNN as snn
import spikefi.utils.io as sfio
from typing import Any, Callable, get_args, Iterable, Literal


# TODO: Review and fix all examples after changes in the init file

SUPPORTED_CASE_STUDIES = Literal[
    'nmnist_cnn', 'nmnist_mlp',
    'gesture_shallow', 'gesture_deep'
]
DEMO_DIR = os.path.dirname(__file__)
WORK_DIR = os.path.join(DEMO_DIR, '..')

sfio.OUT_DIR = os.path.join(WORK_DIR, sfio.OUT_DIR)
sfio.RES_DIR = os.path.join(WORK_DIR, sfio.RES_DIR)
sfio.FIG_DIR = os.path.join(WORK_DIR, sfio.FIG_DIR)
sfio.NET_DIR = os.path.join(WORK_DIR, sfio.NET_DIR)

case_study = None
net_params = None
shape_in = None
device = None
_is_ready = False


def is_demo_ready():
    return _is_ready


def prepare(
        casestudy: SUPPORTED_CASE_STUDIES,
        dev: torch.device = torch.device('cuda'),
        fyamlname: str | None = None
):
    global case_study, net_params, device
    global shape_in, _is_ready
    global Dataset, Network

    if casestudy not in get_args(SUPPORTED_CASE_STUDIES):
        raise ValueError(f"Case study '{case_study}' not added. "
                         "Please modify file 'examples/demo/__init__.py'.")

    case_study = casestudy
    device = dev

    if 'nmnist' in case_study:
        fyaml_name = fyamlname or 'nmnist.yaml'
        shape_in = (2, 34, 34)

        from demo.architectures.nmnist import NmnistDataset as Dataset
        if case_study == 'nmnist_cnn':
            from demo.architectures.nmnist import NmnistCnn as Network
        elif case_study == 'nmnist_mlp':
            from demo.architectures.nmnist import NmnistMlp as Network
    elif 'gesture' in case_study:
        fyaml_name = fyamlname or 'gesture.yaml'
        shape_in = (2, 128, 128)

        from demo.architectures.gesture import GestureDataset as Dataset
        if case_study == 'gesture_shallow':
            from demo.architectures.gesture import GestureShallow as Network
        elif case_study == 'gesture_deep':
            from demo.architectures.gesture import GestureDeep as Network

    net_params = snn.params(os.path.join(DEMO_DIR, f'config/{fyaml_name}'))

    _is_ready = True


def get_net(fpath: str = None, trial: int = None) -> 'Network':
    net = Network(net_params).to(device)
    net_path = fpath or sfio.make_net_filepath(get_fnetname(trial))
    net.load_state_dict(torch.load(net_path, weights_only=True))
    net.eval()

    return net


def get_dataset(train: bool, transform: Callable | None = None) -> 'Dataset':
    return Dataset(
        root_dir=os.path.join(
            WORK_DIR, net_params['training']['path']['root_dir']
        ),
        train=train,
        sampling_time=net_params['simulation']['Ts'],
        sample_length=net_params['simulation']['tSample'],
        transform=transform
    )


def get_loader(
        train: bool,
        transform: Callable | None = None,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[Iterable], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False
) -> DataLoader:
    return DataLoader(
        dataset=get_dataset(train, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def get_tiny_loader(size: int = 1) -> DataLoader:
    loader_iter = iter(get_loader(train=False))
    batches = [next(loader_iter) for _ in range(size)]
    fields = list(zip(*batches))
    tensors = [torch.cat(field, dim=0) for field in fields]

    return DataLoader(TensorDataset(*tensors))


def get_base_fname(train: bool = False) -> str:
    return (
        f"{case_study}{'_train' if train else ''}"
    )


def get_trial() -> int:
    return sfio.calculate_trial(get_base_fname() + '_net.pt', sfio.NET_DIR)


def get_trial_str(trial: int = None) -> str:
    return f" ({trial})" if trial else ""


def get_fnetname(trial: int = None, format: str = 'pt') -> str:
    return f"{get_base_fname()}_net{get_trial_str(trial)}.{format}"


def get_fstaname(trial: int = None, format: str = 'pkl') -> str:
    return f"{get_base_fname()}_stats{get_trial_str(trial)}.{format}"


def get_ffigname(trial: int = None, format: str = 'svg') -> str:
    return f"{get_base_fname()}_train{get_trial_str(trial)}.{format}"
