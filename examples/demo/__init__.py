__all__ = [
    "architectures", "utils", "setup_nmnist", "setup_gesture",
    "Dataset", "Network"
]

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import slayerSNN as snn
import spikefi.utils.io as sfio
from typing import Literal


SUPPORTED_CASE_STUDIES = ['nmnist-lenet', 'nmnist-deep', 'gesture']
DEMO_DIR = os.path.dirname(__file__)
WORK_DIR = os.path.join(DEMO_DIR, '..')

sfio.OUT_DIR = os.path.join(WORK_DIR, sfio.OUT_DIR)
sfio.RES_DIR = os.path.join(WORK_DIR, sfio.RES_DIR)
sfio.FIG_DIR = os.path.join(WORK_DIR, sfio.FIG_DIR)
sfio.NET_DIR = os.path.join(WORK_DIR, sfio.NET_DIR)

case_study = None
dropout_en = None
fyaml_name = None
batch_size = None
to_shuffle = None
net_params = None
shape_in = None

device = torch.device('cuda')

_is_ready = False


def is_demo_ready():
    return _is_ready


def prepare(
        casestudy: Literal['nmnist-lenet', 'nmnist-deep', 'gesture'],
        dropout: bool = False,
        fyamlname=None,
        batchsize=None,
        shuffle=None
):
    global case_study, dropout_en, fyaml_name, batch_size
    global to_shuffle, shape_in, _is_ready
    global Dataset, Network
    global net_params

    if casestudy not in SUPPORTED_CASE_STUDIES:
        raise ValueError(f"Case study '{case_study}' not added. "
                         "Please modify file 'examples/demo/__init__.py'.")

    case_study = casestudy
    dropout_en = dropout

    if 'nmnist' in case_study:
        fyaml_name = fyamlname or 'nmnist.yaml'
        batch_size = batchsize or 12
        to_shuffle = shuffle if shuffle is not None else False
        shape_in = (2, 34, 34)

        from demo.architectures.nmnist import NMNISTDataset as Dataset
        if case_study == 'nmnist-lenet':
            from demo.architectures.nmnist import LeNetNetwork as Network
        elif case_study == 'nmnist-deep':
            from demo.architectures.nmnist import NMNISTNetwork as Network
    elif case_study == 'gesture':
        fyaml_name = fyamlname or 'gesture.yaml'
        batch_size = batchsize or 4
        to_shuffle = shuffle if shuffle is not None else True
        shape_in = (2, 128, 128)

        from demo.architectures.gesture import GestureDataset as Dataset
        from demo.architectures.gesture import GestureNetwork as Network

    net_params = snn.params(os.path.join(DEMO_DIR, f'config/{fyaml_name}'))

    _is_ready = True


def get_net(fpath: str = None, trial: int = None) -> 'Network':
    net = Network(net_params, dropout_en).to(device)
    net_path = fpath or sfio.make_net_filepath(get_fnetname(trial))
    net.load_state_dict(torch.load(net_path, weights_only=True))
    net.eval()

    return net


def get_dataset(train: bool) -> 'Dataset':
    return Dataset(
        root_dir=os.path.join(
            WORK_DIR, net_params['training']['path']['root_dir']
        ),
        split='Train' if train else 'Test',
        sampling_time=net_params['simulation']['Ts'],
        sample_length=net_params['simulation']['tSample'])


def get_loader(train: bool) -> DataLoader:
    return DataLoader(
        dataset=get_dataset(train),
        batch_size=batch_size,
        shuffle=to_shuffle,
        num_workers=4
    )


def get_tiny_loader(size: int = 1) -> DataLoader:
    loader_iter = iter(get_loader(train=False))
    batches = [next(loader_iter) for _ in range(size)]
    fields = list(zip(*batches))
    tensors = [torch.cat(field, dim=0) for field in fields]

    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=False
    )


def get_base_fname(train: bool = False) -> str:
    return (
        f"{case_study}{'-do' if dropout_en else ''}{'_train' if train else ''}"
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
