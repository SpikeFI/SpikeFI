__all__ = ["bitflip", "example", "optimizations", "parametric", "train_golden", "train",
           "Dataset", "Network"]

import os

from torch.utils.data import DataLoader, TensorDataset

import slayerSNN as snn

import spikefi as sfi


# Configuration parameters (modify depending on application)
CASE_STUDY = 'nmnist-lenet'  # 'nmnist-lenet' # 'nmnist-deep' # 'gesture'
DO_ENABLED = False
OUT_DIR = 'out/net'

_exception = ValueError(f"Case study '{CASE_STUDY}' not added.")

# To work on a new case study:
#   - Create a new import case

# Case study imports
if 'nmnist' in CASE_STUDY:
    fyamlname = 'nmnist.yaml'
    batch_size = 12
    shuffle = False
    shape_in = (2, 34, 34)

    from demo.nets.nmnist import NMNISTDataset as Dataset
    if CASE_STUDY == 'nmnist-deep':
        from demo.nets.nmnist import NMNISTNetwork as Network
    elif CASE_STUDY == 'nmnist-lenet':
        from demo.nets.nmnist import LeNetNetwork as Network
    elif CASE_STUDY == 'qnmnist-lenet':
        from demo.nets.nmnist import QLeNetNetwork as Network
    else:
        raise _exception
elif CASE_STUDY == 'gesture':
    fyamlname = 'gesture.yaml'
    batch_size = 4
    shuffle = True
    shape_in = (2, 128, 128)

    from demo.nets.gesture import GestureDataset as Dataset
    from demo.nets.gesture import GestureNetwork as Network
else:
    raise _exception

# No changes needed after this line
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)
base_fname = f"{CASE_STUDY}{'-do' if DO_ENABLED else ''}"

net_params = snn.params(f'demo/config/{fyamlname}')

train_set = Dataset(
    data_path=net_params['training']['path']['dir_train'],
    samples_file=net_params['training']['path']['list_train'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)

test_set = Dataset(
    data_path=net_params['training']['path']['dir_test'],
    samples_file=net_params['training']['path']['list_test'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)

single_loader = DataLoader(TensorDataset(*next(iter(test_loader))), batch_size=batch_size, shuffle=False)

trial_def = sfi.utils.io.calculate_trial(base_fname + '_.pt', OUT_DIR)


def get_fnetname(trial: str = None) -> str:
    return f"{base_fname}_net{trial if trial else trial_def}.pt"


def get_fstaname(trial: str = None) -> str:
    return f"{base_fname}_stats{trial if trial else trial_def}.pkl"


def get_ffigname(trial: str = None) -> str:
    return f"{base_fname}_train{trial if trial else trial_def}.svg"
