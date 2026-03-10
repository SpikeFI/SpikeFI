#############################################################################
# An example of training a fault-free SNN with SpikeFI.                     #
#                                                                           #
# Training or running inference of a fault-free network with SpikeFI is     #
# equivalent to creating an empty FI campaign and executing it with run()   #
# or run_train() methods for training and inference phases, respectively.   #
#############################################################################


from collections.abc import Iterable
from tonic import transforms
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import slayerSNN as snn
import spikefi as sfi
import demo


# Number of training epochs
n_epochs = 50

# Setup the fault simulation demo environment and select case study
demo.prepare(casestudy='nmnist_cnn')

# Create a network instance
net = demo.Network(demo.net_params).to(demo.device)
trial = demo.get_trial()

# Create the dataset loaders for the training and testing sets
train_loader = DataLoader(
    demo.get_cached_dataset(
        train=True,
        transform=transforms.Denoise(filter_time=10000)
    ),
    batch_size=16, shuffle=True,
    num_workers=4, pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    demo.get_cached_dataset(
        train=False,
        transform=transforms.Denoise(filter_time=10000)
    ),
    batch_size=4, shuffle=False,
    num_workers=4, pin_memory=True,
    persistent_workers=True
)

# SNN loss
spike_loss = snn.loss(demo.net_params).to(demo.device)


# Optimizer factory
def make_optimizer(params: Iterable) -> Optimizer:
    return torch.optim.Adam(params, lr=2e-3, weight_decay=1e-4)


# Scheduler factory
def make_scheduler(optimizer: Optimizer) -> LRScheduler:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


print("Training configuration:")
print(f"  - network: {demo.case_study}")
print(f"  - epochs: {n_epochs}")
print(f"  - trial: {trial}")
print(f"  - Ts: {demo.net_params['simulation']['Ts']} ms")
print()

# Create an empty FI campaign, which contains a single Fault Round
# without any Faults
cmpn_name = f"{demo.get_base_fname(train=True)}_golden"
cmpn = sfi.Campaign(net, demo.shape_in, net.slayer, cmpn_name)

# Train the network as an empty campaign (without faults)
golden = cmpn.run_train(
    n_epochs,
    train_loader,
    test_loader,
    spike_loss,
    make_optimizer,
    make_scheduler
)[0]

# Save trained network
cmpn.save_net(golden, demo.get_fnetname(trial).split('.')[0])

# Plot and save the learning curve(s)
sfi.visual.learning_curve(cmpn.export(), format='png')
