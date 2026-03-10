#############################################################################
# An example of fault injections before training.                           #
#                                                                           #
# In this example, faults are injected before the training of the network.  #
# Faults of the selected fault model are injected in one or more sites in   #
# the targeted layers of the network, expressed as a percentage of the      #
# total neurons or synapses, depending on the type of the fault model. The  #
# result of the FI experiments is a newly trained instance of the network   #
# for each fault round, that is, for each fault percentage. Finally, the    #
# networks are exported as .pt files, the campaign information is stored in #
# a .pkl file, and the learning curve(s) are plotted and stored.            #
#                                                                           #
# It is also possible to initialize the 'net' variable to an already        #
# trained instance of a network, and then re-train it with faults.          #
#############################################################################


from collections.abc import Iterable
from tonic import transforms
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import slayerSNN as snn
import spikefi as sfi
from spikefi.fault import Fault
import demo


# Configuration parameters for the optimizations example
# Select one or more layers to target
# (use an empty string '' to target the whole network)
layers = []
# Select the number of training epochs
n_epochs = 20
# Select the percentage(s) of faults to inject
# across the selected layers before training
percent = range(10, 31, 10)
# Learning rate
learn_r = 1e-3
# Select the fault model to be used in the FI experiments
f_model = sfi.fm.PerturbedSynapse(0.5)


# Helper function to nicely print a range
def get_range_str(r: range):
    start, step = r.start, r.step
    stop = r.start + (len(r) - 1) * r.step
    return f"{start}-{stop}-{step}"


# Setup the fault simulation demo environment and select case study
demo.prepare(casestudy='nmnist_cnn')

# Create a network instance
# (or load a trained one to perform re-training with faults)
net = demo.Network(demo.net_params).to(demo.device)

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


# Initialize the campaign object
fm_name = f_model.get_name_snake_case(delimiter='-')
cmpn_name = (
    f"{demo.get_base_fname(train=True)}_{fm_name}_c{get_range_str(percent)}"
)
cmpn = sfi.Campaign(net, demo.shape_in, net.slayer, cmpn_name)

# If specific layers are selected...
if bool(layers) and all(layers):
    lay_inj = layers
    l_sizes = [
        cmpn.layers_info.get_size(f_model.is_synaptic(), l_name)
        for l_name in layers
    ]
else:  # If the entire network is selected...
    # Exclude the output layer from the FI experiments
    lay_inj = cmpn.layers_info.get_injectables()[:-1]
    l_sizes = cmpn.layers_info.get_sizes_inj(f_model.is_synaptic())[:-1]

# Each fault round will result to a new trained network instance
for r, c in enumerate(percent):
    # Inject faults randomly across the targeted layers
    cmpn.then_inject(
        Fault.multiple_random_percent(f_model, c / 100., lay_inj, l_sizes)
    )
cmpn.rounds.pop(0)

# Execute the FI experiments
# (train a new network instance for each fault round)
faulties = cmpn.run_train(
    n_epochs,
    train_loader,
    test_loader,
    spike_loss,
    make_optimizer,
    make_scheduler
)

# Save trained networks
for faulty in faulties:
    cmpn.save_net(faulty)

# Save results in a pickle file
cmpn.save()

# Plot and save the learning curve(s)
figs = sfi.visual.learning_curve(cmpn.export(), format='png')
