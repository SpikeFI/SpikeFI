#############################################################################
# An example of training a fault-free SNN with SpikeFI.                     #
#                                                                           #
# Training or running inference of a fault-free network with SpikeFI is     #
# equivalent to creating an empty FI campaign and executing it with run()   #
# or run_train() methods for training and inference phases, respectively.   #
#############################################################################


import torch
import slayerSNN as snn
import spikefi as sfi
import demo


# Number of training epochs
n_epochs = 20

# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Create a network instance
net = demo.Network(demo.net_params, demo.dropout_en).to(demo.device)
trial = demo.get_trial()

print("Training configuration:")
print(f"  - case study: {demo.case_study}")
print(f"  - dropout: {'yes' if demo.dropout_en else 'no'}")
print(f"  - epochs number: {n_epochs}")
print(f"  - trial: {trial}")
print()

# SNN loss
spike_loss = snn.loss(demo.net_params).to(demo.device)
# Optimizer module
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)

# Create an empty FI campaign, which contains a single Fault Round
# without any Faults
cmpn_name = f"{demo.get_base_fname(train=True)}_golden"
cmpn = sfi.Campaign(net, demo.shape_in, net.slayer, cmpn_name)

# Execute the FI experiments
# (train a new network instance for each fault round)
golden = cmpn.run_train(
    n_epochs,
    demo.get_loader(train=True),
    optimizer,
    spike_loss
)[0]

# Save trained network
cmpn.save_net(golden, demo.get_fnetname(trial).split('.')[0])

# Plot and save the learning curve(s)
sfi.visual.learning_curve(cmpn.export(), format='png')
