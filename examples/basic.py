#############################################################################
# A simple example to demonstrate the basic usage of SpikeFI's features.    #
#                                                                           #
# After loading a trained network (see 'train_golden' example), a campaign  #
# object is initiated. First, three faults are defined, namely fx, fy,      #
# and fz, which are injected in two fault rounds to form various scenarios  #
# of FI experiments. The classification accuracy of the faulty networks     #
# is reported for each experiment and the results are stored in a file.     #
# Continuing, the campaign object is reset and a new FI campaign is         #
# initiated to perform the complete assessment of the network's output      #
# layer by injected one bit-flipped synapse fault at the MSB of every        #
# synaptic weight of the layer. Then, the FI experiment targets each        #
# synapse individually, meaning that only one synapse is faulty at a        #
# time. Finally, results are plotted in a heat map and stored in a file.    #
# This example is the implementation of the pseudo-code presented in the    #
# SpikeFI scientific publication.                                           #
#############################################################################


import os
import torch
import spikefi as sfi
from spikefi.models import (
    DeadNeuron, StuckSynapse, ThresholdFaultNeuron, BitflippedSynapse
)
from spikefi.fault import FaultSite, Fault
from spikefi.core import Campaign
from spikefi.utils.quantization import qargs_from_tensor
import demo


# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Load the network
net = demo.get_net(os.path.join(demo.DEMO_DIR, 'models', demo.get_fnetname()))

# Create a dataset loader for the testing set
test_loader = demo.get_loader(train=False)

# Create a SpikeFI Campaign object for network 'net'
# Spiking-related information is configured by the 'net.slayer' object
cmpn = Campaign(net, demo.shape_in, net.slayer, name='basic-fi')

# Create 3 different faults
# Fault 'fx' is a Dead Neuron in layer SF2
fx = Fault(
    DeadNeuron(),
    FaultSite('SF2', (3, 0, 0))
)
# Fault 'fy' is a Stuck Synapse fault between layers SF1 and SF2
fy = Fault(
    StuckSynapse(10.),
    FaultSite('SF1', (0, 0, 0, 0))
)
# Fault 'fz' is a multiple neuron parametric threshold fault in layer SF1
fz = Fault.multiple_random_absolute(ThresholdFaultNeuron(3), 4, 'SF1')

# Round 0: Inject fault fx (single-fault scenario)
cmpn.inject(fx)
# Round 1: Inject both fy and fz faults (multi-fault scenario)
cmpn.then_inject([fy, fz])

print(cmpn)

# Execute the fault injection experiments, applying all optimizations
cmpn.run(test_loader, opt=sfi.CampaignOptimization.O4)

# Show the results: the classification accuracy of each fault round
for r, perf in enumerate(cmpn.performance):
    print(f"Round {r} performance: {perf.testing.maxAccuracy * 100.0:.2f} %")

# Save the campaign and its results to 'out -> res -> basic-fi.pkl'
cmpn.save()

# Reset the campaign, i.e., remove all faults and fault rounds
cmpn.eject()

# Find scale and zero point for the quantization of the synaptic weights
# of layer 'SF2'
W = getattr(net, 'SF2').weight
scale, zero_point = qargs_from_tensor(W, torch.quint8)

# Create fault model 'fm' to be a Bitflipped Synapse with 8-bit quantized
# integer synaptic weights, targeting the bit 7 (MSB) to be flipped
fm = BitflippedSynapse(7, scale, zero_point, torch.quint8)
# Inject a bit-flipped synapse fault to
# every synapse of layer 'SF2', one at a time
cmpn.inject_complete(fm, ['SF2'])

# Execute the fault injection experiments targeting the synapses
# of whole layer, when only one is faulty at a time
cmpn.run(test_loader)

# Export campaign data to use with visualization tools
cmpn_data = cmpn.export()

# Visualize the results using a heatmap plot
# stored at 'out -> fig -> basic-fi_7_heat.png'
# The 'fig' object can be stored in a pickle file for later use/edit
preserve_dim = 'nmnist' in demo.case_study
fig = sfi.visual.heat(
    cmpn_data,
    preserve_dim=preserve_dim,
    format='png',
    title_suffix=demo.case_study
)

# Save the campaign and its results to 'out -> res -> bitflip_7_SF2.pkl'
cmpn.save(
    sfi.utils.io.make_res_filepath(
        demo.case_study + '_bitflip_7_SF2',
        rename=True
    )
)
