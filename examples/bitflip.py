#############################################################################
# An example to demonstrate a study of the network's resiliency to bitflips #
# in the synaptic weights and the importance of the bit positions.          #
#                                                                           #
# First, the configuration parameters defining the FI campaigns, need to be #
# set. The 'layers' parameter selects one or more layers subject to the FI  #
# experiments; the 'bits' parameter selects the targeted bit positions; and #
# finally, the 'qdtype' parameter selects the precision of the quantized    #
# data type. Next, preparing the demo environment, loading the network,     #
# and creating a data loader for the testing set, a nested for loop         #
# iterates over the targeted layers and bit positions one at a time, in     #
# order to create a separate campaign object for each combination. Each FI  #
# campaign then injects a sample of maximum 250^2 synaptic weights with a   #
# bit-flipped synapse error to the synapses of the current layer at the     #
# current bit position. Each fault is injected as a distinct fault round    #
# containing a single fault each, so that the network's reliability is      #
# examined with perspective to each synaptic weight and faulty bit position.#
# At the end of each FI campaign execution, results are stored in a file    #
# and are visualized using a heat map plot, which is also saved as an image.#
#############################################################################


import os
import torch
import slayerSNN as snn
import spikefi as sfi
from spikefi.utils.quantization import qargs_from_tensor
import demo


# Configuration parameters for the bitflip FI experiments
# Select one or more layers to target (use an empty string '' to target the whole network)
layers = ['SF2']    # For example: 'SF2', 'SF1', 'SC3', 'SC2', 'SC1', ''
# Select the bit positions to target
bits = range(8)     # LSB is bit 0
# Select the precision of the quantized integer synaptic weights
qdtype = torch.quint8

# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Load the network
net = demo.get_net(os.path.join(demo.DEMO_DIR, 'models', demo.get_fnetname()))
# Create a dataset loader for the testing set
test_loader = demo.get_loader(train=False)

# Calculate total number of FI campaigns
cmpns_total = len(layers) * len(bits)
cmpns_count = 0

# For each targeted layer
for lay_name in layers:
    # Find scale and zero point for the quantization of the synaptic weights of the layer
    W = getattr(net, lay_name).weight
    scale, zero_point, _ = qargs_from_tensor(W, qdtype)

    # For each targeted bit position
    for b in bits:
        # Create a SpikeFI Campaign with a descriptive name
        cmpn_name = demo.get_fnetname().removesuffix('.pt') + f"_synapse_bitflip_{lay_name or 'ALL'}_b{b}"
        cmpn = sfi.Campaign(net, demo.shape_in, net.slayer, name=cmpn_name)
        cmpns_count += 1

        # Inject bitflipped synapse faults across 250^2 randomly selected synaptic weights in the layer
        # Creates a separate fault round containing a single fault each
        cmpn.inject_complete(sfi.fm.BitflippedSynapse(b, scale, zero_point, qdtype),
                             [lay_name], fault_sampling_k=250**2)

        # Print status information
        print(f"Campaign {cmpns_count}/{cmpns_total}: '{cmpn.name}'")

        # Execute FI experiments for current targeted layer and bit position
        cmpn.run(test_loader, spike_loss=snn.loss(demo.net_params).to(cmpn.device))

        # Visualize results with a heat map
        # The 'fig' object can be stored in a pickle file for later use/edit
        preserve_dim = 'nmnist' in demo.case_study
        fig = sfi.visual.heat(cmpn.export(), preserve_dim=preserve_dim, format='png')

        # Save results in a pkl file
        cmpn.save()
