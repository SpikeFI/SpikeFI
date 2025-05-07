import torch
import spikefi as sfi
from spikefi.models import DeadNeuron, ParametricNeuron, SaturatedSynapse, BitflippedSynapse
from spikefi.fault import FaultSite, Fault
from spikefi.core import Campaign
import demo


# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Select the network file from directory, e.g., SpikeFI -> out -> net -> nmnist-lenet_net.py
net_path = sfi.utils.io.make_net_filepath(demo.get_fnetname())
# Load the network
net: demo.Network = torch.load(net_path, weights_only=False)
# Configure the network to the evaluation mode (network is already trained)
net.eval()

# Create a dataset loader for the testing set
test_loader = demo.get_loader('Test')

# Create a spikefi Campaign object for network 'net'
# Spiking-related information is configured by the 'net.slayer' object
cmpn = Campaign(net, demo.shape_in, net.slayer, name='basic-fi')

# Create 3 different faults
# Fault 'fx' is a Dead Neuron fault to randonmly target a neuron of layer SF2
fx = Fault(DeadNeuron(), FaultSite('SF2'))
# Fault 'fy' is a Saturated Synapse fault to randonmly target a synapse
# between layers SF1 and its predecessor and set the synaptic weight to 10
fy = Fault(SaturatedSynapse(10), FaultSite('SF1'))
# Fault 'fz' is a multiple (4) neuron parametric fault targetting the threshold parameter theta
# and setting it to half its nominal value in 4 neurons randomly selected across the entire network
fz = Fault(ParametricNeuron('theta', 0.5), [FaultSite(), FaultSite(), FaultSite(), FaultSite()])

# Round 0: Inject fault fx (single-fault scenario)
cmpn.inject([fx])
# Round 1: Inject both fy and fz faults (multiple-fault scenario)
cmpn.then_inject([fy, fz])

# Execute the fault injection experiments, applying all optimizations
cmpn.run(test_loader, opt=sfi.CampaignOptimization.O4)

# Show the results, i.e., the classification accuracy corresponding to each fault round
for r, perf in enumerate(cmpn.performance):
    print(f"Round {r} performance: {perf.testing.maxAccuracy * 100.0} %")

# Save the campaign and its results to spikefi -> out -> res -> basic-fi.pkl
cmpn.save()

# Reset the campaign, i.e., remove all faults and fault rounds
cmpn.eject()

# Find min and max synaptic weights of layer 'SF2' to use for
# the quantization of the layer's synaptic weights
layer = getattr(net, 'SF2')
wmin = layer.weight.min().item()
wmax = layer.weight.max().item()

# Create fault model 'fm' to be a Bitflipped Synapse with 8-bit quantized
# integer synaptic weights, targeting the bit 7 (MSB) to be flipped
fm = BitflippedSynapse(7, wmin, wmax, torch.uint8)
# Inject a bitflipped synapse fault to every synapse of layer 'SF2', one at a time
cmpn.inject_complete(fm, ['SF2'])

# Execute the fault injection experiments targeting the synapses of whole layer,
# when only one is faulty at a time
cmpn.run(test_loader)

# Visualize the results using a heatmap plot stored at spikefi -> fig -> basic-fi_7_heat.png
sfi.visual.heat(cmpn.export(), preserve_dim=True, format='png')

# Save the campaign and its results to spikefi -> out -> res -> bitflip_7_SF2.pkl
cmpn.save(sfi.utils.io.make_res_filepath('bitflip_7_SF2.pkl'))
