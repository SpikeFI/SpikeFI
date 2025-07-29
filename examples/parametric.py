#############################################################################
# An example to demonstrate a study of the parametric neuron faults for     #
# various neuron parameters used by the SRM neurons.                        #
#                                                                           #
# In the beginning, the configuration parameters are specified to describe  #
# the FI campaigns to be carried out. Namely, the 'layers' parameter sets   #
# the layer(s) to target; 'params' selects the SRM neuron parameter(s) to   #
# vary their value; and 'percent' defines the percentage of the faulty      #
# value(s) relatively to the nominal one, which is indicated by the         #
# network's yaml configuration file (e.g., demo -> config -> nmnist.yaml).  #
# Next, the demo environment is prepared for the FI experiments, the        #
# subject network is loaded, and a dataloader for the testing set is        #
# created. Continuing with the main part of the example, a nested for-loop  #
# iterates over all the combinations of the configuration parameters, where #
# the corresponding FI campaigns are set up and executed. Each FI campaign  #
# targets each of the neuron of the selected layer(s) individually, or in   #
# other words, creates a separate fault round for each faulty location. In  #
# addition, a sampling of maximum 250^2 neuron positions are to be injected #
# in the case of very large layers, so that the experiment time is reduced. #
# Those neuron locations are picked up randomly, while in the case that the #
# targeted layer is composed of less neurons, all of them will be subject   #
# of the FI experiments. After an FI campaign is over, the results are      #
# stored in a file and the example continues with the next one. When all    #
# the FI campaigns regarding the same SRM parameter finish, the results are #
# visualized collectively for these experiments using a comparative plot of #
# classification accuracy vs. parameter's deviation.                        #
#############################################################################


import os
import slayerSNN as snn
import spikefi as sfi
import demo


# Configuration parameters for the neuron parametric FI experiments
# Select one or more layers to target (use an empty string '' to target the whole network)
layers = ['SC2']    # For example: 'SF2', 'SF1', 'SC3', 'SC2', 'SC1', ''
# Select one or more neuron parameters to target
params = ['theta']  # For example: 'theta', 'tauSr', 'tauRef'
# Select the percentages of the parameter nominal values to set the faulty values
percent = range(10, 301, 10)    # For example: from 10% to 300% with a step of 10%

# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Load the network
net = demo.get_net(os.path.join(demo.DEMO_DIR, 'models', demo.get_fnetname()))
# Create a dataset loader for the testing set
test_loader = demo.get_loader(train=False)

# Calculate total number of FI campaigns
cmpns_total = len(params) * len(layers) * len(percent)
cmpns_count = 0

# For each targeted parameter
for param in params:
    cmpns_data = []
    # For each targeted layer
    for lay_name in layers:
        # For each faulty percentage of the nominal value
        for cent in percent:
            # Create a SpikeFI Campaign with a descriptive name
            cmpn_name = demo.get_fnetname().removesuffix('.pt') + f"_neuron_{param}_{lay_name or 'ALL'}_c{cent}"
            cmpn = sfi.Campaign(net, demo.shape_in, net.slayer, name=cmpn_name)
            cmpns_count += 1

            # Inject parametric neuron faults across all neurons of the layer
            # with a faulty parameter value set to cent% of the nominal value
            # Creates a separate fault round containing a single fault each
            cmpn.inject_complete(sfi.fm.ParametricNeuron(param, cent / 100.0), [lay_name])

            # Print status information
            print(f"Campaign {cmpns_count}/{cmpns_total}: '{cmpn.name}'")

            # Execute FI experiments for current targeted neuron parameter, layer, and faulty percentage
            cmpn.run(test_loader, spike_loss=snn.loss(demo.net_params).to(cmpn.device))

            # Print the duration of the FI campaign in seconds
            print(f"Duration: {cmpn.duration: .2f} secs")

            # Keep the campaign data aside to use later in the combinatorial plot
            cmpns_data.append(cmpn.export())

            # Save the campaign results in a pkl file
            cmpn.save()

    # Visualize results as a function of the parameter's value deviation
    # Creates one plot per neuron parameter
    # The 'fig' object can be stored in a pickle file for later use/edit
    fig = sfi.visual.plot(cmpns_data, xlabel=f"{param} (% of nominal value)", format='png')
