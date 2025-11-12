#############################################################################
# An example for the demonstration of the SpikeFI optimization techniques   #
# and their effect in the fault simulation acceleration.                    #
#                                                                           #
# The configuration parameters in this example are the following six:       #
# 1. The targeted fault model to be used in the FI experiments;             #
# 2. The layers to be subject for the locations of the injected faults;     #
# 3. The batch size(s) for the dataset to compare;                          #
# 4. The number(s) of faults to be injected;                                #
# 5. The optimization setting(s) to compare;                                #
# 6. The tolerance of the Early Stop optimization (if applied).             #
# After preparing the demo environment, nested for-loops iterate over the   #
# targeted configuration parameters to create FI campaigns for all the FI   #
# experiments to be conducted. In the case that a selected fault number is  #
# less than the size of the targeted layer(s), fault sampling is applied.   #
# If more than one fault numbers are selected, then each FI campaign        #
# targeting the same layer(s), stars with the previous faults plus new ones #
# to meet the fault number target. If early stop optimization is applied    #
# and a tolerance other than zero is selected, a .csv file is stored with   #
# relative information on the number of critical faults. Finally, the       #
# results of each FI campaign are saved in separate pickle files.           #
#############################################################################


from copy import copy
import csv
import numpy as np
import os
from torch.utils.data import DataLoader
import slayerSNN as snn
import spikefi as sfi
import demo


# Configuration parameters for the optimizations example
# Select the fault model to be used in the FI experiments
f_model = sfi.fm.DeadNeuron()
# Select one or more layers to target (use an empty string '' to target the whole network)
layers = ['SF2']  # For example: 'SF2', 'SF1', 'SC3', 'SC2', 'SC1', ''
# Select the batch size(s)
s_batch = [10, 1]
# Select the number(s) of faults (0 means the max possible number = no fault sampling)
n_faults = [30, 100]  # For example: 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 500
# Select the SpikeFI optimizations
opts = [4, 0]  # 0 (no opt), 1 (opt loop order), 2 (late start), 3 (early stop), 4 (all)
# Set the tolerance for the Early Stop optimization (if applied)
es_tol = [0, 2]  # For example: range(71)

# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Load the network
net = demo.get_net(os.path.join(demo.DEMO_DIR, 'models', demo.get_fnetname()))

# Calculate total number of FI campaigns
cmpns_count = 0
cmpns_total = len(layers) * len(s_batch) * len(n_faults) * \
    np.sum(np.where(np.array(opts) >= sfi.CampaignOptimization.O3.value, len(es_tol), 1))

# For each targeted layer
for lay_name in layers:
    rounds = []  # Accumulated fault injection rounds
    rghist = {}  # Round groups histogram - frequencies of faults per layer

    # For each targeted number of faults to be injected
    for k in sorted(n_faults) or [0]:

        # For each targeted batch size
        for bs in s_batch:
            # Create a dataset loader for the testing set with the targeted batch size
            test_loader = DataLoader(dataset=demo.get_dataset(train=False), batch_size=bs, shuffle=demo.to_shuffle)

            # For each targeted optimization
            for o in opts:
                # Create a SpikeFI Campaign with a descriptive name
                cmpn_name = demo.get_fnetname().removesuffix('.pt') \
                    + f"_{f_model.get_name_snake_case()}_{lay_name or 'ALL'}_bs{bs}_k{k}_O{o}"
                cmpn = sfi.Campaign(net, demo.shape_in, net.slayer, name=cmpn_name)

                # Initialize the newly created campaign with the same fault rounds with its predecessor
                # (accumulated fault rounds for different parameter combinations regarding the same layer)
                cmpn.rounds = rounds

                if not k:
                    # If the targeted number of faults is set to 0, then all possible locations will be included to
                    # the FI experiment (no fault sampling case)
                    layer_names = [lay_name] if lay_name else cmpn.layers_info.get_injectables()
                    cmpn.inject_complete(f_model, layer_names)
                else:
                    # Fault sampling
                    # Accumulation of faults if the number of faults changes for the same layer
                    # (more than one numbers in the n_fault list)
                    k_actual = k - len(cmpn.rounds)  # Actual number of (new) faults to be injected

                    if k_actual > 0:
                        if lay_name:
                            # Try to inject k faults
                            cmpn.inject_complete(f_model, lay_name, fault_sampling_k=k_actual)

                            # Fault hyper-sampling
                            while k - len(cmpn.rounds) > 0:
                                cmpn.then_inject(sfi.ff.Fault(f_model, sfi.ff.FaultSite(lay_name)))
                        else:
                            # Equally distribute faults across layers
                            k_lay = int(k_actual / len(cmpn.layers_info.get_injectables()))
                            for lay in cmpn.layers_info.get_injectables():
                                n_lay = len(cmpn.inject_complete(f_model, lay, fault_sampling_k=k_lay))
                                rghist.setdefault(lay, 0)
                                rghist[lay] += n_lay

                            # Inject remaining faults
                            while k - len(cmpn.rounds) > 0:
                                min_lay = min(rghist, key=rghist.get)
                                cmpn.then_inject(sfi.ff.Fault(f_model, sfi.ff.FaultSite(min_lay)))
                                rghist[min_lay] += 1

                # Early Stop optimization tolerance
                durations = []
                N_critical = []

                use_es = o >= sfi.CampaignOptimization.O3.value
                actual_tol = es_tol if use_es and bool(es_tol) else [0]

                for t in actual_tol:
                    # Print status information
                    cmpns_count += 1
                    print(f"\nCampaign {cmpns_count}/{cmpns_total}: '{cmpn.name}'")
                    if use_es:
                        print(f"Early Stop tolerance: {t}")

                    # Execute the current FI experiments
                    cri = cmpn.run(test_loader, spike_loss=snn.loss(demo.net_params).to(cmpn.device),
                                   es_tol=t, opt=sfi.CampaignOptimization(o))

                    durations.append(cmpn.duration)
                    print(f"Campaign duration: {cmpn.duration: .2f} secs")

                    # Get the number of critical faults
                    if cri is not None:
                        N_critical.append(cri.sum().item())
                        print(f"Critical faults #: {N_critical[-1]}")

                # Store results regarding early stop tolerance for post-analysis
                if use_es and bool(es_tol):
                    with open(sfi.utils.io.make_res_filepath(cmpn_name + '_tol.csv', rename=True), mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['tolerance', 'duration', 'N_critical'])
                        writer.writerows(list(zip(es_tol, durations, N_critical)))

                # Store current fault rounds to be used in the next campaign if the targeted layer remains the same
                rounds = copy(cmpn.rounds)

                # Save FI campaign results in a pkl file
                cmpn.save()
