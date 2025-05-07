from copy import copy
import csv
import os
import torch
from torch.utils.data import DataLoader

import slayerSNN as snn

import spikefi as sfi
import demo as cs


fm_type = sfi.fm.DeadNeuron
fm_name = "neuron_dead"
L = ['SC1']  # 'SF2', 'SF1', 'SC3', 'SC2', 'SC1', ''
B = [1]
K = [30]  # 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 500
O = [3]  # noqa E741
T = range(71)

fnetname = cs.get_fnetname()
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for lay_name in L:
    rounds = []  # Accumulated fault injection rounds
    rghist = {}  # Round groups histogram - frequencies of faults per layer
    for k in sorted(K) or [0]:
        for bs in B:
            test_loader = DataLoader(dataset=cs.test_set, batch_size=bs, shuffle=cs.shuffle)
            for o in O:
                cmpn_name = fnetname.removesuffix('.pt') + f"_{fm_name}_{lay_name or 'ALL'}_bs{bs}_k{k}_O{o}"
                cmpn = sfi.Campaign(net, cs.shape_in, net.slayer, name=cmpn_name)
                cmpn.rounds = rounds

                # Fault sampling
                if not k:
                    layer_names = [lay_name] if lay_name else cmpn.layers_info.get_injectables()
                    cmpn.inject_complete(fm_type(), layer_names)
                else:
                    # Accumulation of faults if k changes for the same layer
                    # Actual number of (new) faults to be injected
                    k_actual = k - len(cmpn.rounds)
                    if k_actual > 0:
                        shapes = cmpn.layers_info.shapes_syn if fm_type().is_synaptic() else cmpn.layers_info.shapes_neu

                        if lay_name:
                            # Try to inject k faults
                            cmpn.inject_complete(fm_type(), [lay_name], fault_sampling_k=k_actual)

                            # Fault hyper-sampling
                            while k - len(cmpn.rounds) > 0:
                                cmpn.then_inject([sfi.ff.Fault(fm_type(), sfi.ff.FaultSite(lay_name))])
                        else:
                            # Equally distribute faults across layers
                            k_lay = int(k_actual / len(cmpn.layers_info.get_injectables()))
                            for lay in cmpn.layers_info.get_injectables():
                                n_lay = len(cmpn.inject_complete(fm_type(), [lay], fault_sampling_k=k_lay))
                                rghist.setdefault(lay, 0)
                                rghist[lay] += n_lay

                            # Inject remaining faults
                            while k - len(cmpn.rounds) > 0:
                                min_lay = min(rghist, key=rghist.get)
                                cmpn.then_inject([sfi.ff.Fault(fm_type(), sfi.ff.FaultSite(min_lay))])
                                rghist[min_lay] += 1

                print(cmpn.name)

                # Early-Stop tolerance
                durations = []
                N_critical = []
                for t in T or [0]:
                    cri = cmpn.run(test_loader, spike_loss=snn.loss(cs.net_params).to(cmpn.device),
                                   es_tol=t, opt=sfi.CampaignOptimization(o))

                    durations.append(cmpn.duration)
                    print(f"Campaign duration: {cmpn.duration : .2f} secs")

                    if cri is not None:
                        N_critical.append(cri.sum().item())
                        print(f"Critical faults #: {N_critical[-1]}")

                if len(T) > 1:
                    with open(os.path.join(sfi.utils.io.RES_DIR, cmpn_name + '_tol.csv'), mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['tolerance', 'duration', 'N_critical'])
                        writer.writerows(list(zip(T, durations, N_critical)))

                rounds = copy(cmpn.rounds)
                cmpn.save()
