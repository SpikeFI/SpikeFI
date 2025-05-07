import numpy as np
import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs


EPOCHS_NUM = 20
C_START = 5
C_STOP = 35
C_STEP = 5

fm_type = sfi.fm.PerturbedSynapse
fm_name = "synapse_perturbed"
fm_argu = (0.5,)

# Generalized network/dataset initialization
device = torch.device('cuda')
net = cs.Network(cs.net_params, cs.DO_ENABLED).to(device)

spike_loss = snn.loss(cs.net_params).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)

fnetname = cs.get_fnetname(trial='')
cmpn_name = fnetname.removesuffix('.pt') + f"_train_{fm_name}_c{C_START}-{C_STOP}-{C_STEP}"
cmpn = sfi.Campaign(net, cs.shape_in, net.slayer, cmpn_name)

layers = cmpn.layers_info.get_injectables()[:-1]
for c in range(C_START, C_STOP + 1, C_STEP):
    fc = []
    for lay in [layers[-1]]:
        fm = fm_type(*fm_argu)
        sl = np.prod(cmpn.layers_info.get_shapes(fm.is_synaptic(), lay))
        fcl = sfi.ff.Fault.multiple_random(fm, int(c * sl / 100.), [lay])
        fc.append(fcl)

    cmpn.then_inject(fc)

print(cmpn)
faulties = cmpn.run_train(EPOCHS_NUM, cs.single_loader, optimizer, spike_loss)

for faulty in faulties:
    cmpn.save_faulty(faulty)

cmpn.save()

sfi.visual.learning_curve([cmpn.export()])
