import os
import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs


L = ['SF2']     # 'SF2', 'SF1', 'SC3', 'SC2', 'SC1', ''
B = range(8)    # LSB: bit 0
qdtype = torch.uint8

fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for lay_name in L:
    layer = getattr(net, lay_name)
    wmin = layer.weight.min().item()
    wmax = layer.weight.max().item()
    for b in B:
        cmpn_name = fnetname.removesuffix('.pt') + f"_synapse_bitflip_{lay_name or 'ALL'}_b{b}"
        cmpn = sfi.Campaign(net, cs.shape_in, net.slayer, name=cmpn_name)

        cmpn.inject_complete(sfi.fm.BitflippedSynapse(b, wmin, wmax, qdtype),
                             [lay_name], fault_sampling_k=250**2)

        print(cmpn.name)
        cmpn.run(cs.test_loader, spike_loss=snn.loss(cs.net_params).to(cmpn.device))
        print(f"{cmpn.duration : .2f} secs")

        sfi.visual.heat(cmpn.export(), preserve_dim=True, format='png')
        cmpn.save()
