from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
import torch

import slayerSNN as snn

import demo as cs

EPOCHS_NUM = 200

# Generalized network/dataset initialization
device = torch.device('cuda')
net = cs.Network(cs.net_params, cs.DO_ENABLED).to(device)
trial = cs.trial_def

spike_loss = snn.loss(cs.net_params).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)
stats = snn.utils.stats()

print("Training configuration:")
print(f"  - case study: {cs.CASE_STUDY}")
print(f"  - dropout: {'yes' if cs.DO_ENABLED else 'no'}")
print(f"  - epochs num: {EPOCHS_NUM}")
print(f"  - trial: {trial or 0}")
print()

for epoch in range(EPOCHS_NUM):
    tSt = datetime.now()

    for i, (_, input, target, label) in enumerate(cs.train_loader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input)

        stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.training.numSamples += len(label)

        loss = spike_loss.numSpikes(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        stats.training.lossSum += loss.cpu().data.item()
        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    # Testing
    for i, (_, input, target, label) in enumerate(cs.test_loader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = spike_loss.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(epoch, i)

    stats.update()

    # Save trained network (based on the best testing accuracy)
    if stats.testing.accuracyLog[-1] == stats.testing.maxAccuracy:
        torch.save(net, os.path.join(cs.OUT_DIR, cs.get_fnetname(trial)))

# Save statistics
with open(os.path.join(cs.OUT_DIR, cs.get_fstaname(trial)), 'wb') as stats_file:
    pickle.dump(stats, stats_file)

# Plot and save the training results
plt.figure()
plt.plot(range(1, EPOCHS_NUM + 1), torch.Tensor(stats.training.accuracyLog) * 100., 'b--', label='Training')
plt.plot(range(1, EPOCHS_NUM + 1), torch.Tensor(stats.testing.accuracyLog) * 100., 'g-', label='Testing')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.xticks(ticks=[1] + list(range(10, EPOCHS_NUM + 1, 10)))
plt.xticks(ticks=range(2, EPOCHS_NUM + 1, 2), minor=True)
plt.yticks(ticks=range(0, 101, 10))
plt.yticks(ticks=range(0, 100, 2), minor=True)
plt.grid(visible=True, which='both', axis='both')
plt.xlim((1, EPOCHS_NUM))
plt.ylim((0., 100.))
plt.savefig(os.path.join(cs.OUT_DIR, cs.get_ffigname(trial)))
