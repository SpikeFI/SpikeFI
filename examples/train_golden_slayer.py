#############################################################################
# An example of training a fault-free SNN with SLAYER.                      #
#                                                                           #
# For more information, please refer to the original examples in the        #
# SLAYER framework repository:                                              #
# https://github.com/bamsumit/slayerPytorch/tree/master/example             #
#############################################################################


from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tonic import transforms
import slayerSNN as snn
import spikefi.utils.io as sfio
import demo


# Number of training epochs
epochs = 100

# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='gesture', dev=torch.device('cuda:3'))

# Create a network instance
net = demo.Network(demo.net_params).to(demo.device)
trial = demo.get_trial()

# Create the dataset loaders for the training and testing sets
train_loader = DataLoader(
    demo.get_cached_dataset(
        train=True,
        transform=transforms.Denoise(filter_time=10000)
    ),
    batch_size=8, shuffle=True,
    num_workers=4, pin_memory=True,
    persistent_workers=True
)
test_loader = DataLoader(
    demo.get_cached_dataset(
        train=False,
        transform=transforms.Denoise(filter_time=10000)
    ),
    batch_size=4, shuffle=False,
    num_workers=4, pin_memory=True
)

print("Training configuration:")
print(f"  - network: {demo.case_study}")
print(f"  - epochs: {epochs}")
print(f"  - trial: {trial}")
print(f"  - Ts: {demo.net_params['simulation']['Ts']} ms")
print()

# SNN loss
spike_loss = snn.loss(demo.net_params).to(demo.device)

# Optimizer module
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)

# TODO: Replace slayer statistics with custom metric
# TODO: Update training and testing loops

# Learning statistics
stats = snn.utils.stats()

for epoch in range(epochs):
    tSt = datetime.now()

    # Training loop
    net.train()
    for i, (input, label) in enumerate(train_loader, 0):
        input = input.to(demo.device, non_blocking=True)
        label = label.to(demo.device, non_blocking=True)

        optimizer.zero_grad()

        output = net(input)

        # One-hot vector for labels: target[b, label[b], 0, 0, 0] = 1
        target = (
            torch.zeros_like(output[..., :1])
            .scatter_(1, label.view(-1, 1, 1, 1, 1), 1.0)
        )

        loss = spike_loss.numSpikes(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predict = output.sum(dim=(2, 3, 4)).argmax(dim=1)
            correct = (predict == label).sum().item()
            batch_s = label.size(0)

            stats.training.correctSamples += correct
            stats.training.numSamples += batch_s

            stats.training.lossSum += loss.detach().item() * batch_s

        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    scheduler.step()

    # Testing loop
    net.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader, 0):
            input = input.to(demo.device, non_blocking=True)
            label = label.to(demo.device, non_blocking=True)

            output = net.forward(input)

            # One-hot vector for labels: target[b, label[b], 0, 0, 0] = 1
            target = (
                torch.zeros_like(output[..., :1])
                .scatter_(1, label.view(-1, 1, 1, 1, 1), 1.0)
            )

            predict = output.sum(dim=(2, 3, 4)).argmax(dim=1)
            correct = (predict == label).sum().item()
            batch_s = label.size(0)

            stats.testing.correctSamples += correct
            stats.testing.numSamples += batch_s

            loss = spike_loss.numSpikes(output, target)
            stats.testing.lossSum += loss.detach().item() * batch_s

            stats.print(epoch, i)

    stats.update()

    # Save the trained network instance with the best testing accuracy
    accu = np.asarray(stats.testing.accuracyLog, dtype=float)
    rev_i = np.nanargmax(accu[::-1])
    last_max_epoch = accu.size - 1 - rev_i

    if last_max_epoch == epoch:
        torch.save(
            net.state_dict(),
            sfio.make_net_filepath(demo.get_fnetname(trial))
        )

# Save stats in a pickle file
with open(
    sfio.make_out_filepath(demo.get_fstaname(trial)), 'wb'
) as stats_file:
    pickle.dump(stats, stats_file)

# Plot the training results (learning curves) and save in a .png file
plt.figure()
plt.plot(
    range(1, epochs + 1),
    torch.Tensor(stats.training.accuracyLog) * 100., 'b--', label='Training'
)
plt.plot(
    range(1, epochs + 1),
    torch.Tensor(stats.testing.accuracyLog) * 100., 'g-', label='Testing'
)
plt.xlabel('Epoch #')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.xticks(ticks=[1] + list(range(10, epochs + 1, 10)))
plt.xticks(ticks=range(2, epochs + 1, 2), minor=True)
plt.yticks(ticks=range(0, 101, 10))
plt.yticks(ticks=range(0, 100, 2), minor=True)
plt.grid(visible=True, which='both', axis='both')
plt.xlim((1, epochs))
plt.ylim((0., 100.))
plt.savefig(sfio.make_fig_filepath(demo.get_ffigname(trial, format='png')))
