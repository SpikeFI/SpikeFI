#############################################################################
# An example of training a fault-free SNN with SLAYER.                      #
#                                                                           #
# For more information, please refer to the original examples in the        #
# SLAYER framework repository:                                              #
# https://github.com/bamsumit/slayerPytorch/tree/master/example             #
#############################################################################


from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import torch
import slayerSNN as snn
import spikefi.utils.io as sfio
import demo


# Number of training epochs
n_epochs = 2

# Setup the fault simulation demo environment
# Selects the case study, e.g., the LeNet network without dropout
demo.prepare(casestudy='nmnist-lenet', dropout=False)

# Create a network instance
net = demo.Network(demo.net_params, demo.dropout_en).to(demo.device)
trial = demo.get_trial()

# Create the dataset loaders for the training and testing sets
train_loader = demo.get_loader(train=True)
test_loader = demo.get_loader(train=False)

print("Training configuration:")
print(f"  - case study: {demo.case_study}")
print(f"  - dropout: {'yes' if demo.dropout_en else 'no'}")
print(f"  - epochs number: {n_epochs}")
print(f"  - trial: {trial}")
print()

# SNN loss
spike_loss = snn.loss(demo.net_params).to(demo.device)
# Optimizer module
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)

# Learning statistics
stats = snn.utils.stats()

for epoch in range(n_epochs):
    tSt = datetime.now()

    # Training loop
    for i, (_, input, target, label) in enumerate(train_loader, 0):
        input = input.to(demo.device)
        target = target.to(demo.device)

        # Forward pass
        output = net.forward(input)

        # Gather the training stats
        stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.training.numSamples += len(label)

        # Calculate loss
        loss = spike_loss.numSpikes(output, target)

        # Reset gradients to zero
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Gather training loss stats
        stats.training.lossSum += loss.cpu().data.item()
        # Display training stats
        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    # Testing loop
    # Same steps as in training loop except loss backpropagation and weight update
    for i, (_, input, target, label) in enumerate(test_loader, 0):
        input = input.to(demo.device)
        target = target.to(demo.device)

        output = net.forward(input)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = spike_loss.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(epoch, i)

    # Update stats
    stats.update()

    # Save the trained network instance with the best testing accuracy
    if stats.testing.accuracyLog[-1] == stats.testing.maxAccuracy:
        torch.save(net.state_dict(), sfio.make_net_filepath(demo.get_fnetname(trial)))

# Save stats in a pickle file
with open(sfio.make_out_filepath(demo.get_fstaname(trial)), 'wb') as stats_file:
    pickle.dump(stats, stats_file)

# Plot the training results (learning curves) and save in a .png file
plt.figure()
plt.plot(range(1, n_epochs + 1), torch.Tensor(stats.training.accuracyLog) * 100., 'b--', label='Training')
plt.plot(range(1, n_epochs + 1), torch.Tensor(stats.testing.accuracyLog) * 100., 'g-', label='Testing')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.xticks(ticks=[1] + list(range(10, n_epochs + 1, 10)))
plt.xticks(ticks=range(2, n_epochs + 1, 2), minor=True)
plt.yticks(ticks=range(0, 101, 10))
plt.yticks(ticks=range(0, 100, 2), minor=True)
plt.grid(visible=True, which='both', axis='both')
plt.xlim((1, n_epochs))
plt.ylim((0., 100.))
plt.savefig(sfio.make_fig_filepath(demo.get_ffigname(trial, format='png')))
