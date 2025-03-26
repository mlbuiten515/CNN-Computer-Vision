import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from cnn import CNN
from data_processing import ImagesDataset

alldata = ImagesDataset(r'UCMerced_LandUse\Images')

batch_size = 16
random_seed = 42

dataset_size = len(alldata)
indices = list(range(dataset_size))
split_train = int(np.floor(0.7 * dataset_size))
split_val = split_train + int(np.floor(0.1 * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices, test_incices = indices[:split_train], indices[split_train:split_val], indices[split_val:]


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_incices)

train_loader = DataLoader(alldata, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = DataLoader(alldata, batch_size=batch_size,
                                                sampler=valid_sampler)
test_loader = DataLoader(alldata, batch_size=batch_size,
                                                sampler=test_sampler)

model = CNN()


def accuracy(epoch_idx, test_loader, model, set_type=None):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    if set_type == "train":
        print('\nEpoch{}: Train accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, correct, split_train,
            100. * correct / split_train))

    if set_type == "test":
        print('\nEpoch{}: Validation accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, correct, (split_val - split_train),
            100. * correct / (split_val - split_train)))

    return correct / len(test_loader.dataset)


learning_rate = 1e-3
num_epochs = 20

loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"-------------------------------\nEpoch {epoch}")
    for batch_index, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)
        loss = loss_function(output, labels)
        loss.backward()

        optimizer.step()

    train_accuracy = accuracy(epoch, train_loader, model, set_type="train")
    val_accuracy = accuracy(epoch, validation_loader, model, set_type="test")
