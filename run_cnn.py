import numpy as np
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from cnn import CNN
from data_processing import ImagesDataset


def data_loaders(dataset, batch_size):
    random_seed = 0
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_train = int(np.floor(0.7 * dataset_size))
    split_val = split_train + int(np.floor(0.1 * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices = indices[:split_train]
    val_indices = indices[split_train:split_val]
    test_incices = indices[split_val:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_incices)

    train_loader = DataLoader(alldata, batch_size=batch_size,
                              sampler=train_sampler)
    validation_loader = DataLoader(alldata, batch_size=batch_size,
                                   sampler=valid_sampler)
    test_loader = DataLoader(alldata, batch_size=batch_size,
                             sampler=test_sampler)

    return train_loader, validation_loader, test_loader


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
            epoch_idx, correct, len(test_loader.sampler),
            100. * correct / len(test_loader.sampler)))

    if set_type == "test":
        print('\nEpoch{}: Validation accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, correct, len(test_loader.sampler),
            100. * correct / len(test_loader.sampler)))
    return correct / len(test_loader.sampler)


def train(num_epochs, model, train_loader, validation_loader,
          optimizer, loss_function):

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                               factor=0.5, patience=3,
                                               verbose=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"-------------------------------\nEpoch {epoch}")

        for batch_index, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(data)
            loss = loss_function(output, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(f'Train Loss: {epoch_loss:.4f}')
        train_accuracy = accuracy(epoch, train_loader, model, set_type="train")
        val_accuracy = accuracy(epoch, validation_loader, model, set_type="test")
        scheduler.step(val_accuracy)

    return train_accuracy, val_accuracy


def test(model, test_loader):
    model.eval()

    correct = 0
    length = len(test_loader.sampler)
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / length
    print(f'Final test accuracy: {correct}/{length} ({accuracy:.0f}%)')
    return accuracy


if __name__ == '__main__':

    batch_size = 32
    learning_rate = 1e-3
    wd = 1e-4
    num_epochs = 15

    model = CNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate,
                     weight_decay=wd)

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ColorJitter(0.2, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ])
    alldata = ImagesDataset(r'UCMerced_LandUse\Images', transform)

    train_loader, val_loader, test_loader = data_loaders(alldata, batch_size)
    final_train, final_val = train(num_epochs, model, train_loader, val_loader,
                                   optimizer, loss_function)

    test_accuracy = test(model, test_loader)
