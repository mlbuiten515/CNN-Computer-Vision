import numpy as np
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

print(len(train_indices), len(val_indices), len(test_incices))


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

num_epochs = 10
for epoch in range(num_epochs):
    # Train:   
    for batch_index, (faces, labels) in enumerate(train_loader):
        # ...'
