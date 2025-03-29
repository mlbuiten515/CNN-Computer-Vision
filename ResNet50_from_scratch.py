import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from data_processing import ImagesDataset
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image

# Data Management
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

# Define the residual block class
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    # Define the forward pass of the block
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)
        return x

# ResNet50 [3, 4, 6, 3]
class ResNet(nn.Module): 
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                        stride=stride), nn.BatchNorm2d(out_channels * 4))
            
            #the layer which change the number of channels, out_channels gonna be 64 * 4 = 256
            layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
            self.in_channels = out_channels * 4 
            
            for i in range(num_residual_blocks - 1):
                layers.append(block(self.in_channels, out_channels)) #256 -> 64, 64*4(256) again
                
            return nn.Sequential(*layers)
        
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def test():
    net = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cpu')
    print(y.shape)
    
test()


# Your existing block and ResNet classes remain the same
# [Your block and ResNet classes from above go here]
# Modified Dataset class
class AerialDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Use PIL to open the image instead of matplotlib
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Convert to numpy array and make it writable
        image = np.array(image)
        if not image.flags.writeable:
            image = image.copy()
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor

# [Keep your existing train_model and plot_metrics functions]

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # Your specific dataset path
    data_dir = '/Users/Frankie_C/Documents/GitHub/CNN-Computer-Vision/UCMerced_LandUse/images'
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if not d.startswith('.')])
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    # Load image paths and labels
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            if img_name.endswith(('.tif', '.tiff')):  # UCMerced uses TIFF
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(class_to_idx[class_name])
    
    # Verify we have 2100 images and 21 classes
    print(f"Total images: {len(image_paths)}")
    print(f"Number of classes: {len(set(labels))}")
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # This handles the numpy to tensor conversion properly
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AerialDataset(train_paths, train_labels, transform=transform)
    val_dataset = AerialDataset(val_paths, val_labels, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model - ensure num_classes matches your dataset
    model = ResNet50(img_channels=3, num_classes=len(class_names)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Plot results
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Save the model
    torch.save(model.state_dict(), 'resnet50_ucmerced.pth')

if __name__ == '__main__':
    main()