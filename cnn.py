from torch import nn


# define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # input 256 x 256 x 3

        # Layer 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(2, 2)  # output: 128 x 128 x 16

        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same', dilation=2)
        self.bn2 = nn.BatchNorm2d(32)
        # output: 64 x 64 x 32

        # Layer 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        # output: 32 x 32 x 64

        # Layer 4
        self.conv4 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        # output: 16 x 16 x 128

        # Layer Connecting
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*16*128, 512)
        self.fc2 = nn.Linear(512, 21)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)
        x = self.mp1(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.bn3(x)
        x = self.mp1(x)

        x = self.conv4(x)
        x = self.act(x)
        x = self.bn4(x)
        x = self.mp1(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
