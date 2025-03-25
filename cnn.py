from torch import nn
import torch.nn.functional as F


# define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1,10,5)    # Convolutional layer
        self.bn1 = nn.BatchNorm2d(10)     # batch normalization
        self.mp1 = nn.MaxPool2d(2, 2)     # Max Pooling

        self.conv2 = nn.Conv2d(10,20,3)
        self.bn2 = nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)
        # in: batch*1*28*28, out: batch*10*24*24  -- (28-5+1)
        x = self.conv1(x)
        x = F.relu(x)  # does not change the input size
        x = self.bn1(x)  # does not change the input size
        # in: batch*10*24*24, out: batch*10*12*12
        x = self.mp1(x)

        # in: batch*10*12*12, out: batch*20*10*10  -- (12-3+1)
        x = self.conv2(x)
        x = F.relu(x)

        # 20*10*10 = 2000
        x = x.view(input_size,-1)

        # in: batch*2000  out:batch*500
        x = self.fc1(x)
        x = F.relu(x)

        # in:batch*500 out:batch*10
        x = self.fc2(x)
        return x
