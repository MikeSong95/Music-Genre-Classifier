import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent

class MusicClassifier(nn.Module):
    def __init__(self):
        super(MusicClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 5,kernel_size= 6,stride = 2) 
        self.pool1 = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv2 = nn.Conv2d(in_channels = 5,out_channels = 10,kernel_size= 5,stride = 2)
        self.pool2 = nn.MaxPool2d(4, 4, 1) #kernel_size, stride, padding
        self.fc1 = nn.Linear(490, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 3)
      

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 490)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x