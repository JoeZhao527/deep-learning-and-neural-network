# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(784,10)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.fc1(x))
        return x
        

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.in_hid  = nn.Linear(784,120)
        self.out_hid = nn.Linear(120,10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.tanh(self.in_hid(x))
        x = F.log_softmax(self.out_hid(x))
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1,15,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(15,150,5)
        self.fc1 = nn.Linear(2400,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = F.log_softmax(self.fc1(x))
        return x