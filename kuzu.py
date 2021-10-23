"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        

        #input is an image 28x28, output is a class selection of 10 options
        self._layer1 = nn.Linear(28*28,10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self._layer1(x)
        return F.log_softmax(x, dim=1)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self._layer1 = nn.Linear(28*28,170)
        self._layer2 = nn.Linear(170,10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self._layer1(x)
        x = torch.tanh(x)
        x = self._layer2(x)
        return F.log_softmax(x, dim=1)
         

#Note: code used to run was python kuzu_main.py --net conv --mom 0.3 --lr 0.05 > p1q3.txt

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self._conv_layer1 = nn.Conv2d(1, 16, 5)
        
        self._conv_layer2 = nn.Conv2d(16, 32, 5)
        
        self._linear_layer3 = nn.Linear(512,128)
        
        self._linear_layer4 = nn.Linear(128,10)
        # print("PARAMS:", sum(p.numel() for p in self.parameters() if p.requires_grad))


    def forward(self, x):
        x = self._conv_layer1(x)
        x = torch.relu(x)
        # print("HERE")
        # print(x.shape)
        x = torch.max_pool2d(x,2, stride=2)
        # print("HERE2")
        # print(x.shape)
        x = self._conv_layer2(x)
        x = torch.relu(x)
        # print("HERE3")
        # print(x.shape)
        x = torch.max_pool2d(x,2,stride=2)
        # print("HERE4")
        # print(x.shape)
        x = torch.flatten(x,1)
        # print("HERE5")
        # print(x.shape)
        x = self._linear_layer3(x)
        x = torch.relu(x)
        # print("HERE6")
        # print(x.shape)
        x = self._linear_layer4(x)
        # print("HERE7")
        # print(x.shape)
        return torch.log_softmax(x, dim=1)

        return 0 # CHANGE CODE HERE
