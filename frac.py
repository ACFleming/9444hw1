"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Reached 100 with python frac_main.py --net full2 --hid 15 --lr 0.001 --init 0.35

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self._layer1 = nn.Linear(2,hid)
        self._layer2 = nn.Linear(hid,hid)
        self._layer_output = nn.Linear(hid,1)

    def forward(self, input):

        self.hid1 = torch.tanh(self._layer1(input))
        self.hid2 = torch.tanh(self._layer2(self.hid1))
        input = torch.sigmoid(self._layer_output(self.hid2))
        return input[:,0].view(-1,1)


# Reached 100 with python frac_main.py --net full3 --hid 15 --init 0.35
class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self._layer1 = nn.Linear(2,hid)
        self._layer2 = nn.Linear(hid,hid)
        self._layer3 = nn.Linear(hid,hid)
        self._layer_output = nn.Linear(hid,1)

    def forward(self, input):
        self.hid1 = torch.tanh(self._layer1(input))
        self.hid2 = torch.tanh(self._layer2(self.hid1))
        self.hid3 = torch.tanh(self._layer3(self.hid2))
        input = torch.sigmoid(self._layer_output(self.hid3))
        return input[:,0].view(-1,1)

class DenseNet(torch.nn.Module):
    def __init__(self, hid):
        super(DenseNet, self).__init__()
        self._layer1 = nn.Linear(2,hid)
        self._layer2 = nn.Linear(hid,hid)
        self._layer_output = nn.Linear(hid,1)

    def forward(self, input):
        self.hid1 = torch.tanh(self._layer1(input))
        self.hid2 = torch.tanh(torch.cat(   self._layer2(input),\
                                            self._layer2(self.hid1),dim=1))
        input = torch.sigmoid(torch.cat(    self._layer_output(input),\
                                            self._layer_output(self.hid1),\
                                            self._layer_output(self.hid2), dim=1))
        return input[:,0].view(-1,1)
