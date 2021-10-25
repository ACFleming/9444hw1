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
        print("PARAMS:", sum(p.numel() for p in self.parameters() if p.requires_grad))



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
        print("PARAMS:", sum(p.numel() for p in self.parameters() if p.requires_grad))


    def forward(self, input):
        self.hid1 = torch.tanh(self._layer1(input))
        self.hid2 = torch.tanh(self._layer2(self.hid1))
        self.hid3 = torch.tanh(self._layer3(self.hid2))
        input = torch.sigmoid(self._layer_output(self.hid3))
        return input[:,0].view(-1,1)


# Reached 100 with python frac_main.py --net dense --hid 15 --init 0.35

#python frac_main.py --net dense --hid 14 --lr 0.008 --init 0.5
class DenseNet(torch.nn.Module):
    def __init__(self, hid):
        super(DenseNet, self).__init__()
        self._layer1 = nn.Linear(2,hid)
        self._layer2 = nn.Linear(2+ hid,hid)
        self._layer_output = nn.Linear(2 + hid + hid,1)
        print("PARAMS:", sum(p.numel() for p in self.parameters() if p.requires_grad))


    def forward(self, input):
        # print("INPUT:", input.size())
        self.hid1 = torch.tanh(self._layer1(input))
        # print("HID 1:", self.hid1.size())
        input2 = torch.cat((input,self.hid1),dim=1)
        # print("INPUT 2:", input2.size())
        self.hid2 = torch.tanh(self._layer2(input2))
        # print("HID 2", self.hid2.size())
        input3 = torch.cat((input,self.hid1, self.hid2),dim=1)
        # print("INPUT 3:", input3.size())
        output = torch.sigmoid(self._layer_output(input3))
        # print("OUTPUT:", output.size())
        return output[:,0].view(-1,1)
        
