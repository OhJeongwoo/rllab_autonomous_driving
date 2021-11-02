#!/usr/bin/env python
from __future__ import print_function

##### add python path #####
import sys
import os

from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.utils import shuffle

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
EPS = 1e-6

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, learning_rate):
        super(MLP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        self.z = nn.Linear(self.hidden_layers[self.H - 1], self.action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()

    def forward(self, z, state):
        # concatenate state and action
        x = torch.cat((state, z), axis = 1)

        # forward network and return
        for i in range(0,self.H):
            x = F.relu(self.fc[i](x))
        z = self.z(x)
        return z


    def mlp_loss(self, z, action):
        # z: N * A
        # action: N * A
        return self.loss(z, action)