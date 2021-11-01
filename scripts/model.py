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

from sklearn.utils import shuffle


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, learning_rate, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
        self.z = nn.Linear(self.hidden_layers[self.H - 1], 1)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, state, action):
        x = torch.cat((state, action), axis = 1)

        for i in range(0, self.H):
            x = F.relu(self.fc[i](x))
        z = self.z(x)
        return z


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, learning_rate, device):
        super(Policy, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        for i in range(1, self.H):
            self.fc.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
        self.mu = nn.Linear(self.hidden_layers[self.H - 1], self.action_dim)
        self.sigma = nn.Linear(self.hidden_layers[self.H - 1], self.action_dim)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, state, action):
        x = torch.cat((state, action), axis = 1)

        for i in range(0, self.H):
            x = F.relu(self.fc[i](x))
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        return mu, sigma

