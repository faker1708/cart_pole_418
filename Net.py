
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle


class Net(nn.Module):
    def __init__(self, ):




        super(Net, self).__init__()
        # 64 32

        N_ACTIONS = 2
        N_STATES = 4

        self.fc1 = nn.Linear(N_STATES, 32)#.cuda()
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(32, N_ACTIONS)#.cuda()
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = x#.cuda()
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        actions_value = self.out(x)
        actions_value = actions_value.cpu()
        # actions_value.type(torch.float32)


        return actions_value

