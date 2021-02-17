import torch.nn as nn
import torch
import numpy as np

class Critic(nn.Module):
    def __init__(self,num_action,num_states):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(num_states,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,num_action)
        self.relu = nn.ReLU()
        print(num_states)
    def forward(self,x):
        print(x.size())
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x
class Actor(nn.Module):
    def __init__(self,num_action,num_states,batch_size):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(num_states,32)
        self.gru1 = nn.GRUCell(32,32)
        self.fc2 = nn.Linear(32,num_action)
        self.relu = nn.ReLU()
    def forward(self,x,h0):
        x = self.fc1(x)
        x = self.relu(x)
        h_in = h0.reshape(-1,32)
        h = self.gru1(x,h_in)
        x = self.fc2(h)
        return x,h
    def init_hidden(self):
        return self.fc1.weight.new(1,32).zero_()
