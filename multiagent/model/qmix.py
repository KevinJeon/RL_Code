import torch.nn as nn
import torch
import numpy as np
class Q_agent(nn.Module):
    def __init__(self,num_obs,num_action):
        super(Q_agent,self).__init__()
        self.obs = num_obs
        self.fc1 = nn.Linear(num_obs,32)
        self.gru1 = nn.GRUCell(32,32)
        self.fc2 = nn.Linear(32,num_action)
        self.relu = nn.ReLU()
    def init_hidden(self,batch_size):
        return self.fc1.weight.new(batch_size,32).zero_()
    def forward(self,x,h0):
        x = self.relu(self.fc1(x))
        #print(x.size())
        h = self.gru1(x,h0)
        q = self.fc2(h)
        return q,h

# QMix
class QMix(nn.Module):
    def __init__(self,num_agent,state_dim,embed_dim):
        super(QMix,self).__init__()
        self.num_agent = num_agent
        self.embed_dim = embed_dim
        self.w1 = nn.Linear(state_dim,embed_dim*num_agent)
        self.b1 = nn.Linear(state_dim,embed_dim)
        self.w2 = nn.Linear(state_dim,embed_dim)
        self.elu = nn.ELU()
        self.w3 = nn.Linear(state_dim,self.embed_dim)
        self.relu = nn.ReLU()
        self.w4 = nn.Linear(embed_dim,1)
    
    def forward(self,qs,global_state):
        w1 = torch.abs(self.w1(global_state))
        b1 = self.b1(global_state)
        w1 = w1.view(-1,self.num_agent,self.embed_dim)
        b1 = b1.view(-1,1,self.embed_dim)
        hidden = self.elu(torch.bmm(qs,w1)+b1)
        w2 = torch.abs(self.w2(global_state))
        w2 = w2.view(-1,self.embed_dim,1)
        v = self.w3(global_state)
        v = self.relu(v)
        v = self.w4(v)
        q_tot = torch.bmm(hidden,w2)+v
        return q_tot