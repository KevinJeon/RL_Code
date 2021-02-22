import torch as tr
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_size, num_action):
        super(self, Policy).__init__()
        self.layers = nn.Sequential(
                nn.Linear(obs_size, 64), nn.ReLU(inplace=True), nn.Linear(64,64),nn.ReLU(inplace=True), nn.Linear(64, num_action))
    def forward(self, obs):
        return self.layers(obs)

class Critic(nn.Module):
    def __init__(self, state_size, num_action):
        super(self, Critic).__init__()
        self.layers = nn.Sequential(
                nn.Linear(state_size + num_action, 64), nn.ReLU(inplace=True), nn.Linear(64,64),nn.ReLU(inplace=True), nn.Linear(64, 1))
    def forward(self, state, action):
        state_action = tr.cat([state, action], axis=-1)
        return self.layers(obs)

class MADDPG(object):
    def __init__(self, num_agent, num_action):
        self.num_agent = num_agent
        self.num_action = num_action
        # Predict model
        self.policies = [None] * self.num_agent
        self.critics = [None] * self.num_agent
        # Target model
        self.policies_t = [None] * self.num_agent
        self.critics_t =  [None] * self.num_agent 
        for i in range(num_agent):
            self.policies[i] = Policy(obs_size, num_action)
            self.criticis[i] = Critic(obs_size, num_action)
            
            self.policies_t[i] = Policy(obs_size, num_action)
            self.criticis_t[i] = Critic(obs_size, num_action)
            if self.cuda:
                self.policies[i] = self.policies[i].cuda()
                self.critics[i] = self.critics[i].cuda()
