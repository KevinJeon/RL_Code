import torch.nn as nn
import torch as tr
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as trd
import numpy as np


## Since COMA operates with homogenuous agents, it runs in only simple_spread.

class Critic(nn.Module):
    def __init__(self, all_action, num_state, num_agent, cf_action):
        super(Critic,self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(num_state + all_action + obs_size + num_agent + cf_action, 64), 
                nn.ReLU(inplace=True), 
                nn.Linear(64, 64), 
                nn.ReLU(inplace=True), 
                nn.Linear(64, num_agent))  
    def forward(self, curr_u_cf, state, obs, agent_ind, prev_u_all):
        pair = tr.cat([u_cf, state, obs, agent_ind, prev_u_agent], dim=-1)
        return self.layers(pair)

class Actor(nn.Module):
    def __init__(self, obs_size, num_action, batch_size):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(num_action + obs_size, 64)
        self.gru = nn.GRUCell(64, 64)
        self.fc2 = nn.Linear(64, num_action)
        self.relu(inplace=True)
    def forward(self,x,h0):
        x = self.fc1(x)
        x = self.relu(x)
        h_in = h0.reshape(-1, 64)
        h = self.gru1(x,h_in)
        z = self.fc2(h)
        z = F.softmax(z)
        return x,h
    def init_hidden(self):
        return self.fc1.weight.new(1, 64).zero_()

class COMA(object):
    def __init__(self, obs_sizes, num_agent, num_actions, num_episode, batch_size, 
            lr=0.01, tau=0.01, gamma=0.95, epsilon=[0.2, 0.05], use_gpu=False):
        self.num_agent = num_agent
        self.num_actions = num_actions
        self.num_episode = num_episode
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.policies_t = [None] * self.num_agent
        self.policies = [None] * self.num_agent
        self.start_epsilon = epsilon[0]
        self.end_epsilon = epsilon[1]
        self.critic = Critic(sum(num_actions), sum(obs_sizes), num_agent)
        self.critic_t = Critic(sum(num_actions), sum(obs_sizes), num_agent)
        
        self.policy_optims = [None] * self.num_agent
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.memories = OffpolicyMemory(0, 1e6)

        for i, (obs_size, num_action) in enumerate(zip(obs_sizes, num_actions)):
            self.policies[i] = Actor(obs_size, num_action, batch_size)
            self.policies_t[i] = Actor(obs_size, num_action, batch_size)
            self.policy_optims[i] = optim.Adam(self.policies[i].parameters(), lr=self.lr)

            if self.use_gpu:
                self.policies[i] = self.policies[i].cuda()
                self.poicieies_t[i] = self.policies_t[i].cuda()
        if self.use_gpu:
            self.critic = self.critic.cuda()
            self.critic_t = self.critic_t.cuda()
    def one_hot(self, logit):
        return (logit == logit.max(1, keepdim=True[0]).float()
    def epsilon_decay(self):
        self.epsilon -= (self.start_epsilon - self.end_epsilon) / self.num_episode
    def act(self, _observations, hidden_state):
        pis = []
        hs = []
        us = []
        for i, (obs, policy) in enumerate(zip(_observations, self.policies)):
            obs = tr.from_numpy(obs).float()
            ## need ti detach hidden state?
            pi, h = self.policy(obs.cuda().detach(), hidden_state)
            u = (1 - self.epsilon) * pi + self.epsilon * self.num_action
            print('Eps pi', u)
            u = Categorical(u).sample().long()
            print('Sampled', u)
            pis.append(pi)
            hs.append(h)
            us.append(u)
        return us, pis, hs
    def train(self, batch_size):
        actor_losses = []
        
        # Critic Train

        states, all_acts, rews, next_states, masks, coma_inputs = self.memories.sample(batch_size)

        states = tr.from_numpy(states).float()
        next_states = tr.from_numpy(next_states).float()
        all_acts = tr.from_numpy(all_acts).float()
        rews = tr.from_numpy(rews).float()
        masks = tr.from_numpy(masks).float()
        if self.use_cuda:
            states = states.cuda()
            next_states = next_states.cuda()
            all_acts = all_acts.cuda()
            rws = rews.cuda()
            masks = masks.cuda()


