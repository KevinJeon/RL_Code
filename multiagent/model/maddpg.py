import os
import numpy as np
import torch.nn.functional as F
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils.replay_buffer import OffpolicyMemory

class OU(object):
    def function(self, x, mu=0, theta=0.15, sigma=0.2):
        return theta * (mu -x) + sigma * np.random.randn(len(x))

class Policy(nn.Module):
    def __init__(self, obs_size, num_action):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(obs_size, 64), nn.ReLU(inplace=True), nn.Linear(64,64),nn.ReLU(inplace=True), nn.Linear(64, num_action))
    def forward(self, obs):
        out = self.layers(obs)
        u = tr.rand_like(out)
        pi = F.softmax(out - tr.log(-tr.log(u)), dim=-1)
        return out, pi

class Critic(nn.Module):
    def __init__(self, state_size, num_action, num_agent):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear((state_size + num_action) * num_agent, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64,64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 1))
    
    def forward(self, state, action):
        obs = tr.cat([state, action], dim=-1)
        return self.layers(obs)
        
class MADDPG(object):
    def __init__(self, obs_size, num_agent, num_action, 
            batch_size, gamma=0.99, cuda=False):
        self.num_agent = num_agent
        self.num_action = num_action
        self.gamma = 0.99
        self.tau = 0.01
        self.cuda = cuda
        self.lr = 5e-3
        self.batch_size = batch_size
        # Prediction Network
        self.policies = [None] * self.num_agent
        self.critics =  [None] * self.num_agent
        # Target Network 
        self.critics_t = [None] * self.num_agent
        self.policies_t = [None] * self.num_agent
        # Optimizers
        self.critic_optims = [None] * self.num_agent
        self.policy_optims = [None] * self.num_agent
        # Memories
        self.memories = [None] * self.num_agent
        self.ou = OU()
        # Load Models and Optimizer
        for i in range(num_agent):
            self.policies[i] = Policy(obs_size, num_action)
            self.critics[i] = Critic(obs_size, num_action, num_agent)
            
            self.policies_t[i] = Policy(obs_size, num_action)
            self.critics_t[i] = Critic(obs_size, num_action, num_agent)
            
            self.policies_t[i].load_state_dict(self.policies[i].state_dict())
            self.critics_t[i].load_state_dict(self.critics[i].state_dict())
            self.policy_optims[i] = optim.Adam(self.policies[i].parameters(), lr=self.lr)
            self.critic_optims[i] = optim.Adam(self.critics[i].parameters(), lr=self.lr)
            
            self.memories[i] = OffpolicyMemory(agent_ind=i, capacity=50000) 
            if self.cuda:
                self.policies[i] = self.policies[i].cuda()
                self.critics[i] = self.critics[i].cuda()
        
    def act(self, _observations):
        pis = []
        for i in range(self.num_agent):
            obs = tr.from_numpy(_observations[i]).float()
            _, pi = self.policies[i](obs)
            pis.append(pi.detach().numpy() + self.ou.function(pi.detach().numpy())*0.1)
        return pis

    def update(self):
        for src, trg in zip(self.policies, self.policies_t):
            param_names = list(src.state_dict().keys())
            src_params = src.state_dict()
            trg_params = trg.state_dict()
            for param in param_names:
                trg_params[param] = src_params[param] * self.tau + trg_params[param] * (1 - self.tau)
        for src, trg in zip(self.critics, self.critics_t):
            param_names = list(src.state_dict().keys())
            src_params = src.state_dict()
            trg_params = trg.state_dict()
            for param in param_names:
                trg_params[param] = src_params[param] * self.tau + trg_params[param] * (1 - self.tau)
    def train(self, batch_size): 
        critic_losses, actor_losses = [], []
        for ind, (a, a_t, c, c_t, a_o, c_o) in enumerate(zip(self.policies, self.policies_t, self.critics, self.critics_t, self.policy_optims, self.critic_optims)):
            obss, acts, rews, obss_next, masks, maddpg_inputs = self.memories[ind].sample(batch_size)
            states = np.array([state[0] for state in maddpg_inputs])
            next_states = np.array([next_state[1] for next_state in maddpg_inputs])
            all_acts = np.array([all_act[2] for all_act in maddpg_inputs])
            states = tr.from_numpy(states).float()
            next_states = tr.from_numpy(next_states).float()
            all_acts = tr.from_numpy(all_acts).float()
            obss = tr.from_numpy(obss).float()
            acts = tr.from_numpy(acts).float()
            obss_next = tr.from_numpy(obss_next).float()
            masks = tr.from_numpy(masks).float().view(-1, 1)
            rews = tr.from_numpy(rews).float().view(-1, 1)
            q = c(states, all_acts)
            acts_t = tr.cat([pi(obss_next)[1].detach() for pi in self.policies_t], dim=-1)
            q_t = c_t(next_states, acts_t).detach()
            target = q_t * self.gamma * (1 - masks) + rews
            target = target.float()
            # Update Critic
            critic_loss = nn.MSELoss()(target, q)
            c_o.zero_grad()
            critic_loss.backward()
            tr.nn.utils.clip_grad_norm(c.parameters(), 0.5)
            c_o.step()
            
            # Update Policy
            _, pi = a(obss)
            all_acts[:, self.num_action * ind:self.num_action * (ind + 1)] = pi
            actor_loss = tr.pow(pi, 2).mean() * 1e-3 - c(states, all_acts).mean()
            a_o.zero_grad()
            actor_loss.backward()
            tr.nn.utils.clip_grad_norm(a.parameters(), 0.5)
            a_o.step()
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
        self.update()
        return critic_losses, actor_losses 
    def save(self, save_dir, epoch):
        args = 'epoch{}_lr{}_tau{}_gamma_{}_bs{}'.format(epoch, self.lr, self.tau, self.gamma, self.batch_size)
        for ind, (policy, critic) in enumerate(zip(self.policies, self.critics)):
            tr.save(policy.state_dict(), save_dir + '/actor{}_{}.h5'.format(args, ind))
            tr.save(critic.state_dict(), save_dir + '/critic{}_{}.h5'.format(args, ind))
