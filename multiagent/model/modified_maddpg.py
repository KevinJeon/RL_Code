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
        return out

class Critic(nn.Module):
    def __init__(self, obs_size, all_action):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size + all_action, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64,64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 1))
    
    def forward(self, obs, action):
        obs = tr.cat([obs, action], dim=-1)
        return self.layers(obs)
        
class MADDPG(object):
    def __init__(self, obs_sizes, num_agent, num_actions, 
            batch_size, gamma=0.99, use_gpu=False):
        self.num_agent = num_agent
        self.num_actions = num_actions
        self.gamma = 0.95
        self.tau = 0.01
        self.scale = 0.1
        self.use_gpu = use_gpu
        self.lr = 1e-2
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
        # Check GPU
        print('Cuda Device : {}, GPU Count : {}'.format(tr.cuda.current_device(), tr.cuda.device_count()))
        # Load Models and Optimizer
        all_action = sum(num_actions)
        state_size = sum(obs_sizes)
        self.obs_ind = []
        self.act_ind = []
        ocum, acum = 0, 0
        for i, (obs_size, num_action) in enumerate(zip(obs_sizes, num_actions)):
            print('Agent {} Obs {} Act {}'.format(i, obs_size, num_action))
            self.policies[i] = Policy(obs_size, num_action)
            self.critics[i] = Critic(obs_size, all_action)
            
            self.policies_t[i] = Policy(obs_size, num_action)
            self.critics_t[i] = Critic(obs_size, all_action)
            
            self.policies_t[i].load_state_dict(self.policies[i].state_dict())
            self.critics_t[i].load_state_dict(self.critics[i].state_dict())
            self.policy_optims[i] = optim.Adam(self.policies[i].parameters(), lr=self.lr)
            self.critic_optims[i] = optim.Adam(self.critics[i].parameters(), lr=self.lr)
            
            self.memories[i] = OffpolicyMemory(agent_ind=i, capacity=50000) 
            obs_st_en = [ocum]
            act_st_en = [acum]
            obs_st_en.append(ocum + obs_size)
            act_st_en.append(acum + num_action)
            ocum = ocum + obs_size
            acum = acum + num_action
            self.obs_ind.append(obs_st_en)
            self.act_ind.append(act_st_en)
            if self.use_gpu:
                self.policies[i] = self.policies[i].cuda()
                self.critics[i] = self.critics[i].cuda()
                self.policies_t[i] = self.policies_t[i].cuda()
                self.critics_t[i] = self.critics_t[i].cuda()
        
    def act(self, _observations):
        pis = []
        for i in range(self.num_agent):
            obs = tr.from_numpy(_observations[i]).float()
            pi = self.policies[i](obs.cuda().detach())
            act = F.gumbel_softmax(pi.detach(), hard=True)
            pis.append(act.cpu().numpy())
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
    def one_hot(self, logit):
        onehot = (logit == logit.max(1, keepdim=True)[0]).float()
        return onehot
    def train(self, batch_size): 
        critic_losses, actor_losses = [], []
        for ind, (a, a_t, c, c_t, a_o, c_o) in enumerate(zip(self.policies, self.policies_t, self.critics, self.critics_t, self.policy_optims, self.critic_optims)):
            obss, acts, rews, obss_next, masks, maddpg_inputs = self.memories[ind].sample(batch_size)
            states = np.array([state[0] for state in maddpg_inputs])
            next_states = np.array([next_state[1] for next_state in maddpg_inputs])
            all_acts = np.array([all_act[2] for all_act in maddpg_inputs])
            states = tr.from_numpy(states).cuda().float()
            next_states = tr.from_numpy(next_states).cuda().float()
            all_acts = tr.from_numpy(all_acts).cuda().float()
            obss = tr.from_numpy(obss).cuda().float()
            acts = tr.from_numpy(acts).cuda().float()
            obss_next = tr.from_numpy(obss_next).cuda().float()
            masks = tr.from_numpy(masks).cuda().float().view(-1, 1)
            rews = tr.from_numpy(rews).cuda().float().view(-1, 1)
            q = c(obss, all_acts)
            acts_t = tr.cat([self.one_hot(pi(next_states[:, self.obs_ind[i][0]:self.obs_ind[i][1]].detach())) for i, pi in enumerate(self.policies_t)], dim=-1)
            q_t = c_t(obss_next, acts_t).detach()
            target = q_t * self.gamma * (1 - masks) + rews
            target = target.float()
            # Update Critic
            critic_loss = nn.MSELoss()(target, q)
            c_o.zero_grad()
            critic_loss.backward()
            tr.nn.utils.clip_grad_norm(c.parameters(), 0.5)
            c_o.step()
            
            # Update Policy
            pi = a(obss)
            pi_act = F.gumbel_softmax(pi, hard=True)
            all_acts[:, self.act_ind[ind][0]:self.act_ind[ind][1]] = pi_act
            actor_loss = (pi**2).mean() * 1e-3 - c(obss, all_acts).mean()
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
