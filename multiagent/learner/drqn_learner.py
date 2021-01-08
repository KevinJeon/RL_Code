from utils,replay_buffer import ReplayBuffer
from model.drqn import DRQN
import numpy as np
import torch.nn as nn
import torch

class DRQNLearner:
    def __init__(self,num_agent,num_action,num_obs,max_epi,max_epilen):
        self.num_agent = num_agent
        self.num_action = num_action
        self.num_obs = num_obs
        self.memory = ReplayBuffer(num_agent,num_action,num_obs,max_epi,max_epilen)
        self.q = [DRQN() for i in range(num_agent)]
        self.q_hat = [DRQN() for i in range(num_agent)]
        self.loss = nn.MSELoss()
        for i in range(num_agent):
            self.q_hat[i].load_state_dict(self.q[i].state_dict())

    def action(self):
        # is action selected from target q? -> no, from q

    def train(self,num_epi):
        actions,obss,dones,_,rewards = self.memory.get_episodes(num_epi)
        q_set = torch.empty(num_epi,self.max_epilen-1,self.num_agent,5)
        q_hat_set = torch.empty(num_epi,self.max_epilen-1,self.num_agent,5)
        for i in range(self.num_agent):
            action,obs,done,reward = actions[:,:,i,:],obss[:,:,i,:],dones[:,:,i,:],rewards[:,:,i,:]
            action = torch.tensor(action,dtype=torch.float).squeeze(2)
            obs = torch.tensor(obs,dtype=torch.float).squeeze(2)
            reward = torch.tensor(done,dtype=torch.float).squeeze(2)
            qs,hidden = self.q[i].forward(obs[:,:-1,:])
            # drqn gru hidden state role? -> like lstm            
            qs_hat = self.q_hat[i].forward(obs[:,1:,:])
            q_vals = torch.gather(qs,dim=2,action)
            q_hat_vals = torch.gather(qs_hat,dim=2,action)
            q_set[:,:,i,:] = q_vals
            q_hat_set[:,:,i,:] = q_hat_set
        q_target = reward + self.gamma * q_hat_set
        loss = self.mseloss(q_set,q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optmizer.step()
        self.memory.clear()
