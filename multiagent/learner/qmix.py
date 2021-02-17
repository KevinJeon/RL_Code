import torch.nn as nn
from multiagent.model.qmix import Q_agent,QMix
import torch
import numpy as np
from torchviz import make_dot
import torch.onnx

class QMixLearner:
    '''
    q_mix 부분, 데이터 인풋 전처리 부분 해야 함 

    '''
    def __init__(self,obs_dim,num_agent,num_action,batch_size,writer):
        self.q = [Q_agent(obs_dim[i],num_action[i]) for i in range(num_agent)]
        self.q_target = [Q_agent(obs_dim[i],num_action[i]) for i in range(num_agent)]
        self.q_mix = QMix(num_agent,sum(obs_dim),32)
        self.q_mix_target = QMix(num_agent,sum(obs_dim),32)
        self.memory = ExperienceReplay(num_agent)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.RMSprop(params=self.q_mix.parameters(),lr=0.01)
        self.batch_size =batch_size
        self.num_agent = num_agent
        self.num_action = num_action
        self.count = 0
        self.gamma = 0.95
        self.epsilon = 0.9
        self.writer = writer
    def action(self,obs,h0):
        actions,hs = [],[]
        for i in range(self.num_agent):
            q_vals,h = self.q[i](torch.tensor(obs[i],dtype=torch.float32).view(1,-1),h0[i])
            hs.append(h)
            if np.random.random() > self.epsilon:
                action = q_vals.max(1)[1].item()
            else:
                action = np.random.randint(0,self.num_action[i]-1)
            self.memory.batch['action_idx'][i].append(action)
            action = np.eye(self.num_action[i])[action]
            actions.append(action)
            
        return actions,hs
    def init_hidden(self):
        hs = []
        for agent in range(self.num_agent):
            h = self.q[agent].init_hidden(1)
            hs.append(h)
        return hs
    def train(self,episode):
        q_vals = []
        for i in range(self.num_agent):
            global_states,obs1,rewards,actions,action_idx,done = self.memory.get_trajectories(i)
            h0 = self.q[i].init_hidden(len(obs1))
            q_i,h = self.q[i](obs1,h0)
            q_val = torch.gather(q_i[:-1,:],dim=1,index=action_idx.view(-1,1))
            q_vals.append(q_val)
        q_vals = torch.stack(q_vals,dim=1).transpose(1,2)
        q_target_vals = []
        for i in range(self.num_agent):
            global_states,obs,rewards,actions,action_idx,done = self.memory.get_trajectories(i)
            h0 = self.q_target[i].init_hidden(len(obs))
            q_i,h = self.q_target[i](obs,h0)
            q_val = torch.gather(q_i[:-1,:],dim=1,index=action_idx.view(-1,1))
            q_target_vals.append(q_val)
        q_target_vals = torch.stack(q_target_vals,dim=1).transpose(1,2)
        tot_q_vals = self.q_mix(q_vals,global_states[:-1])
        tot_q_max_vals = self.q_mix_target(q_target_vals,global_states[:-1])
        y_tot = rewards + self.gamma * (1-done)*tot_q_max_vals
        y_tot = y_tot.detach()
        loss = self.loss(tot_q_vals,y_tot)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # target update
        if self.count == 2:
            self._update_target()
            self.count = 0
        else:
            self.count += 1
        self.memory.clear()
        # log
        if episode == 0:
            torch.onnx.export(self.q[-1], (obs1,h0), "output1.onnx")
            torch.onnx.export(self.q_mix, (q_vals,global_states[:-1]), "output.onnx")
        # tensorboard
        self.writer.add_scalar('loss',loss,episode)
        
        #self.writer.add_graph(self.q_mix,(q_vals,global_states[:-1]))
        
    def _update_target(self):
        for i in range(self.num_agent):
            self.q_target[i].load_state_dict(self.q[i].state_dict())
        self.q_mix_target.load_state_dict(self.q_mix.state_dict())

        

class ExperienceReplay:
    def __init__(self,num_agent):
        self.num_agent = num_agent
        self.batch = dict(
            rewards=[[] for i in range(self.num_agent)],
            actions=[[] for i in range(self.num_agent)],
            action_idx=[[] for i in range(self.num_agent)],
            done=[[] for i in range(self.num_agent)],
            obs=[[] for i in range(self.num_agent)],
            global_states=[[] for i in range(self.num_agent)])
    def clear(self):
        self.batch = dict(
            rewards=[[] for i in range(self.num_agent)],
            actions=[[] for i in range(self.num_agent)],
            action_idx=[[] for i in range(self.num_agent)],
            done=[[] for i in range(self.num_agent)],
            obs=[[] for i in range(self.num_agent)],
            global_states=[[] for i in range(self.num_agent)])
    def get_trajectories(self,i):
        obs = torch.tensor(self.batch['obs'][i],dtype=torch.float32)
        global_states = torch.tensor(self.batch['global_states'][i],dtype=torch.float32)
        done = torch.tensor(self.batch['done'][i],dtype=torch.float32)
        rewards = torch.tensor(self.batch['rewards'][i],dtype=torch.float32)
        actions = torch.tensor(self.batch['actions'][i],dtype=torch.float32)
        action_idx = torch.tensor(self.batch['action_idx'][i])
        return global_states,obs,rewards,actions,action_idx,done