import torch
import torch.distributions as F
from torch.distributions import Categorical
from multiagent.model.coma import Critic,Actor
import torch.nn as nn
import numpy as np
class COMA:
    '''
    implementation of Counterfactual Multi-Agent Policy Gradients
    '''
    def __init__(self,num_agent,batch_size,num_action,critic_states,actor_states):
        self.central_critic = Critic(num_action,critic_states)
        self.critic_target= Critic(num_action,critic_states)
        self.critic_target.load_state_dict(self.central_critic.state_dict())
        self.decentral_actors = [Actor(num_action,actor_states,batch_size) for actor in range(num_agent)]
        self.memory = Experience_Replay(num_agent)
        self.batch_size = batch_size
        self.actor_states = actor_states
        self.num_action = num_action
        self.epsilon = 0.9
        self.num_agent = num_agent
        self.gamma = 0.9
        self.actors_optim = [torch.optim.Adam(self.decentral_actors[i].parameters(), lr=0.001) for i in range(num_agent)]
        self.critic_optim = torch.optim.Adam(self.central_critic.parameters(), lr=0.001)
    def action(self,local_obs,prev_agent_actions):
        actions = []
        for i in range(len(local_obs)):
            actor_states = self.actor_process(local_obs[i],prev_agent_actions[i])
            hidden_states = self.decentral_actors[i].init_hidden()
            actor_states = torch.tensor(actor_states).float()
            pi,_ = self.decentral_actors[i](actor_states,hidden_states)
            pi = ((1 - self.epsilon) * pi + torch.ones_like(pi) * self.epsilon/self.num_action)
            action = Categorical(pi).sample()
            self.memory.actions_idx[i].append(action.numpy())
            action = torch.eye(5)[action].view(-1)
            action = action.numpy()

            self.memory.pis[i].append(pi)
        actions.append(action)
        return actions
    def train(self):
        for i in range(self.num_agent):
            curr_exc_agent_actions,global_states,local_obs,prev_actions,pis,rewards,done,actions_idx = \
                self.memory.get_samples(i)
            critic_states = self.critic_process(curr_exc_agent_actions,global_states,local_obs,prev_actions)
            # train actors
            q_fix = self.critic_target(critic_states).detach()
            print(type(q_fix),type(np.array(actions_idx).reshape(-1,1)))
            q = torch.gather(q_fix,dim=1,index=actions_idx.reshape(-1,1))
            # counterfactual baseline
            cb =  torch.sum(pis[i]*q_fix)
            log_pi = torch.log(pis[i])
            advantage = q - cb
            actor_loss = torch.mean(advantage*log_pi)
            self.actors_optim[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decentral_actors[i].parameters(), 5)
            self.actors_optim[i].step()
            # train critic
            qs = self.central_critic(critic_states)
            agent_actions = self.memory.actions[i]
            tds = torch.zeros_like(qs)
            for t in range(len(tds)):
                if t == len(tds)-1:
                    tds[t] = rewards[t][i]
                else:
                    tds[t] = rewards[t][i] + self.gamma * q[t+1]## to fix of step t+1
            print(tds[0],qs[0])
            critic_loss = torch.mean((tds - qs) ** 2)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.central_critic.parameters(), 5)
            self.critic_optim.step()
        self.memory.clear()
    def actor_process(self,local_obs,prev_agent_action):
        return np.reshape(np.concatenate([local_obs,prev_agent_action]),(-1,self.actor_states))
    def critic_process(self,curr_exc_agent_actions,global_states,local_obs,prev_actions):
        if curr_exc_agent_actions == None:
            return torch.cat((global_states,local_obs,prev_actions),1)
        return torch.cat((curr_exc_agent_actions,global_states,local_obs,prev_actions),1)
class Experience_Replay:
    def __init__(self,num_agent):
        '''
        curr_exc_agent_actions,global_states,local_obs,prev_actions,pis
        '''
        self.actions = []
        self.local_obs = []
        self.rewards = []
        self.pis = [[] for i in range(num_agent)]
        self.done = []
        self.num_agent = num_agent
        self.actions_idx = [[] for i in range(num_agent)]
    def clear(self):
        self.actions = []
        self.local_obs = []
        self.rewards = []
        self.pis = [[] for i in range(self.num_agent)]
        self.done = []
        self.actions_idx = [[] for i in range(self.num_agent)]
    def get_samples(self,agent_idx):
        self.actions = np.array(self.actions)
        self.local_obs = np.array(self.local_obs)
        self.rewards = np.array(self.rewards)
        self.done = np.array(self.done)
        curr_exc_agent_actions = np.delete(self.actions[1:,:],(agent_idx),axis=1)
        if self.num_agent == 1:
            curr_exc_agent_actions = None
        else:
            curr_exc_agent_actions = torch.tensor(curr_exc_agent_actions).view(-1,len(self.actions[0][0])-1).float()
        global_states = np.reshape(self.local_obs[1:,],(-1,self.num_agent*len(self.local_obs[0][0])))
        global_states = torch.tensor(global_states).float()
        local_obs = torch.tensor(self.local_obs[1:,agent_idx]).float()
        prev_actions = torch.tensor(self.actions[:-1,:]).view(-1,len(self.actions[0][0])).float()
        pis = self.pis[agent_idx]
        rewards = torch.tensor(self.rewards)
        actions_idx = torch.tensor(self.actions_idx[agent_idx][1:])
        print(global_states.size(),local_obs.size(),prev_actions.size(),actions_idx.size())
        return curr_exc_agent_actions,global_states,local_obs,prev_actions,pis,rewards,self.done,actions_idx


