import numpy as np

class ReplayBuffer(object):
    def __init__(self,num_agent,num_action,num_obs,max_epi,max_epilen):
        self.num_agent = num_agent
        self.num_action = num_action
        self.num_state = num_state
        self.max_epi = max_epi
        self.max_epilen = max_epilen
        self.action = np.empty((self.max_epi,self.max_epilen,self.num_agent,self.num_action))
        self.obs = np.empty((self.max_epi,self.max_epilen,self.num_agent,self.num_obs))
        self.done = np.empty((self.max_epi,self.max_epilen,self.num_agent,1))
        self.policy = np.empty((self.max_epi,self.max_epilen,self.num_agent,self.num_action))
        self.reward = np.empty((self.max_epi,self.max_epilen,self.num_agent,1))
        self.curr_step = 0
        self.curr_epi = 0
    def clear(self):
        self.action = np.empty((self.max_epi,self.max_epilen,self.num_agent,self.num_action))
        self.obs = np.empty((self.max_epi,self.max_epilen,self.num_agent,self.num_obs))
        self.done = np.empty((self.max_epi,self.max_epilen,self.num_agent,1))
        self.policy = np.empty((self.max_epi,self.max_epilen,self.num_agent,self.num_action))
        self.reward = np.empty((self.max_epi,self.max_epilen,self.num_agent,1))
    def add(self,obss,actions,policies,rewards,dones):
        for agent,obs,action,policy,reward,done in zip(range(self.num_agent),obss,actions,policies,rewards,dones):
            self.action[self.curr_epi,self.curr_step,agent,:] = action
            self.obs[self.curr_epi,self.curr_step,agent,:] = obs
            self.done[self.curr_epi,self.curr_step,agent,:] = done
            self.policy[self.curr_epi,self.curr_step,agent,:] = policy
            self.reward[self.curr_epi,self.curr_step,agent,:] = reward
        self.curr_step = self.curr_step + 1
    def next_epi(self):
        self.curr_epi = self.curr_epi + 1
        self.curr_step = 0
    def get_episodes(self,num_epi):
        # check whether empty is zero or None
        # need to reflect about random selection of episode
        episodes = np.random.choice(range(self.max_epi),num_epi,replace=False)[0]
        action = np.zeros((num_epi,self.max_epilen,self.num_agent,self.num_action))
        obs = np.zeros((num_epi,self.max_epilen,self.num_agent,self.num_obs))
        done = np.zeros((num_epi,self.max_epilen,self.num_agent,1))
        policy = np.zeros((num_epi,self.max_epilen,self.num_agent,self.num_action))
        reward = np.zeros((num_epi,self.max_epilen,self.num_agent,1))
        for i,epi in enumerate(episodes):
            action[i] = self.actions[epi]
            obs[i] = self.obs[epi]
            done[i] = self.done[epi]
            policy[i] = self.policy[epi]
            reward[i] = self.reward[epi]
        return action,obs,done,policy,reward
