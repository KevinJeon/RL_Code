import torch as tr
import torch.nn as nn
import torch.optim as optim

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
        self.critic_optims = [None] * self.num_agent
        # Memories
        self.memories = [None] * self.num_agent
        # Load Models and Optimizer
        for i in range(num_agent):
            self.policies[i] = Policy(obs_size, num_action)
            self.criticis[i] = Critic(obs_size, num_action)
            
            self.policies_t[i] = Policy(obs_size, num_action)
            self.criticis_t[i] = Critic(obs_size, num_action)

            self.policy_optims[i] = optim.Adam(self.policies[i].parameters(), lr=0.001)
            self.critic_optims[i] = optim.Adam(self.critics[i].parameters(), lr=0.001)
            
            self.memories[i] = OffpolicyMemory(agent_ind=i, capacity=50000) 
            if self.cuda:
                self.policies[i] = self.policies[i].cuda()
                self.critics[i] = self.critics[i].cuda()
        
    def act(self, observations):
        pis = []
        for i in range(self.num_agent)
            pi = self.policies_t[i](observations[i])
            self.pis.append(pi.detach().numpy())
        return pis
    def update(self):
        for a, a_t, c, c_t in zip(self.policies, self.policies_t, self.critics, self.critics_t):
            srcs = [a.parameters(), c.paramters()]
            trgs =  [a_t.parameters(), c_t.parameters()]
            for src, trg in zip(srcs, trgs):
                trg = self.tau * trg + (1 - self.tau) * src
    def train(self):
        for ind, (a, a_t, c, c_t, a_o, c_o) in enumerate(zip(self.policies, self.policies_t, self.critics, self.critics_t, self.policy_optims, self.critic_optims)):
            obss, acts, rews, obss_next, masks = memory.sample()
            obss = tr.from_numpy(obss, dtype=tr.float)
            acts = tr.from_numpy(acts, dtype=tr.float)
            obss_next = tr.from_numpy(obss_next, dtype=tr.float)
            q = c(obss, acts)
            q_t = c_t(obss_next, a_t(obss_next).detach())
            target = q_t * self.gamma * mask + rews
            
            # Update Critic
            critic_loss = F.mse_loss(target, q)
            c_o.zero_grad()
            critic_loss.backward()
            c_o.step()
            
            # Update Policy


            
