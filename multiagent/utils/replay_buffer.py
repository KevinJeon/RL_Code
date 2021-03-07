import numpy as np
import random
# Replay Memory is for 1 agent
class OffpolicyMemory(object):
    def __init__(self, agent_ind, capacity):
        self._maxsize = int(capacity)
        self._next_ind = 0
        self.agent_ind = agent_ind 
        # Data 
        self.storage = []

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self.storage = []
        self._next_ind = 0

    def add(self, obs, act, rew, obs_next, mask, model_input):
        '''
        Input format : o_t, a_t, r_t, o_{t+1}, T, (model_input: optional)

        '''
        data = (obs, act, rew, obs_next, mask, model_input)
        if self._next_ind >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self._next_ind] = data
        self._next_ind = (self._next_ind + 1) % self._maxsize
    
    def _encode_sample(self, ids):
        obss, acts, rews, obss_next, masks, model_inputs = [], [], [], [], [], []
        for ind in ids:
            data = self.storage[ind]
            obs, act, rew, obs_next, mask, model_input = data
            obss.append(np.array(obs, copy=False))
            acts.append(np.array(act, copy=False))
            rews.append(rew)
            obss_next.append(np.array(obs_next, copy=False))
            masks.append(mask)
            model_inputs.append(model_input)
        return np.array(obss), np.array(acts), np.array(rews), np.array(obss_next), np.array(masks), model_inputs

    def make_index(self, batch_size):
        return [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
    def make_latest_index(self, batch_size):
        ind = [(self._next_ind - 1 -i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(ind)
        return ind
    def sample_index(self, indices):
        return self._encode_sample(indices)
    def sample(self, batch_size):
        '''
        Sampling Data for batch_size
        Input : batch_size (format : int)
        '''
        if batch_size > 0:
            indices = self.make_index(batch_size)
        else:
            indices = range(0, len(self._storage))
        return self._encode_sample(indices)


