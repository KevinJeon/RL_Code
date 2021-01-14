import numpy as np

class ReplayBuffer(object):
    def __init__(self,num_agent,num_action,num_obs,max_epi,max_epilen):
        self._storage = []
        self._maxsize = int(size)
        self._next_ind = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_ind = 0

    def add(self, obs, act, rew, obs_next, done):
        data = (obs, act, rew, obs_next, done)

        if self._next_ind >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_ind] = data
        self._next_ind = (self._next_ind + 1) % self._maxsize

    def _encode_sample(self, indices):
        obss, acts, rews, obss_next, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            obs, act, rew, obs_next, done = data
            obss.append(np.array(obs,copy=False))
            obss_next.append(np.array(obs_next,copy=False))
            acts.append(np.array(act,copy=False))
            rews.append(rew)
            dones.append(done)
        return np.array(obss), np.array(acts), np.array(rews), \
            np.array(obss_next), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0,len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        ind = [(self._next_ind - 1 -i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(ind)
        return ind

    def sample_index(self,indices):
        return self._encode_sample(indices)

    def sample(self,batch_size):
        if batch_size > 0:
            indices = self.make_index(batch_size)
        else:
            indices = range(0, len(self._storage))
        return self._encode_sample(indices)

    def collect(self):
        return self.sample(-1)
        
