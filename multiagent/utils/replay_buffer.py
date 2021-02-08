import numpy as np
import tensorflow as tf
class OffpolicyMemory(tf.Module):
    def __init__(self, num_agent, capacity, obs_size, num_process, num_step):
        super(OffpolicyMemory, self).__init__()
        self.obs_size = obs_size
        self.capcity = capacity
        self.num_agent = num_agent
        self._maxsize = int(size)
        self._next_ind = 0
        
        # Data 
        self.obs = tf.zeros(num_agent, num_step + 1, num_process, *obs_size)
        self.rews = tf.zeros(num_agent, num_step, num_process, 1)
        self.acts = tf.zeros(num_agent, num_step, num_process, num_action)
        self.next_obs = tf.zeros(num_agent, num_step + 1, num_process, *obs_size)
        self.mask = tf.zeros(num_agent, num_step, num_process, 1)
        
    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_ind = 0

    def add_batch(self, obs, act, rew, obs_next, mask):
        '''
        Input format : Tensor format

        '''
        self.obs[self.step + 1].copy_(obs)
        self.act[self.step].copy_(act)
        self.rew[self.step].copy_(rew)
        self.obs_next[self.step].copy_(obs_next)
        self.mask[self.step + 1].copy_(mask)
        if self._next_ind >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_ind] = data
        self._next_ind = (self._next_ind + 1) % self._maxsize
        self.step (self.step + 1) % self.num_step



    def sample(self,batch_size):
        '''
        Sampling Data for batch_size
        Input : batch_size (format : int)
        '''
        if batch_size > 0:
            indices = self.make_index(batch_size)
        else:
            indices = range(0, len(self._storage))
        return self._encode_sample(indices)
class OnpolicyMemory(tf.Module):
    def __init__(self, num_agent, obs_size, num_process, num_action):
        super(OnpolicyMemory, self).__init__()

