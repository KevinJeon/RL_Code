import argparse
import os, sys
sys.path.append('../')
from environment import MultiAgentEnv
import scenarios as scenarios
from model.maddpg import MADDPG
from utils.replay_buffer import OffpolicyMemory
import numpy as np


class RandomPolicy:
    def __init__(self, agent_index, num_agent, batch_size):
        self.agent_index = agent_index
        self.batch_size = batch_size
        self.memory = OffpolicyMemory(agent_index, 50000)
    def act(self, obs):
        return np.array([1, 0, 0, 0, 0, 0, 0])
    def train(self):
        print('-'*10+'TRAIN'+'-'*10)
        obss, acts, rews, obss_next, masks = \
            self.memory.sample(self.batch_size) 
        return None

def parse_args():
    parser = argparse.ArgumentParser('MARL Params')
    parser.add_argument('--num_episode', '-e', default=1000, type=int)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--env_name', '-en', default='simple_spread', type=str)
    parser.add_argument('--max_step', '-ms', default=1000, type=int)
    return parser.parse_args()

def get_trainer(obs_size, num_agent, num_action, trainer_name='maddpg'):
    if trainer_name == 'maddpg':
        trainer = MADDPG(obs_size, num_agent, num_action)
    return trainer

def main(args):
    scenario = scenarios.load(args.env_name+'.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    #env.render()
    trainer = get_trainer(env.observation_space[0].shape[0], env.n, 7, 'maddpg')
    for episode in range(args.num_episode):
        print('-'*10+'EPISODE START'+'-'*10)
        obss = env.reset()
        step = 0
        while True:
            acts = trainer.act(obss)
            step += 1
            obss_next, rews, masks, _ = env.step(acts)
            for i, memory in enumerate(trainer.memories):
                '''
                obss : (18, )
                acts : (7, )
                rews : float
                mask : bool
                '''
                memory.add(obss[i], acts[i], rews[i], obss_next[i], masks[i])
            obss = obss_next
            if (step % args.batch_size) == 0:
                 
                print('-'*10+'TRAIN START'+'-'*10)
                trainer.train(args.batch_size)
            if all(masks):
                break
            if step > args.max_step:
                break
            #env.render()
            



if __name__ == '__main__':
    args = parse_args()
    main(args)
