import argparse
import os, sys
sys.path.append('../')
from environment import MultiAgentEnv
import scenarios as scenarios
from utils.replay_buffer import OffpolicyMemory
import numpy as np


class RandomPolicy:
    def __init__(self, agent_index, num_agent):
        self.agent_index = agent_index
        self.memory = OffpolicyMemory(agent_index, 50000)
    def act(self, obs):
        return np.array([1, 0, 0, 0, 0, 0, 0])
def parse_args():
    parser = argparse.ArgumentParser('MARL Params')
    parser.add_argument('--num_episode', '-e', default=1000, type=int)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--env_name', '-en', default='simple_spread', type=str)
    return parser.parse_args()


def main(args):
    scenario = scenarios.load(args.env_name+'.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    #env.render()
    policies = [RandomPolicy(i, env.n) for i in range(env.n)]
    for episode in range(args.num_episode):
        obss = env.reset()
        while True:
            acts = []
            for i, policy in enumerate(policies):
                acts.append(policy.act(obss[i]))
            print(acts)
            obss_next, rews, masks, _ = env.step(acts)
            for i, policy in enumerate(policies):
                policy.memory.add(obss[i], acts[i], rews[i], obss_next[i], masks[i])
            #env.render()
            



if __name__ == '__main__':
    args = parse_args()
    main(args)
