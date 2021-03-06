import argparse
import os, sys
sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument('--save_dir', '-s', default='./save', type=str)
    parser.add_argument('--save_freq', '-sf', default=100, type=int)
    parser.add_argument('--log_dir', '-l', default='./logs', type=str)
    return parser.parse_args()

def get_trainer(obs_size, num_agent, num_action, batch_size, trainer_name='maddpg'):
    if trainer_name == 'maddpg':
        trainer = MADDPG(obs_size, num_agent, num_action, batch_size=batch_size,) 
    return trainer

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    scenario = scenarios.load(args.env_name+'.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    #env.render()
    writer = SummaryWriter(args.log_dir)
    trainer = get_trainer(env.observation_space[0].shape[0], env.n, 7, args.batch_size, 'maddpg')
    for episode in range(args.num_episode):
        a_losses,c_losses = [[] for _ in range(env.n)], [[] for _ in range(env.n)]
        total_reward = [0] * env.n
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
                total_reward[i] += rews[i]
            obss = obss_next
            if (step % args.batch_size) == 0:
                 
                critic_losses, actor_losses = trainer.train(args.batch_size)
                for i, (closs, aloss) in enumerate(zip(critic_losses, actor_losses)):
                    a_losses[i].append(aloss)
                    c_losses[i].append(closs)
            if all(masks):
                break
            if step > args.max_step:
                break
            env.render()
        for i in range(env.n):
            writer.add_scalar('agent{}/actor_loss'.format(i), sum(a_losses[i])/len(a_losses[i]), episode)
            writer.add_scalar('agent{}/critic_loss'.format(i), sum(c_losses[i])/len(c_losses[i]), episode)
            writer.add_scalar('agent{}/total_reward'.format(i), total_reward[i], episode)
        if episode % args.save_freq == 0:
            trainer.save(args.save_dir)
    writer.close()
            



if __name__ == '__main__':
    args = parse_args()
    main(args)
