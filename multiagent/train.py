import argparse
import os, sys
from datetime import datetime
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
        self.memory = OffpolicyMemory(agent_index, 1e6)
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
    parser.add_argument('--max_step', '-ms', default=100, type=int)
    parser.add_argument('--save_dir', '-s', default='./save', type=str)
    parser.add_argument('--save_freq', '-sf', default=100, type=int)
    parser.add_argument('--log_dir', '-l', default='./logs', type=str)
    return parser.parse_args()

def get_trainer(obs_size, num_agent, num_action, batch_size, trainer_name='maddpg'):
    if trainer_name == 'maddpg':
        trainer = MADDPG(obs_size, num_agent, num_action, batch_size=batch_size,) 
    return trainer

def main(args):
    st_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
        if not os.path.exists(args.log_dir+'/{}'.format(st_time)):
            os.mkdir(args.log_dir+'/{}'.format(st_time))
    scenario = scenarios.load(args.env_name+'.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    #env.render()
    total_step = 0
    writer = SummaryWriter(args.log_dir+'/{}'.format(st_time))
    trainer = get_trainer(env.observation_space[0].shape[0], env.n, 7, args.batch_size, 'maddpg')
    for episode in range(args.num_episode):
        a_losses,c_losses = [[] for _ in range(env.n)], [[] for _ in range(env.n)]
        total_reward = [0] * env.n
        obss = env.reset()
        for step in range(args.max_step):
            acts = trainer.act(obss)
            total_step += 1
            obss_next, rews, masks, _ = env.step(acts)
            for i, memory in enumerate(trainer.memories):
                '''
                obss : (18, )
                acts : (7, )
                rews : float
                mask : bool
                '''
                state = np.reshape(obss, (-1))
                next_state = np.reshape(obss_next, (-1))
                all_acts = np.reshape(acts, (-1))
                model_inputs = (state, next_state, all_acts)
                memory.add(obss[i], acts[i], rews[i], obss_next[i], masks[i], model_inputs)
                total_reward[i] += rews[i]
            obss = obss_next
            
            if (len(trainer.memories[0]) >= args.batch_size) and (total_step % 100 == 0):
                critic_losses, actor_losses = trainer.train(args.batch_size)
                for i, (closs, aloss) in enumerate(zip(critic_losses, actor_losses)):
                    a_losses[i].append(aloss)
                    c_losses[i].append(closs)
                    writer.add_scalar('agent{}/actor_loss'.format(i), aloss, total_step)
                    writer.add_scalar('agent{}/critic_loss'.format(i), closs, total_step)
                continue
            env.render()
        print('-'*10+'EPISODE END! REWARD :{} STEP : {}'.format(total_reward[0], total_step)+'-'*10)
        writer.add_scalar('Total_reward', total_reward[i], episode)
        if episode % args.save_freq == 0:
            trainer.save(args.save_dir, episode)
    writer.close()
            



if __name__ == '__main__':
    args = parse_args()
    main(args)
