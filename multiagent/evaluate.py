import os, sys, argparse, time
sys.path.append('../')
import numpy as np
import torch as tr
import torch.nn.functional as F
import scenarios as scenarios
from environment import MultiAgentEnv
from model.maddpg import MADDPG, Policy

class ZeroShotAgents(object):
    def __init__(self, model_types, load_dirs, obs_sizes, num_action, file_name, agent1, agent2):
        self.models = []
        self.file_name = file_name
        self.num_agent = len(model_types)
        self.load_dirs = self._load_path(load_dirs, agent1, agent2)
        self.model_types = []
        self.num_action = num_action
        for model, pth, obs_size in zip(model_types, self.load_dirs, obs_sizes):
            if model == 'maddpg':
                agent = Policy(obs_size, num_action).cuda()
                agent.load_state_dict(tr.load(pth))
                agent.eval()
            self.models.append(agent)
            self.model_types.append(model)
    def act(self, _observations):
        acts = []
        for i in range(self.num_agent):
            obs = tr.from_numpy(_observations[i]).float()
            pi = self.models[i](obs.cuda().detach())
            action = tr.zeros(self.num_action)
            action[tr.argmax(pi)] = 1
            acts.append(action.cpu().numpy())
        return acts
    def _load_path(self, load_dirs, agent1, agent2):
        real_dirs = []
        chosen = [agent1] * 1  + [agent2] * 1 + [agent1]
        print('-'*10+'{} CHOSEN'.format(chosen)+'-'*10)
        for i, (agent, pth) in enumerate(zip(chosen, load_dirs)):
            mpth = pth + '/{}/{}.h5'.format(agent, self.file_name+str(i))
            real_dirs.append(mpth)
        return real_dirs
def parse_args():
    parser = argparse.ArgumentParser('MARL EVAL Params')
    parser.add_argument('--num_episode', '-e', default=10, type=int)
    parser.add_argument('--env_name', '-en', default='simple_spread', type=str)
    parser.add_argument('--max_step', '-ms', default=25, type=int)
    parser.add_argument('--base_dir', '-l', default=['./zeroshot', './zeroshot', './zeroshot'], type=str)
    parser.add_argument('--file_name', '-f', default='.h5', type=str)
    parser.add_argument('--model_types', '-m', default=['maddpg', 'maddpg', 'maddpg'], type=list)
    return parser.parse_args()

def zero_shot(env, models, num_episode, max_step=25):
    total_avg_rews = []
    for episode in range(num_episode):
        obss = env.reset()
        epi_rews = 0
        for step in range(max_step):
            acts = models.act(obss)
            obss_next, rews, masks, _ = env.step(acts)
            obss = obss_next
            epi_rews += rews[0]
        epi_rews /= args.max_step
        total_avg_rews.append(epi_rews)
    print('Finish!!! Reward : {:.3f}'.format(sum(total_avg_rews)/len(total_avg_rews)))
    return sum(total_avg_rews) / len(total_avg_rews)
def main(args):
    scenario = scenarios.load(args.env_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # Consider the obs size since some algo get states instead observations
    obs_sizes = [env.observation_space[0].shape[0] for _ in range(env.n)]
    agents = os.listdir(args.base_dir[0])
    jd = dict()
    for agent1 in sorted(agents):
        jd[agent1] = dict()
        for agent2 in sorted(agents):
            models = ZeroShotAgents(args.model_types, args.base_dir, obs_sizes, 7, args.file_name, agent1, agent2)
            avg_rews = zero_shot(env, models, args.num_episode, args.max_step)
            jd[agent1][agent2] = avg_rews
    import json
    with open('result.json', 'w') as f:
        json.dump(jd, f)
if __name__ == '__main__':
    args = parse_args()
    main(args) 

