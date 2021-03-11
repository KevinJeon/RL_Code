import os, sys, argparse
sys.path.append('../')
import numpy as np
import torch as tr
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser('MARL EVAL Params')
    parser.add_argument('--num_episode', '-e', default=500, type=int)
    parser.add_argument('--env_name', '-en', default='simple_spread', type=str)
    parser.add_argument('--max_step', '-ms', default=25, type=int)
    parser.add_argument('--load_dir', '-l', default='./save', type=str)
    return parser.parse_args()


def main(args):
    scenario = scenarios.load(args.env_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    models = get_model(args.model)
    for model, ldir in zip(models, args.load_dir):
        ckpt = tr.load(ldir)
        model.policy.load_state_dict(ckpt)
    for episode in range(args.num_episode):
        obss = env.reset()
        epi_rews = 0
        for step in range(args.max_step):
            acts = models.act(obss)
            obss_next, rews, masks, _ = env.step(acts)
            obss = obss_next
            epi_rews += rews[0]
        print(epi_rews)
        env.redner()


if __name__ == '__main__':
    args = parse_args()
    main(args) 

