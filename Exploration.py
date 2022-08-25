# This file includes different algorithms of exploration

from Environment import get_environment
from Agent import RandomAgent
from tqdm import tqdm
import visual

class RandomExploration:
    def __init__(self, args):
        self.env = get_environment(sim=args.sim, config_path=args.config_path)
        self.agent = RandomAgent(self.env, args)
        self.steps = args.steps
        self.resolution = args.resolution

    def start(self, topdown_map_path, seen_map_path):
        for i in tqdm(range(self.steps)):
            self.agent.take_action()

        visual.save_topdown_map(self.env, self.resolution, f'{topdown_map_path}.jpg')
        visual.save_seen_map(self.env, self.agent, self.resolution, f'{seen_map_path}{i}.jpg')



