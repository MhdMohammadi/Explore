# This file includes different algorithms of exploration

from operator import ge, imod
from Environment import get_environment
from Agent import RandomAgent
from tqdm import tqdm

class RandomExploration:
    def __init__(self, sim=None, config_path=None, steps=None):
        self.env = get_environment(sim=sim, config_path=config_path)
        self.agent = RandomAgent(self.env)
        self.steps = steps
    
    def start(self):
        for i in tqdm(range(self.steps)):
            self.agent.take_action()

