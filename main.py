from distutils.command.config import config
from random import Random
from Exploration import RandomExploration
import matplotlib.pyplot as plt

# Selecting Hyper-parameters
sim = 'habitat'
config_path = 'configs/datasets/pointnav/gibson.yaml'
steps = 10

# Create an explorer
exp = RandomExploration(sim=sim, config_path=config_path, steps=steps)

# Start the exploration process
exp.start()
