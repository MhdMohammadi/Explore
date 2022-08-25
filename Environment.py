import habitat
from habitat.utils.visualizations import maps

import matplotlib.pyplot as plt
import os

# Single-tone pattern is used to avoid "kernel dying" problem
main_env = None

# This function is creating an environment based on the given inputs
# This function is assuming that habitat-lab and project's main folder are in the same directory
def get_environment(sim=None, config_path=None):
    global main_env
    if sim == 'habitat':
        if main_env is not None:
            main_env.close()
        os.chdir('../habitat-lab')
        main_env = habitat.Env(config=habitat.get_config(config_path))
        main_env.reset()
        os.chdir('../Explore')
    return main_env

